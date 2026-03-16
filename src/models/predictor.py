"""
Gold price predictor.

Model: Gradient Boosting Regressor + Classifier (sklearn) trained on:
  - Technical indicators
  - Lag features (past N days of key signals)
  - Rolling volatility / momentum statistics
  - Macro context (DXY, SPY, TLT, VIX, Silver, Oil, Gold/Silver ratio)
  - News sentiment

Predicts next-day closing price (regression) and direction (classification).
Also provides 80% prediction interval via quantile regression.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score
from typing import Optional


MODEL_DIR        = os.path.join(os.path.dirname(__file__), "saved")
PRICE_MODEL_PATH = os.path.join(MODEL_DIR, "price_model.joblib")
DIR_MODEL_PATH   = os.path.join(MODEL_DIR, "direction_model.joblib")

# Base features (technical indicators + macro)
BASE_FEATURE_COLS = [
    # Trend
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    # Momentum
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "Stoch_K", "Stoch_D", "Williams_R",
    # Volatility
    "BB_Width", "ATR_14",
    # Price positions
    "Price_vs_SMA20", "Price_vs_SMA50",
    # Returns
    "Return_1d", "Return_5d", "Return_20d",
    # Macro
    "DXY_Close", "SPY_Close", "TLT_Close", "VIX_Close",
    "Silver_Close", "Gold_Silver_Ratio",
    "Oil_Close",        # Crude oil — inflation proxy, inverse-correlated with gold in risk-on
    # Sentiment
    "sentiment",
]

# Lag configuration: (column, [lag_days])
LAG_CONFIG = [
    ("Return_1d",  [1, 2, 3, 5]),
    ("RSI_14",     [1, 2, 3]),
    ("MACD_Hist",  [1, 2]),
    ("ATR_14",     [1, 3]),
    ("VIX_Close",  [1, 2]),
    ("sentiment",  [1, 2]),
    ("Oil_Close",  [1, 2]),   # oil momentum signal
]

# Minimum training rows needed for TimeSeriesSplit(n_splits=5)
_MIN_ROWS = 80


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged versions of key columns and rolling statistics."""
    df = df.copy()
    for col, lags in LAG_CONFIG:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling volatility (short, medium, long windows)
    close = df["Close"]
    pct = close.pct_change()          # pandas 2.2+ compatible (no fill_method)
    df["Rolling_vol_5"]  = pct.rolling(5).std()
    df["Rolling_vol_10"] = pct.rolling(10).std()
    df["Rolling_vol_20"] = pct.rolling(20).std()

    # Price momentum
    df["Momentum_5"]  = close / close.shift(5) - 1
    df["Momentum_10"] = close / close.shift(10) - 1

    # DXY momentum (dollar strengthening = gold headwind)
    if "DXY_Close" in df.columns:
        df["DXY_Return_5"] = df["DXY_Close"].pct_change(5)   # pandas 2.2+ compatible

    # VIX change (rising fear = gold tailwind)
    if "VIX_Close" in df.columns:
        df["VIX_Change_1"] = df["VIX_Close"].pct_change(1)   # pandas 2.2+ compatible

    # Oil momentum (rising oil = inflationary, mixed gold signal)
    if "Oil_Close" in df.columns:
        df["Oil_Return_5"] = df["Oil_Close"].pct_change(5)

    return df


def _build_feature_cols(df: pd.DataFrame) -> list[str]:
    """Build the full list of feature columns that actually exist in df."""
    lag_cols     = [f"{col}_lag{lag}" for col, lags in LAG_CONFIG for lag in lags]
    rolling_cols = [
        "Rolling_vol_5", "Rolling_vol_10", "Rolling_vol_20",
        "Momentum_5", "Momentum_10",
        "DXY_Return_5", "VIX_Change_1", "Oil_Return_5",
    ]
    all_cols = BASE_FEATURE_COLS + lag_cols + rolling_cols
    return [c for c in all_cols if c in df.columns]


def _prepare_features(df: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """Build feature matrix from indicator-enriched + lag-enriched DataFrame."""
    df = df.copy()

    # Attach sentiment
    if sentiment_series is not None:
        s = sentiment_series.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # pandas 2.2+ compatible: reindex then forward-fill separately
        df["sentiment"] = s.reindex(df.index).ffill()

    df = _add_lag_features(df)

    feature_cols = _build_feature_cols(df)

    # Fill missing optional columns with 0
    for col in BASE_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.ffill(inplace=True)
    X.fillna(0, inplace=True)
    return X


def _build_targets(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return df["Close"].shift(-horizon)


def _make_price_pipeline(n_estimators: int = 400) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
    ])


def _make_quantile_pipeline(alpha: float, n_estimators: int = 200) -> Pipeline:
    """Quantile regressor for prediction-interval bound (low or high)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            loss="quantile", alpha=alpha,
            n_estimators=n_estimators, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
    ])


def train(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    horizon: int = 1,
    save: bool = True,
) -> dict:
    """
    Train an ensemble of Gradient Boosting + Random Forest models.
    Uses time-series cross-validation to avoid data leakage.
    Also trains quantile regressors (α=0.1 and α=0.9) for 80% prediction intervals.
    """
    X       = _prepare_features(df, sentiment_series)
    y_price = _build_targets(df, horizon)
    y_dir   = (y_price > df["Close"]).astype(int)

    mask  = y_price.notna()
    X, y_price, y_dir = X[mask], y_price[mask], y_dir[mask]

    valid = X.notna().all(axis=1)
    X, y_price, y_dir = X[valid], y_price[valid], y_dir[valid]

    if len(X) < _MIN_ROWS:
        raise ValueError(
            f"Not enough data — need at least {_MIN_ROWS} clean rows "
            f"(got {len(X)}). Try a longer time period."
        )

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    # Ensure test fold is large enough for reliable metrics
    if len(test_idx) < 10:
        train_idx, test_idx = splits[-2]  # fall back to second-to-last fold

    X_tr, X_te   = X.iloc[train_idx], X.iloc[test_idx]
    yp_tr, yp_te = y_price.iloc[train_idx], y_price.iloc[test_idx]
    yd_tr, yd_te = y_dir.iloc[train_idx], y_dir.iloc[test_idx]

    # ── Price regression: GBM + RF ensemble ──────────────────────────────────
    gbm_price = _make_price_pipeline(n_estimators=400)
    rf_price  = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        )),
    ])
    gbm_price.fit(X_tr, yp_tr)
    rf_price.fit(X_tr, yp_tr)

    y_pred_price = 0.6 * gbm_price.predict(X_te) + 0.4 * rf_price.predict(X_te)
    mape = mean_absolute_percentage_error(yp_te, y_pred_price)

    # ── Quantile bounds (80% prediction interval) ─────────────────────────────
    gbm_low  = _make_quantile_pipeline(alpha=0.10)
    gbm_high = _make_quantile_pipeline(alpha=0.90)
    gbm_low.fit(X_tr, yp_tr)
    gbm_high.fit(X_tr, yp_tr)

    # ── Direction classifier: GBM + RF ensemble ───────────────────────────────
    gbm_dir = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.04,
            subsample=0.8, random_state=42,
        )),
    ])
    rf_dir = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        )),
    ])
    gbm_dir.fit(X_tr, yd_tr)
    rf_dir.fit(X_tr, yd_tr)

    proba_blend = (
        0.6 * gbm_dir.predict_proba(X_te) +
        0.4 * rf_dir.predict_proba(X_te)
    )
    y_pred_dir = (proba_blend[:, 1] >= 0.5).astype(int)
    dir_acc = accuracy_score(yd_te, y_pred_dir)

    # Retrain on full dataset for final models
    gbm_price.fit(X, y_price)
    rf_price.fit(X, y_price)
    gbm_low.fit(X, y_price)
    gbm_high.fit(X, y_price)
    gbm_dir.fit(X, y_dir)
    rf_dir.fit(X, y_dir)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(
            {"gbm": gbm_price, "rf": rf_price, "gbm_low": gbm_low, "gbm_high": gbm_high},
            PRICE_MODEL_PATH,
        )
        joblib.dump({"gbm": gbm_dir, "rf": rf_dir}, DIR_MODEL_PATH)

    # Feature importance from GBM (primary model)
    feat_imp = dict(zip(
        X.columns,
        gbm_price.named_steps["model"].feature_importances_,
    ))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    return {
        "price_mape":          round(mape * 100, 2),
        "direction_accuracy":  round(dir_acc * 100, 2),
        "feature_importances": feat_imp,
        "train_samples":       len(X),
        "test_samples":        len(test_idx),
        "feature_count":       len(X.columns),
    }


def predict(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    horizon: int = 1,
) -> dict:
    """Generate next-bar prediction using saved ensemble models."""
    if not (os.path.exists(PRICE_MODEL_PATH) and os.path.exists(DIR_MODEL_PATH)):
        train(df, sentiment_series, horizon=horizon, save=True)

    price_models = joblib.load(PRICE_MODEL_PATH)
    dir_models   = joblib.load(DIR_MODEL_PATH)

    X        = _prepare_features(df, sentiment_series)
    X_latest = X.iloc[[-1]]

    pred_price = (
        0.6 * price_models["gbm"].predict(X_latest)[0] +
        0.4 * price_models["rf"].predict(X_latest)[0]
    )

    # Quantile bounds (80% prediction interval) — graceful fallback for old models
    pred_low  = price_models["gbm_low"].predict(X_latest)[0]  if "gbm_low"  in price_models else None
    pred_high = price_models["gbm_high"].predict(X_latest)[0] if "gbm_high" in price_models else None

    proba = (
        0.6 * dir_models["gbm"].predict_proba(X_latest)[0] +
        0.4 * dir_models["rf"].predict_proba(X_latest)[0]
    )
    pred_dir = int(proba[1] >= 0.5)

    current_price   = float(df["Close"].iloc[-1])
    expected_change = (pred_price - current_price) / current_price * 100

    result = {
        "current_price":       round(current_price, 2),
        "predicted_price":     round(float(pred_price), 2),
        "predicted_direction": "Up" if pred_dir == 1 else "Down",
        "up_probability":      round(float(proba[1]) * 100, 1),
        "down_probability":    round(float(proba[0]) * 100, 1),
        "expected_change_pct": round(expected_change, 2),
        "horizon_days":        horizon,
    }
    if pred_low is not None:
        result["predicted_price_low"]  = round(float(pred_low),  2)
        result["predicted_price_high"] = round(float(pred_high), 2)
    return result


def retrain_and_predict(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    horizon: int = 1,
) -> dict:
    """Retrain on latest data and return a fresh prediction."""
    metrics    = train(df, sentiment_series, horizon=horizon, save=True)
    prediction = predict(df, sentiment_series, horizon=horizon)
    return {**prediction, **metrics}
