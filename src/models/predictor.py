"""
Gold price predictor.

Model: Gradient Boosting Regressor (sklearn) trained on technical indicators
       + macro context + rolling sentiment.

Predicts next-day closing price (regression) and direction (classification).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score
from typing import Optional


MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
PRICE_MODEL_PATH = os.path.join(MODEL_DIR, "price_model.joblib")
DIR_MODEL_PATH   = os.path.join(MODEL_DIR, "direction_model.joblib")

FEATURE_COLS = [
    # Trend
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    # Momentum
    "RSI_14", "MACD", "MACD_Hist", "Stoch_K", "Stoch_D", "Williams_R",
    # Volatility
    "BB_Width", "ATR_14",
    # Price positions
    "Price_vs_SMA20", "Price_vs_SMA50",
    # Returns
    "Return_1d", "Return_5d", "Return_20d",
    # Macro (optional, filled with 0 if missing)
    "DXY_Close", "SPY_Close", "TLT_Close",
    # Sentiment (optional)
    "sentiment",
]


def _prepare_features(df: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Build feature matrix from an indicator-enriched DataFrame.
    Missing optional columns are filled with 0.
    """
    df = df.copy()

    # Attach sentiment if provided
    if sentiment_series is not None:
        sentiment_series = sentiment_series.copy()
        sentiment_series.index = pd.to_datetime(sentiment_series.index).tz_localize(None)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["sentiment"] = sentiment_series.reindex(df.index, method="ffill")

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLS].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.ffill(inplace=True)
    X.fillna(0, inplace=True)
    return X


def _build_targets(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Next-day close price shifted back `horizon` periods."""
    return df["Close"].shift(-horizon)


def train(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    horizon: int = 1,
    save: bool = True,
) -> dict:
    """
    Train price-regression and direction-classification models.

    Args:
        df:               DataFrame with OHLCV + technical indicators
        sentiment_series: optional daily sentiment scores (pd.Series indexed by date)
        horizon:          prediction horizon in bars (default 1 = next day)
        save:             persist models to disk

    Returns:
        dict with keys: price_mape, direction_accuracy, feature_importances
    """
    X = _prepare_features(df, sentiment_series)
    y_price = _build_targets(df, horizon)
    y_dir   = (y_price > df["Close"]).astype(int)   # 1 = up, 0 = down/flat

    # Drop rows where target is NaN (last `horizon` rows)
    mask = y_price.notna()
    X, y_price, y_dir = X[mask], y_price[mask], y_dir[mask]

    # Remove initial rows with NaN features (indicator warm-up)
    valid = X.notna().all(axis=1)
    X, y_price, y_dir = X[valid], y_price[valid], y_dir[valid]

    if len(X) < 60:
        raise ValueError("Not enough data to train — need at least 60 clean rows.")

    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    split_idx = list(tscv.split(X))
    train_idx, test_idx = split_idx[-1]   # use last fold for evaluation

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_price_train, y_price_test = y_price.iloc[train_idx], y_price.iloc[test_idx]
    y_dir_train, y_dir_test     = y_dir.iloc[train_idx], y_dir.iloc[test_idx]

    # ── Price regression ───────────────────────────────────────────────────────
    price_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])
    price_pipe.fit(X_train, y_price_train)
    y_pred_price = price_pipe.predict(X_test)
    mape = mean_absolute_percentage_error(y_price_test, y_pred_price)

    # ── Direction classifier ───────────────────────────────────────────────────
    dir_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])
    dir_pipe.fit(X_train, y_dir_train)
    y_pred_dir = dir_pipe.predict(X_test)
    dir_acc = accuracy_score(y_dir_test, y_pred_dir)

    # Retrain on full dataset
    price_pipe.fit(X, y_price)
    dir_pipe.fit(X, y_dir)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(price_pipe, PRICE_MODEL_PATH)
        joblib.dump(dir_pipe,   DIR_MODEL_PATH)

    # Feature importance
    feat_imp = dict(zip(
        FEATURE_COLS,
        price_pipe.named_steps["model"].feature_importances_,
    ))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    return {
        "price_mape":          round(mape * 100, 2),
        "direction_accuracy":  round(dir_acc * 100, 2),
        "feature_importances": feat_imp,
        "train_samples":       len(X),
    }


def predict(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    horizon: int = 1,
) -> dict:
    """
    Generate a prediction for the next bar using the saved models.
    If no saved models exist, trains on the fly first.

    Returns:
        predicted_price, predicted_direction ('Up'/'Down'),
        direction_probability, current_price, expected_change_pct
    """
    # Load or train models
    if not (os.path.exists(PRICE_MODEL_PATH) and os.path.exists(DIR_MODEL_PATH)):
        train(df, sentiment_series, horizon=horizon, save=True)

    price_pipe = joblib.load(PRICE_MODEL_PATH)
    dir_pipe   = joblib.load(DIR_MODEL_PATH)

    X = _prepare_features(df, sentiment_series)
    X_latest = X.iloc[[-1]]   # last row = latest bar

    pred_price = float(price_pipe.predict(X_latest)[0])
    pred_dir   = int(dir_pipe.predict(X_latest)[0])
    dir_proba  = dir_pipe.predict_proba(X_latest)[0]

    current_price     = float(df["Close"].iloc[-1])
    expected_change   = (pred_price - current_price) / current_price * 100

    return {
        "current_price":        round(current_price, 2),
        "predicted_price":      round(pred_price, 2),
        "predicted_direction":  "Up" if pred_dir == 1 else "Down",
        "up_probability":       round(float(dir_proba[1]) * 100, 1),
        "down_probability":     round(float(dir_proba[0]) * 100, 1),
        "expected_change_pct":  round(expected_change, 2),
        "horizon_days":         horizon,
    }


def retrain_and_predict(
    df: pd.DataFrame,
    sentiment_series: Optional[pd.Series] = None,
    horizon: int = 1,
) -> dict:
    """Convenience: retrain on latest data and return fresh prediction."""
    metrics = train(df, sentiment_series, horizon=horizon, save=True)
    prediction = predict(df, sentiment_series, horizon=horizon)
    return {**prediction, **metrics}
