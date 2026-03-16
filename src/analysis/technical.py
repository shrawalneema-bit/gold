"""
Technical analysis indicators for gold price prediction.
Pure numpy/pandas — no external ta library required.
"""

import pandas as pd
import numpy as np


# ── Individual indicator functions ────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = ema(series, fast)
    ema_slow    = ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist        = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, window=20, num_std=2):
    mid  = sma(series, window)
    std  = series.rolling(window).std()
    high = mid + num_std * std
    low  = mid - num_std * std
    return high, mid, low


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window=14, smooth=3):
    lowest  = low.rolling(window).min()
    highest = high.rolling(window).max()
    denom   = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / denom
    d = k.rolling(smooth).mean()
    return k, d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window=14) -> pd.Series:
    highest = high.rolling(window).max()
    lowest  = low.rolling(window).min()
    denom   = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - close) / denom


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Commodity Channel Index — measures deviation from average price.
    Values above +100 indicate overbought; below -100 indicate oversold.
    Particularly effective for cyclical commodities like gold.
    """
    tp       = (high + low + close) / 3          # typical price
    sma_tp   = tp.rolling(window).mean()
    mean_dev = tp.rolling(window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    denom = (0.015 * mean_dev).replace(0, np.nan)
    return (tp - sma_tp) / denom


# ── Main enrichment function ───────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a gold OHLCV DataFrame with technical indicators.
    Original OHLCV columns are preserved; indicators are appended.
    """
    df = df.copy()
    c  = df["Close"]
    h  = df["High"]
    l  = df["Low"]
    v  = df.get("Volume", pd.Series(0, index=df.index))

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["SMA_20"]  = sma(c, 20)
    df["SMA_50"]  = sma(c, 50)
    df["SMA_200"] = sma(c, 200)
    df["EMA_12"]  = ema(c, 12)
    df["EMA_26"]  = ema(c, 26)

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["RSI_14"]     = rsi(c, 14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(c)
    df["Stoch_K"], df["Stoch_D"] = stochastic(h, l, c)
    df["Williams_R"] = williams_r(h, l, c)
    df["CCI_20"]     = cci(h, l, c, 20)    # Commodity Channel Index

    # ── Volatility ─────────────────────────────────────────────────────────────
    df["BB_High"], df["BB_Mid"], df["BB_Low"] = bollinger_bands(c)
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"].replace(0, np.nan)
    df["ATR_14"]   = atr(h, l, c)

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["OBV"] = obv(c, v)

    # ── Custom signals ─────────────────────────────────────────────────────────
    df["Price_vs_SMA20"] = (c - df["SMA_20"]) / df["SMA_20"].replace(0, np.nan)
    df["Price_vs_SMA50"] = (c - df["SMA_50"]) / df["SMA_50"].replace(0, np.nan)

    df["Golden_Cross"] = (
        (df["SMA_50"] > df["SMA_200"]) &
        (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
    ).astype(int)
    df["Death_Cross"] = (
        (df["SMA_50"] < df["SMA_200"]) &
        (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))
    ).astype(int)

    # ── Returns ────────────────────────────────────────────────────────────────
    df["Return_1d"]  = c.pct_change(1)
    df["Return_5d"]  = c.pct_change(5)
    df["Return_20d"] = c.pct_change(20)

    return df


# ── Signal interpretation ──────────────────────────────────────────────────────

def _is_valid(val) -> bool:
    """Return True if val is a finite, non-NaN number usable in comparisons."""
    try:
        return pd.notna(val) and np.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Interpret the latest row of indicators into human-readable signals.
    Returns dict with keys: overall, score, signals, rsi, macd, etc.
    NaN/missing values are handled gracefully — no invalid comparisons.
    """
    latest  = df.iloc[-1]
    signals = []
    score   = 0

    close  = latest.get("Close")
    sma50  = latest.get("SMA_50")
    sma200 = latest.get("SMA_200")

    # Trend: price vs SMA-50
    if _is_valid(close) and _is_valid(sma50):
        if close > sma50:
            signals.append(("Trend", "Bullish", "Price above SMA-50"))
            score += 1
        else:
            signals.append(("Trend", "Bearish", "Price below SMA-50"))
            score -= 1
    else:
        signals.append(("Trend", "Neutral", "SMA-50 unavailable (insufficient data)"))

    # Trend: golden/death cross region
    if _is_valid(sma50) and _is_valid(sma200):
        if sma50 > sma200:
            signals.append(("Trend", "Bullish", "SMA-50 above SMA-200 (golden region)"))
            score += 1
        else:
            signals.append(("Trend", "Bearish", "SMA-50 below SMA-200 (death region)"))
            score -= 1
    else:
        signals.append(("Trend", "Neutral", "SMA-200 unavailable (need 200+ bars)"))

    # RSI
    r = latest.get("RSI_14", 50)
    r = r if _is_valid(r) else 50.0
    if r < 30:
        signals.append(("Momentum", "Oversold", f"RSI {r:.1f} — potential reversal up"))
        score += 1
    elif r > 70:
        signals.append(("Momentum", "Overbought", f"RSI {r:.1f} — potential reversal down"))
        score -= 1
    else:
        signals.append(("Momentum", "Neutral", f"RSI {r:.1f}"))

    # MACD
    macd_val = latest.get("MACD", 0)
    macd_sig = latest.get("MACD_Signal", 0)
    if _is_valid(macd_val) and _is_valid(macd_sig):
        if macd_val > macd_sig:
            signals.append(("Momentum", "Bullish", "MACD above signal line"))
            score += 1
        else:
            signals.append(("Momentum", "Bearish", "MACD below signal line"))
            score -= 1
    else:
        signals.append(("Momentum", "Neutral", "MACD unavailable"))

    # CCI
    cci_val = latest.get("CCI_20")
    if _is_valid(cci_val):
        if cci_val > 100:
            signals.append(("Momentum", "Overbought", f"CCI {cci_val:.0f} — above +100 threshold"))
            score -= 1
        elif cci_val < -100:
            signals.append(("Momentum", "Oversold", f"CCI {cci_val:.0f} — below -100 threshold"))
            score += 1
        else:
            signals.append(("Momentum", "Neutral", f"CCI {cci_val:.0f} — within normal range"))

    # Bollinger Bands
    bb_hi = latest.get("BB_High")
    bb_lo = latest.get("BB_Low")
    if _is_valid(close) and _is_valid(bb_hi) and _is_valid(bb_lo):
        if close > bb_hi:
            signals.append(("Volatility", "Overbought", "Price above upper Bollinger Band"))
            score -= 1
        elif close < bb_lo:
            signals.append(("Volatility", "Oversold", "Price below lower Bollinger Band"))
            score += 1
        else:
            signals.append(("Volatility", "Normal", "Price within Bollinger Bands"))
    else:
        signals.append(("Volatility", "Neutral", "Bollinger Bands unavailable"))

    overall = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"

    return {
        "overall":     overall,
        "score":       score,
        "signals":     signals,
        "rsi":         float(r),
        "cci":         float(cci_val) if _is_valid(cci_val) else None,
        "macd":        latest.get("MACD"),
        "macd_signal": latest.get("MACD_Signal"),
        "bb_width":    latest.get("BB_Width"),
        "atr":         latest.get("ATR_14"),
    }
