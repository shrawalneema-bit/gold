"""
Technical analysis indicators for gold price prediction.
Uses the `ta` library for indicator computation.
"""

import pandas as pd
import numpy as np
import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a gold OHLCV DataFrame with a comprehensive set of technical indicators.
    All new columns are appended; original OHLCV columns are preserved.

    Indicators added
    ----------------
    Trend  : SMA_20, SMA_50, SMA_200, EMA_12, EMA_26
    Momentum : RSI_14, MACD, MACD_Signal, MACD_Hist, Stoch_K, Stoch_D, Williams_R
    Volatility : BB_High, BB_Low, BB_Mid, BB_Width, ATR_14
    Volume : OBV, VWAP (approximate daily)
    Custom : Price_vs_SMA20, Price_vs_SMA50, Golden_Cross, Death_Cross
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["SMA_20"]  = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["EMA_12"]  = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"]  = ta.trend.ema_indicator(close, window=26)

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["RSI_14"]     = ta.momentum.rsi(close, window=14)

    macd_obj         = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]       = macd_obj.macd()
    df["MACD_Signal"]= macd_obj.macd_signal()
    df["MACD_Hist"]  = macd_obj.macd_diff()

    stoch            = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["Stoch_K"]    = stoch.stoch()
    df["Stoch_D"]    = stoch.stoch_signal()

    df["Williams_R"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # ── Volatility ─────────────────────────────────────────────────────────────
    bb               = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_High"]    = bb.bollinger_hband()
    df["BB_Low"]     = bb.bollinger_lband()
    df["BB_Mid"]     = bb.bollinger_mavg()
    df["BB_Width"]   = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]

    df["ATR_14"]     = ta.volatility.average_true_range(high, low, close, window=14)

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["OBV"]        = ta.volume.on_balance_volume(close, vol)

    # ── Custom signals ─────────────────────────────────────────────────────────
    df["Price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]
    df["Price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"]

    # Golden Cross (SMA50 crosses above SMA200) / Death Cross (opposite)
    df["Golden_Cross"] = ((df["SMA_50"] > df["SMA_200"]) &
                          (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))).astype(int)
    df["Death_Cross"]  = ((df["SMA_50"] < df["SMA_200"]) &
                          (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))).astype(int)

    # ── Returns ────────────────────────────────────────────────────────────────
    df["Return_1d"]  = close.pct_change(1)
    df["Return_5d"]  = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)

    return df


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Interpret the latest row of indicators into human-readable signals.

    Returns a dict with keys:
        trend, momentum, volatility, overall, details
    """
    latest = df.iloc[-1]
    signals = []
    score   = 0   # +1 bullish, -1 bearish per signal

    # Trend signals
    if latest["Close"] > latest.get("SMA_50", float("nan")):
        signals.append(("Trend", "Bullish", "Price above SMA-50"))
        score += 1
    else:
        signals.append(("Trend", "Bearish", "Price below SMA-50"))
        score -= 1

    if latest.get("SMA_50", 0) > latest.get("SMA_200", 0):
        signals.append(("Trend", "Bullish", "SMA-50 above SMA-200 (golden region)"))
        score += 1
    else:
        signals.append(("Trend", "Bearish", "SMA-50 below SMA-200 (death region)"))
        score -= 1

    # RSI
    rsi = latest.get("RSI_14", 50)
    if rsi < 30:
        signals.append(("Momentum", "Oversold", f"RSI {rsi:.1f} — potential reversal up"))
        score += 1
    elif rsi > 70:
        signals.append(("Momentum", "Overbought", f"RSI {rsi:.1f} — potential reversal down"))
        score -= 1
    else:
        signals.append(("Momentum", "Neutral", f"RSI {rsi:.1f}"))

    # MACD
    if latest.get("MACD", 0) > latest.get("MACD_Signal", 0):
        signals.append(("Momentum", "Bullish", "MACD above signal line"))
        score += 1
    else:
        signals.append(("Momentum", "Bearish", "MACD below signal line"))
        score -= 1

    # Bollinger Bands
    if latest["Close"] > latest.get("BB_High", float("inf")):
        signals.append(("Volatility", "Overbought", "Price above upper Bollinger Band"))
        score -= 1
    elif latest["Close"] < latest.get("BB_Low", 0):
        signals.append(("Volatility", "Oversold", "Price below lower Bollinger Band"))
        score += 1
    else:
        signals.append(("Volatility", "Normal", "Price within Bollinger Bands"))

    if score >= 2:
        overall = "Bullish"
    elif score <= -2:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return {
        "overall": overall,
        "score":   score,
        "signals": signals,
        "rsi":     rsi,
        "macd":    latest.get("MACD", None),
        "macd_signal": latest.get("MACD_Signal", None),
        "bb_width": latest.get("BB_Width", None),
        "atr":     latest.get("ATR_14", None),
    }
