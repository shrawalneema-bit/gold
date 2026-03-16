"""
Live gold price data fetcher using yfinance.
Gold futures ticker: GC=F
Gold ETF (SPDR): GLD
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


GOLD_TICKER = "GC=F"       # Gold Futures (most accurate spot-like price)
GOLD_ETF    = "GLD"        # SPDR Gold Shares ETF (highly liquid proxy)
DXY_TICKER  = "DX-Y.NYB"  # US Dollar Index (inverse correlation with gold)
SPY_TICKER  = "SPY"        # S&P 500 ETF (risk-on/off signal)
TLT_TICKER  = "TLT"        # 20yr Treasury ETF (safe-haven signal)


def fetch_gold_ohlcv(
    period: str = "1y",
    interval: str = "1d",
    ticker: str = GOLD_TICKER,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for gold.

    Args:
        period:   yfinance period string  e.g. '1d','5d','1mo','3mo','6mo','1y','2y','5y'
        interval: bar interval            e.g. '1m','5m','15m','1h','1d','1wk','1mo'
        ticker:   which symbol to use

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period}, interval={interval})")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)
    return df


def fetch_gold_realtime() -> dict:
    """Return the latest available quote for gold futures."""
    tk = yf.Ticker(GOLD_TICKER)
    info = tk.fast_info
    return {
        "ticker":        GOLD_TICKER,
        "last_price":    getattr(info, "last_price", None),
        "previous_close":getattr(info, "previous_close", None),
        "day_high":      getattr(info, "day_high", None),
        "day_low":       getattr(info, "day_low", None),
        "currency":      getattr(info, "currency", "USD"),
        "timestamp":     datetime.utcnow().isoformat(),
    }


def fetch_macro_context(period: str = "1y") -> pd.DataFrame:
    """
    Fetch macro context indicators aligned to gold dates.
    Columns: DXY_Close, SPY_Close, TLT_Close
    """
    tickers = {
        "DXY": DXY_TICKER,
        "SPY": SPY_TICKER,
        "TLT": TLT_TICKER,
    }
    frames = {}
    for name, sym in tickers.items():
        try:
            df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
            if not df.empty:
                frames[f"{name}_Close"] = df["Close"]
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro


def fetch_combined(period: str = "1y") -> pd.DataFrame:
    """
    Fetch gold OHLCV + macro context in one aligned DataFrame.
    """
    gold = fetch_gold_ohlcv(period=period)
    macro = fetch_macro_context(period=period)

    # strip timezone for alignment
    gold.index = gold.index.tz_localize(None) if gold.index.tz else gold.index
    macro.index = macro.index.tz_localize(None) if macro.index.tz else macro.index

    combined = gold.join(macro, how="left")
    combined.ffill(inplace=True)
    return combined
