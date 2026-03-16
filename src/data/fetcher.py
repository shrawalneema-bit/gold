"""
Live gold price data fetcher using yfinance.
Gold futures ticker: GC=F
Gold ETF (SPDR): GLD
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


GOLD_TICKER    = "GC=F"       # Gold Futures (most accurate spot-like price)
GOLD_ETF       = "GLD"        # SPDR Gold Shares ETF (highly liquid proxy)
DXY_TICKER     = "DX-Y.NYB"  # US Dollar Index (inverse correlation with gold)
SPY_TICKER     = "SPY"        # S&P 500 ETF (risk-on/off signal)
TLT_TICKER     = "TLT"        # 20yr Treasury ETF (safe-haven signal)
VIX_TICKER     = "^VIX"       # CBOE Volatility Index (fear → gold rises)
SILVER_TICKER  = "SI=F"       # Silver Futures (gold/silver ratio signal)
USDINR_TICKER  = "USDINR=X"  # USD/INR exchange rate (for MCX price conversion)
NIFTY_TICKER   = "^NSEI"     # Nifty 50 (Indian equity market)

# MCX gold is priced in INR per 10 grams
# COMEX price is in USD per troy oz (1 troy oz = 31.1035 g)
# MCX approx = COMEX_USD × USDINR × (10 / 31.1035)
MCX_CONVERSION = 10 / 31.1035


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
    Columns: DXY_Close, SPY_Close, TLT_Close, VIX_Close, Silver_Close, Gold_Silver_Ratio
    """
    tickers = {
        "DXY":    DXY_TICKER,
        "SPY":    SPY_TICKER,
        "TLT":    TLT_TICKER,
        "VIX":    VIX_TICKER,
        "Silver": SILVER_TICKER,
        "USDINR": USDINR_TICKER,
        "Nifty":  NIFTY_TICKER,
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

    # Derived: Gold/Silver ratio
    if "Silver_Close" in combined.columns:
        combined["Gold_Silver_Ratio"] = combined["Close"] / combined["Silver_Close"].replace(0, float("nan"))

    # Derived: Approximate MCX price (INR per 10g)
    if "USDINR_Close" in combined.columns:
        combined["MCX_Approx"] = combined["Close"] * combined["USDINR_Close"] * MCX_CONVERSION

    return combined


def fetch_india_context() -> dict:
    """
    Fetch Indian market snapshot: USD/INR rate and MCX approximate gold price.
    Returns a dict for display in the dashboard.
    """
    result = {"usdinr": None, "mcx_approx": None, "nifty": None}
    try:
        inr_info = yf.Ticker(USDINR_TICKER).fast_info
        usdinr   = getattr(inr_info, "last_price", None)
        if usdinr:
            result["usdinr"]    = round(usdinr, 2)
            # Get latest COMEX price
            gold_info = yf.Ticker(GOLD_TICKER).fast_info
            comex = getattr(gold_info, "last_price", None)
            if comex:
                result["mcx_approx"] = round(comex * usdinr * MCX_CONVERSION, 0)
    except Exception:
        pass
    try:
        nifty_info   = yf.Ticker(NIFTY_TICKER).fast_info
        result["nifty"] = getattr(nifty_info, "last_price", None)
    except Exception:
        pass
    return result
