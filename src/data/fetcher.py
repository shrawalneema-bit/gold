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
OIL_TICKER     = "CL=F"       # Crude Oil Futures — inflation proxy, mixed gold signal
VIX_TICKER     = "VIXY"       # ProShares VIX ETF — more reliable than ^VIX on cloud
VIX_FALLBACKS  = ["^VIX", "VIXM"]
SILVER_TICKERS = ["SLV", "SIVR", "SI=F"]  # Silver fallback chain
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
        "ticker":         GOLD_TICKER,
        "last_price":     getattr(info, "last_price", None),
        "previous_close": getattr(info, "previous_close", None),
        "day_high":       getattr(info, "day_high", None),
        "day_low":        getattr(info, "day_low", None),
        "currency":       getattr(info, "currency", "USD"),
        "timestamp":      datetime.utcnow().isoformat(),
    }


def fetch_macro_context(period: str = "1y") -> pd.DataFrame:
    """
    Fetch macro context indicators aligned to gold dates.
    Columns: DXY_Close, SPY_Close, TLT_Close, VIX_Close, Silver_Close,
             Oil_Close, Gold_Silver_Ratio, USDINR_Close, Nifty_Close
    """
    tickers = {
        "DXY":    DXY_TICKER,
        "SPY":    SPY_TICKER,
        "TLT":    TLT_TICKER,
        "Silver": SILVER_TICKERS[0],  # primary; fallbacks tried below
        "Oil":    OIL_TICKER,         # Crude oil futures
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

    # VIX — try primary ETF then index fallbacks
    for vix_sym in [VIX_TICKER] + VIX_FALLBACKS:
        try:
            df = yf.Ticker(vix_sym).history(period=period, interval="1d", auto_adjust=True)
            if not df.empty:
                frames["VIX_Close"] = df["Close"]
                break
        except Exception:
            continue

    # Silver — try fallback chain if primary fails
    if "Silver_Close" not in frames:
        for ag_sym in SILVER_TICKERS[1:]:
            try:
                df = yf.Ticker(ag_sym).history(period=period, interval="1d", auto_adjust=True)
                if not df.empty:
                    frames["Silver_Close"] = df["Close"]
                    break
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro


def fetch_combined(period: str = "1y") -> pd.DataFrame:
    """
    Fetch gold OHLCV + macro context in one aligned DataFrame.
    Falls back to gold-only if macro fetch fails entirely.
    """
    gold = fetch_gold_ohlcv(period=period)

    try:
        macro = fetch_macro_context(period=period)
    except Exception:
        macro = pd.DataFrame()

    # Strip timezone for alignment
    gold.index = gold.index.tz_localize(None) if gold.index.tz else gold.index

    if macro.empty:
        # Return gold-only DataFrame — no macro context, but dashboard still works
        return gold

    macro.index = macro.index.tz_localize(None) if macro.index.tz else macro.index

    combined = gold.join(macro, how="left")
    combined.ffill(inplace=True)

    # Derived: Gold/Silver ratio
    if "Silver_Close" in combined.columns:
        silver = combined["Silver_Close"].replace(0, float("nan"))
        combined["Gold_Silver_Ratio"] = combined["Close"] / silver
        combined["Gold_Silver_Ratio"] = combined["Gold_Silver_Ratio"].replace(
            [float("inf"), float("-inf")], float("nan")
        )

    # Derived: Gold/Oil ratio (high ratio = gold outperforming, risk-off signal)
    if "Oil_Close" in combined.columns:
        oil = combined["Oil_Close"].replace(0, float("nan"))
        combined["Gold_Oil_Ratio"] = combined["Close"] / oil
        combined["Gold_Oil_Ratio"] = combined["Gold_Oil_Ratio"].replace(
            [float("inf"), float("-inf")], float("nan")
        )

    # Derived: Approximate MCX price (INR per 10g)
    if "USDINR_Close" in combined.columns:
        combined["MCX_Approx"] = combined["Close"] * combined["USDINR_Close"] * MCX_CONVERSION

    return combined


def _safe_last_price(ticker: str) -> Optional[float]:
    """Return last close price for a ticker, trying fast_info then history fallback."""
    try:
        val = getattr(yf.Ticker(ticker).fast_info, "last_price", None)
        if val and float(val) > 0:
            return float(val)
    except Exception:
        pass
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_india_context() -> dict:
    """
    Fetch Indian market snapshot: USD/INR, MCX approximate gold price, Nifty, VIX.
    """
    result = {"usdinr": None, "mcx_approx": None, "nifty": None, "vix": None, "oil": None}

    usdinr = _safe_last_price(USDINR_TICKER)
    if usdinr:
        result["usdinr"] = round(usdinr, 2)
        comex = _safe_last_price(GOLD_TICKER)
        if comex:
            result["mcx_approx"] = round(comex * usdinr * MCX_CONVERSION, 0)

    result["nifty"] = _safe_last_price(NIFTY_TICKER)

    # Crude oil live price
    oil = _safe_last_price(OIL_TICKER)
    if oil:
        result["oil"] = round(oil, 2)

    # VIX — try each fallback
    for sym in [VIX_TICKER] + VIX_FALLBACKS:
        v = _safe_last_price(sym)
        if v:
            result["vix"] = round(v, 2)
            break

    return result
