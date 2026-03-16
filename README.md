# Gold Price Predictor

A Streamlit dashboard that combines live gold price data, technical chart analysis,
news sentiment, and a machine learning model to forecast gold prices.

## Features

| Module | Details |
|---|---|
| **Live Data** | Gold futures (GC=F) via `yfinance` + macro context (DXY, SPY, TLT) |
| **Charts** | Candlestick + SMA/EMA, Bollinger Bands, RSI, MACD, Volume |
| **Sentiment** | RSS news from Reuters / Kitco / MarketWatch, scored with VADER (gold-tuned lexicon) |
| **ML Model** | Gradient Boosting (sklearn) — price regression + direction classification |
| **Dashboard** | Interactive Streamlit app with Plotly charts, auto-refresh |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) add a free NewsAPI key for richer news coverage
cp .env.example .env
# edit .env and add your NEWS_API_KEY

# Run the dashboard
streamlit run app.py
```

## Project Structure

```
gold/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt
├── .env.example
└── src/
    ├── data/
    │   └── fetcher.py          # Live gold + macro data via yfinance
    ├── analysis/
    │   ├── technical.py        # RSI, MACD, Bollinger Bands, SMA/EMA, etc.
    │   └── sentiment.py        # News fetch + VADER sentiment scoring
    └── models/
        └── predictor.py        # Gradient Boosting price & direction model
```

## How It Works

1. **Data** — `yfinance` pulls OHLCV history for `GC=F` (gold futures) plus macro
   indicators (US Dollar Index, S&P 500, 20yr Treasury) for context.

2. **Technical Analysis** — `ta` library computes 20+ indicators.
   A rule-based `get_signal_summary()` scores each indicator bullish/bearish.

3. **News Sentiment** — RSS feeds are filtered to gold-relevant headlines.
   VADER scores each article. Scores are aggregated into a daily sentiment series
   and fed as an extra feature into the ML model.

4. **Prediction** — A `GradientBoostingRegressor` predicts next-day closing price.
   A `GradientBoostingClassifier` predicts direction (Up/Down) with probabilities.
   Both use time-series cross-validation to avoid data leakage.

5. **Dashboard** — Streamlit renders everything in real-time with interactive
   Plotly charts, signal cards, news table, and feature importance bar chart.

## Disclaimer

This tool is for **educational purposes only**. Gold price prediction is inherently
uncertain. Do not use this for financial decisions.
