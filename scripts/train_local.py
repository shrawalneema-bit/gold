#!/usr/bin/env python3
"""
Local training script for Gold Price Predictor.

Run this on your Mac for heavy ML computation:
    python scripts/train_local.py               # 5y data, horizon=1
    python scripts/train_local.py --period 2y   # quicker run
    python scripts/train_local.py --horizon 3   # predict 3 days ahead

After training, commit the model files to GitHub so Streamlit Cloud
can load them without retraining on each deployment:
    git add src/models/saved/
    git commit -m "update trained model"
    git push
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import fetch_combined
from src.analysis.technical import add_all_indicators
from src.analysis.sentiment import get_sentiment_summary, build_daily_sentiment_series
from src.models.predictor import train


GOLD   = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def bar_chart(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Train Gold Price Predictor (local)")
    parser.add_argument("--period",  default="5y",
                        choices=["1y","2y","3y","5y"],
                        help="Historical data period (default: 5y)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in days (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all feature importances")
    args = parser.parse_args()

    print()
    print(f"{GOLD}{BOLD}🥇 Gold Price Predictor — Local Training{RESET}")
    print(f"{DIM}{'─'*52}{RESET}")
    print(f"  Period:    {args.period}  |  Horizon: {args.horizon} day(s)")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Market data ───────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"  {CYAN}[1/4]{RESET} Fetching market data...")
    try:
        df = fetch_combined(period=args.period)
        df = add_all_indicators(df)
        print(f"        ✓ {len(df)} trading days  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        print(f"        ✓ {len(df.columns)} columns (OHLCV + {len(df.columns)-5} indicators/macro)")
    except Exception as e:
        print(f"        {RED}✗ Failed: {e}{RESET}")
        sys.exit(1)

    # ── Sentiment ─────────────────────────────────────────────────────────────
    print(f"  {CYAN}[2/4]{RESET} Fetching news sentiment...")
    try:
        sentiment_data   = get_sentiment_summary()
        sentiment_series = build_daily_sentiment_series(sentiment_data.get("articles", []))
        n_art = sentiment_data.get("article_count", 0)
        avg_c = sentiment_data.get("avg_compound", 0)
        print(f"        ✓ {n_art} articles  |  avg compound: {avg_c:+.3f}  "
              f"({'Bullish' if avg_c > 0.05 else 'Bearish' if avg_c < -0.05 else 'Neutral'})")
    except Exception as e:
        print(f"        ⚠ Sentiment unavailable ({e}) — training without it")
        sentiment_series = None

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"  {CYAN}[3/4]{RESET} Training ensemble models...")
    print(f"        Models: GBM Regressor + RF Regressor (60/40 blend)")
    print(f"        +       GBM Classifier + RF Classifier (direction)")
    print(f"        +       Quantile GBM α=0.10 / α=0.90 (80% CI bounds)")
    t1 = time.time()
    try:
        metrics = train(df, sentiment_series, horizon=args.horizon, save=True)
    except Exception as e:
        print(f"        {RED}✗ Training failed: {e}{RESET}")
        sys.exit(1)
    elapsed = time.time() - t1
    print(f"        ✓ Training completed in {elapsed:.1f}s")

    # ── Save metadata ─────────────────────────────────────────────────────────
    print(f"  {CYAN}[4/4]{RESET} Saving model metadata...")
    meta_path = os.path.join("src", "models", "saved", "model_meta.json")
    meta = {
        "trained_at":          datetime.utcnow().isoformat() + "Z",
        "period":              args.period,
        "horizon_days":        args.horizon,
        "price_mape":          metrics["price_mape"],
        "direction_accuracy":  metrics["direction_accuracy"],
        "train_samples":       metrics["train_samples"],
        "test_samples":        metrics.get("test_samples", "?"),
        "feature_count":       metrics["feature_count"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"        ✓ Saved to {meta_path}")

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print()
    print(f"{DIM}{'─'*52}{RESET}")
    print(f"{GREEN}{BOLD}  ✅ Training Complete{RESET}  {DIM}({total_time:.1f}s total){RESET}")
    print()
    print(f"  {BOLD}Performance metrics{RESET}  {DIM}(last time-series fold){RESET}")
    print(f"    Price MAPE:          {metrics['price_mape']:.2f}%   "
          f"{DIM}(lower is better){RESET}")
    print(f"    Direction Accuracy:  {metrics['direction_accuracy']:.1f}%   "
          f"{DIM}(>55% is useful){RESET}")
    print(f"    Training samples:    {metrics['train_samples']}")
    print(f"    Test samples:        {metrics.get('test_samples','?')}")
    print(f"    Features:            {metrics['feature_count']}")
    print()

    # Feature importances
    n_feats = 20 if args.verbose else 10
    print(f"  {BOLD}Top {n_feats} features by importance{RESET}")
    fi = metrics["feature_importances"]
    max_imp = max(fi.values()) if fi else 1.0
    for i, (feat, imp) in enumerate(list(fi.items())[:n_feats], 1):
        pct  = imp / max_imp
        bar  = bar_chart(pct, 22)
        print(f"    {i:2d}. {feat:<28} {GOLD}{bar}{RESET}  {imp:.4f}")
    print()

    print(f"  {BOLD}Next step — commit model for Streamlit Cloud:{RESET}")
    print(f"    {DIM}git add src/models/saved/{RESET}")
    print(f"    {DIM}git commit -m \"update trained model ({args.period})\"{RESET}")
    print(f"    {DIM}git push{RESET}")
    print()
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()
