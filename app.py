"""
Gold Price Predictor — Professional Trading Dashboard
"""

import os
import html as html_lib
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional

from src.data.fetcher import fetch_combined, fetch_gold_realtime, fetch_india_context
from src.analysis.technical import add_all_indicators, get_signal_summary
from src.analysis.sentiment import get_sentiment_summary, build_daily_sentiment_series
from src.models.predictor import retrain_and_predict, predict, load_model_meta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Premium dark trading terminal ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Base layout ── */
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #07070f; }
.block-container { padding: 1.5rem 2rem !important; }

/* ── Live pulse animation ── */
@keyframes live-glow {
    0%, 100% { box-shadow: 0 0 4px 1px rgba(38,166,154,0.7); }
    50%       { box-shadow: 0 0 10px 3px rgba(38,166,154,0.3); }
}
@keyframes fade-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

/* ── Top nav bar ── */
.nav-bar {
    background: linear-gradient(90deg, #0a0a18 0%, #0d0d20 100%);
    border-bottom: 1px solid #c9a84c18;
    padding: 10px 0 10px 0;
    margin: -1.5rem -2rem 2rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-left: 2rem;
    padding-right: 2rem;
}
.nav-logo {
    font-size: 1rem;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: 0.5px;
}
.nav-live {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.65rem;
    color: #26a69a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.live-dot {
    width: 7px; height: 7px;
    background: #26a69a;
    border-radius: 50%;
    animation: live-glow 2s infinite;
}
.nav-time { font-size: 0.72rem; color: #444; font-family: 'Space Mono', monospace; }

/* ── Price hero ── */
.price-hero {
    background: linear-gradient(135deg, #0c0c1e 0%, #121230 60%, #0c0c1e 100%);
    border: 1px solid rgba(201,168,76,0.12);
    border-radius: 16px;
    padding: 28px 36px 24px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    animation: fade-in 0.4s ease;
}
.price-hero::after {
    content: '';
    position: absolute;
    top: -120px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(201,168,76,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.price-hero-label {
    font-size: 0.62rem;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.price-hero-main {
    font-size: 4.2rem;
    font-weight: 800;
    color: #c9a84c;
    letter-spacing: -3px;
    font-variant-numeric: tabular-nums;
    font-family: 'Space Mono', monospace;
    line-height: 0.9;
    text-shadow: 0 0 80px rgba(201,168,76,0.18);
    margin-bottom: 10px;
}
.price-hero-change-up   { font-size: 1.15rem; font-weight: 600; color: #26a69a; letter-spacing: -0.3px; }
.price-hero-change-down { font-size: 1.15rem; font-weight: 600; color: #ef5350; letter-spacing: -0.3px; }
.price-hero-meta { font-size: 0.72rem; color: #444; margin-top: 6px; font-family: 'Space Mono', monospace; }

/* ── Metric tiles ── */
.metric-tile {
    background: linear-gradient(135deg, #0c0c1e 0%, #101025 100%);
    border: 1px solid #161628;
    border-left: 3px solid rgba(201,168,76,0.15);
    border-radius: 10px;
    padding: 13px 16px;
    transition: border-left-color 0.25s, background 0.25s;
    cursor: default;
}
.metric-tile:hover {
    border-left-color: rgba(201,168,76,0.55);
    background: linear-gradient(135deg, #0f0f25 0%, #141435 100%);
}
.metric-tile-label {
    font-size: 0.6rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 5px;
}
.metric-tile-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ddd;
    font-variant-numeric: tabular-nums;
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}
.metric-tile-up   { font-size: 0.72rem; color: #26a69a; margin-top: 3px; }
.metric-tile-down { font-size: 0.72rem; color: #ef5350; margin-top: 3px; }
.metric-tile-flat { font-size: 0.72rem; color: #444; margin-top: 3px; }

/* ── India strip ── */
.india-strip {
    background: linear-gradient(90deg, rgba(201,168,76,0.05) 0%, rgba(201,168,76,0.02) 100%);
    border: 1px solid rgba(201,168,76,0.1);
    border-radius: 10px;
    padding: 12px 20px;
    display: flex;
    gap: 36px;
    align-items: center;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.india-item-label { font-size: 0.6rem; color: #555; text-transform: uppercase; letter-spacing: 1px; }
.india-item-value { font-size: 1rem; font-weight: 700; color: #ddd; font-family: 'Space Mono', monospace; }

/* ── Section title ── */
.sec-title {
    font-size: 0.58rem;
    font-weight: 700;
    color: rgba(201,168,76,0.6);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(201,168,76,0.08);
}

/* ── Signal cards ── */
.sig-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 20px; }
.sig-card {
    background: linear-gradient(135deg, #0c0c1e 0%, #101025 100%);
    border: 1px solid #161628;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.sig-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 60%, rgba(201,168,76,0.03) 100%);
}
.sig-card-label {
    font-size: 0.58rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}
.sig-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-bottom: 14px;
}
.badge-bull { background: rgba(38,166,154,0.12); color: #26a69a; border: 1px solid rgba(38,166,154,0.25); }
.badge-bear { background: rgba(239,83,80,0.12);  color: #ef5350; border: 1px solid rgba(239,83,80,0.25); }
.badge-neut { background: rgba(120,120,140,0.10); color: #888;   border: 1px solid rgba(120,120,140,0.2); }
.sig-main {
    font-size: 2.2rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    font-family: 'Space Mono', monospace;
    letter-spacing: -1.5px;
    line-height: 1;
    margin-bottom: 6px;
}
.sig-meta { font-size: 0.7rem; color: #555; line-height: 1.6; }
.sig-conf { font-size: 0.68rem; color: #3a3a5c; margin-top: 4px; }

/* ── Signal strength bar ── */
.strength-track {
    background: #12122a;
    border-radius: 3px;
    height: 5px;
    margin: 11px 0 3px 0;
    overflow: hidden;
    position: relative;
}
.strength-fill { height: 100%; border-radius: 3px; }
.strength-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.55rem;
    color: #2a2a44;
    letter-spacing: 0.5px;
}

/* ── Trade signal card ── */
.trade-card {
    border-radius: 12px;
    padding: 18px 26px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}
.trade-card-buy  { background: linear-gradient(135deg, rgba(38,166,154,0.08) 0%, rgba(38,166,154,0.02) 100%); border: 1px solid rgba(38,166,154,0.18); }
.trade-card-sell { background: linear-gradient(135deg, rgba(239,83,80,0.08) 0%, rgba(239,83,80,0.02) 100%);  border: 1px solid rgba(239,83,80,0.18); }
.trade-card-hold { background: linear-gradient(135deg, rgba(201,168,76,0.07) 0%, rgba(201,168,76,0.02) 100%); border: 1px solid rgba(201,168,76,0.15); }
.trade-action {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    white-space: nowrap;
    min-width: 160px;
}
.trade-bullets { font-size: 0.73rem; color: #666; line-height: 1.8; }
.trade-disclaimer { font-size: 0.62rem; color: #2a2a3a; margin-top: 4px; }

/* ── News cards ── */
.news-card {
    background: linear-gradient(135deg, #0c0c1e 0%, #101025 100%);
    border: 1px solid #161628;
    border-radius: 10px;
    padding: 13px 16px;
    margin-bottom: 9px;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: rgba(201,168,76,0.25); }
.news-title { font-size: 0.88rem; font-weight: 500; color: #ccc; line-height: 1.4; }
.news-title a { color: #ccc; text-decoration: none; }
.news-title a:hover { color: #c9a84c; }
.news-meta { font-size: 0.68rem; color: #444; margin-top: 5px; }
.news-badge { display: inline-block; padding: 1px 7px; border-radius: 8px; font-size: 0.6rem; font-weight: 700; margin-right: 5px; }
.nbull { background: rgba(38,166,154,0.12); color: #26a69a; }
.nbear { background: rgba(239,83,80,0.12); color: #ef5350; }
.nneut { background: rgba(120,120,140,0.10); color: #666; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #121224; margin: 26px 0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #07070f;
    border-right: 1px solid #0f0f22;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #888 !important;
    font-size: 0.78rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0c0c1e;
    border: 1px solid #161628;
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    period   = st.selectbox("Time period",  ["3mo","6mo","1y","2y","5y"], index=2)
    horizon  = st.slider("Forecast horizon (days)", 1, 10, 1)
    interval = st.selectbox("Chart interval", ["1d","1wk"], index=0)
    show_vol = st.checkbox("Show volume", value=True)
    retrain_btn = st.button("🔄 Retrain model", use_container_width=True)

    # Model metadata
    meta = load_model_meta()
    if meta:
        trained_str = meta.get("trained_at","")[:10]
        st.markdown(
            f"<div style='font-size:0.68rem;color:#333;margin-top:4px'>"
            f"Model trained: {trained_str} &nbsp;·&nbsp; "
            f"MAPE {meta.get('price_mape','?')}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#333;line-height:2'>
    <b style='color:#555'>Data</b><br>
    GC=F · DXY · SPY · TLT · VIX<br>Silver · Crude Oil · Nifty<br><br>
    <b style='color:#555'>News</b><br>
    Reddit · yfinance · RSS Feeds<br><br>
    <b style='color:#555'>Sentiment</b><br>
    VADER + 50-term gold lexicon<br>
    Source reliability weighting<br><br>
    <b style='color:#555'>Model</b><br>
    GBM + RF Ensemble (60/40)<br>
    Quantile bounds (80% CI)<br>
    ~45 features · time-series CV<br><br>
    <b style='color:#555'>Local training</b><br>
    <code style='color:#c9a84c;font-size:0.65rem'>make train</code>
    </div>
    """, unsafe_allow_html=True)


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(period, interval):
    df = fetch_combined(period=period)
    return add_all_indicators(df)

@st.cache_data(ttl=300, show_spinner=False)
def load_sentiment():
    return get_sentiment_summary()

@st.cache_data(ttl=60, show_spinner=False)
def load_realtime():
    return fetch_gold_realtime()

@st.cache_data(ttl=60, show_spinner=False)
def load_india():
    return fetch_india_context()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(""):
    try:
        realtime = load_realtime()
        df       = load_data(period, interval)
        india    = load_india()
    except Exception as e:
        st.error(f"Failed to load market data: {e}")
        st.stop()

with st.spinner(""):
    try:
        sentiment = load_sentiment()
    except Exception:
        sentiment = {"label":"Neutral","avg_compound":0.0,"bullish_pct":0,
                     "bearish_pct":0,"neutral_pct":100,"article_count":0,"articles":[]}


# ── Compute core values ───────────────────────────────────────────────────────
last = realtime.get("last_price") or df["Close"].iloc[-1]
prev = realtime.get("previous_close") or df["Close"].iloc[-2]
if not prev:
    prev = df["Close"].iloc[-2]
change = last - prev
pct    = (change / prev * 100) if prev else 0.0
day_hi = realtime.get("day_high") or df["High"].iloc[-1]
day_lo = realtime.get("day_low")  or df["Low"].iloc[-1]

tech             = get_signal_summary(df)
sentiment_series = build_daily_sentiment_series(sentiment.get("articles", []))

with st.spinner(""):
    try:
        if retrain_btn:
            pred = retrain_and_predict(df, sentiment_series, horizon=horizon)
            st.sidebar.success("✓ Model retrained")
        else:
            pred = predict(df, sentiment_series, horizon=horizon)
    except Exception as e:
        st.sidebar.warning(f"Model error: {e}")
        pred = None


# ── Nav bar ───────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="nav-bar">'
    f'  <div class="nav-logo">🥇 Gold Price Predictor</div>'
    f'  <div class="nav-live"><span class="live-dot"></span>Live</div>'
    f'  <div class="nav-time">{datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Price hero ────────────────────────────────────────────────────────────────
chg_cls = "price-hero-change-up" if change >= 0 else "price-hero-change-down"
chg_sym = "▲" if change >= 0 else "▼"
high52  = df["High"].max()
low52   = df["Low"].min()
range_pct = (last - low52) / (high52 - low52) * 100 if high52 != low52 else 50

st.markdown(
    f'<div class="price-hero">'
    f'  <div class="price-hero-label">Gold Futures &nbsp;·&nbsp; GC=F &nbsp;·&nbsp; USD / troy oz</div>'
    f'  <div class="price-hero-main">${last:,.2f}</div>'
    f'  <div class="{chg_cls}">{chg_sym} ${abs(change):.2f} &nbsp; ({pct:+.2f}%)</div>'
    f'  <div class="price-hero-meta">'
    f'    H&nbsp;${day_hi:,.2f} &nbsp;·&nbsp; L&nbsp;${day_lo:,.2f} &nbsp;·&nbsp; '
    f'    {period} range: ${low52:,.0f} – ${high52:,.0f} &nbsp;·&nbsp; '
    f'    Position in range: {range_pct:.0f}%'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Metric tiles ──────────────────────────────────────────────────────────────
def _tile(label, value, sub="", sub_cls="metric-tile-flat", border=""):
    extra = f' style="border-left-color:{border}"' if border else ""
    return (
        f'<div class="metric-tile"{extra}>'
        f'<div class="metric-tile-label">{label}</div>'
        f'<div class="metric-tile-value">{value}</div>'
        f'<div class="{sub_cls}">{sub}</div>'
        f'</div>'
    )

def _tile_chg(label, value, chg_val, sub_str):
    cls = "metric-tile-up" if chg_val >= 0 else "metric-tile-down"
    sym = "▲" if chg_val >= 0 else "▼"
    return _tile(label, value, f"{sym} {sub_str}", cls)

# Gather values
vix_live = india.get("vix")
oil_live = india.get("oil")

vix_prev = df["VIX_Close"].iloc[-2] if "VIX_Close" in df.columns and df["VIX_Close"].notna().sum() > 1 else vix_live
oil_prev = df["Oil_Close"].iloc[-2]  if "Oil_Close" in df.columns and df["Oil_Close"].notna().sum() > 1 else oil_live

dxy_val  = df["DXY_Close"].iloc[-1]  if "DXY_Close"    in df.columns and df["DXY_Close"].notna().any()    else None
dxy_prev = df["DXY_Close"].iloc[-2]  if dxy_val and df["DXY_Close"].notna().sum() > 1 else dxy_val
spy_val  = df["SPY_Close"].iloc[-1]  if "SPY_Close"    in df.columns and df["SPY_Close"].notna().any()    else None
spy_prev = df["SPY_Close"].iloc[-2]  if spy_val and df["SPY_Close"].notna().sum() > 1 else spy_val
sv_val   = df["Silver_Close"].iloc[-1] if "Silver_Close" in df.columns and df["Silver_Close"].notna().any() else None
gsr      = last / sv_val if sv_val and sv_val > 0 else None

tiles_html = ""

tiles_html += _tile_chg("VIX · Fear Index",
    f"{vix_live:.1f}" if vix_live else "—",
    (vix_live - vix_prev) if (vix_live and vix_prev) else 0,
    f"{abs(vix_live-vix_prev):.2f}" if (vix_live and vix_prev) else "—")

tiles_html += _tile_chg("Crude Oil · CL=F",
    f"${oil_live:.2f}" if oil_live else ("$"+f"{df['Oil_Close'].iloc[-1]:.2f}" if "Oil_Close" in df.columns and df["Oil_Close"].notna().any() else "—"),
    (oil_live - oil_prev) if (oil_live and oil_prev) else 0,
    f"${abs(oil_live-oil_prev):.2f}" if (oil_live and oil_prev) else "—")

tiles_html += _tile_chg("DXY · Dollar Index",
    f"{dxy_val:.2f}" if dxy_val else "—",
    (dxy_val - dxy_prev) if (dxy_val and dxy_prev) else 0,
    f"{abs(dxy_val-dxy_prev):.2f}" if (dxy_val and dxy_prev) else "—")

tiles_html += _tile_chg("S&P 500 · SPY",
    f"${spy_val:,.2f}" if spy_val else "—",
    (spy_val - spy_prev) if (spy_val and spy_prev) else 0,
    f"${abs(spy_val-spy_prev):.2f}" if (spy_val and spy_prev) else "—")

tiles_html += _tile("Silver · SLV",
    f"${sv_val:.2f}" if sv_val else "—",
    f"G/S ratio: {gsr:.1f}" if gsr else "—")

tiles_html += _tile("52-wk Range", f"${low52:,.0f}", f"High: ${high52:,.0f}", "metric-tile-flat", "rgba(201,168,76,0.4)")

tile_cols = st.columns(6)
html_parts = [p for p in tiles_html.split('</div>') if p.strip()]
# Re-render tiles in columns cleanly
tile_data = [
    ("VIX · Fear Index",      f"{vix_live:.1f}" if vix_live else "—",
     (f"{'▲' if vix_live and vix_prev and vix_live>=vix_prev else '▼'} {abs(vix_live-vix_prev):.2f}" if (vix_live and vix_prev) else "—"),
     "metric-tile-up" if (vix_live and vix_prev and vix_live>=vix_prev) else "metric-tile-down"),

    ("Crude Oil · CL=F",
     f"${oil_live:.2f}" if oil_live else (f"${df['Oil_Close'].iloc[-1]:.2f}" if "Oil_Close" in df.columns and df["Oil_Close"].notna().any() else "—"),
     (f"{'▲' if oil_live and oil_prev and oil_live>=oil_prev else '▼'} ${abs(oil_live-oil_prev):.2f}" if (oil_live and oil_prev) else "—"),
     "metric-tile-up" if (oil_live and oil_prev and oil_live>=oil_prev) else "metric-tile-down"),

    ("DXY · Dollar Index",    f"{dxy_val:.2f}" if dxy_val else "—",
     (f"{'▲' if dxy_val and dxy_prev and dxy_val>=dxy_prev else '▼'} {abs(dxy_val-dxy_prev):.2f}" if (dxy_val and dxy_prev) else "—"),
     "metric-tile-up" if (dxy_val and dxy_prev and dxy_val>=dxy_prev) else "metric-tile-down"),

    ("S&P 500 · SPY",         f"${spy_val:,.2f}" if spy_val else "—",
     (f"{'▲' if spy_val and spy_prev and spy_val>=spy_prev else '▼'} ${abs(spy_val-spy_prev):.2f}" if (spy_val and spy_prev) else "—"),
     "metric-tile-up" if (spy_val and spy_prev and spy_val>=spy_prev) else "metric-tile-down"),

    ("Silver",                f"${sv_val:.2f}" if sv_val else "—",
     f"G/S ratio: {gsr:.1f}" if gsr else "—", "metric-tile-flat"),

    ("52-wk Range",           f"${low52:,.0f} – ${high52:,.0f}",
     f"Position: {range_pct:.0f}%", "metric-tile-flat"),
]
for col, (lbl, val, sub, sub_cls) in zip(tile_cols, tile_data):
    bdr = "rgba(201,168,76,0.4)" if lbl == "52-wk Range" else ""
    with col:
        st.markdown(_tile(lbl, val, sub, sub_cls, bdr), unsafe_allow_html=True)

# ── India strip ───────────────────────────────────────────────────────────────
mcx, usdinr, nifty = india.get("mcx_approx"), india.get("usdinr"), india.get("nifty")
india_items = [
    ("🇮🇳 MCX Gold (approx)", f"₹{mcx:,.0f}/10g") if mcx else None,
    ("USD / INR",              f"₹{usdinr:,.2f}") if usdinr else None,
    ("Nifty 50",               f"{nifty:,.0f}") if nifty else None,
]
india_items = [x for x in india_items if x]
if india_items:
    items_html = "".join(
        f'<div><div class="india-item-label">{lbl}</div>'
        f'<div class="india-item-value">{val}</div></div>'
        for lbl, val in india_items
    )
    st.markdown(f'<div class="india-strip">{items_html}</div>', unsafe_allow_html=True)


# ── Signal cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Market Signals</div>', unsafe_allow_html=True)

def _badge(label: str) -> str:
    cls = {"Bullish": "badge-bull", "Bearish": "badge-bear"}.get(label, "badge-neut")
    dot = {"Bullish": "▲", "Bearish": "▼"}.get(label, "—")
    return f'<span class="sig-badge {cls}">{dot} {label}</span>'

def _strength_bar(score: int, lo: int = -5, hi: int = 5) -> str:
    pct = (score - lo) / (hi - lo) * 100
    pct = max(0, min(100, pct))
    color = "#26a69a" if score > 0 else "#ef5350" if score < 0 else "#3a3a5c"
    return (
        f'<div class="strength-track">'
        f'  <div class="strength-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f'</div>'
        f'<div class="strength-labels"><span>BEAR</span><span>NEUTRAL</span><span>BULL</span></div>'
    )

rsi_val = tech["rsi"]
rsi_col = "#26a69a" if rsi_val < 40 else "#ef5350" if rsi_val > 65 else "#c9a84c"
cci_str = f" · CCI {tech['cci']:.0f}" if tech.get("cci") is not None else ""
macd_arrow = "▲" if (tech.get("macd") or 0) > (tech.get("macd_signal") or 0) else "▼"

tech_html = (
    f'<div class="sig-card">'
    f'<div class="sig-card-label">Technical Analysis</div>'
    f'{_badge(tech["overall"])}'
    f'<div class="sig-main" style="color:{rsi_col}">{rsi_val:.0f}</div>'
    f'<div class="sig-meta">RSI-14{cci_str} &nbsp;·&nbsp; MACD {macd_arrow} &nbsp;·&nbsp; Score {tech["score"]:+d}/5</div>'
    f'{_strength_bar(tech["score"])}'
    f'</div>'
)

sl = sentiment["label"]
sl_col = "#26a69a" if sl == "Bullish" else "#ef5350" if sl == "Bearish" else "#888"
sent_html = (
    f'<div class="sig-card">'
    f'<div class="sig-card-label">News Sentiment</div>'
    f'{_badge(sl)}'
    f'<div class="sig-main" style="color:{sl_col}">{sentiment["avg_compound"]:+.3f}</div>'
    f'<div class="sig-meta">'
    f'{sentiment["article_count"]} articles &nbsp;·&nbsp; '
    f'<span style="color:#26a69a">▲ {sentiment["bullish_pct"]:.0f}%</span> &nbsp; '
    f'<span style="color:#ef5350">▼ {sentiment["bearish_pct"]:.0f}%</span>'
    f'</div>'
    f'{_strength_bar(int((sentiment["avg_compound"]) * 10), -5, 5)}'
    f'</div>'
)

if pred:
    dir_lbl  = pred["predicted_direction"]
    dir_col  = "#26a69a" if dir_lbl == "Up" else "#ef5350"
    dir_sym  = "▲" if dir_lbl == "Up" else "▼"
    conf_pct = max(pred["up_probability"], pred["down_probability"])
    ci_str   = ""
    if pred.get("predicted_price_low") and pred.get("predicted_price_high"):
        ci_str = (
            f'<div class="sig-conf">80% CI: ${pred["predicted_price_low"]:,.0f} – '
            f'${pred["predicted_price_high"]:,.0f}</div>'
        )
    ml_html = (
        f'<div class="sig-card">'
        f'<div class="sig-card-label">ML Forecast · +{pred["horizon_days"]}d</div>'
        f'<span class="sig-badge {"badge-bull" if dir_lbl=="Up" else "badge-bear"}">'
        f'{dir_sym} {dir_lbl}</span>'
        f'<div class="sig-main" style="color:{dir_col}">${pred["predicted_price"]:,.2f}</div>'
        f'<div class="sig-meta">'
        f'{pred["expected_change_pct"]:+.2f}% change &nbsp;·&nbsp; '
        f'{conf_pct:.0f}% confidence &nbsp;·&nbsp; '
        f'MAPE {pred.get("price_mape","?")}%'
        f'</div>'
        f'{ci_str}'
        f'</div>'
    )
else:
    ml_html = (
        '<div class="sig-card">'
        '<div class="sig-card-label">ML Forecast</div>'
        '<div class="sig-meta" style="margin-top:16px">Model unavailable</div>'
        '</div>'
    )

st.markdown(f'<div class="sig-grid">{tech_html}{sent_html}{ml_html}</div>', unsafe_allow_html=True)


# ── Trading recommendation ────────────────────────────────────────────────────
def _recommendation(tech, sentiment, pred) -> tuple[str, str, str, list[str]]:
    bull, bear = 0, 0
    bullets = []
    ts = tech.get("score", 0)
    if ts >= 2:
        bull += 2; bullets.append(f"Technical: Bullish (score {ts:+d}/5)")
    elif ts <= -2:
        bear += 2; bullets.append(f"Technical: Bearish (score {ts:+d}/5)")
    else:
        bullets.append(f"Technical: Mixed (score {ts:+d}/5)")

    avg_c = sentiment.get("avg_compound", 0)
    if avg_c >= 0.05:
        bull += 1; bullets.append(f"Sentiment: Bullish — {sentiment.get('bullish_pct',0):.0f}% bullish articles")
    elif avg_c <= -0.05:
        bear += 1; bullets.append(f"Sentiment: Bearish — {sentiment.get('bearish_pct',0):.0f}% bearish articles")
    else:
        bullets.append("Sentiment: Neutral")

    if pred:
        cp = max(pred["up_probability"], pred["down_probability"])
        if pred["predicted_direction"] == "Up":
            bull += 1; bullets.append(f"ML Forecast: {pred['expected_change_pct']:+.2f}% ({cp:.0f}% confidence)")
        else:
            bear += 1; bullets.append(f"ML Forecast: {pred['expected_change_pct']:+.2f}% ({cp:.0f}% confidence)")

    if bull >= 3:     return "BUY / LONG",       "trade-card-buy",  "#26a69a", bullets
    elif bear >= 3:   return "SELL / SHORT",      "trade-card-sell", "#ef5350", bullets
    elif bull > bear: return "CAUTIOUS BUY",      "trade-card-buy",  "#26a69a", bullets
    elif bear > bull: return "CAUTIOUS SELL",      "trade-card-sell", "#ef5350", bullets
    else:             return "HOLD / WATCH",      "trade-card-hold", "#c9a84c", bullets

action, trade_cls, trade_col, bullets = _recommendation(tech, sentiment, pred)
bullets_html = "".join(f'<div>· {b}</div>' for b in bullets)

st.markdown(
    f'<div class="trade-card {trade_cls}">'
    f'  <div>'
    f'    <div style="font-size:0.55rem;color:#333;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px">Aggregated Signal (Educational)</div>'
    f'    <div class="trade-action" style="color:{trade_col}">{action}</div>'
    f'  </div>'
    f'  <div>'
    f'    <div class="trade-bullets">{bullets_html}</div>'
    f'    <div class="trade-disclaimer">⚠ Not financial advice — educational purposes only.</div>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Gauges row ────────────────────────────────────────────────────────────────
col_rsi, col_sent, col_bb = st.columns(3)

with col_rsi:
    fig_rsi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        title={"text": "RSI-14", "font": {"size": 12, "color": "#555"}},
        number={"font": {"size": 30, "color": "#ddd", "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0,100], "tickcolor":"#222", "tickfont":{"color":"#333","size":9}},
            "bar": {"color": rsi_col, "thickness": 0.22},
            "bgcolor": "#0c0c1e", "bordercolor": "#161628",
            "steps": [
                {"range":[0,30],   "color":"rgba(38,166,154,0.07)"},
                {"range":[30,70],  "color":"rgba(80,80,100,0.04)"},
                {"range":[70,100], "color":"rgba(239,83,80,0.07)"},
            ],
            "threshold": {"line":{"color":"#c9a84c","width":2},"thickness":0.72,"value":rsi_val},
        },
    ))
    fig_rsi.update_layout(height=200, margin=dict(l=16,r=16,t=36,b=8),
                          paper_bgcolor="#0c0c1e", font_color="#555")
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_sent:
    bull_p = sentiment["bullish_pct"]
    bear_p = sentiment["bearish_pct"]
    neut_p = sentiment["neutral_pct"]
    fig_s = go.Figure(go.Pie(
        labels=["Bullish","Bearish","Neutral"], values=[bull_p, bear_p, neut_p],
        hole=0.68, marker_colors=["#26a69a","#ef5350","#1e1e30"],
        textinfo="none", hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig_s.add_annotation(text=f"<b>{sl}</b>", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=13, color="#c9a84c", family="Space Grotesk"))
    fig_s.update_layout(
        title=dict(text="Sentiment", font=dict(size=12, color="#555"), x=0.5),
        height=200, margin=dict(l=16,r=16,t=36,b=8), paper_bgcolor="#0c0c1e",
        showlegend=True,
        legend=dict(font=dict(size=9,color="#444"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_s, use_container_width=True)

with col_bb:
    close_now = df["Close"].iloc[-1]
    bb_hi = df["BB_High"].iloc[-1] if "BB_High" in df.columns else close_now * 1.02
    bb_lo = df["BB_Low"].iloc[-1]  if "BB_Low"  in df.columns else close_now * 0.98
    bb_pct = max(0.0, min(100.0, (close_now - bb_lo) / (bb_hi - bb_lo) * 100 if bb_hi != bb_lo else 50))
    bb_col = "#ef5350" if bb_pct > 80 else "#26a69a" if bb_pct < 20 else "#c9a84c"
    fig_bb = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(bb_pct, 1),
        title={"text": "Bollinger %B", "font": {"size": 12, "color": "#555"}},
        number={"suffix":"%","font":{"size":30,"color":"#ddd","family":"Space Mono"}},
        gauge={
            "axis": {"range":[0,100],"tickcolor":"#222","tickfont":{"color":"#333","size":9}},
            "bar": {"color": bb_col, "thickness": 0.22},
            "bgcolor": "#0c0c1e", "bordercolor": "#161628",
            "steps": [
                {"range":[0,20],   "color":"rgba(38,166,154,0.07)"},
                {"range":[20,80],  "color":"rgba(80,80,100,0.04)"},
                {"range":[80,100], "color":"rgba(239,83,80,0.07)"},
            ],
        },
    ))
    fig_bb.update_layout(height=200, margin=dict(l=16,r=16,t=36,b=8),
                          paper_bgcolor="#0c0c1e", font_color="#555")
    st.plotly_chart(fig_bb, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Price chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Price Chart</div>', unsafe_allow_html=True)

ci_col, sr_col = st.columns([3, 1])
with ci_col:
    chart_indicators = st.multiselect(
        "Overlay indicators",
        ["SMA_20","SMA_50","SMA_200","EMA_12","EMA_26",
         "BB_High","BB_Low","BB_Mid",
         "Stoch_K","Stoch_D","Williams_R","CCI_20"],
        default=["SMA_20","SMA_50","BB_High","BB_Low"],
        label_visibility="collapsed",
    )
with sr_col:
    show_sr = st.checkbox("Support / Resistance", value=True)

price_overlays = [i for i in chart_indicators if i not in ("Stoch_K","Stoch_D","Williams_R","CCI_20")]
osc_overlays   = [i for i in chart_indicators if i in ("Stoch_K","Stoch_D","Williams_R","CCI_20")]
has_osc = bool(osc_overlays)

row_h = [0.52, 0.22]
if has_osc: row_h.append(0.14)
if show_vol: row_h.append(0.12)
total = sum(row_h)
row_h = [r / total for r in row_h]

subs = ["", "RSI · MACD"]
if has_osc:  subs.append("Oscillators")
if show_vol: subs.append("Volume")

fig = make_subplots(
    rows=len(row_h), cols=1, shared_xaxes=True,
    row_heights=row_h, subplot_titles=subs, vertical_spacing=0.025,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Gold",
    increasing_line_color="#26a69a", increasing_fillcolor="rgba(38,166,154,0.15)",
    decreasing_line_color="#ef5350", decreasing_fillcolor="rgba(239,83,80,0.15)",
), row=1, col=1)

COLORS = {
    "SMA_20":"#c9a84c","SMA_50":"#f57c00","SMA_200":"#ef5350",
    "EMA_12":"#42a5f5","EMA_26":"#1565c0",
    "BB_High":"#4fc3f7","BB_Low":"#4fc3f7","BB_Mid":"rgba(79,195,247,0.4)",
}
for ind in price_overlays:
    if ind in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ind], name=ind,
            line=dict(color=COLORS.get(ind,"#aaa"), width=1.2,
                      dash="dot" if "BB" in ind else "solid"),
            opacity=0.85,
        ), row=1, col=1)

if "BB_High" in price_overlays and "BB_Low" in price_overlays:
    fig.add_trace(go.Scatter(
        x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
        y=pd.concat([df["BB_High"], df["BB_Low"][::-1]]),
        fill="toself", fillcolor="rgba(79,195,247,0.04)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="BB Band",
    ), row=1, col=1)

# RSI + MACD (row 2)
fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI",
    line=dict(color="#ce93d8",width=1.5)), row=2, col=1)
fig.add_hrect(y0=70,y1=100,fillcolor="rgba(239,83,80,0.05)",line_width=0,row=2,col=1)
fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.05)",line_width=0,row=2,col=1)
fig.add_hline(y=70,line_dash="dash",line_color="rgba(239,83,80,0.2)",line_width=1,row=2,col=1)
fig.add_hline(y=30,line_dash="dash",line_color="rgba(38,166,154,0.2)",line_width=1,row=2,col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
    line=dict(color="#4fc3f7",width=1.2)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
    line=dict(color="#ff8a65",width=1.2)), row=2, col=1)
hist_c = ["#26a69a" if v>=0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
    marker_color=hist_c, opacity=0.45), row=2, col=1)

# Oscillators (row 3 if selected)
osc_row = 3 if has_osc else None
OSC_C = {"Stoch_K":"#80cbc4","Stoch_D":"#26a69a","Williams_R":"#ffcc80","CCI_20":"#f48fb1"}
if has_osc and osc_row:
    for ind in osc_overlays:
        if ind in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ind], name=ind,
                line=dict(color=OSC_C.get(ind,"#aaa"),width=1.2)), row=osc_row, col=1)
    if any(i in osc_overlays for i in ("Stoch_K","Stoch_D")):
        for y, c in [(80,"rgba(239,83,80,0.2)"),(20,"rgba(38,166,154,0.2)")]:
            fig.add_hline(y=y,line_dash="dash",line_color=c,line_width=1,row=osc_row,col=1)
    if "CCI_20" in osc_overlays:
        for y, c in [(100,"rgba(239,83,80,0.2)"),(-100,"rgba(38,166,154,0.2)")]:
            fig.add_hline(y=y,line_dash="dash",line_color=c,line_width=1,row=osc_row,col=1)
    if "Williams_R" in osc_overlays:
        for y, c in [(-20,"rgba(239,83,80,0.2)"),(-80,"rgba(38,166,154,0.2)")]:
            fig.add_hline(y=y,line_dash="dash",line_color=c,line_width=1,row=osc_row,col=1)

# Support & Resistance
if show_sr:
    recent = df.tail(120)
    resist = recent["High"].rolling(20, center=True).max().dropna()
    supp   = recent["Low"].rolling(20, center=True).min().dropna()
    if len(resist) > 0 and pd.notna(resist.iloc[-1]):
        r_lvl = resist.iloc[-1]
        fig.add_hline(y=r_lvl, line_dash="dot", line_color="rgba(239,83,80,0.35)", line_width=1.5,
                      annotation_text=f"R  ${r_lvl:,.0f}",
                      annotation_font_color="#ef5350", annotation_font_size=10, row=1, col=1)
    if len(supp) > 0 and pd.notna(supp.iloc[-1]):
        s_lvl = supp.iloc[-1]
        fig.add_hline(y=s_lvl, line_dash="dot", line_color="rgba(38,166,154,0.35)", line_width=1.5,
                      annotation_text=f"S  ${s_lvl:,.0f}",
                      annotation_font_color="#26a69a", annotation_font_size=10, row=1, col=1)

# ML forecast + confidence band
if pred:
    future_x = [df.index[-1] + pd.Timedelta(days=pred["horizon_days"])]
    if pred.get("predicted_price_low") and pred.get("predicted_price_high"):
        p_lo, p_hi = pred["predicted_price_low"], pred["predicted_price_high"]
        cur = df["Close"].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[-1], future_x[0], future_x[0], df.index[-1]],
            y=[cur, p_lo, p_hi, cur],
            fill="toself", fillcolor="rgba(201,168,76,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Band", showlegend=True, hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=future_x, y=[pred["predicted_price"]], mode="markers+text",
        marker=dict(symbol="diamond", size=11, color="#c9a84c", line=dict(color="#fff",width=1)),
        text=[f"  ${pred['predicted_price']:,.0f}"],
        textfont=dict(color="#c9a84c", size=10, family="Space Mono"),
        textposition="middle right",
        name=f"ML +{pred['horizon_days']}d", showlegend=True,
    ), row=1, col=1)

# Volume
vol_row = (osc_row + 1 if osc_row else 3) if show_vol else None
if show_vol and vol_row and "Volume" in df.columns:
    vol_c = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
             for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_c, opacity=0.5), row=vol_row, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#07070f", plot_bgcolor="#0c0c1e",
    height=780,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                font=dict(size=10,color="#555",family="Space Grotesk"),
                bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=8, t=28, b=8),
    font=dict(family="Space Grotesk", color="#555"),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#12122a", bordercolor="rgba(201,168,76,0.15)", font_color="#ccc"),
)
for i in range(1, len(row_h)+1):
    fig.update_xaxes(gridcolor="#0f0f22", zeroline=False,
                     showspikes=True, spikecolor="rgba(201,168,76,0.2)", spikethickness=1,
                     row=i, col=1)
    fig.update_yaxes(gridcolor="#0f0f22", zeroline=False, row=i, col=1)
fig.update_yaxes(title_text="USD", title_font=dict(size=9,color="#333"), row=1, col=1)
fig.update_yaxes(title_text="RSI·MACD", title_font=dict(size=9,color="#333"), row=2, col=1)

st.plotly_chart(fig, use_container_width=True)


# ── Technical signals table ───────────────────────────────────────────────────
with st.expander("🔍 Technical Signal Breakdown"):
    cols_sig = st.columns(2)
    icons = {"Bullish":"🟢","Bearish":"🔴","Oversold":"🔵","Overbought":"🟠","Neutral":"🟡","Normal":"⚪"}
    for i, (cat, sig, desc) in enumerate(tech["signals"]):
        with cols_sig[i % 2]:
            st.markdown(
                f"<div style='background:#0c0c1e;border:1px solid #161628;border-radius:8px;"
                f"padding:10px 14px;margin-bottom:7px'>"
                f"<span style='font-size:0.58rem;color:#333;text-transform:uppercase;letter-spacing:1px'>{cat}</span><br>"
                f"<span style='font-size:0.82rem;color:#ccc'>{icons.get(sig,'⚪')} {sig}</span>"
                f"<span style='font-size:0.72rem;color:#444'> — {desc}</span>"
                f"</div>", unsafe_allow_html=True,
            )

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── News ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Latest Gold News</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🌍 Global", "🇮🇳 India"])
articles = sentiment.get("articles", [])
india_arts  = [a for a in articles if any(
    kw in (a.get("source","") + a.get("title","")).lower()
    for kw in ["india","mcx","rupee","inr","nse","bse","sensex","nifty"]
)]
global_arts = [a for a in articles if a not in india_arts]

def render_news(article_list, max_items=20):
    if not article_list:
        st.markdown(
            '<div class="news-card" style="text-align:center;padding:28px">'
            '<div style="color:#333;font-size:0.8rem">No articles found. '
            'Refreshing every 5 minutes.</div></div>', unsafe_allow_html=True)
        return
    ncols = st.columns(2)
    for i, a in enumerate(article_list[:max_items]):
        pub = a.get("published","")
        pub = pub.strftime("%b %d, %H:%M") if hasattr(pub,"strftime") else str(pub)[:16]
        sv  = a.get("sentiment","Neutral")
        bcls = {"Bullish":"nbull","Bearish":"nbear","Neutral":"nneut"}.get(sv,"nneut")
        score = a.get("compound",0)
        url   = a.get("url","")
        title = html_lib.escape(a.get("title",""))
        src   = html_lib.escape(a.get("source",""))
        th    = f'<a href="{url}" target="_blank">{title}</a>' if url else title
        with ncols[i % 2]:
            st.markdown(
                f'<div class="news-card">'
                f'<div class="news-title">{th}</div>'
                f'<div class="news-meta">'
                f'<span class="news-badge {bcls}">{sv}</span>'
                f'{src} &nbsp;·&nbsp; {pub} &nbsp;·&nbsp; '
                f'<span style="color:#c9a84c;font-family:Space Mono">{score:+.3f}</span>'
                f'</div></div>', unsafe_allow_html=True,
            )

with tab1: render_news(global_arts)
with tab2:
    if india_arts:
        render_news(india_arts)
    else:
        india_filtered = [a for a in articles if any(
            kw in a.get("title","").lower() for kw in ["india","mcx","rupee","rbi","sebi"]
        )]
        render_news(india_filtered if india_filtered else articles[:10])


# ── Feature importances ───────────────────────────────────────────────────────
if pred and "feature_importances" in pred:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    with st.expander("🧠 Model Feature Importances"):
        fi    = pred["feature_importances"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).head(20)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,"#0c0c1e"],[0.5,"rgba(201,168,76,0.4)"],[1,"#c9a84c"]],
                showscale=False,
            ),
        ))
        fig_fi.update_layout(
            template="plotly_dark",
            paper_bgcolor="#07070f", plot_bgcolor="#0c0c1e",
            height=500, xaxis_title="Importance",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=8,r=8,t=8,b=8),
            font=dict(family="Space Grotesk",color="#555",size=11),
        )
        fig_fi.update_xaxes(gridcolor="#0f0f22")
        fig_fi.update_yaxes(gridcolor="#0f0f22")
        st.plotly_chart(fig_fi, use_container_width=True)

        if pred.get("train_samples"):
            mc = st.columns(4)
            mc[0].metric("Train samples",     pred["train_samples"])
            mc[1].metric("Test samples",       pred.get("test_samples","—"))
            mc[2].metric("Features",           pred.get("feature_count","—"))
            mc[3].metric("Direction accuracy", f"{pred.get('direction_accuracy','?')}%")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
fcol1, fcol2 = st.columns([3, 1])
with fcol1:
    st.markdown(
        "<div style='font-size:0.65rem;color:#222'>"
        "Educational purposes only · Not financial advice · "
        "Data: Yahoo Finance · News: Reddit, Google News RSS, yfinance"
        "</div>",
        unsafe_allow_html=True,
    )
with fcol2:
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        elapsed   = time.time() - st.session_state.last_refresh
        remaining = max(0, 300 - int(elapsed))
        if remaining == 0:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()
        else:
            mins, secs = divmod(remaining, 60)
            st.caption(f"Refresh in {mins}:{secs:02d}")
            time.sleep(1)
            st.rerun()
