"""
Gold Price Predictor — Professional Streamlit Dashboard
"""

import html as html_lib
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from src.data.fetcher import fetch_combined, fetch_gold_realtime, fetch_india_context
from src.analysis.technical import add_all_indicators, get_signal_summary
from src.analysis.sentiment import get_sentiment_summary, build_daily_sentiment_series
from src.models.predictor import retrain_and_predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main background */
.stApp {
    background-color: #0a0a0f;
}

/* ── Top header bar ── */
.header-bar {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    border-bottom: 1px solid #c9a84c30;
    padding: 18px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 0.75rem;
    color: #666;
    margin-top: 2px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.header-time {
    font-size: 0.8rem;
    color: #555;
}

/* ── Price ticker cards ── */
.ticker-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.ticker-card {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
.ticker-card:hover { border-color: #c9a84c40; }
.ticker-label {
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.ticker-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e8e8;
    letter-spacing: -0.5px;
}
.ticker-change-up   { font-size: 0.8rem; color: #26a69a; font-weight: 500; }
.ticker-change-down { font-size: 0.8rem; color: #ef5350; font-weight: 500; }
.ticker-change-flat { font-size: 0.8rem; color: #888;    font-weight: 500; }

/* ── Section headers ── */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    color: #c9a84c;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #c9a84c20;
}

/* ── Signal cards ── */
.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.signal-card {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 20px 24px;
}
.signal-card-label {
    font-size: 0.65rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}
.signal-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.badge-bullish  { background: #26a69a20; color: #26a69a; border: 1px solid #26a69a40; }
.badge-bearish  { background: #ef535020; color: #ef5350; border: 1px solid #ef535040; }
.badge-neutral  { background: #88888820; color: #aaa;    border: 1px solid #88888840; }
.signal-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8e8e8;
    margin-bottom: 4px;
}
.signal-meta { font-size: 0.75rem; color: #555; }

/* ── Trading recommendation card ── */
.trade-card {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 28px;
}
.trade-card-buy   { border-color: #26a69a50; }
.trade-card-sell  { border-color: #ef535050; }
.trade-card-hold  { border-color: #c9a84c50; }
.trade-action {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.trade-reason { font-size: 0.78rem; color: #888; line-height: 1.6; }

/* ── Confidence band ── */
.conf-band {
    font-size: 0.72rem;
    color: #555;
    margin-top: 4px;
}

/* ── News cards ── */
.news-card {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: #c9a84c30; }
.news-title {
    font-size: 0.9rem;
    font-weight: 500;
    color: #ddd;
    line-height: 1.4;
    text-decoration: none;
}
.news-title a { color: #ddd; text-decoration: none; }
.news-title a:hover { color: #c9a84c; }
.news-meta { font-size: 0.72rem; color: #555; margin-top: 6px; }
.news-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.65rem;
    font-weight: 600;
    margin-right: 6px;
}
.news-bull { background: #26a69a15; color: #26a69a; }
.news-bear { background: #ef535015; color: #ef5350; }
.news-neut { background: #88888815; color: #888; }

/* ── Divider ── */
.gold-divider {
    border: none;
    border-top: 1px solid #1e1e30;
    margin: 28px 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1a1a2e;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #aaa !important;
    font-size: 0.8rem !important;
}

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 14px;
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
    retrain  = st.button("🔄 Retrain model", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#555; line-height:1.8'>
    <b style='color:#777'>Data</b><br>
    Gold Futures · GC=F · yfinance<br>
    DXY · SPY · TLT · VIX · Silver · Oil<br><br>
    <b style='color:#777'>News</b><br>
    Reddit · yfinance · Google News RSS<br><br>
    <b style='color:#777'>Sentiment</b><br>
    VADER · gold-tuned lexicon (50+ terms)<br>
    Source reliability weighting<br><br>
    <b style='color:#777'>Model</b><br>
    GBM + Random Forest ensemble<br>
    Quantile bounds for 80% interval<br>
    ~45 features · time-series CV
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
        st.error(f"Failed to load data: {e}")
        st.stop()

with st.spinner(""):
    try:
        sentiment = load_sentiment()
    except Exception:
        sentiment = {"label":"Neutral","avg_compound":0.0,"bullish_pct":0,
                     "bearish_pct":0,"neutral_pct":100,"article_count":0,"articles":[]}

# ── Compute values ────────────────────────────────────────────────────────────
last   = realtime.get("last_price") or df["Close"].iloc[-1]
prev   = realtime.get("previous_close") or df["Close"].iloc[-2]
# Prevent division by zero when prev is 0 or None
if not prev:
    prev = df["Close"].iloc[-2]
change = last - prev
pct    = (change / prev * 100) if prev else 0.0

tech             = get_signal_summary(df)
sentiment_series = build_daily_sentiment_series(sentiment.get("articles", []))

with st.spinner(""):
    try:
        pred = retrain_and_predict(df, sentiment_series, horizon=horizon)
        if retrain:
            st.sidebar.success("✓ Model retrained")
    except Exception as e:
        st.sidebar.warning(f"Model error: {e}")
        pred = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
  <div>
    <div class="header-title">🥇 Gold Price Predictor</div>
    <div class="header-subtitle">Live Data · Technical Analysis · News Sentiment · ML Forecast</div>
  </div>
  <div class="header-time">{datetime.utcnow().strftime('%d %b %Y · %H:%M UTC')}</div>
</div>
""", unsafe_allow_html=True)

# ── Ticker bar ────────────────────────────────────────────────────────────────
day_hi  = realtime.get("day_high")  or df["High"].iloc[-1]
day_lo  = realtime.get("day_low")   or df["Low"].iloc[-1]

def _card(label, value, sub, border=""):
    """Render a single ticker card."""
    style = f' style="border-color:{border}"' if border else ""
    st.markdown(
        f'<div class="ticker-card"{style}>'
        f'<div class="ticker-label">{label}</div>'
        f'<div class="ticker-value">{value}</div>'
        f'<div class="ticker-change-flat">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _card_change(label, value, change_val, change_str):
    cls = "ticker-change-up" if change_val >= 0 else "ticker-change-down"
    sym = "▲" if change_val >= 0 else "▼"
    st.markdown(
        f'<div class="ticker-card">'
        f'<div class="ticker-label">{label}</div>'
        f'<div class="ticker-value">{value}</div>'
        f'<div class="{cls}">{sym} {change_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Row 1: Global prices (6 cards)
r1 = st.columns(6)
with r1[0]: _card_change("Gold Futures · GC=F", f"${last:,.2f}", change, f"${abs(change):.2f} ({pct:+.2f}%)")
with r1[1]: _card("Day High", f"${day_hi:,.2f}", "Session high")
with r1[2]: _card("Day Low",  f"${day_lo:,.2f}", "Session low")

# VIX
vix_live = india.get("vix")
with r1[3]:
    if vix_live:
        vix_prev = df["VIX_Close"].iloc[-2] if "VIX_Close" in df.columns and df["VIX_Close"].notna().sum() > 1 else vix_live
        _card_change("VIX · Fear Index", f"{vix_live:.1f}", vix_live - vix_prev, f"{abs(vix_live - vix_prev):.2f} today")
    else:
        _card("VIX · Fear Index", "—", "Loading…")

# Crude oil
oil_live = india.get("oil")
with r1[4]:
    if oil_live:
        oil_prev = df["Oil_Close"].iloc[-2] if "Oil_Close" in df.columns and df["Oil_Close"].notna().sum() > 1 else oil_live
        oil_chg  = oil_live - oil_prev
        _card_change("Crude Oil · CL=F", f"${oil_live:.2f}", oil_chg, f"${abs(oil_chg):.2f} today")
    elif "Oil_Close" in df.columns and df["Oil_Close"].notna().any():
        oil_v = df["Oil_Close"].iloc[-1]
        _card("Crude Oil · CL=F", f"${oil_v:.2f}", "per barrel")
    else:
        _card("Crude Oil", "—", "Loading…")

# Gold/Silver ratio
with r1[5]:
    if "Silver_Close" in df.columns and df["Silver_Close"].notna().any():
        sv = df["Silver_Close"].iloc[-1]
        gsr = df["Close"].iloc[-1] / sv if sv and sv > 0 else None
        if gsr:
            _card("Gold / Silver Ratio", f"{gsr:.1f}", "avg ~65–80")
        else:
            _card("G/S Ratio", "—", "No silver data")
    else:
        _card("G/S Ratio", "—", "No silver data")

# Row 2: India
mcx, usdinr, nifty = india.get("mcx_approx"), india.get("usdinr"), india.get("nifty")
india_cards = [x for x in [mcx, usdinr, nifty] if x]
if india_cards:
    r2 = st.columns(max(len(india_cards), 3))
    ci = 0
    if mcx:
        with r2[ci]: _card("🇮🇳 MCX Gold (approx)", f"₹{mcx:,.0f}", "per 10g · COMEX×INR", "rgba(201,168,76,0.15)")
        ci += 1
    if usdinr:
        with r2[ci]: _card("USD / INR", f"₹{usdinr:,.2f}", "Live exchange rate")
        ci += 1
    if nifty:
        with r2[ci]: _card("Nifty 50", f"{nifty:,.0f}", "NSE index")

# ── Signal cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Market Signals</div>', unsafe_allow_html=True)

def badge(label):
    cls = {"Bullish":"badge-bullish","Bearish":"badge-bearish"}.get(label,"badge-neutral")
    return f'<span class="signal-badge {cls}">{label}</span>'

# Technical card
rsi_val = tech["rsi"]
rsi_col = "#26a69a" if rsi_val < 40 else "#ef5350" if rsi_val > 65 else "#c9a84c"
cci_val = tech.get("cci")
cci_str = f" · CCI {cci_val:.0f}" if cci_val is not None else ""

# Sentiment card
sl = sentiment["label"]

# ML card
if pred:
    dir_lbl = pred["predicted_direction"]
    dir_col = "#26a69a" if dir_lbl == "Up" else "#ef5350"
    conf_pct = max(pred["up_probability"], pred["down_probability"])

tech_html = f"""
<div class="signal-card">
  <div class="signal-card-label">Technical Analysis</div>
  {badge(tech['overall'])}
  <div class="signal-value" style="color:{rsi_col}">{rsi_val:.0f}</div>
  <div class="signal-meta">RSI-14 &nbsp;·&nbsp; Score {tech['score']:+d}/5 &nbsp;·&nbsp; MACD {'▲' if (tech.get('macd') or 0) > (tech.get('macd_signal') or 0) else '▼'}{cci_str}</div>
</div>"""

sent_html = f"""
<div class="signal-card">
  <div class="signal-card-label">News Sentiment</div>
  {badge(sl)}
  <div class="signal-value">{sentiment['avg_compound']:+.3f}</div>
  <div class="signal-meta">
    {sentiment['article_count']} articles &nbsp;·&nbsp;
    <span style='color:#26a69a'>Bullish {sentiment['bullish_pct']:.0f}%</span> &nbsp;
    <span style='color:#ef5350'>Bearish {sentiment['bearish_pct']:.0f}%</span>
  </div>
</div>"""

if pred:
    conf_band_str = ""
    if pred.get("predicted_price_low") and pred.get("predicted_price_high"):
        conf_band_str = (
            f"<div class='conf-band'>80% range: "
            f"${pred['predicted_price_low']:,.0f} – ${pred['predicted_price_high']:,.0f}</div>"
        )
    ml_html = f"""
<div class="signal-card">
  <div class="signal-card-label">ML Forecast · +{pred['horizon_days']}d</div>
  {badge(dir_lbl)}
  <div class="signal-value" style="color:{dir_col}">${pred['predicted_price']:,.2f}</div>
  <div class="signal-meta">
    {pred['expected_change_pct']:+.2f}% &nbsp;·&nbsp;
    Confidence {conf_pct:.0f}% &nbsp;·&nbsp;
    MAPE {pred.get('price_mape','?')}%
  </div>
  {conf_band_str}
</div>"""
else:
    ml_html = '<div class="signal-card"><div class="signal-card-label">ML Forecast</div><div class="signal-meta">Model unavailable</div></div>'

st.markdown(f'<div class="signal-grid">{tech_html}{sent_html}{ml_html}</div>', unsafe_allow_html=True)

# ── Trading Recommendation ────────────────────────────────────────────────────
def _trading_recommendation(tech: dict, sentiment: dict, pred: Optional[dict]) -> tuple[str, str, str]:
    """
    Aggregate technical, sentiment and ML signals into a single trade suggestion.
    Returns (action, color_class, reason_html).
    """
    bull_count = 0
    bear_count = 0
    reasons = []

    # Technical score
    ts = tech.get("score", 0)
    if ts >= 2:
        bull_count += 2
        reasons.append(f"Technical signals bullish (score {ts:+d}/5)")
    elif ts <= -2:
        bear_count += 2
        reasons.append(f"Technical signals bearish (score {ts:+d}/5)")
    else:
        reasons.append(f"Technical signals mixed (score {ts:+d}/5)")

    # Sentiment
    avg_c = sentiment.get("avg_compound", 0)
    if avg_c >= 0.05:
        bull_count += 1
        reasons.append(f"Sentiment bullish ({sentiment.get('bullish_pct',0):.0f}% bullish articles)")
    elif avg_c <= -0.05:
        bear_count += 1
        reasons.append(f"Sentiment bearish ({sentiment.get('bearish_pct',0):.0f}% bearish articles)")
    else:
        reasons.append("Sentiment neutral")

    # ML
    if pred:
        if pred["predicted_direction"] == "Up":
            bull_count += 1
            reasons.append(f"ML forecasts +{pred['expected_change_pct']:.2f}% ({max(pred['up_probability'], pred['down_probability']):.0f}% confidence)")
        else:
            bear_count += 1
            reasons.append(f"ML forecasts {pred['expected_change_pct']:.2f}% ({max(pred['up_probability'], pred['down_probability']):.0f}% confidence)")

    if bull_count >= 3:
        action = "BUY / LONG"
        cls    = "trade-card-buy"
        color  = "#26a69a"
    elif bear_count >= 3:
        action = "SELL / SHORT"
        cls    = "trade-card-sell"
        color  = "#ef5350"
    elif bull_count > bear_count:
        action = "CAUTIOUS BUY"
        cls    = "trade-card-buy"
        color  = "#26a69a"
    elif bear_count > bull_count:
        action = "CAUTIOUS SELL"
        cls    = "trade-card-sell"
        color  = "#ef5350"
    else:
        action = "HOLD / NEUTRAL"
        cls    = "trade-card-hold"
        color  = "#c9a84c"

    reason_html = " &nbsp;·&nbsp; ".join(reasons)
    return action, cls, color, reason_html

action, trade_cls, trade_color, trade_reason = _trading_recommendation(tech, sentiment, pred)
st.markdown(
    f'<div class="trade-card {trade_cls}">'
    f'<div style="font-size:0.65rem;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Aggregated Signal (Educational)</div>'
    f'<div class="trade-action" style="color:{trade_color}">{action}</div>'
    f'<div class="trade-reason">{trade_reason}</div>'
    f'<div style="font-size:0.65rem;color:#333;margin-top:10px">⚠️ Not financial advice — for educational purposes only.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── RSI Gauge + Sentiment Donut + BB gauge ────────────────────────────────────
col_gauge, col_donut, col_bb = st.columns(3)

with col_gauge:
    fig_rsi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        title={"text": "RSI-14", "font": {"size": 13, "color": "#888"}},
        number={"font": {"size": 32, "color": "#e8e8e8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444", "tickfont": {"color":"#666","size":10}},
            "bar": {"color": rsi_col, "thickness": 0.25},
            "bgcolor": "#0f0f1a",
            "bordercolor": "#1e1e30",
            "steps": [
                {"range": [0, 30],  "color": "rgba(38,166,154,0.08)"},
                {"range": [30, 70], "color": "rgba(136,136,136,0.06)"},
                {"range": [70, 100],"color": "rgba(239,83,80,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#c9a84c", "width": 2},
                "thickness": 0.75,
                "value": rsi_val,
            },
        },
    ))
    fig_rsi.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0f0f1a", font_color="#888",
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_donut:
    bull = sentiment["bullish_pct"]
    bear = sentiment["bearish_pct"]
    neut = sentiment["neutral_pct"]
    fig_sent = go.Figure(go.Pie(
        labels=["Bullish", "Bearish", "Neutral"],
        values=[bull, bear, neut],
        hole=0.65,
        marker_colors=["#26a69a", "#ef5350", "#444"],
        textinfo="none",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig_sent.add_annotation(
        text=f"<b>{sl}</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#c9a84c"),
    )
    fig_sent.update_layout(
        title=dict(text="Sentiment Breakdown", font=dict(size=13, color="#888"), x=0.5),
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0f0f1a",
        showlegend=True,
        legend=dict(font=dict(size=10, color="#666"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

with col_bb:
    close_now = df["Close"].iloc[-1]
    bb_hi = df["BB_High"].iloc[-1] if "BB_High" in df.columns else close_now * 1.02
    bb_lo = df["BB_Low"].iloc[-1]  if "BB_Low"  in df.columns else close_now * 0.98
    bb_pct = (close_now - bb_lo) / (bb_hi - bb_lo) * 100 if bb_hi != bb_lo else 50
    # Clamp to [0, 100] in case of extreme moves outside bands
    bb_pct = max(0.0, min(100.0, bb_pct))

    fig_bb = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(bb_pct, 1),
        title={"text": "Bollinger %B", "font": {"size": 13, "color": "#888"}},
        number={"suffix": "%", "font": {"size": 32, "color": "#e8e8e8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444", "tickfont": {"color":"#666","size":10}},
            "bar": {"color": "#c9a84c", "thickness": 0.25},
            "bgcolor": "#0f0f1a",
            "bordercolor": "#1e1e30",
            "steps": [
                {"range": [0,  20], "color": "rgba(38,166,154,0.08)"},
                {"range": [20, 80], "color": "rgba(136,136,136,0.06)"},
                {"range": [80, 100],"color": "rgba(239,83,80,0.08)"},
            ],
        },
    ))
    fig_bb.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0f0f1a", font_color="#888",
    )
    st.plotly_chart(fig_bb, use_container_width=True)

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ── Main chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)

ci_col, sr_col = st.columns([3,1])
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

# Determine sub-chart rows
price_overlays  = [i for i in chart_indicators if i not in ("Stoch_K","Stoch_D","Williams_R","CCI_20")]
osc_overlays    = [i for i in chart_indicators if i in ("Stoch_K","Stoch_D","Williams_R","CCI_20")]
has_osc         = bool(osc_overlays)

rows        = (2 if show_vol else 1) + 1 + (1 if has_osc else 0)   # price + RSI/MACD [+ osc] [+ vol]
row_heights_base = [0.50, 0.22]
if has_osc:
    row_heights_base.append(0.15)
if show_vol:
    row_heights_base.append(0.13)
# Normalise so they sum to 1
total = sum(row_heights_base)
row_heights = [r / total for r in row_heights_base]

subtitle_list = ["", "RSI · MACD"]
if has_osc:
    subtitle_list.append("Oscillators")
if show_vol:
    subtitle_list.append("Volume")

fig = make_subplots(
    rows=len(row_heights), cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    subplot_titles=subtitle_list,
    vertical_spacing=0.03,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Gold",
    increasing_line_color="#26a69a", increasing_fillcolor="rgba(38,166,154,0.13)",
    decreasing_line_color="#ef5350", decreasing_fillcolor="rgba(239,83,80,0.13)",
), row=1, col=1)

# Indicator overlays on price chart
COLORS = {
    "SMA_20": "#c9a84c", "SMA_50": "#f57c00", "SMA_200": "#ef5350",
    "EMA_12": "#42a5f5", "EMA_26": "#1976d2",
    "BB_High": "#4fc3f7", "BB_Low": "#4fc3f7", "BB_Mid": "rgba(79,195,247,0.5)",
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

# RSI (row 2)
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI_14"], name="RSI",
    line=dict(color="#ce93d8", width=1.5),
), row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",  line_width=0, row=2, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.25)",  line_width=1, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.25)", line_width=1, row=2, col=1)

# MACD (row 2, same subplot)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD"], name="MACD",
    line=dict(color="#4fc3f7", width=1.2),
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD_Signal"], name="Signal",
    line=dict(color="#ff8a65", width=1.2),
), row=2, col=1)
hist_c = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_Hist"], name="MACD Hist",
    marker_color=hist_c, opacity=0.5,
), row=2, col=1)

# Oscillator sub-chart (row 3 if present)
osc_row = 3 if has_osc else None
OSC_COLORS = {
    "Stoch_K": "#80cbc4", "Stoch_D": "#26a69a",
    "Williams_R": "#ffcc80", "CCI_20": "#f48fb1",
}
if has_osc and osc_row:
    for ind in osc_overlays:
        if ind in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ind], name=ind,
                line=dict(color=OSC_COLORS.get(ind, "#aaa"), width=1.2),
            ), row=osc_row, col=1)
    # Reference lines for oscillators
    if "Stoch_K" in osc_overlays or "Stoch_D" in osc_overlays:
        fig.add_hline(y=80, line_dash="dash", line_color="rgba(239,83,80,0.25)", line_width=1, row=osc_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="rgba(38,166,154,0.25)", line_width=1, row=osc_row, col=1)
    if "CCI_20" in osc_overlays:
        fig.add_hline(y=100,  line_dash="dash", line_color="rgba(239,83,80,0.25)",  line_width=1, row=osc_row, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="rgba(38,166,154,0.25)", line_width=1, row=osc_row, col=1)
    if "Williams_R" in osc_overlays:
        fig.add_hline(y=-20,  line_dash="dash", line_color="rgba(239,83,80,0.25)",  line_width=1, row=osc_row, col=1)
        fig.add_hline(y=-80,  line_dash="dash", line_color="rgba(38,166,154,0.25)", line_width=1, row=osc_row, col=1)

# Volume row
vol_row = (osc_row + 1 if osc_row else 3) if show_vol else None
if show_vol and vol_row and "Volume" in df.columns:
    vol_c = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
             for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_c, opacity=0.6,
    ), row=vol_row, col=1)

# Support & Resistance — pivot from recent swing highs/lows
if show_sr:
    window = 20
    recent = df.tail(120)
    resistance = recent["High"].rolling(window, center=True).max().dropna()
    support    = recent["Low"].rolling(window, center=True).min().dropna()
    if len(resistance) > 0 and len(support) > 0:
        r_level = resistance.iloc[-1]
        s_level = support.iloc[-1]
        if pd.notna(r_level) and r_level > 0:
            fig.add_hline(y=r_level, line_dash="dot", line_color="rgba(239,83,80,0.4)", line_width=1.5,
                          annotation_text=f"R ${r_level:,.0f}", annotation_font_color="#ef5350",
                          annotation_font_size=10, row=1, col=1)
        if pd.notna(s_level) and s_level > 0:
            fig.add_hline(y=s_level, line_dash="dot", line_color="rgba(38,166,154,0.4)", line_width=1.5,
                          annotation_text=f"S ${s_level:,.0f}", annotation_font_color="#26a69a",
                          annotation_font_size=10, row=1, col=1)

# ML predicted price marker + confidence interval band
if pred:
    future_x = [df.index[-1] + pd.Timedelta(days=pred["horizon_days"])]

    # 80% confidence interval shading (if quantile models were trained)
    if pred.get("predicted_price_low") and pred.get("predicted_price_high"):
        p_lo = pred["predicted_price_low"]
        p_hi = pred["predicted_price_high"]
        # Shade from last known date to forecast point
        shade_x = [df.index[-1], future_x[0], future_x[0], df.index[-1]]
        shade_y = [df["Close"].iloc[-1], p_lo, p_hi, df["Close"].iloc[-1]]
        fig.add_trace(go.Scatter(
            x=shade_x + shade_x[::-1],
            y=[df["Close"].iloc[-1], p_lo, p_lo, df["Close"].iloc[-1]] +
              [df["Close"].iloc[-1], p_hi, p_hi, df["Close"].iloc[-1]],
            fill="toself",
            fillcolor="rgba(201,168,76,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Band",
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    # Point prediction marker
    fig.add_trace(go.Scatter(
        x=future_x, y=[pred["predicted_price"]],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#c9a84c", line=dict(color="#fff", width=1)),
        text=[f"  ML ${pred['predicted_price']:,.0f}"],
        textfont=dict(color="#c9a84c", size=10),
        textposition="middle right",
        name=f"ML Forecast +{pred['horizon_days']}d",
        showlegend=True,
    ), row=1, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0f0f1a",
    height=780,
    xaxis_rangeslider_visible=False,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        font=dict(size=11, color="#888"), bgcolor="rgba(0,0,0,0)",
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    font=dict(family="Inter", color="#888"),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="rgba(201,168,76,0.2)", font_color="#ddd"),
)
total_rows = len(row_heights)
for i in range(1, total_rows + 1):
    fig.update_xaxes(
        gridcolor="#1a1a2e", zeroline=False,
        showspikes=True, spikecolor="rgba(201,168,76,0.25)", spikethickness=1,
        row=i, col=1,
    )
    fig.update_yaxes(gridcolor="#1a1a2e", zeroline=False, row=i, col=1)

fig.update_yaxes(title_text="USD", title_font=dict(size=10, color="#555"), row=1, col=1)
fig.update_yaxes(title_text="RSI/MACD", title_font=dict(size=10, color="#555"), row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Technical signals table ───────────────────────────────────────────────────
with st.expander("🔍 Technical Signal Breakdown"):
    cols_sig = st.columns(2)
    icons = {"Bullish":"🟢","Bearish":"🔴","Oversold":"🔵","Overbought":"🟠","Neutral":"🟡","Normal":"⚪"}
    for i, (cat, sig, desc) in enumerate(tech["signals"]):
        with cols_sig[i % 2]:
            st.markdown(
                f"<div style='background:#0f0f1a;border:1px solid #1e1e30;border-radius:8px;"
                f"padding:10px 14px;margin-bottom:8px'>"
                f"<span style='font-size:0.65rem;color:#555;text-transform:uppercase'>{cat}</span><br>"
                f"<span style='font-size:0.85rem;color:#ddd'>{icons.get(sig,'⚪')} {sig}</span>"
                f"<span style='font-size:0.75rem;color:#666'> — {desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ── News section ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Latest Gold News</div>', unsafe_allow_html=True)

news_tab1, news_tab2 = st.tabs(["🌍 Global", "🇮🇳 India"])
articles = sentiment.get("articles", [])

india_articles = [a for a in articles if any(
    kw in (a.get("source","") + a.get("title","")).lower()
    for kw in ["india","mcx","rupee","inr","nse","bse","sensex","nifty","hindusthan","hindustan"]
)]
global_articles = [a for a in articles if a not in india_articles]

def render_news(article_list, max_items=20):
    if not article_list:
        st.markdown(
            '<div class="news-card" style="text-align:center;padding:32px">'
            '<div style="color:#555">No articles found. Refreshing every 5 minutes.</div>'
            '</div>', unsafe_allow_html=True)
        return
    news_cols = st.columns(2)
    for i, a in enumerate(article_list[:max_items]):
        pub = a.get("published","")
        if hasattr(pub, "strftime"):
            pub = pub.strftime("%b %d, %H:%M")

        sent_val = a.get("sentiment","Neutral")
        bcls     = {"Bullish":"news-bull","Bearish":"news-bear","Neutral":"news-neut"}.get(sent_val,"news-neut")
        score    = a.get("compound", 0)
        url      = a.get("url","")
        title    = html_lib.escape(a.get("title",""))
        src      = html_lib.escape(a.get("source",""))

        title_html = f'<a href="{url}" target="_blank">{title}</a>' if url else title

        with news_cols[i % 2]:
            st.markdown(
                f'<div class="news-card">'
                f'<div class="news-title">{title_html}</div>'
                f'<div class="news-meta">'
                f'<span class="news-badge {bcls}">{sent_val}</span>'
                f'<span>{src}</span> &nbsp;·&nbsp; <span>{pub}</span>'
                f' &nbsp;·&nbsp; <span style="color:#c9a84c">{score:+.3f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

with news_tab1:
    render_news(global_articles)

with news_tab2:
    if india_articles:
        render_news(india_articles)
    else:
        india_filtered = [a for a in articles if any(
            kw in a.get("title","").lower()
            for kw in ["india","mcx","rupee","sovereign gold","rbi","sebi"]
        )]
        render_news(india_filtered if india_filtered else articles[:10])

# ── Feature importances ───────────────────────────────────────────────────────
if pred and "feature_importances" in pred:
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    with st.expander("🧠 Model Feature Importances"):
        fi    = pred["feature_importances"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).head(20)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,"#1a1a2e"],[0.5,"rgba(201,168,76,0.5)"],[1,"#c9a84c"]],
                showscale=False,
            ),
        ))
        fig_fi.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0a0f", plot_bgcolor="#0f0f1a",
            height=520, xaxis_title="Importance",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", color="#888", size=11),
        )
        fig_fi.update_xaxes(gridcolor="#1a1a2e")
        fig_fi.update_yaxes(gridcolor="#1a1a2e")
        st.plotly_chart(fig_fi, use_container_width=True)

        # Model training summary
        if pred.get("train_samples"):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Training samples", pred["train_samples"])
            mc2.metric("Test samples", pred.get("test_samples", "—"))
            mc3.metric("Features", pred.get("feature_count", "—"))
            mc4.metric("Direction accuracy", f"{pred.get('direction_accuracy','?')}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
fcol1, fcol2 = st.columns([3,1])
with fcol1:
    st.markdown(
        "<div style='font-size:0.7rem;color:#333'>"
        "For educational purposes only. Not financial advice. "
        "Data via Yahoo Finance · News via Reddit, Google News RSS &amp; yfinance."
        "</div>",
        unsafe_allow_html=True,
    )
with fcol2:
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        # Non-blocking countdown — updates every second, refreshes at 0
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
            st.caption(f"Refreshing in {mins}:{secs:02d}")
            time.sleep(1)
            st.rerun()
