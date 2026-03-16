"""
Gold Price Predictor — Professional Streamlit Dashboard
"""

import html as html_lib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

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
    DXY · SPY · TLT · VIX · Silver<br><br>
    <b style='color:#777'>News</b><br>
    yfinance · Google News RSS<br><br>
    <b style='color:#777'>Sentiment</b><br>
    VADER (gold-tuned lexicon)<br><br>
    <b style='color:#777'>Model</b><br>
    GBM + Random Forest ensemble<br>
    ~40 features · time-series CV
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
change = last - prev
pct    = change / prev * 100
tech   = get_signal_summary(df)
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
chg_cls = "ticker-change-up" if change >= 0 else "ticker-change-down"
chg_sym = "▲" if change >= 0 else "▼"

def _card(label, value, sub, border=""):
    """Render a single ticker card — no variable interpolation into large f-strings."""
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

# Row 1: Global prices
r1 = st.columns(5)
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

# Gold/Silver ratio
with r1[4]:
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

# Sentiment card
sl = sentiment["label"]

# ML card
if pred:
    dir_lbl = pred["predicted_direction"]
    dir_col = "#26a69a" if dir_lbl == "Up" else "#ef5350"

tech_html = f"""
<div class="signal-card">
  <div class="signal-card-label">Technical Analysis</div>
  {badge(tech['overall'])}
  <div class="signal-value" style="color:{rsi_col}">{rsi_val:.0f}</div>
  <div class="signal-meta">RSI-14 &nbsp;·&nbsp; Score {tech['score']:+d}/5 &nbsp;·&nbsp; MACD {'▲' if (tech.get('macd') or 0) > (tech.get('macd_signal') or 0) else '▼'}</div>
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
    ml_html = f"""
<div class="signal-card">
  <div class="signal-card-label">ML Forecast · +{pred['horizon_days']}d</div>
  {badge(dir_lbl)}
  <div class="signal-value" style="color:{dir_col}">${pred['predicted_price']:,.2f}</div>
  <div class="signal-meta">
    {pred['expected_change_pct']:+.2f}% change &nbsp;·&nbsp;
    Confidence {max(pred['up_probability'], pred['down_probability']):.0f}% &nbsp;·&nbsp;
    MAPE {pred.get('price_mape','?')}%
  </div>
</div>"""
else:
    ml_html = '<div class="signal-card"><div class="signal-card-label">ML Forecast</div><div class="signal-meta">Model unavailable</div></div>'

st.markdown(f'<div class="signal-grid">{tech_html}{sent_html}{ml_html}</div>', unsafe_allow_html=True)

# ── RSI Gauge + Sentiment Donut ───────────────────────────────────────────────
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
    st.plotly_chart(fig_rsi, width="stretch")

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
    st.plotly_chart(fig_sent, width="stretch")

with col_bb:
    close_now = df["Close"].iloc[-1]
    bb_hi = df["BB_High"].iloc[-1] if "BB_High" in df.columns else close_now * 1.02
    bb_lo = df["BB_Low"].iloc[-1]  if "BB_Low"  in df.columns else close_now * 0.98
    bb_mi = df["BB_Mid"].iloc[-1]  if "BB_Mid"  in df.columns else close_now
    bb_pct = (close_now - bb_lo) / (bb_hi - bb_lo) * 100 if bb_hi != bb_lo else 50

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
    st.plotly_chart(fig_bb, width="stretch")

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ── Main chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)

ci_col, sr_col = st.columns([3,1])
with ci_col:
    chart_indicators = st.multiselect(
        "Overlay indicators",
        ["SMA_20","SMA_50","SMA_200","EMA_12","EMA_26","BB_High","BB_Low","BB_Mid"],
        default=["SMA_20","SMA_50","BB_High","BB_Low"],
        label_visibility="collapsed",
    )
with sr_col:
    show_sr = st.checkbox("Support / Resistance", value=True)

rows        = 3 if show_vol else 2
row_heights = [0.55, 0.25, 0.20] if show_vol else [0.65, 0.35]
subtitles   = ["", "RSI · MACD"] + (["Volume"] if show_vol else [])

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    subplot_titles=subtitles,
    vertical_spacing=0.03,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Gold",
    increasing_line_color="#26a69a", increasing_fillcolor="rgba(38,166,154,0.13)",
    decreasing_line_color="#ef5350", decreasing_fillcolor="rgba(239,83,80,0.13)",
), row=1, col=1)

# Indicator overlays
COLORS = {
    "SMA_20": "#c9a84c", "SMA_50": "#f57c00", "SMA_200": "#ef5350",
    "EMA_12": "#42a5f5", "EMA_26": "#1976d2",
    "BB_High": "#4fc3f7", "BB_Low": "#4fc3f7", "BB_Mid": "rgba(79,195,247,0.5)",
}
for ind in chart_indicators:
    if ind in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ind], name=ind,
            line=dict(color=COLORS.get(ind,"#aaa"), width=1.2,
                      dash="dot" if "BB" in ind else "solid"),
            opacity=0.85,
        ), row=1, col=1)

if "BB_High" in chart_indicators and "BB_Low" in chart_indicators:
    fig.add_trace(go.Scatter(
        x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
        y=pd.concat([df["BB_High"], df["BB_Low"][::-1]]),
        fill="toself", fillcolor="rgba(79,195,247,0.04)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="BB Band",
    ), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI_14"], name="RSI",
    line=dict(color="#ce93d8", width=1.5),
), row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",  line_width=0, row=2, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.25)",  line_width=1, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.25)", line_width=1, row=2, col=1)

# MACD
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

# Support & Resistance — pivot from recent swing highs/lows
if show_sr:
    window = 20
    recent = df.tail(120)
    resistance = recent["High"].rolling(window, center=True).max().dropna()
    support    = recent["Low"].rolling(window, center=True).min().dropna()
    if len(resistance) > 0 and len(support) > 0:
        r_level = resistance.iloc[-1]
        s_level = support.iloc[-1]
        fig.add_hline(y=r_level, line_dash="dot", line_color="rgba(239,83,80,0.4)", line_width=1.5,
                      annotation_text=f"R ${r_level:,.0f}", annotation_font_color="#ef5350",
                      annotation_font_size=10, row=1, col=1)
        fig.add_hline(y=s_level, line_dash="dot", line_color="rgba(38,166,154,0.4)", line_width=1.5,
                      annotation_text=f"S ${s_level:,.0f}", annotation_font_color="#26a69a",
                      annotation_font_size=10, row=1, col=1)

# ML predicted price marker
if pred:
    import numpy as np
    future_x = [df.index[-1] + pd.Timedelta(days=pred["horizon_days"])]
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

# Volume
if show_vol and "Volume" in df.columns:
    vol_c = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
             for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_c, opacity=0.6,
    ), row=3, col=1)

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
for i in range(1, rows+1):
    fig.update_xaxes(
        gridcolor="#1a1a2e", zeroline=False,
        showspikes=True, spikecolor="rgba(201,168,76,0.25)", spikethickness=1,
        row=i, col=1,
    )
    fig.update_yaxes(gridcolor="#1a1a2e", zeroline=False, row=i, col=1)

fig.update_yaxes(title_text="USD", title_font=dict(size=10, color="#555"), row=1, col=1)
fig.update_yaxes(title_text="RSI/MACD", title_font=dict(size=10, color="#555"), row=2, col=1)

st.plotly_chart(fig, width="stretch")

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
        # show all articles filtered by India keywords from titles
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
        st.plotly_chart(fig_fi, width="stretch")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
fcol1, fcol2 = st.columns([3,1])
with fcol1:
    st.markdown(
        "<div style='font-size:0.7rem;color:#333'>"
        "For educational purposes only. Not financial advice. "
        "Data via Yahoo Finance · News via Google News RSS & yfinance."
        "</div>",
        unsafe_allow_html=True,
    )
with fcol2:
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        time.sleep(300)
        st.rerun()
