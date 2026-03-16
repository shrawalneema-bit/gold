"""
Gold Price Predictor — Streamlit Dashboard

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from src.data.fetcher import fetch_combined, fetch_gold_realtime
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
period   = st.sidebar.selectbox("Historical period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
horizon  = st.sidebar.slider("Prediction horizon (days)", 1, 10, 1)
interval = st.sidebar.selectbox("Chart interval", ["1d", "1wk"], index=0)
show_vol = st.sidebar.checkbox("Show volume", value=True)
retrain  = st.sidebar.button("🔄 Retrain model on latest data")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- Gold futures (GC=F) via yfinance\n"
    "- Macro: DXY · SPY · TLT · VIX · Silver\n"
    "- News: yfinance · Google News RSS\n"
    "- Sentiment: VADER (gold-tuned lexicon)\n"
    "- ML: GBM + Random Forest ensemble"
)

# ── Cache helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(period: str, interval: str):
    df = fetch_combined(period=period)
    df = add_all_indicators(df)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_sentiment():
    return get_sentiment_summary()

@st.cache_data(ttl=60, show_spinner=False)
def load_realtime():
    return fetch_gold_realtime()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🥇 Gold Price Predictor")
st.caption("Live data · Technical analysis · News sentiment · ML forecast")

with st.spinner("Fetching live gold data…"):
    try:
        realtime = load_realtime()
        df       = load_data(period, interval)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

# ── Real-time ticker ──────────────────────────────────────────────────────────
last   = realtime.get("last_price") or df["Close"].iloc[-1]
prev   = realtime.get("previous_close") or df["Close"].iloc[-2]
change = last - prev
pct    = change / prev * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Gold (GC=F)",  f"${last:,.2f}",  f"{change:+.2f} ({pct:+.2f}%)")
col2.metric("Day High",     f"${realtime.get('day_high') or df['High'].iloc[-1]:,.2f}")
col3.metric("Day Low",      f"${realtime.get('day_low') or df['Low'].iloc[-1]:,.2f}")

# VIX
if "VIX_Close" in df.columns and df["VIX_Close"].notna().any():
    vix_now  = df["VIX_Close"].iloc[-1]
    vix_prev = df["VIX_Close"].iloc[-2]
    col4.metric("VIX (Fear)", f"{vix_now:.1f}", f"{vix_now - vix_prev:+.1f}")
else:
    col4.metric("VIX (Fear)", "—")

# Gold/Silver ratio
if "Gold_Silver_Ratio" in df.columns and df["Gold_Silver_Ratio"].notna().any():
    gsr = df["Gold_Silver_Ratio"].iloc[-1]
    col5.metric("Gold/Silver Ratio", f"{gsr:.1f}")
else:
    col5.metric("Data as of", realtime.get("timestamp", "—")[:10])

st.markdown("---")

# ── Sentiment ─────────────────────────────────────────────────────────────────
with st.spinner("Analysing news sentiment…"):
    try:
        sentiment = load_sentiment()
    except Exception:
        sentiment = {"label": "Neutral", "avg_compound": 0.0, "bullish_pct": 0,
                     "bearish_pct": 0, "neutral_pct": 100, "article_count": 0, "articles": []}

sent_label = sentiment["label"]
sent_color = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(sent_label, "⚪")

# ── Technical signals ─────────────────────────────────────────────────────────
tech       = get_signal_summary(df)
tech_color = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(tech["overall"], "⚪")

# ── ML Prediction ─────────────────────────────────────────────────────────────
sentiment_series = build_daily_sentiment_series(sentiment.get("articles", []))

with st.spinner("Running prediction model…"):
    try:
        pred = retrain_and_predict(df, sentiment_series, horizon=horizon)
        if retrain:
            st.sidebar.success("Model retrained!")
    except Exception as e:
        st.warning(f"Prediction unavailable: {e}")
        pred = None

# ── Summary cards ─────────────────────────────────────────────────────────────
st.subheader("📊 Signals Summary")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"### {tech_color} Technical")
    st.metric("Signal", tech["overall"])
    st.caption(f"RSI: {tech['rsi']:.1f}  |  Score: {tech['score']:+d}/5")

with c2:
    st.markdown(f"### {sent_color} Sentiment")
    st.metric("News Mood", sent_label)
    st.caption(
        f"Compound: {sentiment['avg_compound']:+.3f}  |  "
        f"{sentiment['article_count']} articles  |  "
        f"🟢 {sentiment['bullish_pct']:.0f}%  🔴 {sentiment['bearish_pct']:.0f}%"
    )

with c3:
    if pred:
        dir_icon = "📈" if pred["predicted_direction"] == "Up" else "📉"
        st.markdown(f"### {dir_icon} ML Forecast")
        st.metric(
            f"Predicted price (+{pred['horizon_days']}d)",
            f"${pred['predicted_price']:,.2f}",
            f"{pred['expected_change_pct']:+.2f}%",
        )
        st.caption(
            f"Up {pred['up_probability']:.0f}% / Down {pred['down_probability']:.0f}%  |  "
            f"MAPE {pred.get('price_mape', '?')}%  |  "
            f"Dir. acc {pred.get('direction_accuracy', '?')}%  |  "
            f"{pred.get('feature_count', '?')} features"
        )
    else:
        st.markdown("### ❓ ML Forecast")
        st.info("Model not available")

st.markdown("---")

# ── Chart ─────────────────────────────────────────────────────────────────────
st.subheader("📈 Gold Price Chart")

chart_indicators = st.multiselect(
    "Overlay indicators",
    ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "BB_High", "BB_Low", "BB_Mid"],
    default=["SMA_20", "SMA_50", "BB_High", "BB_Low"],
)

rows         = 3 if show_vol else 2
row_heights  = [0.55, 0.25, 0.20] if show_vol else [0.65, 0.35]
subplot_titles = ["Gold Price (USD)", "RSI · MACD"] + (["Volume"] if show_vol else [])

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
    vertical_spacing=0.04,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Gold",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# Overlays
COLORS = {
    "SMA_20": "#ffb300", "SMA_50": "#fb8c00", "SMA_200": "#e53935",
    "EMA_12": "#42a5f5", "EMA_26": "#1565c0",
    "BB_High": "rgba(100,181,246,0.4)", "BB_Low": "rgba(100,181,246,0.4)",
    "BB_Mid":  "rgba(100,181,246,0.7)",
}
for ind in chart_indicators:
    if ind in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ind], name=ind,
            line=dict(color=COLORS.get(ind, "#aaa"), width=1,
                      dash="dot" if "BB" in ind else "solid"),
            opacity=0.9,
        ), row=1, col=1)

if "BB_High" in chart_indicators and "BB_Low" in chart_indicators:
    fig.add_trace(go.Scatter(
        x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
        y=pd.concat([df["BB_High"], df["BB_Low"][::-1]]),
        fill="toself", fillcolor="rgba(100,181,246,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="BB Band", showlegend=False,
    ), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI_14"], name="RSI-14",
    line=dict(color="#ce93d8", width=1.5),
), row=2, col=1)
for level, color in [(70, "rgba(239,83,80,0.3)"), (30, "rgba(38,166,154,0.3)")]:
    fig.add_hline(y=level, line_dash="dash", line_color=color, row=2, col=1)

# MACD
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD"], name="MACD",
    line=dict(color="#4fc3f7", width=1.2),
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD_Signal"], name="Signal",
    line=dict(color="#ff8a65", width=1.2),
), row=2, col=1)
hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_Hist"], name="MACD Hist",
    marker_color=hist_colors, opacity=0.6,
), row=2, col=1)

# Volume
if show_vol and "Volume" in df.columns:
    vol_colors = [
        "#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.7,
    ), row=3, col=1)

fig.update_layout(
    template="plotly_dark", height=750,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=40, b=20),
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="RSI / MACD",  row=2, col=1)
if show_vol:
    fig.update_yaxes(title_text="Volume", row=3, col=1)

st.plotly_chart(fig, width='stretch')

# ── Technical signal details ──────────────────────────────────────────────────
with st.expander("🔍 Technical Signal Details"):
    sig_rows = []
    icons = {"Bullish": "🟢", "Bearish": "🔴", "Oversold": "🔵",
             "Overbought": "🟠", "Neutral": "🟡", "Normal": "⚪"}
    for category, signal, desc in tech["signals"]:
        sig_rows.append({
            "Category":    category,
            "Signal":      f"{icons.get(signal, '⚪')} {signal}",
            "Description": desc,
        })
    st.table(pd.DataFrame(sig_rows))

# ── News section ──────────────────────────────────────────────────────────────
st.subheader("📰 Latest Gold News & Sentiment")
articles = sentiment.get("articles", [])
if articles:
    sent_icons = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
    for a in articles[:20]:
        pub = a.get("published", "")
        if hasattr(pub, "strftime"):
            pub = pub.strftime("%b %d, %H:%M")
        icon  = sent_icons.get(a.get("sentiment", "Neutral"), "⚪")
        score = a.get("compound", 0)
        url   = a.get("url", "")
        title = a.get("title", "No title")
        src   = a.get("source", "")

        # Clickable headline
        title_md = f"[{title}]({url})" if url else title
        st.markdown(
            f"{icon} **{title_md}**  \n"
            f"<span style='color:gray;font-size:0.8em'>{src} · {pub} · score {score:+.3f}</span>",
            unsafe_allow_html=True,
        )
        st.divider()
else:
    st.info("No recent gold news found. News updates every 5 minutes.")

# ── Feature importances ───────────────────────────────────────────────────────
if pred and "feature_importances" in pred:
    with st.expander("🧠 Model Feature Importances (top 20)"):
        fi     = pred["feature_importances"]
        fi_df  = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).head(20)
        fi_fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color="#ffb300",
        ))
        fi_fig.update_layout(
            template="plotly_dark", height=520,
            xaxis_title="Importance", yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fi_fig, width='stretch')

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=False)
if auto_refresh:
    time.sleep(300)
    st.rerun()
