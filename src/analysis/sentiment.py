"""
News fetching and sentiment analysis for gold price prediction.

Sources (in priority order):
  1. Reddit JSON API  — no auth, works from any cloud server
  2. yfinance news    — via ETF tickers (GLD, IAU, GDX)
  3. RSS feeds        — Bing News, Yahoo Finance, BBC
  4. NewsAPI          — optional, requires free key in .env as NEWS_API_KEY
"""

import os
import re
import html
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Expanded gold-tuned lexicon with 50+ domain-specific terms
_GOLD_LEXICON = {
    # Core gold terms
    "bullion": 1.5, "safe-haven": 2.0, "safe haven": 2.0,
    "gold demand": 1.5, "gold supply": -0.5, "gold reserves": 1.0,
    "central bank buying": 2.5, "central bank selling": -2.0,
    "etf outflows": -1.5, "etf inflows": 1.5,
    "store of value": 1.5, "inflation hedge": 2.0,
    "alltime high": 1.5, "record high": 1.5, "breakout": 1.0,
    "mining output": -0.5, "gold production": -0.3,
    "jewelry demand": 1.0, "chinese demand": 1.5, "india demand": 1.5,

    # Macro — inflation / monetary policy
    "inflation": 1.5, "stagflation": 2.0, "hyperinflation": 2.5,
    "deflation": -1.0, "disinflation": -0.8,
    "rate hike": -2.0, "rate cut": 2.0, "rate pause": 1.0,
    "rate pivot": 1.5, "pivot": 1.0,
    "fed hawkish": -2.0, "fed dovish": 2.0,
    "quantitative easing": 1.5, "qe": 1.0,
    "quantitative tightening": -1.5, "qt": -0.8, "tapering": -1.5,
    "interest rate": -0.5, "real yield": -1.5,
    "treasury yield": -1.0, "bond yield": -1.0,
    "yield inversion": 1.5, "inverted yield": 1.5,

    # Dollar
    "dollar strength": -2.0, "dollar weakness": 2.0,
    "dollar index": -0.8, "dxy": -0.5,
    "usd strong": -1.5, "usd weak": 1.5,
    "currency debasement": 2.0,

    # Risk / geopolitics
    "geopolitical": 1.0, "uncertainty": 1.0, "risk-off": 1.5, "risk off": 1.5,
    "tariff": 1.0, "sanctions": 1.0, "war": 1.5, "conflict": 1.0,
    "recession": -1.5, "economic slowdown": -0.8, "gdp contraction": -0.8,
    "banking crisis": 2.0, "bank failure": 1.5, "financial crisis": 2.0,
    "debt ceiling": 1.0, "sovereign debt": 0.8, "default risk": 1.5,

    # Technical trading
    "oversold": 1.0, "overbought": -1.0,
    "support level": 0.5, "resistance level": -0.3,
    "short squeeze": 1.0, "buying opportunity": 1.5,
    "profit taking": -0.8, "selling pressure": -1.0,
    "bearish": -1.5, "bullish": 1.5,
}
_analyzer.lexicon.update(_GOLD_LEXICON)

GOLD_KEYWORDS = re.compile(
    r"\bgold\b|\bxau\b|\bbullion\b|\bprecious metal\b|\bgold futures\b"
    r"|\bgold price\b|\bmcx\b|\bcomex\b|\bgld\b|\biau\b",
    re.IGNORECASE,
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# Source reliability weights — higher = more trusted for gold sentiment
# Financial publications get more weight than general news or social media
SOURCE_WEIGHTS = {
    "reuters":       1.4,
    "bloomberg":     1.4,
    "marketwatch":   1.3,
    "wsj":           1.3,
    "financial times": 1.3,
    "ft.com":        1.3,
    "kitco":         1.5,   # dedicated gold news site
    "goldprice":     1.4,
    "world gold council": 1.5,
    "yahoo finance": 1.1,
    "bbc":           1.1,
    "cnbc":          1.1,
    "reddit":        0.7,   # social media: down-weight
    "wallstreetbets": 0.5,  # highly speculative
}

RSS_FEEDS = {
    "Bing: gold price":   "https://www.bing.com/news/search?q=gold+price&format=RSS",
    "Bing: India gold":   "https://www.bing.com/news/search?q=MCX+gold+India+price&format=RSS",
    "Yahoo GLD":          "https://finance.yahoo.com/rss/headline?s=GLD",
    "BBC Business":       "https://feeds.bbci.co.uk/news/business/rss.xml",
}

REDDIT_SOURCES = [
    "https://www.reddit.com/r/Gold/new.json?limit=30",
    "https://www.reddit.com/r/investing/search.json?q=gold+price&sort=new&t=week&limit=25",
    "https://www.reddit.com/r/wallstreetbets/search.json?q=gold&sort=new&t=day&limit=20",
]


def _source_weight(source: str) -> float:
    """Return reliability weight for a given source name."""
    src_lower = source.lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if key in src_lower:
            return weight
    return 1.0


# ── Reddit (most reliable from cloud, no auth needed) ─────────────────────────

def _fetch_reddit(url: str, timeout: int = 10) -> list[dict]:
    """Fetch posts from Reddit JSON API. No auth required for GET requests."""
    try:
        headers = {"User-Agent": "GoldPricePredictor/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        articles = []
        for post in posts:
            p = post.get("data", {})
            title   = p.get("title", "")
            summary = p.get("selftext", "")[:300] or p.get("url", "")
            url_    = f"https://reddit.com{p.get('permalink', '')}"
            created = p.get("created_utc", 0)
            try:
                published = datetime.utcfromtimestamp(float(created))
            except Exception:
                published = datetime.utcnow()
            if title:
                articles.append({
                    "title":     title,
                    "summary":   summary,
                    "published": published,
                    "source":    f"Reddit r/{p.get('subreddit', 'Gold')}",
                    "url":       url_,
                })
        return articles
    except Exception:
        return []


# ── yfinance news ─────────────────────────────────────────────────────────────

def _fetch_yfinance_news() -> list[dict]:
    """Fetch news via yfinance for gold-related ETFs."""
    tickers   = ["GLD", "IAU", "GDX", "SLV", "GOLD"]
    seen_urls = set()
    articles  = []
    for sym in tickers:
        try:
            items = yf.Ticker(sym).news or []
            for item in items:
                # yfinance news schema changed in 0.2.x — handle both formats
                url = (
                    item.get("link")
                    or item.get("url")
                    or item.get("canonicalUrl", {}).get("url", "")
                    if isinstance(item, dict) else ""
                )
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                pub_ts = item.get("providerPublishTime") or 0
                try:
                    published = datetime.utcfromtimestamp(int(pub_ts))
                except Exception:
                    published = datetime.utcnow()
                title   = item.get("title", "") if isinstance(item, dict) else ""
                summary = (item.get("summary", "") or "") if isinstance(item, dict) else ""
                source  = item.get("publisher", "Yahoo Finance") if isinstance(item, dict) else "Yahoo Finance"
                if title:
                    articles.append({
                        "title": title, "summary": summary[:300],
                        "published": published, "source": source, "url": url,
                    })
        except Exception:
            continue
    return articles


# ── RSS feeds ─────────────────────────────────────────────────────────────────

def _parse_rss(content: bytes, source_name: str) -> list[dict]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    ns      = {"atom": "http://www.w3.org/2005/Atom"}
    channel = root.find("channel")
    if channel is not None:
        t = channel.find("title")
        if t is not None and t.text:
            source_name = t.text.strip()
        items = channel.findall("item")
    else:
        items = root.findall("atom:entry", ns) or root.findall("entry")

    articles = []
    for item in items[:40]:
        def txt(tag):
            el = item.find(tag) or item.find(f"atom:{tag}", ns)
            if el is None:
                return ""
            text = el.text or "".join(el.itertext())
            return re.sub(r"<[^>]+>", " ", text).strip()

        title   = txt("title")
        summary = txt("description") or txt("summary") or ""
        link    = txt("link")
        if not link:
            el = item.find("link")
            if el is not None:
                link = el.get("href", "")
        pub_str = txt("pubDate") or txt("published") or txt("updated")

        try:
            published = parsedate_to_datetime(pub_str).replace(tzinfo=None)
        except Exception:
            try:
                published = pd.to_datetime(pub_str).to_pydatetime().replace(tzinfo=None)
            except Exception:
                published = datetime.utcnow()

        if title:
            articles.append({
                "title": title, "summary": summary[:300],
                "published": published, "source": source_name, "url": link,
            })
    return articles


def _fetch_rss(feed_url: str, source_name: str, timeout: int = 10) -> list[dict]:
    try:
        resp = requests.get(feed_url, timeout=timeout, headers=_HEADERS)
        resp.raise_for_status()
        return _parse_rss(resp.content, source_name)
    except Exception:
        return []


# ── NewsAPI (optional) ────────────────────────────────────────────────────────

def _newsapi_fetch(api_key: str) -> list[dict]:
    from_date = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    try:
        resp = requests.get("https://newsapi.org/v2/everything", timeout=10, params={
            "q": "gold price", "from": from_date, "sortBy": "publishedAt",
            "language": "en", "pageSize": 50, "apiKey": api_key,
        })
        resp.raise_for_status()
        articles = []
        for art in resp.json().get("articles", []):
            articles.append({
                "title":     art.get("title", ""),
                "summary":   (art.get("description") or "")[:300],
                "published": art.get("publishedAt", datetime.utcnow().isoformat()),
                "source":    art.get("source", {}).get("name", "NewsAPI"),
                "url":       art.get("url", ""),
            })
        return articles
    except Exception:
        return []


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_gold_news(max_articles: int = 60) -> list[dict]:
    """
    Fetch gold-related news from multiple sources with a reliable fallback chain.
    Returns deduplicated list sorted by recency.
    """
    all_articles = []

    # 1. Reddit — most reliable from cloud servers, no auth
    for url in REDDIT_SOURCES:
        all_articles.extend(_fetch_reddit(url))

    # 2. yfinance news via ETF tickers
    all_articles.extend(_fetch_yfinance_news())

    # 3. RSS feeds
    for name, url in RSS_FEEDS.items():
        all_articles.extend(_fetch_rss(url, name))

    # 4. NewsAPI (optional)
    api_key = os.getenv("NEWS_API_KEY", "")
    if api_key:
        all_articles.extend(_newsapi_fetch(api_key))

    # Filter to gold-relevant
    gold_articles = [
        a for a in all_articles
        if GOLD_KEYWORDS.search(a.get("title", "")) or
           GOLD_KEYWORDS.search(a.get("summary", ""))
    ]
    # If filter is too aggressive, keep everything
    if len(gold_articles) < 5:
        gold_articles = all_articles

    # Deduplicate on title prefix
    seen, unique = set(), []
    for a in gold_articles:
        key = a.get("title", "")[:80].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    unique.sort(key=lambda x: x.get("published", datetime.min), reverse=True)
    return unique[:max_articles]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_article(article: dict) -> dict:
    """
    Score an article using VADER + gold-tuned lexicon.
    Applies source reliability weighting to the compound score.
    """
    text   = f"{article.get('title', '')}. {article.get('summary', '')}"
    scores = _analyzer.polarity_scores(text)

    # Reliability-weighted compound: scale toward 0 for less-trusted sources
    weight   = _source_weight(article.get("source", ""))
    raw_comp = scores["compound"]
    # Weight > 1 amplifies signal; weight < 1 dampens it
    weighted_comp = max(-1.0, min(1.0, raw_comp * weight))

    result = dict(article)
    result["compound"]  = round(weighted_comp, 4)
    result["raw_compound"] = round(raw_comp, 4)
    result["source_weight"] = weight
    result["sentiment"] = (
        "Bullish" if weighted_comp >= 0.05
        else "Bearish" if weighted_comp <= -0.05
        else "Neutral"
    )
    return result


def get_sentiment_summary(articles: Optional[list[dict]] = None) -> dict:
    if articles is None:
        articles = fetch_gold_news()
    scored = [score_article(a) for a in articles]
    if not scored:
        return {"avg_compound": 0.0, "label": "Neutral", "bullish_pct": 0.0,
                "bearish_pct": 0.0, "neutral_pct": 100.0, "article_count": 0, "articles": []}
    compounds = [a["compound"] for a in scored]
    n   = len(compounds)
    avg = sum(compounds) / n
    b   = sum(1 for c in compounds if c >= 0.05)
    br  = sum(1 for c in compounds if c <= -0.05)
    return {
        "avg_compound":  round(avg, 4),
        "label":         "Bullish" if avg >= 0.05 else "Bearish" if avg <= -0.05 else "Neutral",
        "bullish_pct":   round(100 * b / n, 1),
        "bearish_pct":   round(100 * br / n, 1),
        "neutral_pct":   round(100 * (n - b - br) / n, 1),
        "article_count": n,
        "articles":      scored,
    }


def build_daily_sentiment_series(articles: list[dict]) -> pd.Series:
    scored = [score_article(a) for a in articles]
    rows   = []
    for a in scored:
        pub = a.get("published", datetime.utcnow())
        if isinstance(pub, str):
            try:
                pub = pd.to_datetime(pub).to_pydatetime()
            except Exception:
                continue
        # Ensure pub is a datetime object before calling .date()
        if not isinstance(pub, datetime):
            try:
                pub = pd.Timestamp(pub).to_pydatetime()
            except Exception:
                continue
        rows.append({"date": pub.date(), "compound": a["compound"]})
    if not rows:
        return pd.Series(dtype=float, name="sentiment")
    df    = pd.DataFrame(rows)
    daily = df.groupby("date")["compound"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily.name  = "sentiment"
    return daily
