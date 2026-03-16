"""
News fetching and sentiment analysis for gold price prediction.

Source priority chain (all work from cloud servers):
  1. yfinance news  — built-in, most reliable, no rate limits
  2. Google News RSS — free, no key, works from servers
  3. Yahoo Finance RSS for GC=F — free, no key
  4. Optional: NewsAPI (requires free key in .env as NEWS_API_KEY)
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Gold-specific lexicon boosts
_GOLD_LEXICON = {
    "bullion": 1.5,
    "safe-haven": 2.0,
    "safe haven": 2.0,
    "inflation": 1.5,
    "stagflation": 2.0,
    "recession": -1.5,
    "rate hike": -2.0,
    "rate cut": 2.0,
    "geopolitical": 1.0,
    "uncertainty": 1.0,
    "dollar strength": -2.0,
    "dollar weakness": 2.0,
    "fed hawkish": -2.0,
    "fed dovish": 2.0,
    "central bank buying": 2.5,
    "etf outflows": -1.5,
    "etf inflows": 1.5,
    "tariff": 1.0,
    "sanctions": 1.0,
    "war": 1.5,
    "conflict": 1.0,
    "interest rate": -0.5,
}
_analyzer.lexicon.update(_GOLD_LEXICON)

# RSS feeds that reliably work from cloud/server environments
RSS_FEEDS = {
    # Bing News RSS — very reliable from cloud servers, no auth required
    "Bing: gold price":   "https://www.bing.com/news/search?q=gold+price&format=RSS",
    "Bing: gold futures": "https://www.bing.com/news/search?q=gold+futures+XAU+bullion&format=RSS",
    "Bing: India gold":   "https://www.bing.com/news/search?q=MCX+gold+price+India&format=RSS",
    # BBC Business — reliable public RSS
    "BBC Business":       "https://feeds.bbci.co.uk/news/business/rss.xml",
    # Yahoo Finance ETF headlines
    "Yahoo GLD":          "https://finance.yahoo.com/rss/headline?s=GLD",
    "Yahoo GC=F":         "https://finance.yahoo.com/rss/headline?s=GC%3DF",
}

GOLD_KEYWORDS = re.compile(
    r"\bgold\b|\bxau\b|\bbullion\b|\bprecious metal\b|\bgold futures\b|\bgold price\b",
    re.IGNORECASE,
)


def _parse_rss_xml(content: bytes, source_name: str) -> list[dict]:
    """Parse raw RSS/Atom XML bytes into article dicts."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    channel = root.find("channel")

    if channel is not None:
        # RSS 2.0
        t = channel.find("title")
        if t is not None and t.text:
            source_name = t.text.strip()
        items = channel.findall("item")
    else:
        # Atom
        items = root.findall("atom:entry", ns) or root.findall("entry")

    articles = []
    for item in items[:40]:
        def txt(tag):
            # try direct child, then with atom ns
            el = item.find(tag) or item.find(f"atom:{tag}", ns)
            if el is None:
                return ""
            # strip CDATA / HTML tags from summary
            text = el.text or "".join(el.itertext())
            text = re.sub(r"<[^>]+>", " ", text)
            return text.strip()

        title   = txt("title")
        summary = txt("description") or txt("summary") or txt("content")
        link    = txt("link")
        # <link> in Atom can be an element with href attribute
        if not link:
            el = item.find("link")
            if el is not None:
                link = el.get("href", "")
        pub_str = txt("pubDate") or txt("published") or txt("updated")

        try:
            published = parsedate_to_datetime(pub_str).replace(tzinfo=None) if pub_str else datetime.utcnow()
        except Exception:
            try:
                published = pd.to_datetime(pub_str).to_pydatetime().replace(tzinfo=None)
            except Exception:
                published = datetime.utcnow()

        if title:
            articles.append({
                "title":     title,
                "summary":   summary[:300],
                "published": published,
                "source":    source_name,
                "url":       link,
            })
    return articles


def _fetch_rss(feed_url: str, source_name: str, timeout: int = 10) -> list[dict]:
    """Fetch and parse one RSS feed. Returns [] on any failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        resp = requests.get(feed_url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return _parse_rss_xml(resp.content, source_name)
    except Exception:
        return []


def _fetch_yfinance_news(tickers: list[str] = None) -> list[dict]:
    """
    Fetch news via yfinance — works reliably from any environment.
    Uses ETF tickers (GLD, SLV, IAU) which have richer news than futures (GC=F).
    """
    if tickers is None:
        # ETFs first — they have the most news articles on yfinance
        tickers = ["GLD", "IAU", "SLV", "GDX", "GOLD", "GC=F"]

    seen_urls = set()
    articles  = []

    for sym in tickers:
        try:
            tk    = yf.Ticker(sym)
            items = tk.news or []
            for item in items:
                url = item.get("link") or item.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                pub_ts = item.get("providerPublishTime") or item.get("published", 0)
                try:
                    published = datetime.utcfromtimestamp(int(pub_ts))
                except Exception:
                    published = datetime.utcnow()

                title   = item.get("title", "")
                summary = item.get("summary", "") or ""
                source  = item.get("publisher", "") or item.get("source", {}).get("name", "Yahoo Finance")

                if title:
                    articles.append({
                        "title":     title,
                        "summary":   summary[:300],
                        "published": published,
                        "source":    source,
                        "url":       url,
                    })
        except Exception:
            continue

    return articles


def _newsapi_fetch(api_key: str, query: str = "gold price", days: int = 3) -> list[dict]:
    """Fetch from NewsAPI (requires free key at newsapi.org)."""
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey":   api_key,
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
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


def fetch_gold_news(max_articles: int = 60) -> list[dict]:
    """
    Fetch gold-related news from multiple sources with a reliable fallback chain.
    Returns deduplicated list sorted by recency.
    """
    all_articles = []

    # 1. yfinance news (most reliable from cloud)
    all_articles.extend(_fetch_yfinance_news())

    # 2. Google News RSS + Yahoo Finance RSS
    for name, url in RSS_FEEDS.items():
        all_articles.extend(_fetch_rss(url, name))

    # 3. NewsAPI (optional)
    api_key = os.getenv("NEWS_API_KEY", "")
    if api_key:
        all_articles.extend(_newsapi_fetch(api_key))

    # Filter to gold-relevant
    gold_articles = [
        a for a in all_articles
        if GOLD_KEYWORDS.search(a.get("title", "")) or GOLD_KEYWORDS.search(a.get("summary", ""))
    ]

    # If filter is too aggressive, keep all yfinance articles
    if len(gold_articles) < 5:
        gold_articles = all_articles

    # Deduplicate by title
    seen, unique = set(), []
    for a in gold_articles:
        key = a["title"][:60].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    # Sort newest first
    unique.sort(key=lambda x: x.get("published", datetime.min), reverse=True)
    return unique[:max_articles]


def score_article(article: dict) -> dict:
    """Add VADER compound sentiment score to an article dict."""
    text   = f"{article.get('title', '')}. {article.get('summary', '')}"
    scores = _analyzer.polarity_scores(text)
    result = dict(article)
    result["compound"]  = scores["compound"]
    result["pos"]       = scores["pos"]
    result["neg"]       = scores["neg"]
    result["neu"]       = scores["neu"]
    result["sentiment"] = (
        "Bullish" if scores["compound"] >= 0.05
        else "Bearish" if scores["compound"] <= -0.05
        else "Neutral"
    )
    return result


def get_sentiment_summary(articles: Optional[list[dict]] = None) -> dict:
    """Aggregate sentiment across articles into a summary dict."""
    if articles is None:
        articles = fetch_gold_news()

    scored = [score_article(a) for a in articles]

    if not scored:
        return {
            "avg_compound":  0.0,
            "label":         "Neutral",
            "bullish_pct":   0.0,
            "bearish_pct":   0.0,
            "neutral_pct":   100.0,
            "article_count": 0,
            "articles":      [],
        }

    compounds = [a["compound"] for a in scored]
    n         = len(compounds)
    avg       = sum(compounds) / n
    bullish   = sum(1 for c in compounds if c >= 0.05)
    bearish   = sum(1 for c in compounds if c <= -0.05)
    neutral   = n - bullish - bearish
    label     = "Bullish" if avg >= 0.05 else "Bearish" if avg <= -0.05 else "Neutral"

    return {
        "avg_compound":  round(avg, 4),
        "label":         label,
        "bullish_pct":   round(100 * bullish / n, 1),
        "bearish_pct":   round(100 * bearish / n, 1),
        "neutral_pct":   round(100 * neutral / n, 1),
        "article_count": n,
        "articles":      scored,
    }


def build_daily_sentiment_series(articles: list[dict]) -> pd.Series:
    """Aggregate scored articles into a daily average compound score series."""
    scored = [score_article(a) for a in articles]
    rows   = []
    for a in scored:
        pub = a.get("published", datetime.utcnow())
        if isinstance(pub, str):
            try:
                pub = pd.to_datetime(pub).to_pydatetime()
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
