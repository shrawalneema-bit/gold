"""
News fetching and sentiment analysis for gold price prediction.

Sources (all free, no API key required):
  - RSS feeds from Reuters, MarketWatch, Kitco, Investing.com
  - Optional: NewsAPI (requires free API key in .env as NEWS_API_KEY)
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Gold-specific lexicon boosts (VADER allows custom words)
_GOLD_LEXICON = {
    "bullion": 1.5,
    "safe-haven": 2.0,
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
}
_analyzer.lexicon.update(_GOLD_LEXICON)


RSS_FEEDS = {
    "Reuters Commodities": "https://feeds.reuters.com/reuters/businessNews",
    "Kitco Gold News":     "https://www.kitco.com/rss/",
    "MarketWatch":         "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "Investing.com Gold":  "https://www.investing.com/rss/news_25.rss",
}

GOLD_KEYWORDS = re.compile(
    r"\bgold\b|\bxau\b|\bbullion\b|\bprecious metal\b|\bgold futures\b|\bgold price\b",
    re.IGNORECASE,
)


def _fetch_rss(feed_url: str, timeout: int = 8) -> list[dict]:
    """Parse an RSS feed using stdlib xml; return list of {title, summary, published} dicts."""
    try:
        resp = requests.get(feed_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        # Handle both RSS and Atom namespaces
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        channel = root.find("channel")
        source_title = feed_url
        if channel is not None:
            t = channel.find("title")
            if t is not None and t.text:
                source_title = t.text
            items = channel.findall("item")
        else:
            items = root.findall("atom:entry", ns)

        articles = []
        for item in items[:30]:
            def txt(tag):
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            title   = txt("title")
            summary = txt("description") or txt("summary") or txt("atom:summary")
            link    = txt("link")
            pub_str = txt("pubDate") or txt("published") or txt("updated")

            try:
                published = parsedate_to_datetime(pub_str) if pub_str else datetime.utcnow()
                published = published.replace(tzinfo=None)
            except Exception:
                published = datetime.utcnow()

            articles.append({
                "title":     title,
                "summary":   summary,
                "published": published,
                "source":    source_title,
                "url":       link,
            })
        return articles
    except Exception:
        return []


def _newsapi_fetch(api_key: str, query: str = "gold price", days: int = 3) -> list[dict]:
    """Fetch from NewsAPI (requires free key)."""
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey":   api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title":     art.get("title", ""),
                "summary":   art.get("description", ""),
                "published": art.get("publishedAt", ""),
                "source":    art.get("source", {}).get("name", "NewsAPI"),
                "url":       art.get("url", ""),
            })
        return articles
    except Exception:
        return []


def fetch_gold_news(max_articles: int = 60) -> list[dict]:
    """
    Fetch gold-related news from RSS feeds + optional NewsAPI.
    Returns a list of article dicts, filtered to gold-relevant content.
    """
    all_articles = []

    # RSS feeds
    for name, url in RSS_FEEDS.items():
        articles = _fetch_rss(url)
        all_articles.extend(articles)

    # NewsAPI (optional)
    api_key = os.getenv("NEWS_API_KEY", "")
    if api_key:
        all_articles.extend(_newsapi_fetch(api_key))

    # Filter to gold-relevant articles
    gold_articles = [
        a for a in all_articles
        if GOLD_KEYWORDS.search(a["title"]) or GOLD_KEYWORDS.search(a["summary"])
    ]

    # De-duplicate by title
    seen = set()
    unique = []
    for a in gold_articles:
        key = a["title"][:60].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    # Sort by recency
    unique.sort(key=lambda x: x.get("published", datetime.min), reverse=True)
    return unique[:max_articles]


def score_article(article: dict) -> dict:
    """Add VADER compound sentiment score to an article dict."""
    text = f"{article['title']}. {article['summary']}"
    scores = _analyzer.polarity_scores(text)
    article = dict(article)
    article["compound"]   = scores["compound"]
    article["pos"]        = scores["pos"]
    article["neg"]        = scores["neg"]
    article["neu"]        = scores["neu"]
    article["sentiment"]  = (
        "Bullish" if scores["compound"] >= 0.05
        else "Bearish" if scores["compound"] <= -0.05
        else "Neutral"
    )
    return article


def get_sentiment_summary(articles: Optional[list[dict]] = None) -> dict:
    """
    Aggregate sentiment across all articles into a summary dict.

    Returns:
        avg_compound: float in [-1, 1]
        label: 'Bullish' | 'Bearish' | 'Neutral'
        bullish_pct, bearish_pct, neutral_pct
        article_count
        articles: list of scored articles
    """
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
    avg       = sum(compounds) / len(compounds)
    bullish   = sum(1 for c in compounds if c >= 0.05)
    bearish   = sum(1 for c in compounds if c <= -0.05)
    neutral   = len(compounds) - bullish - bearish
    n         = len(compounds)

    label = "Bullish" if avg >= 0.05 else "Bearish" if avg <= -0.05 else "Neutral"

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
    """
    Aggregate scored articles by date into a daily average compound score.
    Useful for joining sentiment as a feature in the ML pipeline.
    """
    scored = [score_article(a) for a in articles]
    rows = []
    for a in scored:
        pub = a["published"]
        if isinstance(pub, str):
            try:
                pub = pd.to_datetime(pub).to_pydatetime()
            except Exception:
                continue
        rows.append({"date": pub.date(), "compound": a["compound"]})

    if not rows:
        return pd.Series(dtype=float, name="sentiment")

    df = pd.DataFrame(rows)
    daily = df.groupby("date")["compound"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily.name  = "sentiment"
    return daily
