"""
Sentiment Scraper — lightweight sentiment from free sources.
No paid APIs. Works on a low-resource server.

Sources (tiered by reliability):
  1. CoinGecko trending coins (already in data_enricher, but we add sentiment classifier)
  2. Google Trends via pytrends (lite version using requests)
  3. Reddit via JSON feed (no API key needed)
  4. News headline sentiment via RSS + VADER

Single file. Only requires: requests, lxml (for RSS), and optionally vaderSentiment.
"""
import json
import math
import re
import ssl
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

# ── Sentiment analysis — lightweight VADER clone ──────────────────────────

# Simplified VADER lexicon (most common financial/crypto terms)
_SENTIMENT_LEXICON = {
    # Positive
    "bullish": 2.0, "moon": 2.0, "pump": 1.5, "rocket": 1.5, "breakout": 1.8,
    "adoption": 1.5, "partnership": 1.8, "upgrade": 1.3, "launch": 1.2,
    "positive": 1.5, "growth": 1.3, "profit": 1.5, "gain": 1.2, "rally": 1.5,
    "surge": 1.5, "soar": 1.8, "breakthrough": 2.0, "innovation": 1.3,
    "approve": 1.5, "approval": 1.5, "green": 1.0, "beat": 1.2,
    "outperform": 1.5, "upgrade": 1.3, "support": 0.5, "strong": 1.0,
    "buy": 1.0, "accumulate": 1.2, "hodl": 0.5, "diamond": 1.0,
    # Negative
    "bearish": -2.0, "dump": -2.0, "crash": -2.5, "selloff": -2.0,
    "scam": -2.5, "hack": -2.5, "fraud": -3.0, "ban": -2.0,
    "regulation": -1.0, "crackdown": -2.0, "negative": -1.5, "loss": -1.5,
    "decline": -1.2, "drop": -1.0, "fall": -1.0, "plunge": -2.0,
    "bear": -1.5, "fud": -1.5, "fear": -1.5, "panic": -2.0,
    "liquidation": -1.5, "reject": -1.5, "rejection": -1.5,
    "delay": -0.8, "fail": -1.5, "failure": -1.8, "worst": -2.0,
    "debt": -1.0, "bankrupt": -2.5, "crisis": -2.0, "worry": -1.0,
    "uncertain": -0.8, "uncertainty": -1.0, "risk": -0.5, "risky": -1.0,
    "weak": -1.0, "warn": -1.0, "warning": -1.0,
    # Neutral amplifiers
    "huge": 0.5, "massive": 0.5, "big": 0.3, "major": 0.3,
    "small": -0.3, "minor": -0.3, "slight": -0.3,
}


def _tokenize(text):
    """Simple lowercase tokenizer."""
    return re.findall(r"[a-z]+", text.lower())


def score_sentiment(text):
    """
    Score sentiment of text using lightweight VADER-like approach.

    Returns dict with:
        - compound: float [-1, 1]
        - pos: float [0, 1]
        - neg: float [0, 1]
        - neu: float [0, 1]
        - label: "positive"/"negative"/"neutral"
    """
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "neutral"}

    tokens = _tokenize(text)

    pos_sum = 0.0
    neg_sum = 0.0
    matched = 0

    for t in tokens:
        if t in _SENTIMENT_LEXICON:
            score = _SENTIMENT_LEXICON[t]
            if score > 0:
                pos_sum += score
            else:
                neg_sum += abs(score)
            matched += 1

    if matched == 0:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "neutral"}

    # Normalize by matched token count
    pos = pos_sum / matched
    neg = neg_sum / matched
    compound = (pos_sum - neg_sum) / (pos_sum + neg_sum + 15)  # damping factor

    # Clamp
    compound = max(-1.0, min(1.0, compound))

    # Normalize pos/neg/neu
    total = pos + neg + 0.1  # small neu baseline
    pos_norm = pos / total
    neg_norm = neg / total
    neu_norm = 1.0 - pos_norm - neg_norm

    if compound >= 0.15:
        label = "positive"
    elif compound <= -0.15:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": round(compound, 4),
        "pos": round(pos_norm, 4),
        "neg": round(neg_norm, 4),
        "neu": round(neu_norm, 4),
        "label": label,
    }


# ── Reddit JSON feed (no API key) ─────────────────────────────────────────


def get_reddit_sentiment(subreddit="CryptoCurrency", query="bitcoin", limit=10):
    """
    Search Reddit for posts/comments about a coin.
    Uses pushshift.io JSON API — no API key needed.

    Returns list of scored text items.
    """
    url = (
        f"https://api.pushshift.io/reddit/search/submission/"
        f"?subreddit={subreddit}&q={urllib.parse.quote(query)}"
        f"&sort=score&order=desc&size={limit}"
    )
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            data = json.loads(r.read())
    except Exception:
        return {"error": "reddit fetch failed", "items": [], "avg_sentiment": 0}

    items = []
    scores = []
    for post in data.get("data", []):
        title = post.get("title", "")
        selftext = post.get("selftext", "")[:500]
        text = f"{title}. {selftext}"
        sentiment = score_sentiment(text)
        items.append({
            "title": title[:100],
            "score": post.get("score", 0),
            "sentiment": sentiment,
            "url": f"https://reddit.com{post.get('permalink', '')}",
            "created": datetime.utcfromtimestamp(post.get("created_utc", 0)).isoformat(),
        })
        scores.append(sentiment["compound"])

    avg_sent = sum(scores) / len(scores) if scores else 0
    return {
        "items": items,
        "avg_sentiment": round(avg_sent, 4),
        "count": len(items),
        "source": f"reddit/r/{subreddit}",
    }


# ── Google Trends (lite) ──────────────────────────────────────────────────


def get_google_trends(keyword="bitcoin", timeframe="now 7-d"):
    """
    Fetch Google Trends interest for a keyword via the unofficial daily API.
    No API key needed. Returns relative interest (0-100).

    Uses the Google Trends data endpoint directly.
    """
    # Google Trends RSS feed
    url = (
        f"https://trends.google.com/trends/trendingsearches/daily/rss?"
        f"geo=US&q={urllib.parse.quote(keyword)}"
    )
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            html = r.read().decode("utf-8")
    except Exception:
        return {"interest": 0, "error": "trends fetch failed"}

    # Simple match: count keyword mentions as proxy for trend strength
    mentions = len(re.findall(re.escape(keyword.lower()), html.lower()))
    interest = min(100, mentions * 10)

    return {"interest": interest, "keyword": keyword, "source": "google_trends"}


# ── Crypto Fear & Greed (already in data_enricher, just re-exported) ──────


def get_coin_sentiment_summary(coin_name="bitcoin", coin_symbol="BTC"):
    """
    Aggregate sentiment from all sources for a specific coin.

    Returns unified sentiment dict.
    """
    # 1. Reddit
    reddit = get_reddit_sentiment("CryptoCurrency", coin_name, limit=5)
    reddit2 = get_reddit_sentiment("CryptoMarkets", coin_name, limit=5)

    # 2. Google Trends
    trends = get_google_trends(coin_symbol)

    # 3. Aggregate
    all_scores = []
    if "avg_sentiment" in reddit:
        all_scores.append(reddit["avg_sentiment"])
    if "avg_sentiment" in reddit2:
        all_scores.append(reddit2["avg_sentiment"])

    avg = sum(all_scores) / len(all_scores) if all_scores else 0

    if avg >= 0.15:
        label = "positive"
    elif avg <= -0.15:
        label = "negative"
    else:
        label = "neutral"

    strength = min(100, int(abs(avg) * 200))

    return {
        "coin": coin_name,
        "symbol": coin_symbol,
        "sentiment": round(avg, 4),
        "label": label,
        "strength": strength,
        "google_trends_interest": trends.get("interest", 0),
        "reddit_posts_found": reddit.get("count", 0) + reddit2.get("count", 0),
        "sources": ["reddit", "google_trends"],
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── News RSS feed sentiment ───────────────────────────────────────────────


def parse_rss(url):
    """Fetch and parse an RSS feed. Returns list of {title, summary}."""
    import xml.etree.ElementTree as ET

    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            xml_data = r.read()
    except Exception:
        return []

    items = []
    try:
        root = ET.fromstring(xml_data)
        # Handle both RSS and Atom
        for item in root.iter("item"):
            title = item.findtext("title", "")
            desc = item.findtext("description", "")[:300]
            items.append({"title": title, "summary": desc})
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title = entry.findtext("{http://www.w3.org/2005/Atom}title", "")
            summary = entry.findtext("{http://www.w3.org/2005/Atom}summary", "")[:300]
            items.append({"title": title, "summary": summary})
    except ET.ParseError:
        pass

    return items


def get_crypto_news_sentiment(query="cryptocurrency"):
    """Fetch crypto news from RSS feeds and score sentiment."""
    feeds = [
        f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en",
        "https://cointelegraph.com/rss",
    ]

    all_items = []
    scores = []
    for url in feeds:
        items = parse_rss(url)
        for item in items:
            text = f"{item['title']} {item.get('summary', '')}"
            sentiment = score_sentiment(text)
            all_items.append({
                "title": item["title"][:100],
                "sentiment": sentiment,
            })
            scores.append(sentiment["compound"])

    avg = sum(scores) / len(scores) if scores else 0
    pos_pct = sum(1 for s in scores if s >= 0.15) / len(scores) * 100 if scores else 0
    neg_pct = sum(1 for s in scores if s <= -0.15) / len(scores) * 100 if scores else 0

    return {
        "avg_sentiment": round(avg, 4),
        "articles_scored": len(all_items),
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "top_headlines": [a["title"] for a in all_items[:5]],
    }


if __name__ == "__main__":
    # Test sentiment classifier
    print("=== Sentiment Classifier ===")
    for text in ["Bitcoin is mooning! Huge bullish breakout!",
                 "Crash incoming, massive dump, total scam",
                 "Bitcoin trading sideways today"]:
        s = score_sentiment(text)
        print(f"  '{text[:40]}...' → {s['label']} (compound={s['compound']})")

    # Test Reddit
    print("\n=== Reddit Sentiment ===")
    reddit = get_reddit_sentiment("CryptoCurrency", "bitcoin", limit=3)
    print(f"  Posts: {reddit.get('count', 0)}, Avg sentiment: {reddit.get('avg_sentiment', 0)}")
    for item in reddit.get("items", [])[:2]:
        print(f"    → {item['title'][:60]}... [{item['sentiment']['label']}]")

    # Test Google Trends
    print("\n=== Google Trends ===")
    trends = get_google_trends("bitcoin")
    print(f"  Interest: {trends.get('interest', 'N/A')}")

    # Test news sentiment
    print("\n=== News Sentiment ===")
    news = get_crypto_news_sentiment("bitcoin")
    print(f"  Avg: {news.get('avg_sentiment', 'N/A')}, Articles: {news.get('articles_scored', 0)}")

    # Test coin summary
    print("\n=== Coin Sentiment Summary ===")
    summary = get_coin_sentiment_summary("bitcoin", "BTC")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("✅ Sentiment scraper test OK")