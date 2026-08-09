"""
Market Intelligence — external data enrichment for AI trading prompts.
Pulls Fear & Greed, trending coins, news headlines, global metrics.
All free APIs, cached (60s TTL), fails silently on any error.

Usage:
    from market_intel import get_market_brief
    brief = get_market_brief()  # "Fear:45(Neutral) BTC.dom:52% Trending:SOL,INJ,CASH"
"""

import json, time, ssl, urllib.request
from pathlib import Path

CACHE = Path("/tmp/market_intel_cache.json")
TTL = 60
CTX = ssl.create_default_context()


def _get(url, timeout=8):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _cached(key, fetcher):
    try:
        raw = CACHE.read_text()
        c = json.loads(raw)
    except Exception:
        c = {}
    e = c.get(key)
    if e and time.time() - e["ts"] < TTL:
        return e["data"]
    data = fetcher()
    if data is not None:
        c[key] = {"ts": time.time(), "data": data}
        CACHE.write_text(json.dumps(c))
    return data


# ── Data sources ────────────────────────────────────────────────────────────

def get_fear_greed():
    def fetch():
        d = _get("https://api.alternative.me/fng/?limit=1")
        if d and d.get("data"):
            e = d["data"][0]
            return {"value": int(e.get("value", 50)), "label": e.get("value_classification", "Neutral")}
        return None
    return _cached("fng", fetch)


def get_trending():
    def fetch():
        d = _get("https://api.coingecko.com/api/v3/search/trending")
        if d and d.get("coins"):
            return [c["item"]["symbol"].upper() for c in d["coins"][:7] if c.get("item", {}).get("symbol")]
        return None
    return _cached("trending", fetch)


def get_global_market():
    def fetch():
        d = _get("https://api.coingecko.com/api/v3/global")
        if d and d.get("data"):
            g = d["data"]
            return {
                "btc_dominance": g.get("market_cap_percentage", {}).get("btc"),
                "eth_dominance": g.get("market_cap_percentage", {}).get("eth"),
                "total_mcap_change": g.get("market_cap_change_percentage_24h_usd"),
            }
        return None
    return _cached("global", fetch)


def get_news_headlines():
    def fetch():
        try:
            d = _get("https://cryptopanic.com/api/v1/posts/?auth_token=public&kind=news&filter=hot")
            if d and d.get("results"):
                return [p.get("title", "")[:120] for p in d["results"][:5]]
        except Exception:
            pass
        return None
    return _cached("news", fetch)


# ── Composite brief for AI prompt ──────────────────────────────────────────

def get_market_brief():
    """Return a compact 1-3 line string for the AI prompt header."""
    parts = []

    fng = get_fear_greed()
    if fng:
        parts.append(f"Fear:{fng['value']}({fng['label']})")

    gm = get_global_market()
    if gm:
        btc_d = gm.get("btc_dominance")
        if btc_d:
            parts.append(f"BTC.dom:{btc_d:.1f}%")
        chg = gm.get("total_mcap_change")
        if chg is not None:
            arrow = "▲" if chg > 0 else "▼"
            parts.append(f"Mcap24h:{arrow}{abs(chg):.1f}%")

    trending = get_trending()
    if trending:
        parts.append(f"Hot:{','.join(trending[:5])}")

    news = get_news_headlines()
    if news:
        parts.append(f"News:{news[0][:80]}")

    return " | ".join(parts) if parts else ""


def get_coin_sentiment_hint(coins: list[str]) -> str:
    """Check if any of our candidate coins are trending on CoinGecko."""
    trending = get_trending() or []
    hits = [c for c in coins if c.upper() in [t.upper() for t in trending]]
    if hits:
        return f"[TRENDING: {','.join(hits)}]"
    return ""


# ── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    brief = get_market_brief()
    print(f"Market brief: {brief}")
    print(f"Sentiment hint: {get_coin_sentiment_hint(['SOL', 'BTC', 'CASHCAT'])}")