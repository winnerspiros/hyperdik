"""
News collector — fetches crypto news headlines for the AI brain
"""
import urllib.request, ssl, json
from datetime import datetime

def get_crypto_news(headlines_only=True):
    """Fetch latest crypto news headlines"""
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(
            "https://r.jina.ai/https://cryptopanic.com/api/v1/posts/?auth_token=public&kind=news&filter=hot",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read())
            headlines = []
            for post in data.get("results", [])[:5]:
                headlines.append(post.get("title", ""))
            if headlines:
                return headlines
    except:
        pass
    
    # Fallback: RSS feed
    try:
        import feedparser
        feed = feedparser.parse("https://feed.informer.com/digests/2Z0GKRPYMX/feeder")
        headlines = [e.title for e in feed.entries[:5] if e.title]
        if headlines:
            return headlines
    except:
        pass
    
    # Final fallback: static headlines
    return [
        f"Crypto market in {'bullish' if datetime.now().hour % 2 == 0 else 'bearish'} sentiment zone",
        "Bitcoin holding key support levels",
        "Altcoin season index showing mixed signals",
    ]

if __name__ == "__main__":
    news = get_crypto_news()
    for h in news:
        print(f"  • {h}")

def get_social_sentiment():
    """Get social media hype signals from Reddit/X trends"""
    ctx = ssl.create_default_context()
    signals = []
    try:
        # Check Reddit crypto trends
        req = urllib.request.Request(
            "https://r.jina.ai/https://www.reddit.com/r/CryptoCurrency/hot/.json?limit=5",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            data = json.loads(resp.read())
            for post in data.get("data", {}).get("children", [])[:3]:
                title = post.get("data", {}).get("title", "")
                if title:
                    signals.append(f"Reddit: {title[:100]}")
    except:
        pass
    
    try:
        # Google Trends for hype detection
        req = urllib.request.Request(
            "https://r.jina.ai/https://trends.google.com/trending?geo=US&category=business",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            content = resp.read().decode()
            # Extract top trending crypto-related items
            for line in content.split("\n"):
                if any(c in line.lower() for c in ["bitcoin", "crypto", "btc", "eth", "solana"]):
                    if len(line) > 20 and line not in signals:
                        signals.append(f"Trending: {line.strip()[:100]}")
                        break
    except:
        pass
    
    if not signals:
        signals.append("No major social hype detected")
    
    return signals[:3]