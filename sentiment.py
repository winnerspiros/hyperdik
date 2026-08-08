"""
Sentiment Analyzer — news and social sentiment for crypto trading
Sources: News headlines, social mentions scoring
"""
import time
import json
import urllib.request
import ssl
import re

class SentimentAnalyzer:
    """Analyze news/social sentiment for crypto assets"""
    
    # Cryptocurrency keyword mapping
    COIN_KEYWORDS = {
        "BTC": ["bitcoin", "btc", "#bitcoin", "#btc"],
        "ETH": ["ethereum", "eth", "#ethereum", "#eth"],
        "SOL": ["solana", "sol", "#solana", "#sol"],
        "XRP": ["xrp", "ripple", "#xrp", "#ripple"],
        "DOGE": ["dogecoin", "doge", "#dogecoin", "#doge"],
        "ADA": ["cardano", "ada", "#cardano", "#ada"],
        "LINK": ["chainlink", "link", "#chainlink"],
        "AVAX": ["avalanche", "avax", "#avalanche", "#avax"],
        "DOT": ["polkadot", "dot", "#polkadot"],
    }
    
    def __init__(self):
        self.ctx = ssl.create_default_context()
    
    def _fetch_json(self, url, timeout=10):
        """Fetch and parse JSON from URL"""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, context=self.ctx, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None
    
    def _fetch_text(self, url, timeout=10):
        """Fetch text content from URL"""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, context=self.ctx, timeout=timeout) as resp:
                return resp.read().decode()
        except Exception:
            return None
    
    def get_crypto_news_sentiment(self, coin="BTC"):
        """Get news headlines related to a crypto asset and score sentiment"""
        keywords = self.COIN_KEYWORDS.get(coin, [coin.lower()])
        query = " OR ".join(keywords)
        
        # Use NewsAPI-like free source (GDELT, or alternative)
        # Primary: CryptoCompare news API (free, no key needed)
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&limit=10"
        data = self._fetch_json(url)
        
        headlines = []
        if data and data.get("Data"):
            for article in data["Data"][:10]:
                headlines.append({
                    "title": article.get("title", ""),
                    "body": article.get("body", "")[:500],
                    "source": article.get("source", "cryptocompare"),
                    "published": article.get("published_on", 0),
                    "categories": article.get("categories", ""),
                })
        
        # Backup: use a free RSS/news aggregator
        if not headlines:
            # Try Alternative: Fear & Greed Index
            fng_url = "https://api.alternative.me/fng/?limit=1"
            fng = self._fetch_json(fng_url)
            if fng and fng.get("data"):
                headlines.append({
                    "title": f"Fear & Greed Index: {fng['data'][0].get('value', 'N/A')} - {fng['data'][0].get('value_classification', 'N/A')}",
                    "body": "",
                    "source": "alternative.me",
                    "published": fng['data'][0].get('timestamp', int(time.time())),
                    "categories": "sentiment",
                })
        
        # Score each headline using simple keyword sentiment
        positive_words = {"bullish", "surge", "gain", "rally", "breakthrough", "adoption",
                          "positive", "growth", "upgrade", "launch", "partnership", "green",
                          "profits", "institutional", "approval", "all-time", "high"}
        negative_words = {"bearish", "crash", "dump", "loss", "ban", "hack", "scam",
                          "negative", "decline", "drop", "fear", "uncertainty", "volatile",
                          "regulation", "crackdown", "investigation", "fine", "fraud"}
        
        total_score = 0
        for h in headlines:
            text = (h["title"] + " " + h["body"]).lower()
            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)
            h["score"] = pos_count - neg_count
            total_score += h["score"]
        
        avg_score = total_score / max(len(headlines), 1)
        sentiment = "bullish" if avg_score > 1 else ("bearish" if avg_score < -1 else "neutral")
        
        return {
            "coin": coin,
            "sentiment": sentiment,
            "score": avg_score,
            "headlines": headlines,
            "count": len(headlines),
        }
    
    def get_market_fear_greed(self):
        """Get Crypto Fear & Greed Index"""
        data = self._fetch_json("https://api.alternative.me/fng/?limit=1")
        if data and data.get("data"):
            entry = data["data"][0]
            return {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
                "timestamp": entry.get("timestamp", int(time.time())),
            }
        return {"value": 50, "classification": "Neutral", "timestamp": int(time.time())}
    
    def get_market_overview(self):
        """Get overall market sentiment"""
        fng = self.get_market_fear_greed()
        
        # Get top coins sentiment
        coins = ["BTC", "ETH", "SOL", "XRP"]
        sentiments = {}
        for coin in coins:
            s = self.get_crypto_news_sentiment(coin)
            sentiments[coin] = {"sentiment": s["sentiment"], "score": s["score"]}
        
        # Get recent crypto headlines from CryptoCompare
        all_news = self._fetch_json("https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit=5")
        top_stories = []
        if all_news and all_news.get("Data"):
            for article in all_news["Data"][:5]:
                top_stories.append({
                    "title": article.get("title", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                })
        
        return {
            "fear_greed": fng,
            "coin_sentiment": sentiments,
            "top_stories": top_stories,
            "overall": fng["classification"],
        }

    def get_reddit_sentiment_simple(self, coin="BTC"):
        """Get Reddit sentiment via pushshift or similar free API"""
        # Use pushshift.io for Reddit data (free, no key)
        subreddits = "cryptocurrency+bitcoin+ethereum+cryptomarkets"
        coin_names = self.COIN_KEYWORDS.get(coin, [coin.lower()])
        
        try:
            url = f"https://api.pullpush.io/reddit/search/submission/?subreddit={subreddits}&q={'%20'.join(coin_names[:2])}&size=10&sort=desc&sort_type=created_utc"
            data = self._fetch_json(url)
            
            posts = []
            if data and data.get("data"):
                for post in data["data"][:10]:
                    posts.append({
                        "title": post.get("title", ""),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "created": post.get("created_utc", 0),
                    })
            
            if posts:
                avg_reddit_score = sum(p["score"] for p in posts) / len(posts)
                return {"coin": coin, "posts": len(posts), "avg_score": avg_reddit_score}
        except Exception:
            pass
        
        return {"coin": coin, "posts": 0, "avg_score": 0}

    def get_composite_sentiment(self, coin="BTC"):
        """Combine news + fear+greed + social for a composite sentiment score"""
        news = self.get_crypto_news_sentiment(coin)
        fng = self.get_market_fear_greed()
        reddit = self.get_reddit_sentiment_simple(coin)
        
        # Weighted composite score (-10 to +10)
        news_weight = 0.4
        fng_weight = 0.4
        reddit_weight = 0.2
        
        # Convert FNG (0-100) to -10 to +10 scale
        fng_score = (fng["value"] - 50) / 5  # -10 to +10
        
        # News score (-10 to +10)
        news_score = max(-10, min(10, news["score"] * 3))
        
        # Reddit score
        reddit_score = max(-5, min(5, reddit.get("avg_score", 0) / 10))
        
        composite = (news_score * news_weight + fng_score * fng_weight + reddit_score * reddit_weight)
        
        if composite > 2:
            sentiment = "bullish"
        elif composite < -2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return {
            "coin": coin,
            "composite": round(composite, 2),
            "sentiment": sentiment,
            "components": {
                "news": {"score": round(news_score, 2), "headlines": len(news["headlines"])},
                "fear_greed": {"value": fng["value"], "classification": fng["classification"]},
                "reddit": {"posts": reddit.get("posts", 0)},
            }
        }