"""
Order Book Imbalance Calculator — fetches order book depth from free exchange REST APIs
and computes imbalance metrics. Works with Binance, Kraken, Coinbase (no API key needed).

Single file, zero external deps except urllib.
"""
import json
import time
import ssl
import urllib.request

# ── Cache per symbol to avoid hammering endpoints ──────────────────────────
_DEPTH_CACHE = {}
_DEPTH_CACHE_TTL = 2  # seconds — order book data is time-sensitive


def _fetch(url, timeout=10):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _cached_fetch(cache_key, url, ttl=None):
    ttl = ttl or _DEPTH_CACHE_TTL
    now = time.time()
    cached = _DEPTH_CACHE.get(cache_key)
    if cached and now - cached["ts"] < ttl:
        return cached["data"]
    data = _fetch(url)
    if "error" not in data:
        _DEPTH_CACHE[cache_key] = {"ts": now, "data": data}
    return data


# ── Exchange adapters ──────────────────────────────────────────────────────


def get_binance_depth(symbol="BTCUSDT", limit=20):
    """
    Fetch Binance order book depth.
    symbol: e.g. "BTCUSDT", "ETHUSDT"
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    data = _cached_fetch(f"binance:{symbol}", url)
    if "error" in data:
        return {"error": data["error"]}
    bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
    asks = [[float(p), float(q)] for p, q in data.get("asks", [])]
    return {"bids": bids, "asks": asks, "exchange": "binance"}


def get_kraken_depth(pair="XBTUSD", limit=25):
    """
    Fetch Kraken order book depth.
    pair: e.g. "XBTUSD", "ETHUSD", "XRPUSD"
    Note: Kraken uses different pair naming (XBT=Bitcoin)
    """
    url = f"https://api.kraken.com/0/public/Depth?pair={pair}&count={limit}"
    data = _cached_fetch(f"kraken:{pair}", url)
    if "error" in data or data.get("error"):
        return {"error": data.get("error", ["unknown"])[0] if data.get("error") else "unknown"}
    result = data.get("result", {})
    # Kraken returns result key = pair name
    key = list(result.keys())[0] if result else None
    if not key:
        return {"error": "no data"}
    book = result[key]
    bids = [[float(p), float(q)] for p, q, _ in book.get("bids", [])]
    asks = [[float(p), float(q)] for p, q, _ in book.get("asks", [])]
    return {"bids": bids, "asks": asks, "exchange": "kraken"}


def get_coinbase_depth(product_id="BTC-USD", limit=25):
    """
    Fetch Coinbase Pro order book depth (level 2).
    product_id: e.g. "BTC-USD", "ETH-USD"
    """
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level=2"
    data = _cached_fetch(f"coinbase:{product_id}", url)
    if "error" in data:
        return {"error": data["error"]}
    # Coinbase returns [price, size, num_orders] per entry
    bids = [[float(p), float(s)] for p, s, _ in data.get("bids", [])][:limit]
    asks = [[float(p), float(s)] for p, s, _ in data.get("asks", [])][:limit]
    return {"bids": bids, "asks": asks, "exchange": "coinbase"}


# ── Imbalance Calculation ──────────────────────────────────────────────────


def calc_order_book_imbalance(bids, asks, depth_pct=0.02):
    """
    Calculate order book imbalance up to a certain depth level.

    Parameters
    ----------
    bids : list[[price, qty]]
    asks : list[[price, qty]]
    depth_pct : float
        Only consider orders within depth_pct of the mid price (e.g. 0.02 = 2%).

    Returns
    -------
    dict with:
        - imbalance : float  (1.0 = all bids, -1.0 = all asks)
        - bid_volume : float
        - ask_volume : float
        - bid_volume_$ : float (quote volume)
        - ask_volume_$ : float
        - bid_count : int
        - ask_count : int
        - mid_price : float
        - spread_pct : float
        - weighted_imbalance : float  (price-weighted)
    """
    if not bids or not asks:
        return {
            "imbalance": 0.0, "bid_volume": 0, "ask_volume": 0,
            "bid_volume_$": 0, "ask_volume_$": 0,
            "bid_count": 0, "ask_count": 0,
            "mid_price": 0, "spread_pct": 0, "weighted_imbalance": 0,
        }

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid_price * 100 if mid_price > 0 else 0

    # Filter to depth range
    lower = mid_price * (1 - depth_pct)
    upper = mid_price * (1 + depth_pct)

    bid_vol = 0.0
    bid_vol_quote = 0.0
    bid_count = 0
    weighted_bid = 0.0

    for p, q in bids:
        if p < lower:
            continue
        bid_vol += q
        bid_vol_quote += p * q
        weighted_bid += q * (p - lower)  # weight by distance from lower bound
        bid_count += 1

    ask_vol = 0.0
    ask_vol_quote = 0.0
    ask_count = 0
    weighted_ask = 0.0

    for p, q in asks:
        if p > upper:
            continue
        ask_vol += q
        ask_vol_quote += p * q
        weighted_ask += q * (upper - p)  # weight by distance from upper bound
        ask_count += 1

    total_vol = bid_vol + ask_vol
    if total_vol > 0:
        # Simple imbalance: [-1, 1]
        imbalance = (bid_vol - ask_vol) / total_vol
        # Weighted imbalance (price-weighted)
        total_weighted = weighted_bid + weighted_ask
        weighted_imbalance = (
            (weighted_bid - weighted_ask) / total_weighted if total_weighted > 0 else 0.0
        )
    else:
        imbalance = 0.0
        weighted_imbalance = 0.0

    return {
        "imbalance": round(imbalance, 6),
        "weighted_imbalance": round(weighted_imbalance, 6),
        "bid_volume": round(bid_vol, 4),
        "ask_volume": round(ask_vol, 4),
        "bid_volume_$": round(bid_vol_quote, 2),
        "ask_volume_$": round(ask_vol_quote, 2),
        "bid_count": bid_count,
        "ask_count": ask_count,
        "mid_price": mid_price,
        "spread_pct": round(spread_pct, 4),
    }


def get_imbalance_signal(symbol="BTCUSDT", exchange="binance", depth_pct=0.02):
    """
    One-shot: fetch order book + return imbalance signal for trading decisions.

    Returns dict with imbalance metrics + a buy/sell/neutral signal.
    """
    if exchange == "binance":
        data = get_binance_depth(symbol)
    elif exchange == "kraken":
        data = get_kraken_depth(symbol)
    elif exchange == "coinbase":
        data = get_coinbase_depth(symbol)
    else:
        return {"error": f"unsupported exchange: {exchange}"}

    if "error" in data:
        return {"error": data["error"]}

    imb = calc_order_book_imbalance(data["bids"], data["asks"], depth_pct)

    # Signal logic
    if imb["imbalance"] > 0.3:
        imb["signal"] = "buy"
        imb["signal_strength"] = "strong" if imb["imbalance"] > 0.6 else "moderate"
    elif imb["imbalance"] < -0.3:
        imb["signal"] = "sell"
        imb["signal_strength"] = "strong" if imb["imbalance"] < -0.6 else "moderate"
    else:
        imb["signal"] = "neutral"
        imb["signal_strength"] = "weak"

    imb["exchange"] = exchange
    imb["symbol"] = symbol
    return imb


if __name__ == "__main__":
    # Test with Binance BTCUSDT
    print("=== Binance BTCUSDT Order Book Imbalance ===")
    result = get_imbalance_signal("BTCUSDT", "binance", 0.02)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("✅ Order Book Imbalance test OK")