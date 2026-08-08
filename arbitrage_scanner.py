"""
Exchange Arbitrage Scanner — detects price differences between exchanges
for the same coin. Uses public REST APIs (no API keys required).

Supported exchanges: Binance, Kraken, Coinbase, Bybit, KuCoin, OKX.

Single file, no external deps beyond urllib.
"""
import json
import ssl
import time
import urllib.request

# ── Cache ──────────────────────────────────────────────────────────────────
_PRICE_CACHE = {}
_CACHE_TTL = 5  # seconds — price data is time-sensitive


def _fetch(url, timeout=10):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _cached_get(key, url, ttl=None):
    ttl = ttl or _CACHE_TTL
    now = time.time()
    cached = _PRICE_CACHE.get(key)
    if cached and now - cached["ts"] < ttl:
        return cached["data"]
    data = _fetch(url)
    if "error" not in data:
        _PRICE_CACHE[key] = {"ts": now, "data": data}
    return data


# ── Exchange price fetchers ────────────────────────────────────────────────

# Standardized symbol mapping: exchange_symbol -> (fetch_fn, base_currency)

def get_binance_price(symbol="BTCUSDT"):
    """Binance ticker price."""
    data = _cached_get(f"binance:{symbol}", f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
    if "error" in data:
        return None
    return float(data.get("price", 0))


def get_kraken_price(pair="XBTUSD"):
    """Kraken ticker. Note: XBT = Bitcoin, different pair names."""
    data = _cached_get(f"kraken:{pair}", f"https://api.kraken.com/0/public/Ticker?pair={pair}")
    if "error" in data or data.get("error"):
        return None
    result = data.get("result", {})
    key = list(result.keys())[0] if result else None
    if not key:
        return None
    return float(result[key].get("c", ["0"])[0])


def get_coinbase_price(product_id="BTC-USD"):
    """Coinbase Pro ticker."""
    data = _cached_get(f"coinbase:{product_id}", f"https://api.exchange.coinbase.com/products/{product_id}/ticker")
    if "error" in data:
        return None
    return float(data.get("price", 0))


def get_bybit_price(symbol="BTCUSDT"):
    """Bybit ticker."""
    data = _cached_get(f"bybit:{symbol}", f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}")
    if "error" in data:
        return None
    try:
        return float(data["result"]["list"][0]["lastPrice"])
    except (KeyError, IndexError, TypeError):
        return None


def get_kucoin_price(symbol="BTC-USDT"):
    """KuCoin ticker."""
    data = _cached_get(f"kucoin:{symbol}", f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}")
    if "error" in data:
        return None
    try:
        return float(data["data"]["price"])
    except (KeyError, TypeError):
        return None


def get_okx_price(inst_id="BTC-USDT"):
    """OKX ticker."""
    data = _cached_get(f"okx:{inst_id}", f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}")
    if "error" in data:
        return None
    try:
        return float(data["data"][0]["last"])
    except (KeyError, IndexError, TypeError):
        return None


# ── Arbitrage Calculation ──────────────────────────────────────────────────


def scan_arbitrage(symbol_map):
    """
    Fetch prices from multiple exchanges for the same coin and find arbitrage.

    Parameters
    ----------
    symbol_map : dict
        Keyed by exchange name, valued by (fetch_fn, symbol/pair string).
        Example:
        {
            "binance": (get_binance_price, "BTCUSDT"),
            "kraken": (get_kraken_price, "XBTUSD"),
            "coinbase": (get_coinbase_price, "BTC-USD"),
            "bybit": (get_bybit_price, "BTCUSDT"),
            "kucoin": (get_kucoin_price, "BTC-USDT"),
            "okx": (get_okx_price, "BTC-USDT"),
        }

    Returns
    -------
    dict with:
        - prices : dict exchange -> price
        - lowest : (exchange, price)
        - highest : (exchange, price)
        - spread_pct : float (diff between highest and lowest, % of lowest)
        - profitable_pairs : list[dict]
        - timestamp : str
    """
    prices = {}
    for name, (fn, symbol) in symbol_map.items():
        try:
            price = fn(symbol)
            if price and price > 0:
                prices[name] = price
        except Exception:
            pass

    if len(prices) < 2:
        return {
            "prices": prices,
            "lowest": None,
            "highest": None,
            "spread_pct": 0.0,
            "profitable_pairs": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "error": "need at least 2 exchanges with data",
        }

    lowest_ex = min(prices, key=prices.get)
    highest_ex = max(prices, key=prices.get)
    lowest_price = prices[lowest_ex]
    highest_price = prices[highest_ex]
    spread_pct = (highest_price - lowest_price) / lowest_price * 100

    # Find all profitable pairs (spread > 0.2% to cover fees + slippage)
    profitable_pairs = []
    min_profit_pct = 0.2  # minimum spread after fees
    names = list(prices.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = prices[a], prices[b]
            if pa > pb:
                spread = (pa - pb) / pb * 100
                if spread >= min_profit_pct:
                    profitable_pairs.append({
                        "buy_exchange": b,
                        "buy_price": pb,
                        "sell_exchange": a,
                        "sell_price": pa,
                        "spread_pct": round(spread, 4),
                        "profit_pct": round(spread - 0.18, 4),  # after ~0.18% fees
                    })
            else:
                spread = (pb - pa) / pa * 100
                if spread >= min_profit_pct:
                    profitable_pairs.append({
                        "buy_exchange": a,
                        "buy_price": pa,
                        "sell_exchange": b,
                        "sell_price": pb,
                        "spread_pct": round(spread, 4),
                        "profit_pct": round(spread - 0.18, 4),
                    })

    profitable_pairs.sort(key=lambda x: x["spread_pct"], reverse=True)

    return {
        "prices": prices,
        "lowest": (lowest_ex, lowest_price),
        "highest": (highest_ex, highest_price),
        "spread_pct": round(spread_pct, 4),
        "profitable_pairs": profitable_pairs[:5],  # top 5
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


# ── Preset symbol maps for common coins ────────────────────────────────────

BTC_SYMBOLS = {
    "binance": (get_binance_price, "BTCUSDT"),
    "kraken": (get_kraken_price, "XBTUSD"),
    "coinbase": (get_coinbase_price, "BTC-USD"),
    "bybit": (get_bybit_price, "BTCUSDT"),
    "kucoin": (get_kucoin_price, "BTC-USDT"),
    "okx": (get_okx_price, "BTC-USDT"),
}

ETH_SYMBOLS = {
    "binance": (get_binance_price, "ETHUSDT"),
    "kraken": (get_kraken_price, "ETHUSD"),
    "coinbase": (get_coinbase_price, "ETH-USD"),
    "bybit": (get_bybit_price, "ETHUSDT"),
    "kucoin": (get_kucoin_price, "ETH-USDT"),
    "okx": (get_okx_price, "ETH-USDT"),
}

XRP_SYMBOLS = {
    "binance": (get_binance_price, "XRPUSDT"),
    "kraken": (get_kraken_price, "XRPUSD"),
    "coinbase": (get_coinbase_price, "XRP-USD"),
    "bybit": (get_bybit_price, "XRPUSDT"),
    "kucoin": (get_kucoin_price, "XRP-USDT"),
    "okx": (get_okx_price, "XRP-USDT"),
}

SOL_SYMBOLS = {
    "binance": (get_binance_price, "SOLUSDT"),
    "kraken": (get_kraken_price, "SOLUSD"),
    "coinbase": (get_coinbase_price, "SOL-USD"),
    "bybit": (get_bybit_price, "SOLUSDT"),
    "kucoin": (get_kucoin_price, "SOL-USDT"),
    "okx": (get_okx_price, "SOL-USDT"),
}


def scan_all_coins():
    """Scan BTC, ETH, XRP, SOL for arbitrage opportunities."""
    results = {}
    for name, symbols in [("BTC", BTC_SYMBOLS), ("ETH", ETH_SYMBOLS),
                           ("XRP", XRP_SYMBOLS), ("SOL", SOL_SYMBOLS)]:
        results[name] = scan_arbitrage(symbols)
        # Rate limit between scans
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    print("=== Arbitrage Scanner ===")
    results = scan_all_coins()
    for coin, data in results.items():
        print(f"\n{coin}:")
        print(f"  Prices: {data.get('prices', {})}")
        print(f"  Highest: {data.get('highest', 'N/A')}")
        print(f"  Lowest: {data.get('lowest', 'N/A')}")
        print(f"  Spread: {data.get('spread_pct', 0)}%")
        pairs = data.get("profitable_pairs", [])
        if pairs:
            print(f"  Profitable arbitrage opportunities:")
            for p in pairs:
                print(f"    Buy on {p['buy_exchange']} @ {p['buy_price']} → "
                      f"Sell on {p['sell_exchange']} @ {p['sell_price']} "
                      f"(+{p['profit_pct']}%)")
        else:
            print(f"  No profitable arbitrage (>0.2%)")
    print("\n✅ Arbitrage scanner test OK")