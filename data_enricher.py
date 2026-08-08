"""
Data Enricher — pulls massive context from CoinGecko, Fear & Greed, trending,
news, whales, funding, and economic calendar. Returns a single rich formatted
string for AI consumption. All APIs are free, no keys needed. Cached to avoid
hammering free endpoints (1 min TTL).
"""
import json, time, ssl, urllib.request
from datetime import datetime
from pathlib import Path

# ── caching ─────────────────────────────────────────────────────────────────
CACHE_FILE = Path("/tmp/gecko_cache.json")
CACHE_TTL  = 60  # seconds — don't re-fetch within 1 minute


def _cache_get(key):
    """Return cached value for *key*, or None if missing/stale."""
    try:
        raw = CACHE_FILE.read_text()
        cache = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    entry = cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["ts"] > CACHE_TTL:
        return None  # expired
    return entry["data"]


def _cache_set(key, data):
    """Store *data* under *key* in the JSON cache file."""
    try:
        raw = CACHE_FILE.read_text()
        cache = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    cache[key] = {"ts": time.time(), "data": data}
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ── safe single-shot fetcher ────────────────────────────────────────────────

def _get_json(url, timeout=10):
    """GET *url*, return parsed JSON or None.  No API key needed."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ── data sources ────────────────────────────────────────────────────────────

def get_fear_greed():
    """Dict with value/classification or None."""
    cached = _cache_get("fear_greed")
    if cached:
        return cached
    data = _get_json("https://api.alternative.me/fng/?limit=1")
    if data and "data" in data and len(data["data"]) > 0:
        entry = data["data"][0]
        result = {
            "value": entry.get("value", "?"),
            "classification": entry.get("value_classification", "?"),
        }
        _cache_set("fear_greed", result)
        return result
    return None


def get_global_data():
    """Dict with total_market_cap, btc_dominance, etc. or None."""
    cached = _cache_get("global")
    if cached:
        return cached
    data = _get_json("https://api.coingecko.com/api/v3/global")
    if data and "data" in data:
        g = data["data"]
        result = {
            "total_market_cap": g.get("total_market_cap", {}).get("eur"),
            "btc_dominance": g.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": g.get("market_cap_percentage", {}).get("eth"),
            "active_cryptocurrencies": g.get("active_cryptocurrencies"),
            "total_volume": g.get("total_volume", {}).get("eur"),
        }
        _cache_set("global", result)
        return result
    return None


def get_trending():
    """List of top-5 trending coin names or empty list."""
    cached = _cache_get("trending")
    if cached:
        return cached
    data = _get_json("https://api.coingecko.com/api/v3/search/trending")
    coins = []
    if data and "coins" in data:
        for item in data["coins"][:5]:
            coin = item.get("item", {})
            coins.append({
                "name": coin.get("name", "?"),
                "symbol": coin.get("symbol", "?").upper(),
                "market_cap_rank": coin.get("market_cap_rank"),
                "price_btc": coin.get("price_btc"),
                "score": coin.get("score", 0),
            })
    _cache_set("trending", coins)
    return coins


# ── CoinGecko coin price lookup ─────────────────────────────────────────────

COINGECKO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "ADA": "cardano", "DOT": "polkadot", "LINK": "chainlink", "AVAX": "avalanche-2",
    "DOGE": "dogecoin", "APT": "aptos", "ATOM": "cosmos", "ARB": "arbitrum",
    "OP": "optimism", "INJ": "injective-protocol", "BCH": "bitcoin-cash",
    "BNB": "binancecoin", "BONK": "bonk", "CRV": "curve-dao-token",
    "ENA": "ethena", "FIL": "filecoin", "HYPE": "hyperliquid", "ICP": "internet-computer",
    "LTC": "litecoin", "NEAR": "near", "PEPE": "pepe", "SEI": "sei-network",
    "SHIB": "shiba-inu", "SUI": "sui", "TON": "the-open-network",
    "TRX": "tron", "UNI": "uniswap", "WIF": "dogwifhat", "HODL": "hodl",
    "HOODIE": "hoodie", "CASH": "cash", "LAB": "lab", "CAT": "cat",
}


def _coingecko_id(symbol):
    """Return CoinGecko ID for *symbol*, or the lowercased symbol as fallback."""
    s = symbol.upper().replace("-EUR", "").replace("EUR", "").strip()
    return COINGECKO_ID_MAP.get(s, s.lower())


def get_coin_prices(symbols):
    """
    Batch price + 24h change + volume for a list of coin symbols.
    Returns dict: {symbol: {"price", "change_24h", "volume", "market_cap", "high_24h", "low_24h"}}
    """
    if not symbols:
        return {}
    # Deduplicate and map to CoinGecko IDs
    ids = sorted(set(_coingecko_id(s) for s in symbols if s))
    if not ids:
        return {}

    cache_key = "prices_" + "_".join(ids)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    url = (f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}"
           f"&vs_currencies=eur&include_24hr_vol=true&include_24hr_change=true"
           f"&include_market_cap=true&include_last_updated_at=true")
    data = _get_json(url)
    prices = {}
    if data:
        for coin_id in ids:
            entry = data.get(coin_id, {})
            if not entry:
                continue
            price = entry.get("eur")
            change = entry.get("eur_24h_change")
            volume = entry.get("eur_24h_vol")
            mcap = entry.get("eur_market_cap")
            # Map back to the first matching raw symbol
            raw_sym = None
            for s in symbols:
                if _coingecko_id(s) == coin_id:
                    raw_sym = s
                    break
            if raw_sym and price is not None:
                prices[raw_sym] = {
                    "price": price,
                    "change_24h": change,
                    "volume": volume,
                    "market_cap": mcap,
                }
    _cache_set(cache_key, prices)
    return prices


# ── external module integration (imports are lazy/optional) ─────────────────

def _safe_import(module_name, function_name):
    """Try to import and return a callable, or return None."""
    try:
        mod = __import__(module_name, fromlist=[function_name])
        return getattr(mod, function_name, None)
    except Exception:
        return None


# ── main public entry point ─────────────────────────────────────────────────

def build_rich_context(holdings, eur_tickers):
    """
    Build a single, rich, formatted context string with ALL data sources.

    Parameters
    ----------
    holdings : dict
        {symbol: qty}  e.g. {"AVAX": 5.0, "XRP": 20.0}
    eur_tickers : list[dict]
        List of ticker dicts with keys 'symbol', 'bid', 'ask', 'change_24h', 'volume_24h'.

    Returns
    -------
    str  — a clean, emoji-rich text block ready for LLM context.
    """
    sections = []
    ts = datetime.now().strftime("%a %d %b %H:%M UTC")

    # ── 1. MARKET OVERVIEW ────────────────────────────────────────────────
    overview_parts = [f"⏰ {ts}"]

    fear_greed = get_fear_greed()
    if fear_greed:
        overview_parts.append(
            f"😨 Fear & Greed: {fear_greed['value']} ({fear_greed['classification']})"
        )

    global_data = get_global_data()
    if global_data:
        mcap = global_data.get("total_market_cap")
        btc_dom = global_data.get("btc_dominance")
        vol = global_data.get("total_volume")
        parts = []
        if mcap:
            parts.append(f"🌍 Global market cap: €{_fmt_big(mcap)}")
        if btc_dom is not None:
            parts.append(f"BTC dominance: {btc_dom:.1f}%")
        if vol:
            parts.append(f"24h vol: €{_fmt_big(vol)}")
        if parts:
            overview_parts.append(" | ".join(parts))

    sections.append("📊 MARKET OVERVIEW")
    sections.extend("  " + line for line in overview_parts)

    # ── 2. TRENDING ───────────────────────────────────────────────────────
    trending = get_trending()
    if trending:
        names = [f"{c['name']} ({c['symbol']})#{c.get('market_cap_rank', '?')}" for c in trending]
        sections.append(f"\n🔥 TRENDING (CoinGecko): {'  |  '.join(names)}")

    # ── 3. POSITION PRICES ────────────────────────────────────────────────
    hold_symbols = list(holdings.keys()) if holdings else []
    ticker_symbols = [t.get("symbol", "") for t in (eur_tickers or [])]

    # Build set of symbols we need CoinGecko data for
    lookup_symbols = set()
    for s in hold_symbols:
        lookup_symbols.add(s)
    for t in ticker_symbols:
        clean = t.upper().replace("-EUR", "").replace("EUR", "").strip()
        if clean:
            lookup_symbols.add(clean)

    coin_prices = get_coin_prices(list(lookup_symbols)) if lookup_symbols else {}

    # ── Holdings section ───────────────────────────────────────────────────
    if holdings:
        sections.append("\n💰 YOUR POSITIONS:")
        for sym, qty in sorted(holdings.items(), key=lambda x: x[0]):
            cg = coin_prices.get(sym, {})
            price = cg.get("price")
            chg = cg.get("change_24h")
            vol = cg.get("volume")
            val_str = f"€{qty * price:,.2f}" if price and qty else "?"
            price_str = f"€{price:,.4f}" if price else "?"
            chg_str = f"{chg:+.1f}%" if chg is not None else "?"
            vol_str = f"€{_fmt_big(vol)}" if vol else "?"
            sections.append(f"  {sym}: {price_str} ({chg_str}) vol:{vol_str} | qty={qty} val={val_str}")

    # ── Top tickers section ────────────────────────────────────────────────
    if eur_tickers:
        sections.append("\n📈 TOP EUR TICKERS (24h):")
        for t in eur_tickers[:8]:
            sym = t.get("symbol", "?")
            bid = t.get("bid", "?")
            ask = t.get("ask", "?")
            chg = t.get("change_24h", "?")
            vol = t.get("volume_24h", "?")
            sections.append(f"  {sym}: bid={bid} ask={ask} 24h={chg}% vol={vol}")

        # Also show CoinGecko data as enrichment
        cg_lines = []
        for t in eur_tickers[:8]:
            raw_sym = t.get("symbol", "").replace("-EUR", "").strip()
            cg = coin_prices.get(raw_sym, {})
            if cg:
                mcap = cg.get("market_cap")
                mcap_str = f"€{_fmt_big(mcap)}" if mcap else "?"
                cg_lines.append(f"    {raw_sym}: mcap={mcap_str} 24h_chg={cg.get('change_24h', '?'):+.1f}%" if cg.get('change_24h') is not None else f"    {raw_sym}: mcap={mcap_str}")
        if cg_lines:
            sections.append("  CoinGecko enrichment:")
            sections.extend(cg_lines)

    # ── 4. NEWS ────────────────────────────────────────────────────────────
    news_fn = _safe_import("news_collector", "get_crypto_news")
    if news_fn:
        try:
            headlines = news_fn()
            if headlines:
                sections.append("\n📰 NEWS:")
                for h in headlines[:5]:
                    sections.append(f"  • {h[:120]}")
        except Exception:
            pass  # fail silently

    # ── 5. WHALES ──────────────────────────────────────────────────────────
    whale_fn = _safe_import("whale_tracker", "get_whale_movements")
    if whale_fn:
        try:
            whales = whale_fn()
            if whales:
                sections.append("\n🐋 WHALES:")
                for w in whales[:3]:
                    sections.append(f"  • {w}")
        except Exception:
            pass

    # ── 6. FUNDING / OI ────────────────────────────────────────────────────
    funding_fn = _safe_import("funding_signals", "get_funding_data")
    if funding_fn:
        try:
            funding = funding_fn()
            if funding:
                sections.append("\n💸 FUNDING / OI:")
                for f in funding[:4]:
                    sections.append(f"  • {f}")
        except Exception:
            pass

    # ── 7. ECONOMIC CALENDAR ───────────────────────────────────────────────
    econ_fn = _safe_import("economic_calendar", "get_upcoming_economic_events")
    if econ_fn:
        try:
            events = econ_fn()
            if events:
                sections.append("\n📅 ECONOMIC CALENDAR:")
                for e in events[:4]:
                    sections.append(f"  • {e}")
        except Exception:
            pass

    # ── assemble ───────────────────────────────────────────────────────────
    return "\n".join(sections)


# ── helpers ─────────────────────────────────────────────────────────────────

def _fmt_big(n):
    """Format a large number: 1_500_000_000 -> \"1.5B\" """
    if n is None:
        return "?"
    try:
        n = float(n)
    except (ValueError, TypeError):
        return str(n)
    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.2f}"


# ── quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  DATA ENRICHER — TEST RUN")
    print("=" * 62)

    # Simulate holdings + tickers like what the trading bot would pass
    test_holdings = {"AVAX": 5, "XRP": 20}
    test_tickers = [
        {"symbol": "BTC-EUR", "bid": 51234, "ask": 51250, "change_24h": -2.1, "volume_24h": "12.5B"},
        {"symbol": "ETH-EUR", "bid": 2780, "ask": 2785, "change_24h": 1.3, "volume_24h": "8.2B"},
        {"symbol": "SOL-EUR", "bid": 142, "ask": 143, "change_24h": 3.4, "volume_24h": "3.1B"},
    ]

    context = build_rich_context(test_holdings, test_tickers)
    print()
    print(context)
    print()
    print("=" * 62)
    print("  ✅ DATA ENRICHER READY")
    print("=" * 62)