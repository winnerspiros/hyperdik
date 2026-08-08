#!/usr/bin/env python3
"""
Cross-Exchange Reference — Binance public data for detecting HL-specific manipulation.

Fetches Binance prices, candles, and volume via public REST API (no auth).
Compares against HyperLiquid data to flag divergences that may indicate
HL-specific manipulation or activity.

Cache TTLs:
  - Price: 30 seconds
  - Candles/volume: 5 minutes

Rate limit: 1200 req/min for public endpoints. 1s delay between calls.

Usage:
    from cross_exchange import (
        get_binance_price,
        get_cross_divergence,
        get_binance_volume_ratio,
        get_cross_exchange_context,
    )

    bnb = get_binance_price("ETH")          # → float
    div = get_cross_divergence("ETH", 3450.0)  # → {divergence_pct, direction, confidence}
    ratio = get_binance_volume_ratio("BTC", hl_vol=5000000)  # → float
    ctx = get_cross_exchange_context("SOL", 142.0)  # → AI prompt enrichment string
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

log = logging.getLogger("cross_exchange")

# ── Binance symbol map ──────────────────────────────────────────────────────
# HL coin name → Binance USDT perpetual-equivalent symbol (spot is close enough)
HL_TO_BINANCE: dict[str, str] = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "DOT":  "DOTUSDT",
    "ADA":  "ADAUSDT",
    "XRP":  "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "SUI":  "SUIUSDT",
    "BNB":  "BNBUSDT",
    "ARB":  "ARBUSDT",
}

# Reverse map for convenience
BINANCE_TO_HL: dict[str, str] = {v: k for k, v in HL_TO_BINANCE.items()}

# ── Cache TTLs ──────────────────────────────────────────────────────────────
PRICE_CACHE_TTL = 30       # seconds — prices move fast
CANDLE_CACHE_TTL = 300     # seconds — volume/ohlc changes slower
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
REQUEST_DELAY = 1.0        # seconds between API calls (be gentle)
_UA = {"User-Agent": "Mozilla/5.0"}

# ── Caches ──────────────────────────────────────────────────────────────────
_price_cache: dict[str, tuple[float, float]] = {}  # symbol → (ts, price)
_candle_cache: dict[str, tuple[float, list]] = {}  # symbol → (ts, klines)


def _binance_symbol(coin: str) -> str | None:
    """Map an HL coin name to a Binance symbol. Case-insensitive."""
    uc = coin.upper()
    return HL_TO_BINANCE.get(uc)


def _fetch_json(url: str, timeout: int = 8) -> dict | list | None:
    """Fetch JSON from a URL via urllib. Returns None on any error."""
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        log.debug("Binance fetch failed: %s", url, exc_info=True)
        return None


def _fetch_price_direct(symbol: str) -> float | None:
    """Fetch current spot price from Binance ticker/price endpoint."""
    url = f"{BINANCE_BASE_URL}/ticker/price?symbol={symbol}"
    data = _fetch_json(url)
    if isinstance(data, dict) and "price" in data:
        return float(data["price"])
    return None


def _fetch_klines(symbol: str, interval: str = "15m", limit: int = 50) -> list | None:
    """
    Fetch klines from Binance.

    Returns list of candles:
      [openTime, open, high, low, close, volume, closeTime,
       quoteVolume, trades, takerBuyVol, takerBuyQuoteVol, ignore]
    """
    url = f"{BINANCE_BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = _fetch_json(url)
    if isinstance(data, list) and len(data) > 0:
        return data
    return None


def _fetch_24hr_ticker(symbol: str) -> dict | None:
    """Fetch 24hr ticker stats for volume, high, low, etc."""
    url = f"{BINANCE_BASE_URL}/ticker/24hr?symbol={symbol}"
    data = _fetch_json(url)
    if isinstance(data, dict) and "lastPrice" in data:
        return data
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def get_binance_price(coin: str) -> float | None:
    """
    Get current Binance spot price for a coin.

    Args:
        coin: HL coin name (e.g. 'BTC', 'ETH')

    Returns:
        Float price or None if unavailable.
    """
    symbol = _binance_symbol(coin)
    if not symbol:
        log.warning("No Binance symbol for coin: %s", coin)
        return None

    # Check cache
    now = time.time()
    cached = _price_cache.get(symbol)
    if cached and (now - cached[0]) < PRICE_CACHE_TTL:
        return cached[1]

    # Fetch
    price = _fetch_price_direct(symbol)
    if price is not None:
        _price_cache[symbol] = (now, price)
        time.sleep(REQUEST_DELAY)
        return price

    # Stale cache fallback
    if cached:
        log.debug("Binance price fetch failed for %s, using stale cache", symbol)
        return cached[1]
    return None


def get_binance_candles(coin: str, interval: str = "15m", limit: int = 50) -> list | None:
    """
    Get Binance klines/candles for a coin.

    Returns list of candles or None.
    Cached for CANDLE_CACHE_TTL seconds.
    """
    symbol = _binance_symbol(coin)
    if not symbol:
        return None

    now = time.time()
    cached = _candle_cache.get(symbol)
    if cached and (now - cached[0]) < CANDLE_CACHE_TTL:
        return cached[1]

    klines = _fetch_klines(symbol, interval, limit)
    if klines is not None:
        _candle_cache[symbol] = (now, klines)
        time.sleep(REQUEST_DELAY)
        return klines

    if cached:
        return cached[1]
    return None


def get_cross_divergence(coin: str, hl_price: float | None) -> dict:
    """
    Compare HL price against Binance spot price to detect divergence.

    Args:
        coin: HL coin name
        hl_price: Current mid/mark price on HyperLiquid

    Returns:
        {
            "divergence_pct": float,       # % difference (HL vs Binance)
            "direction": "premium" | "discount" | "neutral",
            "confidence": "high" | "medium" | "low" | "none",
            "binance_price": float | None,
            "hl_price": float | None,
            "flagged": bool,               # True if >1% divergence
        }
    """
    result = {
        "divergence_pct": 0.0,
        "direction": "neutral",
        "confidence": "low",
        "binance_price": None,
        "hl_price": hl_price,
        "flagged": False,
    }

    if hl_price is None or hl_price <= 0:
        result["confidence"] = "none"
        return result

    bnb = get_binance_price(coin)
    result["binance_price"] = bnb

    if bnb is None or bnb <= 0:
        result["confidence"] = "none"
        return result

    # Compute divergence: positive = HL higher (premium), negative = HL lower (discount)
    divergence_pct = ((hl_price - bnb) / bnb) * 100.0
    result["divergence_pct"] = round(divergence_pct, 3)

    abs_div = abs(divergence_pct)

    if abs_div <= 0.5:
        result["direction"] = "neutral"
        result["confidence"] = "high"
    elif abs_div <= 1.0:
        result["direction"] = "premium" if divergence_pct > 0 else "discount"
        result["confidence"] = "medium"
    elif abs_div <= 3.0:
        result["direction"] = "premium" if divergence_pct > 0 else "discount"
        result["confidence"] = "medium"
        result["flagged"] = True  # >1% divergence
    else:
        result["direction"] = "premium" if divergence_pct > 0 else "discount"
        result["confidence"] = "high"  # Large divergence is clearly detectable
        result["flagged"] = True

    return result


def get_binance_volume_ratio(coin: str, hl_volume: float | None = None) -> float:
    """
    Compare 24h volume on Binance vs HL.

    If hl_volume provided, returns HL_volume / Binance_volume ratio.
    A ratio >> 1.0 means HL-specific activity is disproportionate.
    If hl_volume not provided, returns Binance 24h quote volume alone.

    Uses Binance 24hr ticker for volume (primary), falls back to klines aggregation.

    Args:
        coin: HL coin name
        hl_volume: HL 24h volume (quote, USD) — if None, returns Binance volume only

    Returns:
        Float: volume ratio if hl_volume given, else Binance 24h quote volume
        Returns -1.0 if Binance data unavailable.
    """
    symbol = _binance_symbol(coin)
    if not symbol:
        return -1.0

    bnb_volume: float | None = None

    # Try 24hr ticker first (gives precise 24h volume)
    ticker = _fetch_24hr_ticker(symbol)
    if ticker:
        try:
            bnb_volume = float(ticker.get("quoteVolume", 0))
        except (ValueError, TypeError):
            pass
        time.sleep(REQUEST_DELAY)

    # Fallback: sum 15m candle quote volumes (approximate 24h = 96 candles)
    if bnb_volume is None or bnb_volume <= 0:
        klines = get_binance_candles(coin, limit=96)
        if klines:
            try:
                bnb_volume = sum(float(c[7]) for c in klines if len(c) > 7)  # quoteVolume
            except (ValueError, TypeError, IndexError):
                bnb_volume = None

    if bnb_volume is None or bnb_volume <= 0:
        return -1.0

    if hl_volume is not None and hl_volume > 0:
        return round(hl_volume / bnb_volume, 4)

    return round(bnb_volume, 2)


def get_cross_exchange_context(coin: str, hl_price: float | None) -> str:
    """
    Build a compact context string for AI prompt enrichment.

    Includes:
      - Binance price vs HL price comparison
      - Divergence flagging
      - Volume comparison
      - Interpretation hints for manipulation detection

    Args:
        coin: HL coin name
        hl_price: Current HyperLiquid mid/mark price

    Returns:
        Compact string (≤120 chars) for AI context injection, or empty string.
    """
    parts: list[str] = []

    # Price divergence
    div = get_cross_divergence(coin, hl_price)
    if div["confidence"] != "none" and div["binance_price"]:
        parts.append(
            f"BNB:{div['binance_price']:.2f} HL:{hl_price:.2f} "
            f"({div['divergence_pct']:+.2f}% {div['direction']})"
        )
    elif div["binance_price"]:
        parts.append(f"BNB:{div['binance_price']:.2f} HL:?")
    elif hl_price:
        parts.append(f"BNB:? HL:{hl_price:.2f}")

    # Manipulation flag
    if div["flagged"]:
        if div["direction"] == "premium":
            parts.append("⚠ HL_PREMIUM>1% — possible HL-only pump or order book manipulation")
        else:
            parts.append("⚠ HL_DISCOUNT>1% — possible HL-only dump or liquidity squeeze")

    # Volume ratio hint (use cached Binance volume for brevity)
    vol_ratio = get_binance_volume_ratio(coin, hl_volume=None)
    if vol_ratio > 0:
        vol_str = f"BNB_24h_vol:{vol_ratio:.0f}"
        parts.append(vol_str)

    return " | ".join(parts) if parts else ""


# ── Utility: bulk fetch for the whole universe ──────────────────────────────

def get_all_binance_prices(coins: list[str] | None = None) -> dict[str, float | None]:
    """
    Fetch Binance prices for all (or a subset of) tradable coins.

    Returns:
        {coin_name: price or None, ...}
    """
    if coins is None:
        coins = list(HL_TO_BINANCE.keys())

    results: dict[str, float | None] = {}
    for coin in coins:
        results[coin] = get_binance_price(coin)
    return results


def get_cross_divergence_all(
    coins: list[str] | None = None,
    hl_mids: dict[str, float] | None = None,
) -> dict[str, dict]:
    """
    Get cross-exchange divergence for all (or a subset of) coins.

    Args:
        coins: List of HL coin names (defaults to all mapped coins)
        hl_mids: Dict of coin → mid price on HL (if None, uses empty dict)

    Returns:
        {coin: divergence_result_dict, ...}
    """
    if coins is None:
        coins = list(HL_TO_BINANCE.keys())
    if hl_mids is None:
        hl_mids = {}

    results: dict[str, dict] = {}
    for coin in coins:
        results[coin] = get_cross_divergence(coin, hl_mids.get(coin))
    return results


def invalidate_cache() -> None:
    """Clear all caches. Useful after prolonged downtime."""
    _price_cache.clear()
    _candle_cache.clear()
    log.debug("Cross-exchange caches invalidated")
