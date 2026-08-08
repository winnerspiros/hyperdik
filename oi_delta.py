#!/usr/bin/env python3
"""
Open Interest Delta Signal — WebSocket-driven, per-coin OI tracking.

Driven by Hyperliquid WebSocket `openInterest` channel data (stored in
hyperliquid_ws._latest_data['open_interest']).

Rules (OI/Price Divergence Detection):
  1. Rising OI + Falling Price = New shorts opening   → BEARISH
  2. Falling OI + Rising Price = Shorts covering       → BEARISH (fakeout, reversal likely)
  3. Rising OI + Rising Price  = New longs entering    → BULLISH
  4. Falling OI + Falling Price = Longs capitulating   → BULLISH (reversal likely)

Also maintains backward compatibility with the original get_oi_delta()
and OIDeltaSignal interface used by hyperliquid_strategy.py.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Tuple

log = logging.getLogger("oi_delta")

# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_LOOKBACK_SECONDS = 900   # 15 minutes
MAX_HISTORY_LENGTH = 120          # ~30 min at typical update rates
OI_SIGNIFICANT_PCT = 0.5          # % change to consider OI move significant
PRICE_SIGNIFICANT_PCT = 0.10      # % change to consider price move significant

# ── Per-coin OI History ─────────────────────────────────────────────────────
# {coin: deque of (timestamp_sec, oi_value)}
_oi_history: dict[str, Deque[Tuple[float, float]]] = {}

# Per-coin price snapshots for divergence detection
# {coin: [(timestamp_sec, price), ...]}
_price_history: dict[str, Deque[Tuple[float, float]]] = {}


def _clean_history(coin: str, lookback: float = DEFAULT_LOOKBACK_SECONDS):
    """Remove entries older than lookback seconds."""
    now = time.time()
    cutoff = now - lookback
    if coin in _oi_history:
        while _oi_history[coin] and _oi_history[coin][0][0] < cutoff:
            _oi_history[coin].popleft()
    if coin in _price_history:
        while _price_history[coin] and _price_history[coin][0][0] < cutoff:
            _price_history[coin].popleft()


def _init_coin(coin: str):
    """Ensure history deques exist for a coin."""
    coin = coin.upper()
    if coin not in _oi_history:
        _oi_history[coin] = deque(maxlen=MAX_HISTORY_LENGTH)
    if coin not in _price_history:
        _price_history[coin] = deque(maxlen=MAX_HISTORY_LENGTH)
    return coin


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class OIDeltaSignal:
    """Backward-compatible OI delta signal dataclass."""
    symbol: str
    oi_current: float
    oi_previous: float
    oi_delta_pct: float
    price_current: float
    price_previous: float
    price_delta_pct: float
    signal: str  # trend_confirm_buy | trend_confirm_sell | exhaustion_bullish | exhaustion_bearish | neutral
    confidence: float
    description: str


@dataclass
class OISignalResult:
    """New WebSocket-driven signal format."""
    coin: str
    direction: str       # "bullish" | "bearish" | "neutral"
    confidence: float    # 0-100
    signal_type: str     # "new_shorts" | "shorts_covering" | "new_longs" | "longs_capitulating" | "neutral"
    description: str
    oi_delta_pct: float
    price_delta_pct: float
    oi_current: float
    oi_start: float
    lookback_seconds: int


# ── Internal helpers ────────────────────────────────────────────────────────

def _fetch_ws_oi(coin: str) -> Optional[float]:
    """Read latest OI for a coin from the WebSocket data store."""
    try:
        from hyperliquid_ws import _latest_data, _state_lock
        with _state_lock:
            oi_data = _latest_data.get("open_interest", {})
            entry = oi_data.get(coin.upper())
            if entry and isinstance(entry, dict):
                return float(entry.get("oi", 0))
    except Exception as e:
        log.debug(f"WS OI read error for {coin}: {e}")
    return None


def _compute_delta(coin: str, lookback: float = DEFAULT_LOOKBACK_SECONDS) -> dict:
    """
    Compute OI delta over the lookback window.
    Returns dict with oi_start, oi_end, oi_delta_pct, price_start, price_end, price_delta_pct.
    """
    coin = _init_coin(coin)
    now = time.time()
    _clean_history(coin, lookback)

    oi_hist = _oi_history.get(coin, deque())
    px_hist = _price_history.get(coin, deque())

    if len(oi_hist) < 2:
        return {"oi_delta_pct": 0.0, "price_delta_pct": 0.0,
                "oi_start": 0.0, "oi_end": 0.0,
                "price_start": 0.0, "price_end": 0.0,
                "data_points": len(oi_hist)}

    oi_start = oi_hist[0][1]
    oi_end = oi_hist[-1][1]
    oi_delta_pct = (oi_end - oi_start) / oi_start * 100 if oi_start > 0 else 0.0

    price_start = px_hist[0][1] if px_hist else 0.0
    price_end = px_hist[-1][1] if px_hist else 0.0
    price_delta_pct = (price_end - price_start) / price_start * 100 if price_start > 0 else 0.0

    return {
        "oi_delta_pct": round(oi_delta_pct, 3),
        "price_delta_pct": round(price_delta_pct, 3),
        "oi_start": oi_start,
        "oi_end": oi_end,
        "price_start": price_start,
        "price_end": price_end,
        "data_points": len(oi_hist),
    }


def _record_oi_reading(coin: str, oi_value: float, price: float):
    """Record an OI+price snapshot into the history rings."""
    coin = _init_coin(coin)
    now = time.time()
    _oi_history[coin].append((now, oi_value))
    _price_history[coin].append((now, price))


# ── Core: get_oi_signal (new, WebSocket-driven) ────────────────────────────

def get_oi_signal(
    coin: str,
    current_price: float,
    lookback_minutes: int = 15,
    force_refresh: bool = False,
) -> OISignalResult:
    """
    Produce an OI divergence signal driven by real-time WebSocket data.

    Reads latest OI from hyperliquid_ws._latest_data['open_interest'],
    records it in the per-coin history ring, and compares the delta vs
    price delta over the lookback window.

    Args:
        coin:            Symbol (e.g. "BTC")
        current_price:   Current mark/mid price
        lookback_minutes: How far back to compare (default 15 min)
        force_refresh:   If True, always pull from WS; if False, ok to use
                         last cached reading if very recent.

    Returns:
        OISignalResult with direction, confidence, signal_type, description.
    """
    lookback = lookback_minutes * 60
    coin = coin.upper()

    # Try to read latest OI from WebSocket
    oi = _fetch_ws_oi(coin)
    if oi is not None and oi > 0:
        _record_oi_reading(coin, oi, current_price)

    # Compute delta over lookback window
    delta = _compute_delta(coin, lookback)

    oi_delta_pct = delta["oi_delta_pct"]
    price_delta_pct = delta["price_delta_pct"]
    data_points = delta["data_points"]

    # Not enough data → neutral
    if data_points < 2:
        return OISignalResult(
            coin=coin,
            direction="neutral",
            confidence=5.0,
            signal_type="neutral",
            description="insufficient_data",
            oi_delta_pct=0.0,
            price_delta_pct=0.0,
            oi_current=oi or 0.0,
            oi_start=0.0,
            lookback_seconds=lookback,
        )

    oi_rising = oi_delta_pct >= OI_SIGNIFICANT_PCT
    oi_falling = oi_delta_pct <= -OI_SIGNIFICANT_PCT
    price_rising = price_delta_pct >= PRICE_SIGNIFICANT_PCT
    price_falling = price_delta_pct <= -PRICE_SIGNIFICANT_PCT

    # ── Divergence detection ──

    if oi_rising and price_falling:
        # New shorts opening — bearish
        confidence = min(85, 30 + abs(oi_delta_pct) * 4 + abs(price_delta_pct) * 6)
        return OISignalResult(
            coin=coin,
            direction="bearish",
            confidence=round(confidence, 1),
            signal_type="new_shorts",
            description=f"new_shorts_opening:oi+{oi_delta_pct:.1f}%_price{price_delta_pct:.1f}%",
            oi_delta_pct=oi_delta_pct,
            price_delta_pct=price_delta_pct,
            oi_current=delta["oi_end"],
            oi_start=delta["oi_start"],
            lookback_seconds=lookback,
        )

    elif oi_falling and price_rising:
        # Shorts covering — bearish fakeout, reversal likely
        confidence = min(75, 25 + abs(oi_delta_pct) * 4 + abs(price_delta_pct) * 5)
        return OISignalResult(
            coin=coin,
            direction="bearish",
            confidence=round(confidence, 1),
            signal_type="shorts_covering",
            description=f"shorts_covering:oi{oi_delta_pct:.1f}%_price+{price_delta_pct:.1f}%",
            oi_delta_pct=oi_delta_pct,
            price_delta_pct=price_delta_pct,
            oi_current=delta["oi_end"],
            oi_start=delta["oi_start"],
            lookback_seconds=lookback,
        )

    elif oi_rising and price_rising:
        # New longs entering — bullish
        confidence = min(85, 30 + abs(oi_delta_pct) * 4 + abs(price_delta_pct) * 6)
        return OISignalResult(
            coin=coin,
            direction="bullish",
            confidence=round(confidence, 1),
            signal_type="new_longs",
            description=f"new_longs_entering:oi+{oi_delta_pct:.1f}%_price+{price_delta_pct:.1f}%",
            oi_delta_pct=oi_delta_pct,
            price_delta_pct=price_delta_pct,
            oi_current=delta["oi_end"],
            oi_start=delta["oi_start"],
            lookback_seconds=lookback,
        )

    elif oi_falling and price_falling:
        # Longs capitulating — bullish reversal signal
        confidence = min(80, 30 + abs(oi_delta_pct) * 4 + abs(price_delta_pct) * 5)
        return OISignalResult(
            coin=coin,
            direction="bullish",
            confidence=round(confidence, 1),
            signal_type="longs_capitulating",
            description=f"longs_capitulating:oi{oi_delta_pct:.1f}%_price{price_delta_pct:.1f}%",
            oi_delta_pct=oi_delta_pct,
            price_delta_pct=price_delta_pct,
            oi_current=delta["oi_end"],
            oi_start=delta["oi_start"],
            lookback_seconds=lookback,
        )

    # ── No clear divergence ──
    else:
        return OISignalResult(
            coin=coin,
            direction="neutral",
            confidence=10.0,
            signal_type="neutral",
            description=f"no_divergence:oi{oi_delta_pct:+.1f}%_price{price_delta_pct:+.1f}%",
            oi_delta_pct=oi_delta_pct,
            price_delta_pct=price_delta_pct,
            oi_current=delta["oi_end"],
            oi_start=delta["oi_start"],
            lookback_seconds=lookback,
        )


# ── Layer interface for continuous_predictor.py ─────────────────────────────

def _layer_oi_delta(coin: str, mid: float,
                    lookback_minutes: int = 15) -> dict[str, float]:
    """
    OI delta layer compatible with continuous_predictor's layer interface.

    Returns {"up": 0-100, "down": 0-100, "flat": 0-100, "confidence": 0-100}.
    """
    sig = get_oi_signal(coin, mid, lookback_minutes=lookback_minutes, force_refresh=True)

    if sig.direction == "bullish":
        up = 40 + sig.confidence * 0.5
        down = max(10, 35 - sig.confidence * 0.3)
        flat = max(10, 25 - sig.confidence * 0.2)
    elif sig.direction == "bearish":
        up = max(10, 35 - sig.confidence * 0.3)
        down = 40 + sig.confidence * 0.5
        flat = max(10, 25 - sig.confidence * 0.2)
    else:
        up = 33
        down = 33
        flat = 34

    total = up + down + flat
    if total > 0:
        up = up / total * 100
        down = down / total * 100
        flat = flat / total * 100

    return {
        "up": round(up, 1),
        "down": round(down, 1),
        "flat": round(flat, 1),
        "confidence": round(sig.confidence, 1),
    }


# ── Backward compat: get_oi_delta (used by hyperliquid_strategy.py) ─────────

# Legacy cache
_oi_cache: dict[str, tuple[float, float, float]] = {}  # symbol → (ts, oi, price)
_oi_cache_ttl = 120


def update_oi_cache(symbol: str, oi: float, price: float) -> None:
    """Update OI cache for backward-compatible delta computation."""
    global _oi_cache
    now = time.time()
    key = symbol.upper()
    if key in _oi_cache:
        prev_ts, _, _ = _oi_cache[key]
        if now - prev_ts < _oi_cache_ttl:
            return
    _oi_cache[key] = (now, oi, price)


def get_oi_delta(
    symbol: str,
    oi_current: float,
    price_current: float,
) -> OIDeltaSignal:
    """
    Backward-compatible OI delta signal (used by hyperliquid_strategy.py).

    Computes OI delta from cached previous values.
    Fallback: also records into the WS-driven history ring.
    """
    global _oi_cache
    key = symbol.upper()
    now = time.time()

    # Also feed into WS-driven history if price is meaningful
    if price_current > 0 and oi_current > 0:
        _record_oi_reading(key, oi_current, price_current)

    if key not in _oi_cache:
        _oi_cache[key] = (now, oi_current, price_current)
        return OIDeltaSignal(
            symbol=symbol, oi_current=oi_current, oi_previous=oi_current,
            oi_delta_pct=0, price_current=price_current, price_previous=price_current,
            price_delta_pct=0, signal="neutral", confidence=0,
            description="oi_first_sample",
        )

    prev_ts, prev_oi, prev_price = _oi_cache[key]
    _oi_cache[key] = (now, oi_current, price_current)

    if prev_oi <= 0 or prev_price <= 0:
        return OIDeltaSignal(
            symbol=symbol, oi_current=oi_current, oi_previous=prev_oi,
            oi_delta_pct=0, price_current=price_current, price_previous=prev_price,
            price_delta_pct=0, signal="neutral", confidence=0,
            description="invalid_previous",
        )

    oi_delta = (oi_current - prev_oi) / prev_oi * 100
    price_delta = (price_current - prev_price) / prev_price * 100

    signal = "neutral"
    confidence = 0.0
    desc = ""

    if abs(oi_delta) < 0.3 or abs(price_delta) < 0.1:
        signal = "neutral"
        confidence = 10
        desc = "oi_flat"
    elif oi_delta > 0.5:
        if price_delta > 0.2:
            signal = "trend_confirm_buy"
            confidence = min(80, 40 + oi_delta * 5 + price_delta * 8)
            desc = f"OI_building_bullish:oi+{oi_delta:.1f}%_price+{price_delta:.1f}%"
        elif price_delta < -0.2:
            signal = "trend_confirm_sell"
            confidence = min(80, 40 + oi_delta * 5 + abs(price_delta) * 8)
            desc = f"OI_building_bearish:oi+{oi_delta:.1f}%_price{price_delta:.1f}%"
        else:
            signal = "neutral"
            confidence = 25
            desc = f"OI_building_flat_price:oi+{oi_delta:.1f}%"
    elif oi_delta < -0.5:
        if price_delta > 0.2:
            signal = "exhaustion_bullish"
            confidence = min(70, 35 + abs(oi_delta) * 5 + price_delta * 8)
            desc = f"OI_falling_rally:oi{oi_delta:.1f}%_price+{price_delta:.1f}%"
        elif price_delta < -0.2:
            signal = "exhaustion_bearish"
            confidence = min(70, 35 + abs(oi_delta) * 5 + abs(price_delta) * 8)
            desc = f"OI_falling_dump:oi{oi_delta:.1f}%_price{price_delta:.1f}%"
        else:
            signal = "neutral"
            confidence = 20
            desc = f"OI_falling_flat_price:oi{oi_delta:.1f}%"

    return OIDeltaSignal(
        symbol=symbol,
        oi_current=oi_current,
        oi_previous=prev_oi,
        oi_delta_pct=round(oi_delta, 3),
        price_current=price_current,
        price_previous=prev_price,
        price_delta_pct=round(price_delta, 3),
        signal=signal,
        confidence=round(confidence, 1),
        description=desc,
    )


def get_oi_context(
    symbol: str,
    oi_current: float,
    price_current: float,
) -> str:
    """Get compact OI context for AI/strategy (backward-compatible)."""
    sig = get_oi_delta(symbol, oi_current, price_current)

    if sig.signal == "neutral":
        return f"OI:flat({sig.oi_delta_pct:+.2f}%)"

    emoji_map = {
        "trend_confirm_buy": "🟢",
        "trend_confirm_sell": "🔴",
        "exhaustion_bullish": "⚠️🟢",
        "exhaustion_bearish": "⚠️🔴",
    }
    emoji = emoji_map.get(sig.signal, "")

    return (
        f"{emoji} OI:{sig.signal} "
        f"oi={sig.oi_delta_pct:+.2f}% "
        f"px={sig.price_delta_pct:+.2f}% "
        f"conf={sig.confidence:.0f}%"
    )


# ── Stats / Diagnostics ─────────────────────────────────────────────────────

def get_oi_history(coin: str) -> list[dict]:
    """Return the full OI history for a coin (for debugging/dashboards)."""
    coin = coin.upper()
    _init_coin(coin)
    return [
        {"timestamp": ts, "oi": oi}
        for ts, oi in _oi_history.get(coin, deque())
    ]


def get_active_coins() -> list[str]:
    """Return list of coins with OI history tracked."""
    return [c for c, hist in _oi_history.items() if len(hist) > 0]


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("OI Delta Signal — WebSocket-driven layer")
    print("=" * 60)

    # Test without live WS
    sig = get_oi_signal("BTC", 65000.0)
    print(f"\nBTC @ $65,000 (no data yet):")
    print(f"  direction={sig.direction} conf={sig.confidence:.0f}% type={sig.signal_type}")

    # Simulate a few readings
    _record_oi_reading("BTC", 1_000_000_000, 65000.0)
    time.sleep(0.01)
    _record_oi_reading("BTC", 1_020_000_000, 65500.0)
    time.sleep(0.01)
    _record_oi_reading("BTC", 1_040_000_000, 66000.0)

    sig = get_oi_signal("BTC", 66000.0, lookback_minutes=15)
    print(f"\nBTC — OI rising + price rising (new longs):")
    print(f"  direction={sig.direction} conf={sig.confidence:.0f}% type={sig.signal_type}")
    print(f"  oi_delta={sig.oi_delta_pct:+.2f}% price_delta={sig.price_delta_pct:+.2f}%")
    print(f"  desc={sig.description}")

    # Falling OI test
    _record_oi_reading("ETH", 500_000_000, 3500.0)
    time.sleep(0.01)
    _record_oi_reading("ETH", 490_000_000, 3450.0)
    time.sleep(0.01)
    _record_oi_reading("ETH", 480_000_000, 3400.0)

    sig = get_oi_signal("ETH", 3400.0, lookback_minutes=15)
    print(f"\nETH — OI falling + price falling (longs capitulating):")
    print(f"  direction={sig.direction} conf={sig.confidence:.0f}% type={sig.signal_type}")
    print(f"  desc={sig.description}")

    # Layer output
    layer = _layer_oi_delta("BTC", 66000.0)
    print(f"\nLayer format for BTC @ $66k:")
    print(f"  {layer}")

    # Backward compat
    old_sig = get_oi_delta("BTC", 1_050_000_000, 66100.0)
    print(f"\nBackward-compat get_oi_delta:")
    print(f"  signal={old_sig.signal} conf={old_sig.confidence:.0f}%")
    print(f"  context={get_oi_context('BTC', 1_050_000_000, 66100.0)}")

    print(f"\nActive coins: {get_active_coins()}")
    print("Done.")
