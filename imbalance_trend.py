#!/usr/bin/env python3
"""
Bid/Ask Imbalance Trend — is buying pressure building or fading?

Reads from hyperliquid_ws._latest_data["orderbooks"] which updates on every
l2Book tick. Tracks the bid/ask imbalance ratio over time to determine whether
buying/selling pressure is accelerating or decelerating.

The TREND of imbalance is often more predictive than the snapshot:
- Imbalance +0.3 and rising → buying pressure building → bullish
- Imbalance +0.3 and falling → buying pressure fading → bearish reversal
- Imbalance -0.3 and falling → selling pressure building → bearish
- Imbalance -0.3 and rising → selling pressure fading → bullish reversal
"""

from __future__ import annotations
import time
import logging
from collections import deque
from typing import Optional

log = logging.getLogger("imbalance_trend")

# {coin: deque([(timestamp, imbalance_ratio, bid_depth, ask_depth), ...], maxlen=30)}
_imbalance_history: dict[str, deque] = {}
MAX_HISTORY = 30  # ~30 ticks of l2Book history


def _get_current_imbalance(coin: str) -> tuple[float, float, float, float]:
    """Get current bid/ask imbalance from WS orderbook.
    Returns (imbalance, bid_depth, ask_depth, timestamp) or (0, 0, 0, 0)."""
    try:
        from hyperliquid_ws import _latest_data
        ob = _latest_data.get("orderbooks", {}).get(coin, {})
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return 0.0, 0.0, 0.0, 0.0

        def _sz(level):
            if isinstance(level, dict):
                return float(level.get("sz", 0))
            return float(level[1]) if len(level) > 1 else 0.0

        total_bid = sum(_sz(b) for b in bids[:10])
        total_ask = sum(_sz(a) for a in asks[:10])
        total = total_bid + total_ask
        if total <= 0:
            return 0.0, total_bid, total_ask, time.time()
        imbalance = (total_bid - total_ask) / total  # -1 to +1
        return imbalance, total_bid, total_ask, time.time()
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _record_tick(coin: str):
    """Record current imbalance snapshot to history."""
    imb, bid_d, ask_d, ts = _get_current_imbalance(coin)
    if ts == 0:
        return
    if coin not in _imbalance_history:
        _imbalance_history[coin] = deque(maxlen=MAX_HISTORY)
    _imbalance_history[coin].append((ts, imb, bid_d, ask_d))


def get_imbalance_trend(coin: str) -> dict:
    """
    Analyze imbalance trend over recorded history.

    Returns:
        {"trend": "building_buy"|"fading_buy"|"building_sell"|"fading_sell"|
                  "stable"|"no_data",
         "current_imbalance": -1 to +1,
         "trend_slope": float (change per tick),
         "depth_ratio": bid_depth/ask_depth,
         "signal": "bullish"|"bearish"|"neutral",
         "confidence": 0-100}
    """
    # Record fresh tick
    _record_tick(coin)

    history = _imbalance_history.get(coin)
    if not history or len(history) < 5:
        imb, _, _, _ = _get_current_imbalance(coin)
        return {"trend": "no_data", "current_imbalance": imb,
                "trend_slope": 0, "depth_ratio": 1.0,
                "signal": "neutral", "confidence": 0}

    # Get last N ticks
    ticks = list(history)
    current_imb = ticks[-1][1]
    bid_depth = ticks[-1][2]
    ask_depth = ticks[-1][3]
    depth_ratio = bid_depth / max(ask_depth, 0.001)

    # Simple linear trend on last 10 ticks
    n = min(10, len(ticks))
    recent = ticks[-n:]
    x_vals = list(range(n))
    y_vals = [t[1] for t in recent]

    # Compute slope via simple linear regression
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    num = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    den = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    slope = num / max(den, 0.001)  # Change in imbalance per tick

    # ── Classify trend ──
    slope_threshold = 0.002  # Meaningful slope per tick

    if abs(slope) < slope_threshold:
        trend = "stable"
    elif slope > 0 and current_imb > 0:
        trend = "building_buy"
    elif slope > 0 and current_imb <= 0:
        trend = "fading_sell"  # Imbalance was negative, now normalizing → bullish
    elif slope < 0 and current_imb < 0:
        trend = "building_sell"
    elif slope < 0 and current_imb >= 0:
        trend = "fading_buy"  # Imbalance was positive, now dropping → bearish
    else:
        trend = "stable"

    # ── Direction + confidence ──
    slope_mag = abs(slope) * 100  # Scale for confidence calc

    if trend == "building_buy":
        signal = "bullish"
        confidence = min(70, 20 + slope_mag * 3)
    elif trend == "fading_sell":
        signal = "bullish"
        confidence = min(60, 15 + slope_mag * 2.5)
    elif trend == "building_sell":
        signal = "bearish"
        confidence = min(70, 20 + slope_mag * 3)
    elif trend == "fading_buy":
        signal = "bearish"
        confidence = min(60, 15 + slope_mag * 2.5)
    else:
        signal = "neutral"
        confidence = 5

    # Boost confidence if depth ratio strongly confirms
    if trend == "building_buy" and depth_ratio > 1.5:
        confidence = min(80, confidence + 10)
    elif trend == "building_sell" and depth_ratio < 0.67:
        confidence = min(80, confidence + 10)

    return {
        "trend": trend,
        "current_imbalance": round(current_imb, 4),
        "trend_slope": round(slope, 6),
        "depth_ratio": round(depth_ratio, 3),
        "signal": signal,
        "confidence": round(confidence, 1),
    }


def _layer_imbalance_trend(coin: str, mid: float) -> dict[str, float]:
    """Continuous predictor layer format."""
    trend = get_imbalance_trend(coin)
    direction = trend["signal"]
    confidence = trend["confidence"]

    if direction == "bullish":
        return {"up": 38 + confidence * 0.3, "down": 32 - confidence * 0.15,
                "flat": 30 - confidence * 0.15, "confidence": confidence}
    elif direction == "bearish":
        return {"up": 32 - confidence * 0.15, "down": 38 + confidence * 0.3,
                "flat": 30 - confidence * 0.15, "confidence": confidence}
    else:
        return {"up": 33, "down": 33, "flat": 34, "confidence": confidence * 0.2}


def get_imbalance_context(coin: str) -> str:
    """Compact AI context string."""
    t = get_imbalance_trend(coin)
    if t["trend"] == "no_data":
        return ""
    return (f"OBtrend:{t['trend']} "
            f"imb={t['current_imbalance']:+.3f} "
            f"depth={t['depth_ratio']:.1f}x "
            f"slope={t['trend_slope']:+.4f}")


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Imbalance Trend module loaded.")
    print("  get_imbalance_trend(coin)")
    print("  _layer_imbalance_trend(coin, mid)")
    print("  get_imbalance_context(coin)")
    # Test with no data
    r = get_imbalance_trend("SOL")
    print(f"  SOL: {r['trend']} imb={r['current_imbalance']:.4f} conf={r['confidence']}")
