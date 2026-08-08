#!/usr/bin/env python3
"""
Taker Buy/Sell Ratio — real-time order flow aggression indicator.

Reads from hyperliquid_ws._latest_data["trade_history"] (rolling window of
last 100 trades per coin). Computes the percentage of trades that were
aggressive buys vs sells over a configurable lookback window.

This is one of the strongest short-term predictors in crypto — it tells you
who is in control RIGHT NOW, not 15 minutes ago from a candle.

Signal types:
- taker_buy_dominant: 60-70% buys → bullish momentum
- taker_sell_dominant: 60-70% sells → bearish momentum  
- aggressive_buying:   >70% buys → strong bullish, possible breakout
- aggressive_selling:  >70% sells → strong bearish, possible breakdown
- neutral:             40-60% → no clear aggression
"""

from __future__ import annotations
import time
import logging
from collections import deque
from typing import Optional

log = logging.getLogger("taker_ratio")

# Per-coin cache: {coin: (timestamp, result_dict)}
_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 2.0  # seconds — trades stream updates frequently


def _get_trade_history(coin: str) -> list[dict]:
    """Get rolling trade window from WebSocket data."""
    try:
        from hyperliquid_ws import _latest_data
        history = _latest_data.get("trade_history", {}).get(coin)
        if history is None:
            return []
        return list(history)
    except Exception:
        return []


def get_taker_ratio(coin: str, lookback_seconds: float = 30.0) -> dict:
    """
    Compute aggressive buy percentage over the last N seconds of trades.

    Returns:
        {"buy_pct": 0-100, "sell_pct": 0-100, "total_trades": N,
         "buy_volume": float, "sell_volume": float,
         "signal": "taker_buy_dominant"|"taker_sell_dominant"|"aggressive_buying"|
                   "aggressive_selling"|"neutral"|"no_data",
         "confidence": 0-100, "direction": "bullish"|"bearish"|"neutral"}
    """
    now = time.time()
    ck = f"{coin}_{lookback_seconds}"

    # Cache hit
    if ck in _cache:
        ts, result = _cache[ck]
        if now - ts < CACHE_TTL:
            return result

    trades = _get_trade_history(coin)
    if not trades:
        result = {"buy_pct": 50, "sell_pct": 50, "total_trades": 0,
                  "buy_volume": 0, "sell_volume": 0,
                  "signal": "no_data", "confidence": 0, "direction": "neutral"}
        _cache[ck] = (now, result)
        return result

    # Filter to lookback window
    cutoff_ms = (now - lookback_seconds) * 1000
    window = [t for t in trades if t.get("ts", 0) >= cutoff_ms]

    if len(window) < 5:
        result = {"buy_pct": 50, "sell_pct": 50, "total_trades": len(window),
                  "buy_volume": 0, "sell_volume": 0,
                  "signal": "no_data", "confidence": 0, "direction": "neutral"}
        _cache[ck] = (now, result)
        return result

    buys = [t for t in window if t.get("side") == "B"]
    sells = [t for t in window if t.get("side") == "A"]
    total = len(buys) + len(sells)

    if total == 0:
        buy_pct = 50.0
    else:
        buy_pct = len(buys) / total * 100

    sell_pct = 100 - buy_pct
    buy_vol = sum(float(t.get("size", 0)) for t in buys)
    sell_vol = sum(float(t.get("size", 0)) for t in sells)

    # ── Classify signal ──
    if buy_pct >= 70:
        signal = "aggressive_buying"
        direction = "bullish"
        confidence = min(80, 40 + (buy_pct - 70) * 2)
    elif buy_pct >= 60:
        signal = "taker_buy_dominant"
        direction = "bullish"
        confidence = 20 + (buy_pct - 60) * 1.5
    elif buy_pct <= 30:
        signal = "aggressive_selling"
        direction = "bearish"
        confidence = min(80, 40 + (30 - buy_pct) * 2)
    elif buy_pct <= 40:
        signal = "taker_sell_dominant"
        direction = "bearish"
        confidence = 20 + (40 - buy_pct) * 1.5
    else:
        signal = "neutral"
        direction = "neutral"
        confidence = 5

    # Volume-weighted adjustment: if buy volume >> sell volume even at 55%,
    # it's more significant than the trade count suggests
    total_vol = buy_vol + sell_vol
    if total_vol > 0:
        vol_buy_pct = buy_vol / total_vol * 100
        if abs(vol_buy_pct - buy_pct) > 15:
            # Volume and count disagree — size matters more
            if vol_buy_pct >= 65:
                direction = "bullish"
                confidence = min(75, confidence + 15)
            elif vol_buy_pct <= 35:
                direction = "bearish"
                confidence = min(75, confidence + 15)

    result = {
        "buy_pct": round(buy_pct, 1),
        "sell_pct": round(sell_pct, 1),
        "total_trades": total,
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
        "signal": signal,
        "direction": direction,
        "confidence": round(confidence, 1),
    }
    _cache[ck] = (now, result)
    return result


def _layer_taker_ratio(coin: str, mid: float,
                       lookback_seconds: float = 60.0) -> dict[str, float]:
    """
    Continuous predictor layer format.

    Returns {"up", "down", "flat", "confidence"} for ensemble integration.
    Uses 60s lookback (one daemon cycle) for stable signal.
    """
    ratio = get_taker_ratio(coin, lookback_seconds)

    direction = ratio["direction"]
    confidence = ratio["confidence"]

    if direction == "bullish":
        return {"up": 40 + confidence * 0.35, "down": 30 - confidence * 0.2,
                "flat": 30 - confidence * 0.15, "confidence": confidence}
    elif direction == "bearish":
        return {"up": 30 - confidence * 0.2, "down": 40 + confidence * 0.35,
                "flat": 30 - confidence * 0.15, "confidence": confidence}
    else:
        return {"up": 33, "down": 33, "flat": 34, "confidence": max(5, confidence * 0.2)}


def get_taker_context(coin: str) -> str:
    """Compact AI context string for prompt enrichment."""
    r = get_taker_ratio(coin, 60.0)
    if r["signal"] == "no_data":
        return ""
    vol_str = ""
    if r["total_trades"] > 0:
        vol_str = f" vol:B{r['buy_volume']:.1f}/S{r['sell_volume']:.1f}"
    return (f"Taker:{r['buy_pct']:.0f}%buy "
            f"({r['total_trades']}t{vol_str}) "
            f"{r['signal']}")


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Taker Ratio module loaded.")
    print("  get_taker_ratio(coin, lookback_seconds=30)")
    print("  _layer_taker_ratio(coin, mid, lookback_seconds=60)")
    print("  get_taker_context(coin)")
    print()
    # Test with empty data
    result = get_taker_ratio("SOL", 30)
    print(f"  SOL (no data): {result['signal']} conf={result['confidence']}")
    # Test layer format
    layer = _layer_taker_ratio("SOL", 100.0)
    print(f"  SOL layer: up={layer['up']} down={layer['down']} flat={layer['flat']} conf={layer['confidence']}")
