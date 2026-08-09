#!/usr/bin/env python3
"""
Liquidation Monitor — reads REAL liquidation events from the userFills WebSocket stream.

Unlike liquidation_zones.py (estimates from order book depth) and liquidation_cascade.py
(estimates from OI delta), this module reads ACTUAL liquidation fills from the
hyperliquid_ws._latest_data['fills'] stream, which contains real-time fill confirmations
with a `liquidation` boolean flag.

Provides:
  get_liquidation_activity(coin)      → {recent_liquidations, cascade_risk, direction, ...}
  get_liquidation_context(coin)       → compact string for AI prompt enrichment
  get_self_liquidations()             → list of our own liquidated positions
  has_self_liquidation(coin)          → bool — were we recently liquidated on this coin?
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Optional

log = logging.getLogger("liq_monitor")

# ── Configuration ──────────────────────────────────────────────────────────

# Time windows for liquidation activity tracking (seconds)
WINDOW_RECENT = 300       # 5 min — "recent liquidations" count
WINDOW_CASCADE = 900      # 15 min — cascade risk assessment
WINDOW_SELF_LIQ = 600     # 10 min — self-liquidation recency

# Cascade risk thresholds
CASCADE_HIGH_THRESHOLD = 5     # 5+ liquidations in 15 min = high risk
CASCADE_MODERATE_THRESHOLD = 2  # 2-4 liquidations = moderate risk

# Direction thresholds (when ratio > 0.7, declare direction)
DIRECTION_BIAS_THRESHOLD = 0.7


def _get_fills() -> list[dict]:
    """Get current fills snapshot from the shared WebSocket state."""
    try:
        from hyperliquid_ws import get_latest
        data = get_latest()
        return data.get("fills", [])
    except Exception as e:
        log.debug(f"Failed to read fills from hyperliquid_ws: {e}")
        return []


def _get_liquidation_fills(coin: Optional[str] = None, window_seconds: Optional[int] = None) -> list[dict]:
    """
    Get fills flagged as liquidations, optionally filtered by coin and time window.

    Args:
        coin: Filter to specific coin (None = all coins)
        window_seconds: Only return fills within this many seconds of now (None = all)

    Returns:
        List of fill dicts with liquidation=True, most recent first
    """
    fills = _get_fills()
    now_ms = int(time.time() * 1000)
    cutoff_ms = None
    if window_seconds is not None:
        cutoff_ms = now_ms - (window_seconds * 1000)

    result = []
    for f in fills:
        if not f.get("liquidation", False):
            continue
        if coin and f.get("coin", "").upper() != coin.upper():
            continue
        if cutoff_ms is not None and f.get("time", 0) < cutoff_ms:
            continue
        result.append(f)

    # Sort by time descending (most recent first)
    result.sort(key=lambda x: x.get("time", 0), reverse=True)
    return result


def _classify_liquidation_direction(fills: list[dict]) -> str | None:
    """
    Classify the direction of liquidation activity.

    In Hyperliquid, userFills provides:
      - 'dir' field: "Close Long" (long liquidation) or "Close Short" (short liquidation)
      - 'side' field: "A" (ask/sell) = long liquidation, "B" (bid/buy) = short liquidation

    Returns:
        'longs_getting_wrecked' | 'shorts_getting_wrecked' | 'mixed' | None
    """
    if not fills:
        return None

    long_count = 0
    short_count = 0

    for f in fills:
        dir_val = f.get("dir", "")
        side_val = f.get("side", "")

        # Primary: use 'dir' field
        if "Close Long" in dir_val:
            long_count += 1
        elif "Close Short" in dir_val:
            short_count += 1
        # Fallback: use 'side' field
        elif side_val == "A":
            long_count += 1  # selling = long liquidation
        elif side_val == "B":
            short_count += 1  # buying = short liquidation

    total = long_count + short_count
    if total == 0:
        return None

    long_ratio = long_count / total
    short_ratio = short_count / total

    if long_ratio >= DIRECTION_BIAS_THRESHOLD:
        return "longs_getting_wrecked"
    elif short_ratio >= DIRECTION_BIAS_THRESHOLD:
        return "shorts_getting_wrecked"
    else:
        return "mixed"


def _compute_cascade_risk(count_15m: int) -> str | None:
    """
    Compute cascade risk level from liquidation count in the 15-min window.

    Returns: 'high' | 'moderate' | 'low' | None
    """
    if count_15m >= CASCADE_HIGH_THRESHOLD:
        return "high"
    elif count_15m >= CASCADE_MODERATE_THRESHOLD:
        return "moderate"
    elif count_15m > 0:
        return "low"
    return None


def get_liquidation_activity(coin: str) -> dict[str, Any]:
    """
    Get liquidation activity for a specific coin.

    Reads real liquidation fills from the hyperliquid_ws userFills stream.

    Returns:
        {
            "coin": str,
            "recent_liquidations": int,     # count in last 5 min
            "liquidations_15m": int,        # count in last 15 min
            "cascade_risk": "high" | "moderate" | "low" | None,
            "direction": "longs_getting_wrecked" | "shorts_getting_wrecked" | "mixed" | None,
            "latest_liq_time_ms": int | None,  # timestamp of most recent liq
            "latest_liq_price": float | None,  # price of most recent liq
            "has_data": bool,                  # True if any fills exist (liq or not)
        }
    """
    coin_upper = coin.upper()

    # Get fills with liquidation flag in different time windows
    fills_5m = _get_liquidation_fills(coin=coin_upper, window_seconds=WINDOW_RECENT)
    fills_15m = _get_liquidation_fills(coin=coin_upper, window_seconds=WINDOW_CASCADE)

    count_5m = len(fills_5m)
    count_15m = len(fills_15m)

    # Direction from fills in the wider window for better accuracy
    direction = _classify_liquidation_direction(fills_15m) if fills_15m else None
    cascade_risk = _compute_cascade_risk(count_15m)

    # Latest liquidation details
    latest_time = None
    latest_price = None
    if fills_15m:
        latest = fills_15m[0]  # already sorted desc
        latest_time = latest.get("time", None)
        latest_price = float(latest.get("px", 0)) if latest.get("px") else None

    # Check if we have any data at all for this coin
    all_fills = _get_fills()
    has_data = any(f.get("coin", "").upper() == coin_upper for f in all_fills)

    return {
        "coin": coin_upper,
        "recent_liquidations": count_5m,
        "liquidations_15m": count_15m,
        "cascade_risk": cascade_risk,
        "direction": direction,
        "latest_liq_time_ms": latest_time,
        "latest_liq_price": latest_price,
        "has_data": has_data,
    }


def get_liquidation_context(coin: str) -> str:
    """
    Build compact liquidation context string for AI prompt enrichment.

    Reads real liquidation fills (not estimated zones) from the WebSocket stream.

    Returns format examples:
      "Liq:3 SOL longs | Cascade:high"
      "Liq:1 AVAX short | Cascade:moderate"
      "Liq:none"
      "Liq:none (no data)" — when WebSocket isn't running

    The output is designed to be <50 chars, fitting within the ~150 token context budget.
    """
    activity = get_liquidation_activity(coin)

    if not activity.get("has_data", False):
        return "Liq:none (no data)"

    count = activity["recent_liquidations"]
    direction = activity.get("direction")
    cascade = activity.get("cascade_risk")

    if count == 0:
        return "Liq:none"

    parts = [f"Liq:{count} {coin.upper()}"]
    if direction == "longs_getting_wrecked":
        parts.append("longs")
    elif direction == "shorts_getting_wrecked":
        parts.append("shorts")
    elif direction == "mixed":
        parts.append("mixed")

    if cascade:
        parts.append(f"Cascade:{cascade}")

    return " ".join(parts)


def get_self_liquidations(coin: Optional[str] = None) -> list[dict]:
    """
    Get fills where our own position was liquidated.

    These are fills with liquidation=True and a significant negative closedPnl.
    In userFills, EVERY liquidation fill is a self-liquidation (since userFills
    only reports fills involving the authenticated user). We distinguish by:
      - If closedPnl is negative (we lost money), it's a self-liquidation
      - If closedPnl is 0 or we can't determine, any liquidation flag counts

    Args:
        coin: Optional coin filter. None = return all self-liquidations.

    Returns:
        List of fill dicts, most recent first
    """
    liq_fills = _get_liquidation_fills(coin=coin, window_seconds=WINDOW_SELF_LIQ)

    # Filter for fills where we likely lost money (negative closedPnl)
    self_liqs = []
    for f in liq_fills:
        pnl_str = f.get("closedPnl", "0")
        try:
            pnl = float(pnl_str)
        except (ValueError, TypeError):
            pnl = 0.0

        # A self-liquidation typically has negative PnL
        # But we also include fills where PnL is 0 (unknown/maker-side)
        if pnl < 0 or pnl == 0:
            self_liqs.append(f)

    return self_liqs


def has_self_liquidation(coin: str) -> bool:
    """
    Quick check: were we recently liquidated on this coin?

    Args:
        coin: Coin ticker

    Returns:
        True if a self-liquidation was detected in the recent window
    """
    liqs = get_self_liquidations(coin=coin.upper())
    return len(liqs) > 0


def get_self_liquidation_summary() -> str:
    """
    Compact summary of our own liquidations for AI context.

    Returns:
        String like "⚠️SELF-LIQ: SOL(-$12.50) | AVAX(-$8.20)" or empty string
    """
    liqs = get_self_liquidations()

    if not liqs:
        return ""

    # Group by coin
    by_coin: dict[str, list[dict]] = defaultdict(list)
    for f in liqs:
        coin = f.get("coin", "UNKNOWN").upper()
        by_coin[coin].append(f)

    parts = []
    for coin, fills in by_coin.items():
        total_pnl = 0.0
        for f in fills:
            try:
                total_pnl += float(f.get("closedPnl", 0))
            except (ValueError, TypeError):
                pass
        if total_pnl < 0:
            parts.append(f"{coin}(-${abs(total_pnl):.2f})")
        else:
            parts.append(f"{coin}({len(fills)}x)")

    return "⚠️SELF-LIQ:" + " | ".join(parts)


def get_all_activity_summary() -> str:
    """
    Get a summary of liquidation activity across all coins with recent activity.

    Returns:
        String like "SOL:3(L), AVAX:1(S) | Cascade:SOL=high" or empty string
    """
    all_fills = _get_liquidation_fills(window_seconds=WINDOW_CASCADE)

    if not all_fills:
        return ""

    # Group by coin
    by_coin: dict[str, list[dict]] = defaultdict(list)
    for f in all_fills:
        coin = f.get("coin", "UNKNOWN").upper()
        by_coin[coin].append(f)

    # Build per-coin summaries
    coin_parts = []
    cascade_parts = []
    for coin, fills in sorted(by_coin.items()):
        direction = _classify_liquidation_direction(fills)
        dir_char = {
            "longs_getting_wrecked": "L",
            "shorts_getting_wrecked": "S",
            "mixed": "M",
        }.get(direction, "?")

        coin_parts.append(f"{coin}:{len(fills)}({dir_char})")

        risk = _compute_cascade_risk(len(fills))
        if risk and risk in ("high", "moderate"):
            cascade_parts.append(f"{coin}={risk}")

    result = " | ".join(coin_parts)
    if cascade_parts:
        result += " | Cascade:" + " ".join(cascade_parts)

    return result


# ── Self-test ──────────────────────────────────────────────────────────────


# ── Cascade Detection (merged from liquidation_cascade.py, Aug 9) ──
class CascadeSignal:
    """Signal from liquidation cascade analysis."""
    symbol: str
    cascade_detected: bool
    cascade_side: str  # 'longs_liquidated' or 'shorts_liquidated'
    severity: float  # 0-1, how severe the cascade
    bounce_probability: float  # 0-1, probability of reversal
    lai: float  # Liquidation Asymmetry Index (-1 to 1)
    suggested_action: str  # 'BUY' | 'SELL' | 'HOLD'
    confidence: float
    reason: str

def detect_cascade_from_candles(
    candles: list[dict],
    symbol: str = "",
    lookback_bars: int = 12,
) -> CascadeSignal:
    """
    Detect liquidation cascade patterns from candle data (proxy method).
    
    Signs of a liquidation cascade in candles:
    1. Large candle(s) with high volume → forced selling/buying
    2. Long wicks on reversal → liquidation exhaustion
    3. Volume spike > 3x average → cascade volume
    
    Strategy: Look for exhaustion patterns after cascade candles.
    """
    if len(candles) < lookback_bars:
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=0, bounce_probability=0,
            lai=0, suggested_action="HOLD", confidence=0,
            reason="insufficient_candles",
        )
    
    recent = candles[-lookback_bars:]
    
    # Compute average volume
    volumes = [float(c.get("v", c.get("volume", 0))) for c in recent[:-2]]
    avg_vol = sum(volumes) / len(volumes) if volumes else 1
    
    # Check last 2 candles
    last = recent[-1]
    prev = recent[-2]
    prev_vol = float(prev.get("v", prev.get("volume", 0)))
    last_vol = float(last.get("v", last.get("volume", 0)))
    
    prev_o = float(prev.get("o", prev.get("open", 0)))
    prev_c = float(prev.get("c", prev.get("close", 0)))
    last_o = float(last.get("o", last.get("open", 0)))
    last_c = float(last.get("c", last.get("close", 0)))
    prev_h = float(prev.get("h", prev.get("high", 0)))
    prev_l = float(prev.get("l", prev.get("low", 0)))
    last_h = float(last.get("h", last.get("high", 0)))
    last_l = float(last.get("l", last.get("low", 0)))
    
    if avg_vol <= 0:
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=0, bounce_probability=0,
            lai=0, suggested_action="HOLD", confidence=0,
            reason="zero_volume",
        )
    
    # Detect cascade candle: high volume + large range
    cascade_vol_threshold = 2.5  # 2.5x average volume
    cascade_range_threshold = 2.0  # 2x ATR
    
    prev_range_pct = abs(prev_c - prev_o) / prev_o * 100 if prev_o > 0 else 0
    last_range_pct = abs(last_c - last_o) / last_o * 100 if last_o > 0 else 0
    
    prev_is_cascade = (prev_vol > avg_vol * cascade_vol_threshold and 
                       prev_range_pct > 1.5)
    last_is_cascade = (last_vol > avg_vol * cascade_vol_threshold and 
                       last_range_pct > 1.5)
    
    if not prev_is_cascade and not last_is_cascade:
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=0, bounce_probability=0,
            lai=0, suggested_action="HOLD", confidence=0,
            reason="no_cascade_pattern",
        )
    
    # Determine cascade direction
    cascade_bearish = False
    cascade_bullish = False
    
    if prev_is_cascade:
        if prev_c < prev_o * 0.98:  # Big red candle
            cascade_bearish = True
        elif prev_c > prev_o * 1.02:  # Big green candle
            cascade_bullish = True
    
    # Check for exhaustion/reversal signal
    # Bearish cascade exhaustion: hammer/doji after big red
    # Bullish cascade exhaustion: shooting star after big green
    
    severity = max(prev_vol, last_vol) / (avg_vol * cascade_vol_threshold)
    severity = min(1.0, severity)
    
    bounce_prob = 0.0
    action = "HOLD"
    conf = 0.0
    reason = ""
    
    if cascade_bearish:
        # Long liquidation cascade - look for exhaustion (hammer/bullish reversal)
        lower_wick = (last_l - min(last_o, last_c)) / abs(last_o - last_c) if abs(last_o - last_c) > 0 else 0
        is_hammer = (lower_wick > 0.6 and last_c > last_o * 0.998)
        is_bullish_engulf = (last_c > last_o and last_o <= prev_c and last_c >= prev_o)
        
        if is_hammer or is_bullish_engulf:
            bounce_prob = min(0.80, 0.35 + severity * 0.4)
            action = "BUY"
            conf = min(80, 35 + severity * 40)
            reason = f"liq_cascade_exhaustion:hammer" if is_hammer else f"liq_cascade_exhaustion:engulf"
        else:
            # Cascade ongoing, wait
            bounce_prob = 0.2
            action = "HOLD"
            conf = 25
            reason = "liq_cascade_ongoing:waiting_for_exhaustion"
    
    elif cascade_bullish:
        # Short liquidation cascade - look for exhaustion (shooting star/bearish reversal)
        upper_wick = (last_h - max(last_o, last_c)) / abs(last_o - last_c) if abs(last_o - last_c) > 0 else 0
        is_shooting_star = (upper_wick > 0.6 and last_c < last_o * 1.002)
        is_bearish_engulf = (last_c < last_o and last_o >= prev_c and last_c <= prev_o)
        
        if is_shooting_star or is_bearish_engulf:
            bounce_prob = min(0.80, 0.35 + severity * 0.4)
            action = "SELL"
            conf = min(80, 35 + severity * 40)
            reason = f"liq_cascade_exhaustion:star" if is_shooting_star else f"liq_cascade_exhaustion:engulf"
        else:
            bounce_prob = 0.2
            action = "HOLD"
            conf = 25
            reason = "liq_cascade_ongoing:waiting_for_exhaustion"
    
    return CascadeSignal(
        symbol=symbol,
        cascade_detected=True,
        cascade_side="longs" if cascade_bearish else "shorts",
        severity=round(severity, 3),
        bounce_probability=round(bounce_prob, 3),
        lai=1.0 if cascade_bearish else -1.0,
        suggested_action=action,
        confidence=round(conf, 1),
        reason=reason,
    )

if __name__ == "__main__":
    import sys

    coin = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    print(f"=== Liquidation Monitor Test ({coin}) ===")
    print()

    # Test 1: get_liquidation_activity
    print("1. get_liquidation_activity:")
    activity = get_liquidation_activity(coin)
    for k, v in activity.items():
        print(f"   {k}: {v}")

    # Test 2: get_liquidation_context
    print(f"\n2. get_liquidation_context: {get_liquidation_context(coin)!r}")

    # Test 3: self-liquidations
    print(f"\n3. has_self_liquidation: {has_self_liquidation(coin)}")
    self_liqs = get_self_liquidations(coin)
    print(f"   Self-liquidations: {len(self_liqs)} fills")
    for liq in self_liqs[:3]:
        print(f"   - {liq.get('coin')} @ {liq.get('px')} x {liq.get('sz')} | pnl={liq.get('closedPnl')}")

    # Test 4: self-liquidation summary
    print(f"\n4. get_self_liquidation_summary: {get_self_liquidation_summary()!r}")

    # Test 5: all activity
    print(f"\n5. get_all_activity_summary: {get_all_activity_summary()!r}")

    print("\n=== Done ===")
