#!/usr/bin/env python3
"""
Liquidation Zones — high-level module for liquidation-aware trading decisions.

Builds on liquidation_heatmap to provide:
  1. Periodic fetching with 5-min cache expiry
  2. get_liquidation_context()  → compact AI prompt enrichment
  3. estimate_safe_stop()       → stops placed outside major liquidation clusters
  4. detect_cascade_risk()      → warns when price approaches major liquidation zones

Heuristic foundation:
  Since Hyperliquid has no direct "liquidation levels" REST endpoint, we infer
  liquidation zones from order book depth clusters (l2_snapshot) and open interest
  concentration (meta_and_asset_ctxs). Deep order book levels are where stops
  cluster — more resting orders → more liquidations absorbed there.

Data flow:
  l2_snapshot(coin) → parse bids/asks → cluster nearby levels → score by depth
  meta_and_asset_ctxs() → open interest → leverage proxy for cascade risk
  → LiquidationHeatmap → LiquidationZones public API
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from liquidation_heatmap import (
    LiquidationHeatmap,
    LiquidationLevel,
    fetch_heatmap,
    format_heatmap_for_ai,
    format_liquidation_for_ai,
    _HEATMAP_TTL,
)

log = logging.getLogger("liq_zones")

# ── Safety margins ──
SAFE_STOP_BUFFER_PCT = 0.003   # 0.3% buffer beyond the nearest cluster edge
CLUSTER_DANGER_THRESHOLD = 0.4  # Depth score threshold for "major" cluster
CASCADE_WARNING_DISTANCE = 2.0  # Within N% = approaching cascade territory


@dataclass
class CascadeRiskAssessment:
    """Structured cascade risk analysis."""
    coin: str
    current_price: float
    risk_level: str           # 'none', 'low', 'moderate', 'high', 'critical'
    risk_score: float         # 0-1
    approaching_zone: Optional[LiquidationLevel]  # Nearest major cluster
    distance_pct: float       # Distance to that cluster
    direction: str            # 'below' (long liqs approaching) or 'above' (short liqs)
    estimated_liquidation_size: float  # USD notional at risk
    summary: str              # Human-readable summary


def get_liquidation_context(coin: str, current_price: float = 0.0) -> str:
    """
    Build compact liquidation context string for AI prompt enrichment.

    Returns format: 'Liq clusters: $6.20 (120K), $5.90 (85K) | ...'
    Falls back to empty string on any error.

    Uses the heatmap module's 5-min cache — repeated calls within the TTL
    return cached data without re-fetching from the API.
    """
    return format_heatmap_for_ai(coin, current_price)


def estimate_safe_stop(coin: str, side: str, current_price: float = 0.0) -> float:
    """
    Estimate a safe stop-loss price outside major liquidation clusters.

    Strategy: place the stop just BEYOND the outermost significant liquidation
    cluster, so our stop isn't resting at a level where liquidations cascade.

    For LONG positions (side='long' or 'buy'):
      - Stop-loss goes BELOW entry → look at long_liq_zones (bid clusters)
      - Place stop below the deepest bid cluster that's below current price
      - This avoids getting stopped out by a long liquidation cascade

    For SHORT positions (side='short' or 'sell'):
      - Stop-loss goes ABOVE entry → look at short_liq_zones (ask clusters)
      - Place stop above the deepest ask cluster that's above current price
      - This avoids getting stopped out by a short liquidation cascade

    Args:
        coin: Coin ticker (e.g. 'BTC', 'ETH')
        side: 'long', 'buy', 'short', or 'sell'
        current_price: Current mid price (0 = auto-fetch)

    Returns:
        Recommended stop price, or 0.0 if no data available
    """
    hm = fetch_heatmap(coin, current_price)
    price = hm.current_price if hm.current_price > 0 else current_price

    if price <= 0:
        return 0.0

    side_lower = side.lower()

    if side_lower in ("long", "buy"):
        # For longs: stop goes below entry, below long liquidation clusters
        zones = hm.long_liq_zones  # bid clusters below price
        if not zones:
            # Fallback: place stop 2% below current price
            return round(price * 0.98, 2)

        # Find the deepest significant cluster below price
        # (the one that would cause the biggest cascade if hit)
        below_clusters = [z for z in zones if z.price < price]
        if not below_clusters:
            return round(price * 0.98, 2)

        # Sort by price descending (closest to price first)
        below_clusters.sort(key=lambda z: z.price, reverse=True)
        # Find the lowest-price major cluster (farthest down with high depth)
        major_clusters = [z for z in below_clusters if z.depth_score >= CLUSTER_DANGER_THRESHOLD]
        if major_clusters:
            # Place stop below the outermost major cluster
            outermost = min(major_clusters, key=lambda z: z.price)
            safe_stop = outermost.price * (1 - SAFE_STOP_BUFFER_PCT)
            return round(safe_stop, 2)

        # No major clusters: place stop below the farthest cluster with a buffer
        farthest = min(below_clusters, key=lambda z: z.price)
        safe_stop = farthest.price * (1 - SAFE_STOP_BUFFER_PCT)
        return round(safe_stop, 2)

    elif side_lower in ("short", "sell"):
        # For shorts: stop goes above entry, above short liquidation clusters
        zones = hm.short_liq_zones  # ask clusters above price
        if not zones:
            return round(price * 1.02, 2)

        above_clusters = [z for z in zones if z.price > price]
        if not above_clusters:
            return round(price * 1.02, 2)

        # Sort by price ascending (closest to price first)
        above_clusters.sort(key=lambda z: z.price)
        major_clusters = [z for z in above_clusters if z.depth_score >= CLUSTER_DANGER_THRESHOLD]
        if major_clusters:
            outermost = max(major_clusters, key=lambda z: z.price)
            safe_stop = outermost.price * (1 + SAFE_STOP_BUFFER_PCT)
            return round(safe_stop, 2)

        farthest = max(above_clusters, key=lambda z: z.price)
        safe_stop = farthest.price * (1 + SAFE_STOP_BUFFER_PCT)
        return round(safe_stop, 2)

    else:
        log.warning(f"estimate_safe_stop: unknown side '{side}', returning 0")
        return 0.0


def detect_cascade_risk(coin: str, current_price: float = 0.0) -> dict:
    """
    Detect if price is approaching a major liquidation cluster (potential cascade).

    Returns a dict with:
      - risk_level: 'none' | 'low' | 'moderate' | 'high' | 'critical'
      - risk_score: float 0-1
      - approaching_zone: dict with price, size, notional of nearest major cluster
      - direction: 'below' (price near long-liq cluster → risk of cascade down)
                   'above' (price near short-liq cluster → risk of cascade up)
      - distance_pct: how close price is to the dangerous cluster
      - estimated_liquidation_usd: notional value at risk
      - summary: human-readable explanation
      - safe_stop_long: recommended stop for long positions
      - safe_stop_short: recommended stop for short positions
    """
    hm = fetch_heatmap(coin, current_price)
    price = hm.current_price if hm.current_price > 0 else current_price

    if price <= 0:
        return {
            "risk_level": "none",
            "risk_score": 0.0,
            "approaching_zone": None,
            "direction": "none",
            "distance_pct": 999.0,
            "estimated_liquidation_usd": 0.0,
            "summary": "No price data available",
            "safe_stop_long": 0.0,
            "safe_stop_short": 0.0,
        }

    # Check proximity to major clusters
    nearest_zone = None
    nearest_distance = float("inf")
    nearest_direction = "none"

    # Check long liquidation zones (below price — price falling toward them)
    for zone in hm.long_liq_zones:
        if zone.depth_score < CLUSTER_DANGER_THRESHOLD:
            continue
        dist = abs(zone.distance_pct)  # distance_pct is negative for zones below
        if dist < nearest_distance:
            nearest_distance = dist
            nearest_zone = zone
            nearest_direction = "below"

    # Check short liquidation zones (above price — price rising toward them)
    for zone in hm.short_liq_zones:
        if zone.depth_score < CLUSTER_DANGER_THRESHOLD:
            continue
        dist = abs(zone.distance_pct)
        if dist < nearest_distance:
            nearest_distance = dist
            nearest_zone = zone
            nearest_direction = "above"

    # Determine risk level using cascade_score as primary signal
    # and proximity as secondary confirmation
    cascade_score = hm.cascade_risk_score

    if nearest_zone is None:
        risk_level = "none"
    elif cascade_score > 0.6 and nearest_distance < 0.5:
        risk_level = "critical"
    elif cascade_score > 0.5 and nearest_distance < 1.0:
        risk_level = "high"
    elif cascade_score > 0.35:
        risk_level = "moderate"
    elif cascade_score > 0.15:
        risk_level = "low"
    else:
        risk_level = "none"

    # Override: if there's a very close major cluster (<0.1%), always at least moderate
    if risk_level == "none" and nearest_distance < 0.1:
        risk_level = "moderate"
    elif risk_level == "none" and nearest_distance < 0.5 and hm.max_depth_score > 0.5:
        risk_level = "low"

    # Build summary
    if nearest_zone:
        notional_k = nearest_zone.notional_usd / 1000
        summary = (
            f"{risk_level.upper()} cascade risk: "
            f"price {nearest_distance:.2f}% from {nearest_direction} "
            f"liq cluster at ${nearest_zone.price:,.2f} "
            f"(${notional_k:.0f}K notional, "
            f"score={nearest_zone.depth_score:.2f})"
        )
    else:
        summary = f"{risk_level.upper()} cascade risk: no major clusters detected"

    # Compute safe stops
    safe_long = estimate_safe_stop(coin, "long", price)
    safe_short = estimate_safe_stop(coin, "short", price)

    return {
        "risk_level": risk_level,
        "risk_score": round(cascade_score, 4),
        "approaching_zone": (
            {
                "price": nearest_zone.price,
                "size": nearest_zone.total_size,
                "notional_usd": nearest_zone.notional_usd,
                "depth_score": nearest_zone.depth_score,
                "distance_pct": nearest_zone.distance_pct,
                "order_count": nearest_zone.order_count,
            }
            if nearest_zone else None
        ),
        "direction": nearest_direction,
        "distance_pct": round(nearest_distance, 2) if nearest_zone else 999.0,
        "estimated_liquidation_usd": nearest_zone.notional_usd if nearest_zone else 0.0,
        "summary": summary,
        "safe_stop_long": safe_long,
        "safe_stop_short": safe_short,
        "open_interest_usd": hm.open_interest_usd,
        "oracle_price": hm.oracle_price,
    }


def get_all_liquidation_context(coins: list[str], prices: dict[str, float] = None) -> str:
    """
    Build combined liquidation context for multiple coins.

    Useful for portfolio-level AI context enrichment.

    Args:
        coins: List of coin tickers
        prices: Optional dict mapping coin → current price

    Returns:
        Combined context string, e.g.:
        'BTC: Long liq $62K(1.2M) Short $65K(800K) | ETH: Long $2.8K(500K)'
    """
    prices = prices or {}
    parts = []
    for coin in coins:
        ctx = get_liquidation_context(coin, prices.get(coin.upper(), 0.0))
        if ctx:
            parts.append(f"{coin.upper()}: {ctx}")
    return " | ".join(parts)


# ── Cache inspection ──

def get_cache_stats() -> dict:
    """Return cache statistics for monitoring."""
    from liquidation_heatmap import _heatmap_cache, _heatmap_cache_ts
    now = time.time()
    return {
        "cached_coins": list(_heatmap_cache.keys()),
        "cache_count": len(_heatmap_cache),
        "cache_age_seconds": round(now - _heatmap_cache_ts, 1),
        "cache_ttl": _HEATMAP_TTL,
        "cache_valid": (now - _heatmap_cache_ts) < _HEATMAP_TTL if _heatmap_cache else False,
    }


def invalidate_cache(coin: str = None):
    """Invalidate the heatmap cache for a specific coin or all coins."""
    from liquidation_heatmap import _heatmap_cache, _heatmap_cache_ts
    if coin is None:
        _heatmap_cache.clear()
        log.info("Liquidation heatmap cache cleared (all coins)")
    else:
        _heatmap_cache.pop(coin.upper(), None)
        log.info(f"Liquidation heatmap cache cleared for {coin}")


# ── Self-test ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("=" * 60)
    print("LIQUIDATION ZONES — SELF-TEST")
    print("=" * 60)

    test_coins = ["BTC", "ETH", "SOL"]

    for coin in test_coins:
        print(f"\n{'─' * 40}")
        print(f"  {coin}")
        print(f"{'─' * 40}")

        # 1. Context string
        ctx = get_liquidation_context(coin)
        print(f"  Context: {ctx}")

        # 2. Safe stops
        safe_long = estimate_safe_stop(coin, "long")
        safe_short = estimate_safe_stop(coin, "short")
        print(f"  Safe stop (long):  ${safe_long:,.2f}" if safe_long > 0 else "  Safe stop (long):  N/A")
        print(f"  Safe stop (short): ${safe_short:,.2f}" if safe_short > 0 else "  Safe stop (short): N/A")

        # 3. Cascade risk
        risk = detect_cascade_risk(coin)
        print(f"  Cascade risk: {risk['risk_level'].upper()} (score={risk['risk_score']:.3f})")
        print(f"  {risk['summary']}")
        if risk["approaching_zone"]:
            z = risk["approaching_zone"]
            print(f"  Nearest zone: ${z['price']:,.2f} ({z['distance_pct']:+.2f}%) "
                  f"${z['notional_usd']:,.0f} notional")

    # 4. Multi-coin context
    print(f"\n{'─' * 40}")
    print("  Multi-coin context:")
    multi = get_all_liquidation_context(["BTC", "ETH"])
    print(f"  {multi}")

    # 5. Cache stats
    print(f"\n{'─' * 40}")
    stats = get_cache_stats()
    print(f"  Cache: {stats['cache_count']} coins, age={stats['cache_age_seconds']:.1f}s, "
          f"valid={stats['cache_valid']}")

    print(f"\n{'=' * 60}")
    print("✅ All tests complete")
