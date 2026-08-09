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

# ── Heatmap engine merged inline (Aug 9, was liquidation_heatmap.py) ──

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
    if coin is None:
        _heatmap_cache.clear()
        log.info("Liquidation heatmap cache cleared (all coins)")
    else:
        _heatmap_cache.pop(coin.upper(), None)
        log.info(f"Liquidation heatmap cache cleared for {coin}")


# ── Self-test ──

# ── Heatmap Engine (merged from liquidation_heatmap.py) ──

log = logging.getLogger("liq_heatmap")

# ── Configuration ──
MIN_DEPTH_USD = 50_000        # Minimum USD notional for a level to be "significant"
MAX_LIQ_LEVELS = 8            # Max liquidation levels to return per coin
CLUSTER_PRICE_PCT = 0.002     # 0.2% — merge nearby levels into clusters
DEPTH_WEIGHT_N = 0.3          # Weight for order count in depth scoring
DEPTH_WEIGHT_SZ = 0.7         # Weight for total size in depth scoring


@dataclass
class LiquidationLevel:
    """A price level where significant liquidation activity is expected."""
    price: float
    side: str               # 'bid' (below price → long liqs) or 'ask' (above → short liqs)
    total_size: float       # Total coin size at this cluster
    notional_usd: float     # USD notional value
    order_count: int        # Number of resting orders
    depth_score: float      # Composite score (0-1), higher = more significant
    distance_pct: float     # Distance from current price (negative = below)


@dataclass
class LiquidationHeatmap:
    """Full liquidation heatmap for a single coin."""
    coin: str
    current_price: float
    oracle_price: float
    open_interest_usd: float
    timestamp: float
    long_liq_zones: list[LiquidationLevel]   # Levels BELOW price (long liquidations)
    short_liq_zones: list[LiquidationLevel]  # Levels ABOVE price (short liquidations)
    max_depth_score: float                   # Highest individual cluster score
    cascade_risk_score: float                # 0-1 overall cascade risk


# ── Cache ──
_heatmap_cache: dict[str, LiquidationHeatmap] = {}
_heatmap_cache_ts: float = 0.0
_HEATMAP_TTL = 300  # 5 minutes


# ── Core Fetcher ──

def _fetch_order_book(coin: str) -> Optional[dict]:
    """DISABLED — triggers 429 rate limits per coin."""
    return None


def _fetch_asset_context(coin: str) -> Optional[dict]:
    """DISABLED — triggers 429 rate limits (fetches ALL coins metadata per call)."""
    return None


def _compute_depth_score(sz: float, n: int, max_sz: float, max_n: int) -> float:
    """Compute composite depth score (0-1) for a level or cluster."""
    if max_sz <= 0 and max_n <= 0:
        return 0.0
    sz_score = (sz / max_sz) if max_sz > 0 else 0.0
    n_score = (n / max_n) if max_n > 0 else 0.0
    return DEPTH_WEIGHT_SZ * sz_score + DEPTH_WEIGHT_N * n_score


def _cluster_levels(levels: list[dict], current_price: float,
                    is_bids: bool,
                    global_max_sz: float = 0.0,
                    global_max_n: int = 0) -> list[LiquidationLevel]:
    """
    Cluster nearby order book levels into liquidation zones.

    Levels within CLUSTER_PRICE_PCT of current_price are merged into one zone.
    Returns top clusters sorted by depth score (most significant first).

    Args:
        global_max_sz, global_max_n: Max values across ALL sides for proper
            normalization. If 0, uses local max (for standalone use).
    """
    if not levels:
        return []

    # Convert to (price, sz, n)
    parsed = []
    for lvl in levels:
        px = float(lvl.get("px", 0))
        sz = float(lvl.get("sz", 0))
        n = int(lvl.get("n", 0))
        if px > 0:
            parsed.append((px, sz, n))

    if not parsed:
        return []

    # Find max sz and n for normalization (use global if provided)
    max_sz = global_max_sz if global_max_sz > 0 else max(p[1] for p in parsed)
    max_n = global_max_n if global_max_n > 0 else max(p[2] for p in parsed)

    # Cluster nearby levels
    cluster_threshold = current_price * CLUSTER_PRICE_PCT
    clusters: list[dict] = []  # [{price, total_sz, total_n, count}]

    for px, sz, n in parsed:
        merged = False
        for cluster in clusters:
            if abs(px - cluster["price"]) <= cluster_threshold:
                # Weighted average price
                total_sz_new = cluster["total_sz"] + sz
                cluster["price"] = (cluster["price"] * cluster["total_sz"] + px * sz) / total_sz_new
                cluster["total_sz"] = total_sz_new
                cluster["total_n"] += n
                cluster["count"] += 1
                merged = True
                break
        if not merged:
            clusters.append({
                "price": px,
                "total_sz": sz,
                "total_n": n,
                "count": 1,
            })

    # Score and filter
    side = "bid" if is_bids else "ask"
    results = []
    for cluster in clusters:
        notional = cluster["total_sz"] * cluster["price"]
        if notional < MIN_DEPTH_USD:
            continue
        score = _compute_depth_score(cluster["total_sz"], cluster["total_n"], max_sz, max_n)
        distance_pct = (cluster["price"] - current_price) / current_price * 100
        results.append(LiquidationLevel(
            price=round(cluster["price"], 2),
            side=side,
            total_size=round(cluster["total_sz"], 4),
            notional_usd=round(notional, 0),
            order_count=cluster["total_n"],
            depth_score=round(score, 4),
            distance_pct=round(distance_pct, 2),
        ))

    # Sort by depth score descending
    results.sort(key=lambda x: x.depth_score, reverse=True)
    return results[:MAX_LIQ_LEVELS]


def _compute_cascade_risk(long_zones: list[LiquidationLevel],
                          short_zones: list[LiquidationLevel],
                          open_interest_usd: float,
                          book_depth_total_usd: float) -> float:
    """
    Compute cascade risk score (0-1).

    Factors:
    - Depth concentration: are there a few very deep levels? (many clusters = distributed = safer)
    - Proximity: are major clusters close to current price?
    - Open interest ratio: higher OI relative to visible book depth = more leverage = more risk
    - Asymmetry: one-sided depth imbalance (lopsided clusters = directional risk)

    Score is deliberately capped below ~0.7 for normal conditions — only
    genuinely dangerous setups reach >0.7.
    """
    if book_depth_total_usd <= 0:
        return 0.0

    all_clusters = long_zones + short_zones
    if not all_clusters:
        return 0.0

    # 1) Concentration risk: how concentrated is depth?
    #    Few clusters with high scores = dangerous; many clusters = distributed risk
    cluster_count = len(all_clusters)
    if cluster_count <= 1:
        concentration = 1.0   # Single cluster dominates
    elif cluster_count <= 2:
        concentration = 0.7
    elif cluster_count <= 4:
        concentration = 0.4
    else:
        concentration = 0.2

    # Also factor in how dominant the top cluster is vs others
    if cluster_count >= 2:
        scores = sorted([c.depth_score for c in all_clusters], reverse=True)
        top_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else 0
        if top_score > 0 and (top_score - second_score) / max(top_score, 0.01) > 0.5:
            concentration = min(1.0, concentration + 0.2)

    # 2) Proximity risk: weighted by how close major clusters are
    proximity_score = 0.0
    for zone in all_clusters[:3]:  # Top 3 clusters by depth score
        dist = abs(zone.distance_pct)
        if dist < 0.1:           # Within 0.1% — right at the level
            proximity_score += 1.0
        elif dist < 0.3:         # Within 0.3%
            proximity_score += 0.9
        elif dist < 0.5:         # Within 0.5%
            proximity_score += 0.7
        elif dist < 1.0:         # Within 1%
            proximity_score += 0.5
        elif dist < 2.0:         # Within 2%
            proximity_score += 0.3
        elif dist < 5.0:         # Within 5%
            proximity_score += 0.1
    proximity_score = min(1.0, proximity_score / 3.0)

    # 3) Asymmetry risk: one-sided depth imbalance
    #    If all deep clusters are on one side → directional cascade risk
    bid_total = sum(z.depth_score for z in long_zones)
    ask_total = sum(z.depth_score for z in short_zones)
    total_depth_score = bid_total + ask_total
    if total_depth_score > 0:
        asymmetry = abs(bid_total - ask_total) / total_depth_score
    else:
        asymmetry = 0.0

    # 4) Leverage proxy: OI relative to book depth
    #    Very high OI with thin book = potential for cascade
    if book_depth_total_usd > 0:
        oi_ratio = min(1.0, open_interest_usd / (book_depth_total_usd * 50))
    else:
        oi_ratio = 0.0

    # Weighted composite (clamped to 0-1)
    risk = (0.25 * concentration +
            0.35 * proximity_score +
            0.15 * asymmetry +
            0.25 * oi_ratio)
    return round(min(1.0, risk), 4)


# ── Public API ──

def fetch_heatmap(coin: str, current_price: float = 0.0) -> LiquidationHeatmap:
    """
    Fetch full liquidation heatmap for a coin.

    Uses cached result if available and not expired (5-min TTL).
    Falls back gracefully if API calls fail.
    """
    global _heatmap_cache, _heatmap_cache_ts

    coin_upper = coin.upper()
    now = time.time()

    # Check cache
    if coin_upper in _heatmap_cache:
        cached = _heatmap_cache[coin_upper]
        if now - cached.timestamp < _HEATMAP_TTL:
            return cached

    # Fetch fresh data
    book = _fetch_order_book(coin_upper)
    ctx = _fetch_asset_context(coin_upper)

    # Determine current price
    if current_price <= 0:
        if ctx and ctx.get("midPx", 0) > 0:
            current_price = ctx["midPx"]
        elif ctx and ctx.get("markPx", 0) > 0:
            current_price = ctx["markPx"]
        elif ctx and ctx.get("oraclePx", 0) > 0:
            current_price = ctx["oraclePx"]

    oracle_price = ctx.get("oraclePx", current_price) if ctx else current_price
    open_interest = ctx.get("openInterest", 0) if ctx else 0

    # Parse order book levels
    bid_zones: list[LiquidationLevel] = []
    ask_zones: list[LiquidationLevel] = []
    book_depth_total = 0.0

    if book:
        levels = book.get("levels", [])
        if levels and len(levels) >= 2:
            # Compute global max sz/n across ALL levels for consistent normalization
            all_parsed = []
            for side_levels in levels[:2]:
                for lvl in side_levels:
                    all_parsed.append((
                        float(lvl.get("px", 0)),
                        float(lvl.get("sz", 0)),
                        int(lvl.get("n", 0)),
                    ))
            global_max_sz = max((p[1] for p in all_parsed), default=0.0)
            global_max_n = max((p[2] for p in all_parsed), default=0)

            # levels[0] = bids (below price → long liquidation zones)
            # levels[1] = asks (above price → short liquidation zones)
            bid_zones = _cluster_levels(levels[0], current_price, is_bids=True,
                                        global_max_sz=global_max_sz,
                                        global_max_n=global_max_n)
            ask_zones = _cluster_levels(levels[1], current_price, is_bids=False,
                                        global_max_sz=global_max_sz,
                                        global_max_n=global_max_n)

            book_depth_total = sum(z.notional_usd for z in bid_zones + ask_zones)

    max_score = 0.0
    all_zones = bid_zones + ask_zones
    if all_zones:
        max_score = max(z.depth_score for z in all_zones)

    cascade_risk = _compute_cascade_risk(bid_zones, ask_zones, open_interest, book_depth_total)

    heatmap = LiquidationHeatmap(
        coin=coin_upper,
        current_price=round(current_price, 2),
        oracle_price=round(oracle_price, 2),
        open_interest_usd=round(open_interest, 0),
        timestamp=now,
        long_liq_zones=bid_zones,
        short_liq_zones=ask_zones,
        max_depth_score=round(max_score, 4),
        cascade_risk_score=cascade_risk,
    )

    _heatmap_cache[coin_upper] = heatmap
    _heatmap_cache_ts = now
    return heatmap


def format_heatmap_for_ai(coin: str, current_price: float = 0.0) -> str:
    """
    Format liquidation heatmap as a compact AI prompt string.

    Returns something like:
    'Liq clusters: $6.20 (120K), $5.90 (85K) | Short liq: $6.80 (200K) | Cascade risk: 0.35'
    """
    hm = fetch_heatmap(coin, current_price)

    parts = []

    # Long liquidation zones (below price)
    if hm.long_liq_zones:
        long_strs = []
        for z in hm.long_liq_zones[:3]:
            notional_k = z.notional_usd / 1000
            long_strs.append(f"${z.price:,.2f}({notional_k:.0f}K)")
        if long_strs:
            parts.append("Long liq: " + ", ".join(long_strs))

    # Short liquidation zones (above price)
    if hm.short_liq_zones:
        short_strs = []
        for z in hm.short_liq_zones[:3]:
            notional_k = z.notional_usd / 1000
            short_strs.append(f"${z.price:,.2f}({notional_k:.0f}K)")
        if short_strs:
            parts.append("Short liq: " + ", ".join(short_strs))

    # Cascade risk
    if hm.cascade_risk_score > 0.1:
        risk_label = "HIGH" if hm.cascade_risk_score > 0.5 else "MOD" if hm.cascade_risk_score > 0.25 else "LOW"
        parts.append(f"Cascade:{risk_label}({hm.cascade_risk_score:.2f})")

    # OI context (smart formatting — use K or M)
    if hm.open_interest_usd > 0:
        if hm.open_interest_usd >= 1_000_000:
            oi_m = hm.open_interest_usd / 1_000_000
            parts.append(f"OI:${oi_m:.1f}M")
        elif hm.open_interest_usd >= 1_000:
            oi_k = hm.open_interest_usd / 1_000
            parts.append(f"OI:${oi_k:.0f}K")
        else:
            parts.append(f"OI:${hm.open_interest_usd:.0f}")

    return " | ".join(parts) if parts else ""


def format_liquidation_for_ai(coin: str, current_price: float = 0.0) -> str:
    """
    Alias for format_heatmap_for_ai — matches the expected import in context_enricher.py.
    """
    return format_heatmap_for_ai(coin, current_price)


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
