#!/usr/bin/env python3
"""
Dynamic Price Extremes — track rolling highs across timeframes.
Feeds into TP targets (replace fixed percentages) and AI exit decisions.

Usage:
    ext = get_extremes(coin, mids)  # returns dict with 5m/1h/4h/24h highs
    tp_pct = ext["1h_high_pct"]     # % above current price
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

# Cache TTLs: how long before we re-fetch candles
_CACHE_TTL = {
    "5m": 60,    # 1 min
    "1h": 300,   # 5 min
    "4h": 600,   # 10 min
    "24h": 1800, # 30 min
}

# Per-coin cache: {coin: {tf: (timestamp, high_price)}}
_cache: dict[str, dict[str, tuple[float, float]]] = {}


@dataclass
class PriceExtremes:
    """Rolling price highs AND lows across timeframes for a coin."""
    coin: str
    current: float
    # Highs
    high_5m: float = 0.0
    high_1h: float = 0.0
    high_4h: float = 0.0
    high_24h: float = 0.0
    # Lows
    low_5m: float = 0.0
    low_1h: float = 0.0
    low_4h: float = 0.0
    low_24h: float = 0.0
    # Percentage above/below current
    pct_5m: float = 0.0
    pct_1h: float = 0.0
    pct_4h: float = 0.0
    pct_24h: float = 0.0
    pct_low_1h: float = 0.0
    pct_low_4h: float = 0.0
    pct_low_24h: float = 0.0
    # Range position (0=at low, 100=at high)
    range_pos_1h: float = 50.0
    range_pos_4h: float = 50.0
    range_pos_24h: float = 50.0
    timestamp: float = field(default_factory=time.time)


def get_extremes(coin: str, mids: dict = None) -> PriceExtremes | None:
    """
    Get rolling price extremes for a coin. Caches per-timeframe.
    
    Returns PriceExtremes with highs and percentage distances.
    None if coin not found or no data.
    """
    mids = mids or {}
    current = float(mids.get(coin, 0))
    if current <= 0:
        return None
    
    now = time.time()
    coin_upper = coin.upper()
    
    if coin_upper not in _cache:
        _cache[coin_upper] = {}
    
    result = PriceExtremes(coin=coin_upper, current=current)
    
    timeframe_map = {
        "5m": ("high_5m", "low_5m", "pct_5m", "pct_low_5m", "range_pos_5m", 60, 5),
        "1h": ("high_1h", "low_1h", "pct_1h", "pct_low_1h", "range_pos_1h", 3600, 24),
        "4h": ("high_4h", "low_4h", "pct_4h", "pct_low_4h", "range_pos_4h", 14400, 50),
        "24h": ("high_24h", "low_24h", "pct_24h", "pct_low_24h", "range_pos_24h", 86400, 200),
    }
    
    for tf, (high_attr, low_attr, pct_high_attr, pct_low_attr, range_attr, candle_interval_s, n_candles) in timeframe_map.items():
        # Check cache
        if tf in _cache[coin_upper]:
            ts, high, low = _cache[coin_upper][tf]
            if now - ts < _CACHE_TTL[tf]:
                setattr(result, high_attr, high)
                setattr(result, low_attr, low)
                continue
        
        # Fetch candles
        try:
            from hyperliquid_daemon import _fetch_candles_cached
            candles = _fetch_candles_cached(coin_upper, tf, n_candles + 10)
            if candles and len(candles) >= 2:
                highs = [float(c.get("high", c.get("h", 0))) for c in candles]
                lows  = [float(c.get("low", c.get("l", 0))) for c in candles]
                highs = [h for h in highs if h > 0]
                lows  = [l for l in lows if l > 0]
                if highs and lows:
                    rolling_high = max(highs)
                    rolling_low  = min(lows)
                    _cache[coin_upper][tf] = (now, rolling_high, rolling_low)
                    setattr(result, high_attr, rolling_high)
                    setattr(result, low_attr, rolling_low)
        except Exception:
            pass
    
    # Calculate percentages and range position
    for high_attr, low_attr, pct_high, pct_low, range_attr in [
        ("high_5m", "low_5m", "pct_5m", "pct_low_5m", "range_pos_5m"),
        ("high_1h", "low_1h", "pct_1h", "pct_low_1h", "range_pos_1h"),
        ("high_4h", "low_4h", "pct_4h", "pct_low_4h", "range_pos_4h"),
        ("high_24h", "low_24h", "pct_24h", "pct_low_24h", "range_pos_24h"),
    ]:
        high = getattr(result, high_attr, 0)
        low  = getattr(result, low_attr, 0)
        if high > 0 and current > 0:
            setattr(result, pct_high, round((high - current) / current * 100, 2))
        if low > 0 and current > 0:
            setattr(result, pct_low, round((current - low) / current * 100, 2))
        if high > 0 and low > 0 and high > low:
            setattr(result, range_attr, round((current - low) / (high - low) * 100, 1))
    
    return result


def get_dynamic_tp_targets(coin: str, is_long: bool, entry_price: float, mids: dict = None) -> dict:
    """
    Build dynamic TP targets based on real price extremes instead of fixed R-multiples.
    
    Returns: {tp1_pct, tp2_pct, tp3_pct, tp1_price, tp2_price, tp3_price, 
              target_source: "1h_high" | "4h_high" | "atr_fallback"}
    
    Long: targets are recent highs above entry
    Short: targets are recent lows below entry
    """
    ext = get_extremes(coin, mids)
    targets = {"tp1_pct": 1.5, "tp2_pct": 2.5, "tp3_pct": 4.0, "target_source": "atr_fallback"}
    
    if not ext:
        return targets
    
    if is_long:
        # Look at distance to recent highs
        candidates = []
        if ext.pct_1h > 0.3:
            candidates.append((ext.pct_1h, "1h_high"))
        if ext.pct_4h > 0.5:
            candidates.append((ext.pct_4h, "4h_high"))
        if ext.pct_24h > 0.5:
            candidates.append((ext.pct_24h, "24h_high"))
        if ext.pct_5m > 0.15:
            candidates.append((ext.pct_5m, "5m_high"))
        
        if candidates:
            # Sort by distance (nearest first for TP1, farthest for TP3)
            candidates.sort()
            if len(candidates) >= 1:
                targets["tp1_pct"] = max(0.5, candidates[0][0] * 0.7)
                targets["target_source"] = candidates[0][1]
            if len(candidates) >= 2:
                targets["tp2_pct"] = max(targets["tp1_pct"] + 0.3, candidates[1][0] * 0.85)
            if len(candidates) >= 3:
                targets["tp3_pct"] = max(targets["tp2_pct"] + 0.3, candidates[2][0] * 0.95)
            elif len(candidates) >= 2:
                targets["tp3_pct"] = candidates[-1][0] * 0.95  # farthest
    else:
        # Short: look at distance to recent lows (negative = below current)
        # For now, invert: we need to know how far below us the recent lows are
        pass
    
    return targets


def format_extremes_for_ai(coin: str, current_price: float, mids: dict = None) -> str:
    """Compact range context for AI: highs, lows, range position per timeframe."""
    ext = get_extremes(coin, mids)
    if not ext:
        return ""
    
    parts = []
    parts.append(f"${current_price:.4f}")
    
    # 1h range: "1h: hi=$X(+2%) lo=$Y(-1%) pos=70%"
    if ext.high_1h > 0 and ext.low_1h > 0:
        parts.append(f"1h: hi={ext.pct_1h:+.1f}% lo=-{ext.pct_low_1h:.1f}% rng={ext.range_pos_1h:.0f}%")
    elif ext.high_1h > 0:
        parts.append(f"1h_hi={ext.pct_1h:+.1f}%")
    
    # 4h range
    if ext.high_4h > 0 and ext.low_4h > 0:
        parts.append(f"4h: hi={ext.pct_4h:+.1f}% lo=-{ext.pct_low_4h:.1f}% rng={ext.range_pos_4h:.0f}%")
    elif ext.high_4h > 0:
        parts.append(f"4h_hi={ext.pct_4h:+.1f}%")
    
    # 24h range (key: daily support/resistance)
    if ext.high_24h > 0 and ext.low_24h > 0:
        parts.append(f"24h: hi={ext.pct_24h:+.1f}% lo=-{ext.pct_low_24h:.1f}% rng={ext.range_pos_24h:.0f}%")
    elif ext.high_24h > 0:
        parts.append(f"24h_hi={ext.pct_24h:+.1f}%")
    
    return " | ".join(parts)
