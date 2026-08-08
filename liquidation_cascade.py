#!/usr/bin/env python3
"""
Liquidation Cascade Detector — from MethodAlgo research paper.
Detects mass liquidation events and signals reversal entry points.

Key findings from the paper:
  1. Markets bounce after ONE-SIDED mass liquidations (Hawkes self-exciting process)
  2. Gap Reversal: enter when price exhausts a liquidation cascade
  3. Liquidation Asymmetry Index (LAI): when balanced, stay out
  4. DeFi liquidations cause bigger overshoot than CeFi

Implementation: Uses Hyperliquid API to fetch recent liquidation data
and computes cascade probability + reversal signals.
"""

import math
import time
import logging
from typing import Optional
from dataclasses import dataclass, field

log = logging.getLogger("liqcascade")

# ── Cache ──
_last_liq_fetch = 0.0
_liq_cache_ttl = 30  # seconds
_liq_cache: dict = {}


@dataclass
class LiquidationEvent:
    symbol: str
    timestamp: int  # ms
    side: str  # 'long' or 'short'
    size_usd: float
    price: float
    is_whale: bool  # >$50K


@dataclass 
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


def fetch_hyperliquid_liquidations(hl_client, symbol: str = "", limit: int = 100) -> list[dict]:
    """
    Fetch recent liquidations from Hyperliquid API.
    Uses the userFills or liquidation feed endpoints.
    """
    global _last_liq_fetch, _liq_cache
    
    cache_key = f"liq_{symbol}_{limit}"
    now = time.time()
    if cache_key in _liq_cache and now - _last_liq_fetch < _liq_cache_ttl:
        return _liq_cache[cache_key]
    
    try:
        # Try userFills with liquidation filter
        # Hyperliquid doesn't have a direct "all liquidations" endpoint,
        # but we can use the info endpoint for recent liquidation data
        info = hl_client.info()
        
        # Get recent liquidations from the meta or use alternative approach
        # Actually, Hyperliquid's WebSocket streams liquidation events
        # For now, use the /info endpoint's liquidation-related fields
        
        liquidations = []
        
        # Try to get liquidation data from userFills (our own fills)
        try:
            user = hl_client.user_state()
            # Check if there's a fills endpoint
            fills = hl_client.fills() if hasattr(hl_client, 'fills') else []
            for f in fills:
                if f.get('liquidationMarkPrice') or 'liquidation' in str(f.get('type', '')).lower():
                    liquidations.append({
                        'symbol': f.get('coin', ''),
                        'timestamp': f.get('time', 0),
                        'side': 'long' if f.get('side', '') == 'B' else 'short',
                        'size_usd': float(f.get('px', 0)) * float(f.get('sz', 0)),
                        'price': float(f.get('px', 0)),
                    })
        except Exception:
            pass
        
        _liq_cache[cache_key] = liquidations
        _last_liq_fetch = now
        return liquidations
        
    except Exception as e:
        log.warning(f"Liquidation fetch error: {e}")
        return []


def detect_cascade_from_oi_delta(
    oi_current: float,
    oi_previous: float,
    price_change_pct: float,
    symbol: str = "",
) -> CascadeSignal:
    """
    Detect liquidation cascades using Open Interest change + price action.
    
    This is a proxy method when direct liquidation data isn't available.
    The logic (from MethodAlgo paper):
    
    - Large OI drop + large price drop = LONG liquidations (cascade down)
      → High probability of bounce up
    - Large OI drop + large price spike = SHORT liquidations (cascade up)
      → High probability of reversal down
    - Small OI change = no cascade
    """
    if oi_previous <= 0:
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=0, bounce_probability=0,
            lai=0, suggested_action="HOLD", confidence=0,
            reason="no_oi_data",
        )
    
    oi_delta_pct = (oi_current - oi_previous) / oi_previous * 100
    
    # Cascade thresholds
    # OI drop > 2% in a short period = potential liquidation cascade
    if abs(oi_delta_pct) < 1.0:
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=0, bounce_probability=0,
            lai=0, suggested_action="HOLD", confidence=0,
            reason="oi_stable",
        )
    
    cascade_side = ""
    bounce_prob = 0.0
    severity = min(1.0, abs(oi_delta_pct) / 10.0)
    
    # OI dropped significantly
    if oi_delta_pct < -2.0:
        if price_change_pct < -1.0:
            # Longs being liquidated = cascade down → likely bounce up
            cascade_side = "longs_liquidated"
            # Higher OI drop + bigger price drop = more forced selling = stronger bounce
            bounce_prob = min(0.85, 0.3 + severity * 0.5 + abs(price_change_pct) * 0.05)
            action = "BUY"
            conf = min(85, 40 + severity * 40)
            reason = f"long_liq_cascade:OI_{oi_delta_pct:.1f}%_price_{price_change_pct:.1f}%"
        elif price_change_pct > 1.0:
            # Shorts being liquidated = cascade up → likely reversal down
            cascade_side = "shorts_liquidated"
            bounce_prob = min(0.85, 0.3 + severity * 0.5 + price_change_pct * 0.05)
            action = "SELL"
            conf = min(85, 40 + severity * 40)
            reason = f"short_liq_cascade:OI_{oi_delta_pct:.1f}%_price_{price_change_pct:.1f}%"
        else:
            # OI dropping without price move = position closing, not cascade
            return CascadeSignal(
                symbol=symbol, cascade_detected=False,
                cascade_side="", severity=severity, bounce_prob=0,
                lai=0, suggested_action="HOLD", confidence=0,
                reason=f"oi_drop_no_cascade:{oi_delta_pct:.1f}%",
            )
    elif oi_delta_pct > 2.0:
        # OI rising = new positions, not a cascade
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=0, bounce_prob=0,
            lai=0, suggested_action="HOLD", confidence=0,
            reason=f"oi_building:{oi_delta_pct:.1f}%",
        )
    else:
        return CascadeSignal(
            symbol=symbol, cascade_detected=False,
            cascade_side="", severity=severity, bounce_prob=bounce_prob,
            lai=0, suggested_action="HOLD", confidence=0,
            reason=f"oi_mild_change:{oi_delta_pct:.1f}%",
        )
    
    # Compute LAI (Liquidation Asymmetry Index)
    # Positive LAI → more longs liquidated → bullish reversal signal
    # Negative LAI → more shorts liquidated → bearish reversal signal
    lai = -oi_delta_pct / 10  # negative OI delta (drops) = positive LAI
    lai = max(-1.0, min(1.0, lai))
    
    return CascadeSignal(
        symbol=symbol,
        cascade_detected=True,
        cascade_side=cascade_side,
        severity=round(severity, 3),
        bounce_probability=round(bounce_prob, 3),
        lai=round(lai, 3),
        suggested_action=action,
        confidence=round(conf, 1),
        reason=reason,
    )


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


def get_liquidation_context(candles: list[dict], symbol: str = "") -> str:
    """
    Get a compact liquidation cascade context string for AI/strategy.
    """
    sig = detect_cascade_from_candles(candles, symbol)
    
    if not sig.cascade_detected:
        return f"LiqCascade:none"
    
    emoji = "🩸" if sig.severity > 0.7 else "⚠️"
    return (
        f"{emoji} LiqCascade:{sig.cascade_side} "
        f"severity={sig.severity:.2f} "
        f"bounce={sig.bounce_probability:.0%} "
        f"action={sig.suggested_action} "
        f"conf={sig.confidence:.0f}% "
        f"({sig.reason})"
    )
