#!/usr/bin/env python3
"""
VWAP Mean Reversion Signal — from StrategyBender research.
Daily VWAP acts as an intraday "fair value" anchor.
Price reverts to VWAP in ranging markets; trends away in momentum markets.

Strategy:
  1. Compute session VWAP (volume-weighted average price)
  2. Calculate VWAP bands (1σ, 2σ using standard deviation or ATR)
  3. Fade extremes in SIDEWAYS regime (mean reversion)
  4. Follow trend in TRENDING regimes (trend-day override)
  
From: StrategyBender "VWAP Bands Mean-Reversion" + VWAP strategy research
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class VWAPSignal(str, Enum):
    MEAN_REVERT_BUY = "vwap_mean_revert_buy"    # Price below lower band → buy
    MEAN_REVERT_SELL = "vwap_mean_revert_sell"   # Price above upper band → sell
    TREND_BUY = "vwap_trend_buy"                 # Price above VWAP trending → buy pullback
    TREND_SELL = "vwap_trend_sell"               # Price below VWAP trending → sell pullback
    NEUTRAL = "vwap_neutral"


@dataclass
class VWAPResult:
    vwap: float
    upper_band_1: float  # +1σ
    lower_band_1: float  # -1σ
    upper_band_2: float  # +2σ
    lower_band_2: float  # -2σ
    current_price: float
    deviation_pct: float  # How far from VWAP in %
    deviation_sigma: float  # How many sigmas away
    signal: VWAPSignal
    confidence: float  # 0-100
    description: str


def compute_vwap(candles: list[dict]) -> float:
    """
    Compute VWAP from OHLCV candles.
    VWAP = Σ(typical_price * volume) / Σ(volume)
    where typical_price = (high + low + close) / 3
    """
    if not candles:
        return 0.0
    
    total_pv = 0.0
    total_vol = 0.0
    
    for c in candles:
        h = float(c.get("h", c.get("high", 0)))
        l = float(c.get("l", c.get("low", 0)))
        cl = float(c.get("c", c.get("close", 0)))
        v = float(c.get("v", c.get("volume", 0)))
        
        typical = (h + l + cl) / 3
        total_pv += typical * v
        total_vol += v
    
    return total_pv / total_vol if total_vol > 0 else 0.0


def compute_vwap_bands(
    candles: list[dict],
    num_bars: int = 96,  # ~24h of 15m bars
) -> VWAPResult:
    """
    Compute VWAP with standard deviation bands for mean reversion.
    
    Uses 1σ and 2σ bands. Standard deviation of price from VWAP is 
    volume-weighted for accuracy.
    """
    if len(candles) < 10:
        return VWAPResult(
            vwap=0, upper_band_1=0, lower_band_1=0,
            upper_band_2=0, lower_band_2=0,
            current_price=0, deviation_pct=0, deviation_sigma=0,
            signal=VWAPSignal.NEUTRAL, confidence=0,
            description="insufficient_data",
        )
    
    recent = candles[-min(num_bars, len(candles)):]
    vwap = compute_vwap(recent)
    
    if vwap <= 0:
        return VWAPResult(
            vwap=0, upper_band_1=0, lower_band_1=0,
            upper_band_2=0, lower_band_2=0,
            current_price=0, deviation_pct=0, deviation_sigma=0,
            signal=VWAPSignal.NEUTRAL, confidence=0,
            description="vwap_zero",
        )
    
    # Compute volume-weighted standard deviation
    prices = []
    weights = []
    for c in recent:
        h = float(c.get("h", c.get("high", 0)))
        l = float(c.get("l", c.get("low", 0)))
        cl = float(c.get("c", c.get("close", 0)))
        v = float(c.get("v", c.get("volume", 0)))
        typical = (h + l + cl) / 3
        prices.append(typical)
        weights.append(v)
    
    prices = np.array(prices)
    weights = np.array(weights)
    
    if weights.sum() <= 0:
        return VWAPResult(
            vwap=vwap, upper_band_1=vwap, lower_band_1=vwap,
            upper_band_2=vwap, lower_band_2=vwap,
            current_price=prices[-1], deviation_pct=0, deviation_sigma=0,
            signal=VWAPSignal.NEUTRAL, confidence=0,
            description="zero_weights",
        )
    
    # Volume-weighted std
    avg = np.average(prices, weights=weights)
    variance = np.average((prices - avg) ** 2, weights=weights)
    std = np.sqrt(variance)
    
    current_price = prices[-1]
    deviation_pct = (current_price - vwap) / vwap * 100
    deviation_sigma = (current_price - vwap) / std if std > 0 else 0
    
    upper_1 = vwap + std
    lower_1 = vwap - std
    upper_2 = vwap + 2 * std
    lower_2 = vwap - 2 * std
    
    # Determine signal
    signal = VWAPSignal.NEUTRAL
    confidence = 0.0
    desc = ""
    
    # Check trend state: is price consistently above/below VWAP?
    recent_closes = [float(c.get("c", c.get("close", 0))) for c in recent[-12:]]
    above_count = sum(1 for p in recent_closes if p > vwap)
    is_trending_above = above_count >= 9  # 75% above VWAP
    is_trending_below = above_count <= 3   # 75% below VWAP
    
    if abs(deviation_sigma) >= 2.0:
        # Extreme deviation — strong mean reversion signal
        if deviation_sigma <= -2.0 and not is_trending_below:
            signal = VWAPSignal.MEAN_REVERT_BUY
            confidence = min(85, 50 + abs(deviation_sigma) * 12)
            desc = f"VWAP_extreme_low:{deviation_sigma:.1f}σ_below"
        elif deviation_sigma >= 2.0 and not is_trending_above:
            signal = VWAPSignal.MEAN_REVERT_SELL
            confidence = min(85, 50 + abs(deviation_sigma) * 12)
            desc = f"VWAP_extreme_high:{deviation_sigma:.1f}σ_above"
    elif abs(deviation_sigma) >= 1.0:
        # Moderate deviation
        if deviation_sigma <= -1.0:
            if is_trending_below:
                # Trending below VWAP — don't fade, trade with trend
                signal = VWAPSignal.TREND_SELL
                confidence = 45
                desc = f"VWAP_trend_below:{deviation_sigma:.1f}σ"
            else:
                signal = VWAPSignal.MEAN_REVERT_BUY
                confidence = min(65, 35 + abs(deviation_sigma) * 15)
                desc = f"VWAP_revert_buy:{deviation_sigma:.1f}σ"
        else:
            if is_trending_above:
                signal = VWAPSignal.TREND_BUY
                confidence = 45
                desc = f"VWAP_trend_above:{deviation_sigma:.1f}σ"
            else:
                signal = VWAPSignal.MEAN_REVERT_SELL
                confidence = min(65, 35 + abs(deviation_sigma) * 15)
                desc = f"VWAP_revert_sell:{deviation_sigma:.1f}σ"
    else:
        # Near VWAP — neutral
        signal = VWAPSignal.NEUTRAL
        confidence = 20
        desc = f"VWAP_near:{deviation_sigma:.1f}σ"
    
    return VWAPResult(
        vwap=round(vwap, 6),
        upper_band_1=round(upper_1, 6),
        lower_band_1=round(lower_1, 6),
        upper_band_2=round(upper_2, 6),
        lower_band_2=round(lower_2, 6),
        current_price=round(current_price, 6),
        deviation_pct=round(deviation_pct, 4),
        deviation_sigma=round(deviation_sigma, 2),
        signal=signal,
        confidence=round(confidence, 1),
        description=desc,
    )


def get_vwap_context(candles: list[dict], symbol: str = "") -> str:
    """
    Get compact VWAP context for AI/strategy.
    """
    result = compute_vwap_bands(candles)
    
    if result.vwap <= 0:
        return "VWAP:n/a"
    
    signal_map = {
        VWAPSignal.MEAN_REVERT_BUY: "🟢 MR_BUY",
        VWAPSignal.MEAN_REVERT_SELL: "🔴 MR_SELL",
        VWAPSignal.TREND_BUY: "📈 TREND_BUY",
        VWAPSignal.TREND_SELL: "📉 TREND_SELL",
        VWAPSignal.NEUTRAL: "⚪ NEUTRAL",
    }
    
    return (
        f"VWAP:{signal_map.get(result.signal, '?')} "
        f"@{result.vwap:.4f} "
        f"dev={result.deviation_pct:+.2f}% "
        f"({result.deviation_sigma:+.1f}σ) "
        f"conf={result.confidence:.0f}% "
        f"bands=[{result.lower_band_2:.4f}..{result.upper_band_2:.4f}]"
    )
