#!/usr/bin/env python3
"""
CVD Divergence Detection — from Bookmap research.
Detects when price and volume delta diverge (a leading reversal signal).

Types:
  1. Bearish: price makes higher high, CVD makes lower high → distribution
  2. Bullish: price makes lower low, CVD makes higher low → accumulation
  3. Hidden bullish: price makes higher low, CVD makes lower low → continuation
  4. Hidden bearish: price makes lower high, CVD makes higher high → continuation

From: Bookmap CVD Trading Strategy + Reddit r/algotrading discussions
"""

import numpy as np
from typing import Optional


def compute_cvd(ohlcv: list[dict]) -> list[float]:
    """
    Compute Cumulative Volume Delta from OHLCV candles.
    Uses close-vs-open per candle as proxy for buy/sell pressure.
    
    For each candle:
      - If close > open: buyers won → +volume
      - If close < open: sellers won → -volume
      - If close == open: 0 (no delta)
    
    Returns list of cumulative delta values (same length as input).
    """
    cvd = []
    cum = 0.0
    for c in ohlcv:
        o = float(c.get("o", c.get("open", 0)))
        cl = float(c.get("c", c.get("close", 0)))
        v = float(c.get("v", c.get("volume", 0)))
        
        if cl > o:
            cum += v
        elif cl < o:
            cum -= v
        # else: doji, no delta change
        
        cvd.append(cum)
    return cvd


def find_swing_points(series: list[float], lookback: int = 5) -> list[tuple[int, float, str]]:
    """
    Find swing highs and lows in a series.
    Returns list of (index, value, 'high'|'low').
    """
    swings = []
    for i in range(lookback, len(series) - lookback):
        window = series[i - lookback:i + lookback + 1]
        val = series[i]
        if val == max(window) and val > max(series[i - lookback:i]) and val >= max(series[i + 1:i + lookback + 1]):
            swings.append((i, val, "high"))
        elif val == min(window) and val < min(series[i - lookback:i]) and val <= min(series[i + 1:i + lookback + 1]):
            swings.append((i, val, "low"))
    return swings


def detect_cvd_divergence(
    ohlcv: list[dict],
    lookback_swing: int = 5,
    recent_bars: int = 30,
) -> dict:
    """
    Detect CVD divergences in recent price action.
    
    Returns:
        dict with:
          - divergence: 'bearish' | 'bullish' | 'hidden_bullish' | 'hidden_bearish' | 'none'
          - confidence: 0-100
          - description: human-readable reason
          - price_cvd_gap: how severe the divergence is (bigger = stronger)
    """
    if len(ohlcv) < lookback_swing * 2 + 2:
        return {"divergence": "none", "confidence": 0, "description": "", "price_cvd_gap": 0}
    
    closes = [float(c.get("c", c.get("close", 0))) for c in ohlcv]
    cvd = compute_cvd(ohlcv)
    
    # Only look at recent bars
    recent_closes = closes[-recent_bars:]
    recent_cvd = cvd[-recent_bars:]
    
    price_swings = find_swing_points(recent_closes, lookback_swing)
    cvd_swings = find_swing_points(recent_cvd, lookback_swing)
    
    if len(price_swings) < 2:
        return {"divergence": "none", "confidence": 0, "description": "not_enough_swings", "price_cvd_gap": 0}
    
    # Get last two price swing points
    price_highs = [(i, v) for i, v, t in price_swings if t == "high"]
    price_lows = [(i, v) for i, v, t in price_swings if t == "low"]
    cvd_highs = [(i, v) for i, v, t in cvd_swings if t == "high"]
    cvd_lows = [(i, v) for i, v, t in cvd_swings if t == "low"]
    
    result = {"divergence": "none", "confidence": 0, "description": "", "price_cvd_gap": 0}
    
    # --- Bearish Divergence: price higher high, CVD lower high ---
    if len(price_highs) >= 2 and len(cvd_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]  # (idx, val)
        c1, c2 = cvd_highs[-2], cvd_highs[-1]
        
        if p2[1] > p1[1] * 1.002 and c2[1] < c1[1] * 0.998:
            gap = (p2[1] / p1[1] - 1) - (c2[1] / c1[1] - 1)
            conf = min(95, 50 + abs(gap) * 200)
            result = {
                "divergence": "bearish",
                "confidence": round(conf, 1),
                "description": "CVD_bearish_div:price_HH_CVD_LH",
                "price_cvd_gap": round(gap, 4),
            }
    
    # --- Bullish Divergence: price lower low, CVD higher low ---
    if len(price_lows) >= 2 and len(cvd_lows) >= 2:
        p1, p2 = price_lows[-2], price_lows[-1]
        c1, c2 = cvd_lows[-2], cvd_lows[-1]
        
        if p2[1] < p1[1] * 0.998 and c2[1] > c1[1] * 1.002:
            gap = (c2[1] / c1[1] - 1) - (p1[1] / p2[1] - 1)
            conf = min(95, 50 + abs(gap) * 200)
            result = {
                "divergence": "bullish",
                "confidence": round(conf, 1),
                "description": "CVD_bullish_div:price_LL_CVD_HL",
                "price_cvd_gap": round(gap, 4),
            }
    
    # --- Hidden Bullish: price higher low, CVD lower low (continuation) ---
    if result["divergence"] == "none" and len(price_lows) >= 2 and len(cvd_lows) >= 2:
        p1, p2 = price_lows[-2], price_lows[-1]
        c1, c2 = cvd_lows[-2], cvd_lows[-1]
        
        if p2[1] > p1[1] * 1.002 and c2[1] < c1[1] * 0.998:
            gap = (p2[1] / p1[1] - 1) - (c1[1] / c2[1] - 1)
            conf = min(85, 45 + abs(gap) * 150)
            result = {
                "divergence": "hidden_bullish",
                "confidence": round(conf, 1),
                "description": "CVD_hidden_bullish:price_HL_CVD_LL",
                "price_cvd_gap": round(gap, 4),
            }
    
    # --- Hidden Bearish: price lower high, CVD higher high (continuation) ---
    if result["divergence"] == "none" and len(price_highs) >= 2 and len(cvd_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]
        c1, c2 = cvd_highs[-2], cvd_highs[-1]
        
        if p2[1] < p1[1] * 0.998 and c2[1] > c1[1] * 1.002:
            gap = (c2[1] / c1[1] - 1) - (p1[1] / p2[1] - 1)
            conf = min(85, 45 + abs(gap) * 150)
            result = {
                "divergence": "hidden_bearish",
                "confidence": round(conf, 1),
                "description": "CVD_hidden_bearish:price_LH_CVD_HH",
                "price_cvd_gap": round(gap, 4),
            }
    
    return result


def get_cvd_context(ohlcv: list[dict]) -> str:
    """
    Get a compact CVD context string for AI/strategy consumption.
    """
    div = detect_cvd_divergence(ohlcv)
    
    if div["divergence"] == "none":
        # Just report CVD trend
        cvd_vals = compute_cvd(ohlcv)
        if len(cvd_vals) >= 10:
            recent_slope = cvd_vals[-1] - cvd_vals[-10]
            if recent_slope > 0:
                return f"CVD:rising({div['confidence']:.0f}%)"
            elif recent_slope < 0:
                return f"CVD:falling({div['confidence']:.0f}%)"
            else:
                return "CVD:flat"
        return "CVD:insufficient_data"
    
    direction = "📉BEARISH" if "bearish" in div["divergence"] else "📈BULLISH"
    return f"{direction} CVD div:{div['description']} conf={div['confidence']:.0f}% gap={div['price_cvd_gap']:.4f}"


# ── Absorption Detection (CryptoFlowEngine research) ────────────────────────

def detect_absorption(ohlcv: list[dict], lookback: int = 10) -> dict:
    """
    Detect absorption: large opposing wall getting eaten.
    
    Absorption = high volume candle that reverses after hitting a level.
    - Bullish absorption: price dips, heavy volume, closes near high (sellers absorbed)
    - Bearish absorption: price spikes, heavy volume, closes near low (buyers absorbed)
    
    Returns: {"type": "bullish"/"bearish"/"none", "confidence": 0-100, "candle_idx": int}
    """
    if len(ohlcv) < lookback + 3:
        return {"type": "none", "confidence": 0}
    
    # Get recent candles
    recent = ohlcv[-(lookback + 3):]
    
    # Average volume for context
    avg_vol = sum(float(c.get("v", c.get("volume", 0))) for c in recent[:-3]) / max(len(recent[:-3]), 1)
    
    best_absorption = {"type": "none", "confidence": 0, "candle_idx": -1}
    
    for i in range(len(recent) - 2, 1, -1):
        c = recent[i]
        prev = recent[i-1]
        nxt = recent[i+1]
        
        vol = float(c.get("v", c.get("volume", 0)))
        h = float(c.get("h", c.get("high", 0)))
        l = float(c.get("l", c.get("low", 0)))
        o = float(c.get("o", c.get("open", 0)))
        cl = float(c.get("c", c.get("close", 0)))
        prev_c = float(prev.get("c", prev.get("close", 0)))
        
        if h == l or avg_vol <= 0:
            continue
        
        vol_ratio = vol / avg_vol
        if vol_ratio < 1.5:
            continue  # Not enough volume to qualify as absorption
        
        # Position of close within the candle range (0=low, 1=high)
        close_pos = (cl - l) / (h - l)
        
        # Bullish absorption: price went down, heavy vol, closed near high
        if cl < prev_c and close_pos > 0.65 and vol_ratio > 1.8:
            conf = min(80, vol_ratio * 30)
            if conf > best_absorption["confidence"]:
                best_absorption = {"type": "bullish", "confidence": round(conf, 1),
                                   "candle_idx": i, "description": "sellers_absorbed"}
        
        # Bearish absorption: price went up, heavy vol, closed near low
        if cl > prev_c and close_pos < 0.35 and vol_ratio > 1.8:
            conf = min(80, vol_ratio * 30)
            if conf > best_absorption["confidence"]:
                best_absorption = {"type": "bearish", "confidence": round(conf, 1),
                                   "candle_idx": i, "description": "buyers_absorbed"}
    
    return best_absorption


def detect_stacked_imbalance(ohlcv: list[dict], consecutive: int = 3) -> dict:
    """
    Detect stacked imbalance: multiple consecutive candles with delta in same direction.
    
    Uses close position within candle range as delta proxy:
    - Close > 65% of range = bullish delta
    - Close < 35% of range = bearish delta
    
    Returns: {"direction": "bullish"/"bearish"/"none", "count": int, "confidence": 0-100}
    """
    if len(ohlcv) < consecutive + 1:
        return {"direction": "none", "count": 0, "confidence": 0}
    
    recent = ohlcv[-consecutive:]
    directions = []
    
    for c in recent:
        h = float(c.get("h", c.get("high", 0)))
        l = float(c.get("l", c.get("low", 0)))
        cl = float(c.get("c", c.get("close", 0)))
        
        if h == l:
            continue
        
        close_pos = (cl - l) / (h - l)
        if close_pos > 0.65:
            directions.append("bullish")
        elif close_pos < 0.35:
            directions.append("bearish")
        else:
            directions.append("neutral")
    
    if len(directions) < consecutive:
        return {"direction": "none", "count": 0, "confidence": 0}
    
    # Check for streaks
    bull_count = 0; bear_count = 0
    for d in directions:
        if d == "bullish":
            bull_count += 1
            bear_count = 0
        elif d == "bearish":
            bear_count += 1
            bull_count = 0
        else:
            bull_count = 0
            bear_count = 0
    
    if bull_count >= consecutive:
        return {"direction": "bullish", "count": bull_count,
                "confidence": min(85, bull_count * 25)}
    elif bear_count >= consecutive:
        return {"direction": "bearish", "count": bear_count,
                "confidence": min(85, bear_count * 25)}
    
    return {"direction": "none", "count": 0, "confidence": 0}


def get_absorption_stacked_context(ohlcv: list[dict]) -> str:
    """Combined absorption + stacked imbalance context for AI."""
    parts = []
    
    abs_result = detect_absorption(ohlcv)
    if abs_result["type"] != "none" and abs_result["confidence"] > 30:
        emoji = "🟢" if abs_result["type"] == "bullish" else "🔴"
        parts.append(f"{emoji}Absorption:{abs_result['type']}({abs_result['confidence']:.0f}%)")
    
    stacked = detect_stacked_imbalance(ohlcv)
    if stacked["direction"] != "none" and stacked["confidence"] > 40:
        emoji = "📈" if stacked["direction"] == "bullish" else "📉"
        parts.append(f"{emoji}Stacked:{stacked['direction']}x{stacked['count']}({stacked['confidence']:.0f}%)")
    
    return " | ".join(parts) if parts else ""
