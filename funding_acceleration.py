#!/usr/bin/env python3
"""
Funding Rate Acceleration Signal

Detects whether funding rate is accelerating (building FOMO/distress)
or decelerating (crowd unwinding). 

Key insight: The DIRECTION and SPEED of funding rate change is more 
predictive than the absolute level. Accelerating positive funding = 
FOMO building = potential top. Decelerating extreme funding = crowd 
unwinding = reversal imminent.

Used by: continuous_predictor funding_oi layer, perfect_predictor
"""
from typing import Optional
from collections import defaultdict
import time

# In-memory history for funding rate tracking
_funding_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
# {coin: [(timestamp, funding_rate), ...]}
MAX_HISTORY = 20  # Keep last 20 funding snapshots per coin


def record_funding(coin: str, rate: float, timestamp: float = None):
    """Record a funding rate data point. Call every cycle."""
    ts = timestamp or time.time()
    _funding_history[coin].append((ts, rate))
    if len(_funding_history[coin]) > MAX_HISTORY:
        _funding_history[coin] = _funding_history[coin][-MAX_HISTORY:]


def get_funding_acceleration(coin: str) -> dict:
    """
    Compute funding rate slope (acceleration/deceleration).
    
    Returns: {
        "slope": float (rate change per hour, annualized),
        "direction": "accelerating_long" | "accelerating_short" | "decelerating" | "flat",
        "confidence": 0-100,
        "current_rate": float,
        "hours_of_data": float,
    }
    """
    history = _funding_history.get(coin, [])
    if len(history) < 3:
        return {"slope": 0, "direction": "flat", "confidence": 0, 
                "current_rate": history[-1][1] if history else 0, "hours_of_data": 0}
    
    # Extract rates and timestamps
    times = [h[0] for h in history]
    rates = [h[1] for h in history]
    current_rate = rates[-1]
    
    # Time span in hours
    time_span_hours = (times[-1] - times[0]) / 3600
    if time_span_hours < 0.1:
        return {"slope": 0, "direction": "flat", "confidence": 10,
                "current_rate": current_rate, "hours_of_data": time_span_hours}
    
    # Linear regression: rate = slope * time + intercept
    n = len(times)
    mean_t = sum(times) / n
    mean_r = sum(rates) / n
    
    num = sum((times[i] - mean_t) * (rates[i] - mean_r) for i in range(n))
    den = sum((times[i] - mean_t) ** 2 for i in range(n))
    
    if abs(den) < 1e-10:
        return {"slope": 0, "direction": "flat", "confidence": 0,
                "current_rate": current_rate, "hours_of_data": time_span_hours}
    
    slope_per_second = num / den
    slope_per_hour = slope_per_second * 3600  # Rate change per hour
    
    # Direction classification
    abs_slope = abs(slope_per_hour)
    
    if abs_slope < 0.00001:  # < 0.001%/hr change — essentially flat
        direction = "flat"
        confidence = 10
    elif slope_per_hour > 0.00003:  # Accelerating positive (>0.003%/hr/hr)
        if current_rate > 0:
            direction = "accelerating_long"  # FOMO building
            confidence = min(85, abs_slope * 500000)
        else:
            direction = "accelerating_short"  # Short squeeze building
            confidence = min(85, abs_slope * 500000)
    elif slope_per_hour < -0.00003:  # Decelerating
        direction = "decelerating"
        confidence = min(75, abs_slope * 500000)
    else:
        direction = "flat"
        confidence = 20
    
    return {
        "slope": round(slope_per_hour, 8),
        "direction": direction,
        "confidence": round(confidence, 1),
        "current_rate": current_rate,
        "hours_of_data": round(time_span_hours, 2),
    }


def funding_accel_signal(coin: str, current_rate: float) -> dict:
    """
    Convert funding acceleration into a directional signal for the continuous predictor.
    
    Returns {"up": 0-100, "down": 0-100, "flat": 0-100, "confidence": 0-100}
    """
    # Record this data point
    record_funding(coin, current_rate)
    
    accel = get_funding_acceleration(coin)
    direction = accel["direction"]
    confidence = accel["confidence"]
    
    if direction == "accelerating_long":
        # FOMO building → contrarian: expect reversal DOWN
        return {"up": 30, "down": 50, "flat": 20, "confidence": confidence}
    elif direction == "accelerating_short":
        # Panic building → contrarian: expect reversal UP
        return {"up": 50, "down": 30, "flat": 20, "confidence": confidence}
    elif direction == "decelerating":
        # Funding unwinding → whichever side was paying is losing conviction
        # Look at current rate sign
        if current_rate > 0.00005:
            # Longs were paying, now decelerating → bullish (shorts capitulating)
            return {"up": 45, "down": 30, "flat": 25, "confidence": confidence * 0.7}
        elif current_rate < -0.00003:
            # Shorts were paying, now decelerating → bearish (longs capitulating)
            return {"up": 30, "down": 45, "flat": 25, "confidence": confidence * 0.7}
        else:
            return {"up": 33, "down": 33, "flat": 34, "confidence": 10}
    else:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)
    
    # Simulate accelerating positive funding (FOMO building)
    base_time = time.time() - 3600
    for i in range(15):
        t = base_time + i * 240  # Every 4 minutes
        rate = 0.0001 + i * 0.00005  # Rising
        record_funding("TEST", rate, t)
    
    result = get_funding_acceleration("TEST")
    print(f"Accelerating positive: {result['direction']} slope={result['slope']:.8f}/hr conf={result['confidence']:.0f}%")
    
    signal = funding_accel_signal("TEST", result['current_rate'])
    print(f"Signal: ↑{signal['up']:.0f} ↓{signal['down']:.0f} →{signal['flat']:.0f} conf={signal['confidence']:.0f}%")
    
    print("\n✓ Funding acceleration detector ready")
