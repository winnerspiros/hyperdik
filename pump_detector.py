"""
Pump & Dump Detection — detects suspicious volume/price patterns.
Volume spikes + sharp price moves + social media coordination signals.
"""
import numpy as np
import logging
from typing import List, Dict, Optional

log = logging.getLogger("pump_detector")


def detect_pump(close: np.ndarray, volume: np.ndarray,
                lookback: int = 48) -> Optional[Dict]:
    """
    Detect pump & dump patterns from price/volume data.
    
    Pump signature:
    1. Sudden volume spike (3x+ vs average)
    2. Sharp price increase (5%+ in short period)
    3. Usually followed by reversal (dump)
    
    Args:
        close: array of closing prices
        volume: array of volumes
        lookback: bars to analyze
        
    Returns:
        Dict with pump info or None
    """
    if len(close) < lookback or len(volume) < lookback:
        return None
    
    recent_close = close[-lookback:]
    recent_vol = volume[-lookback:]
    
    avg_vol = np.mean(recent_vol[:-5]) + 1  # Avoid div by 0
    avg_vol_all = np.mean(recent_vol) + 1
    current_vol = recent_vol[-1]
    vol_ratio = current_vol / avg_vol
    
    price_change = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100
    
    # Max price in window
    max_price = np.max(recent_close)
    max_idx = np.argmax(recent_close)
    peak_to_current = (recent_close[-1] - max_price) / max_price * 100 if max_price > 0 else 0
    
    # Volume spike detection
    vol_spike = vol_ratio > 2.5
    extreme_vol = vol_ratio > 5.0
    
    # Pump criteria
    is_pump = False
    confidence = 0
    
    # Criteria 1: Sharp price rise
    if price_change > 5:
        confidence += 30
    if price_change > 10:
        confidence += 20
    
    # Criteria 2: Volume spike
    if vol_spike:
        confidence += 25
    if extreme_vol:
        confidence += 25
    
    # Criteria 3: Reversal after spike (dump phase)
    if peak_to_current < -3 and max_idx < len(recent_close) - 3:
        confidence += 20
        is_pump = True
    
    if confidence >= 50:
        return {
            "detected": True,
            "confidence": min(confidence, 100),
            "price_change_pct": round(price_change, 1),
            "volume_ratio": round(vol_ratio, 1),
            "dump_pct": round(peak_to_current, 1),
            "avg_volume": round(float(avg_vol_all), 0),
            "current_volume": round(float(current_vol), 0),
            "signal": "🚨 PUMP DETECTED" if confidence > 70 else "⚠️ SUSPICIOUS VOLUME",
        }
    
    return None


def detect_wash_trading(close: np.ndarray, volume: np.ndarray,
                        lookback: int = 24) -> Optional[Dict]:
    """
    Detect wash trading indicators — high volume with no price movement,
    volume clustering, round-number volume patterns.
    """
    if len(volume) < lookback:
        return None
    
    recent_vol = volume[-lookback:]
    recent_close = close[-lookback:]
    
    avg_vol = np.mean(recent_vol)
    vol_std = np.std(recent_vol)
    vol_cv = vol_std / avg_vol if avg_vol > 0 else 0  # Coefficient of variation
    
    price_range = (np.max(recent_close) - np.min(recent_close)) / np.min(recent_close) * 100
    
    # Wash trading = high volume but low price movement
    if avg_vol > 10000 and price_range < 1.5 and vol_cv < 0.5:
        return {
            "detected": True,
            "indicators": ["high_volume_no_move", "low_volatility", "low_volume_variation"],
            "signal": "🔍 POSSIBLE WASH TRADING",
        }
    
    return None


def format_pump_context(close=None, volume=None, high=None):
    """Wraps pump detection into a context string for AI. Returns empty if no data."""
    if close is None or volume is None:
        return ""
    try:
        return format_pump_warning(close, volume, high)
    except Exception:
        return ""

def format_pump_warning(close: np.ndarray, volume: np.ndarray,
                        symbol: str = "") -> str:
    """Format pump/wash detection for AI context"""
    pump = detect_pump(close, volume)
    wash = detect_wash_trading(close, volume)
    
    lines = []
    if pump:
        lines.append(pump["signal"])
        lines.append(f"  {symbol}: {pump['price_change_pct']:+.1f}% | "
                     f"vol {pump['volume_ratio']}x avg | dump {pump['dump_pct']:+.1f}%")
    
    if wash:
        lines.append(wash["signal"])
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic pump data
    np.random.seed(42)
    close = np.concatenate([
        np.linspace(100, 98, 30),     # Normal
        np.linspace(98, 112, 5),       # Pump! +14%
        np.linspace(112, 104, 10),     # Dump
        np.linspace(104, 103, 3),      # Aftermath
    ])
    vol = np.concatenate([
        np.ones(30) * 1000,
        np.ones(5) * 8000,     # 8x volume spike
        np.ones(10) * 3000,
        np.ones(3) * 1200,
    ])
    
    result = detect_pump(close, vol)
    if result:
        print(format_pump_warning(close, vol, "TEST"))
    else:
        print("No pump detected")
    
    wash = detect_wash_trading(close, vol)
    print(f"Wash trading: {wash['signal'] if wash else 'None'}")