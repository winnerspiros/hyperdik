"""
Market Structure Detection — identifies swing highs/lows, trend direction,
breaks of structure (BOS), and change of character (CHOCH) from OHLCV data.
Based on ICT/SMC / Smart Money Concepts methodology.
"""
import numpy as np
from typing import Optional, List, Tuple


def find_swing_points(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                      left_bars: int = 3, right_bars: int = 3):
    """
    Find swing highs and swing lows in price data.
    
    A swing high = a high that is higher than `left_bars` candles before AND after.
    A swing low = a low that is lower than `left_bars` candles before AND after.
    
    Args:
        closes: array of closing prices
        highs: array of high prices
        lows: array of low prices
        left_bars: bars to look left of the pivot
        right_bars: bars to look right of the pivot
        
    Returns:
        (swing_highs, swing_lows) — each is list of (index, price)
    """
    length = len(closes)
    swing_highs = []
    swing_lows = []
    
    for i in range(left_bars, length - right_bars):
        # Swing High
        is_high = True
        for j in range(1, left_bars + 1):
            if highs[i] <= highs[i - j]:
                is_high = False
                break
        if is_high:
            for j in range(1, right_bars + 1):
                if highs[i] <= highs[i + j]:
                    is_high = False
                    break
        if is_high:
            swing_highs.append((i, highs[i]))
        
        # Swing Low
        is_low = True
        for j in range(1, left_bars + 1):
            if lows[i] >= lows[i - j]:
                is_low = False
                break
        if is_low:
            for j in range(1, right_bars + 1):
                if lows[i] >= lows[i + j]:
                    is_low = False
                    break
        if is_low:
            swing_lows.append((i, lows[i]))
    
    return swing_highs, swing_lows


def classify_market_structure(closes, highs, lows, left_bars=3, right_bars=3,
                               lookback: int = 30) -> dict:
    """
    Classify market structure based on recent swing points.
    
    Returns:
        dict with:
          - trend: "bullish" | "bearish" | "ranging" | "unknown"
          - structure: "higher_highs_higher_lows" | "lower_highs_lower_lows" | ... 
          - bos: True if recent break of structure
          - choch: True if recent change of character
          - description: human-readable string
    """
    if closes is None or len(closes) < left_bars + right_bars + 5:
        return {"trend": "unknown", "structure": "unknown", "bos": False, 
                "choch": False, "description": "Insufficient data"}
    
    # Use recent data
    recent_close = closes[-lookback:] if len(closes) > lookback else closes
    offset = len(closes) - len(recent_close)
    
    # Adjust for the slice
    h = highs[-lookback:] if len(highs) > lookback else highs
    l = lows[-lookback:] if len(lows) > lookback else lows
    
    swing_highs, swing_lows = find_swing_points(recent_close, h, l, left_bars, right_bars)
    
    if len(swing_highs) < 2 and len(swing_lows) < 2:
        return {"trend": "ranging", "structure": "no_clear_points",
                "bos": False, "choch": False,
                "description": "No clear swing points in recent data"}
    
    # Check HH/HL pattern (bullish)
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_h = swing_highs[-1][1]
        prev_h = swing_highs[-2][1]
        last_l = swing_lows[-1][1]
        prev_l = swing_lows[-2][1]
        
        is_hh = last_h > prev_h
        is_hl = last_l > prev_l
        is_lh = last_h < prev_h
        is_ll = last_l < prev_l
        
        # Break of structure: price broke above previous high (bullish BOS) 
        # or below previous low (bearish BOS)
        bullish_bos = swing_highs[-1][0] > swing_highs[-2][0] and is_hh
        bearish_bos = swing_lows[-1][0] > swing_lows[-2][0] and is_ll
        
        # Change of character: trend reversal
        if len(swing_highs) >= 4 and len(swing_lows) >= 3:
            # Check for HH/HH/HH then LH = bearish CHOCH
            all_hh = all(swing_highs[i][1] > swing_highs[i-1][1] for i in range(-3, 0))
            last_lh = swing_highs[-1][1] < swing_highs[-2][1]
            choch_bearish = all_hh and last_lh and bearish_bos
            
            # Check for LL/LL/LL then HL = bullish CHOCH
            all_ll = all(swing_lows[i][1] < swing_lows[i-1][1] for i in range(-3, 0))
            last_hl = swing_lows[-1][1] > swing_lows[-2][1]
            choch_bullish = all_ll and last_hl and bullish_bos
        else:
            choch_bullish = False
            choch_bearish = False
        
        if is_hh and is_hl:
            return {"trend": "bullish", "structure": "higher_highs_higher_lows",
                    "bos": bullish_bos, "choch": choch_bullish if choch_bullish else choch_bearish,
                    "description": f"Bullish trend — HH/HL. {'BOS!' if bullish_bos else ''} {'CHOCH!' if choch_bullish else ''}"}
        elif is_lh and is_ll:
            return {"trend": "bearish", "structure": "lower_highs_lower_lows",
                    "bos": bearish_bos, "choch": choch_bearish if choch_bearish else choch_bullish,
                    "description": f"Bearish trend — LH/LL. {'BOS!' if bearish_bos else ''} {'CHOCH!' if choch_bearish else ''}"}
        elif is_hh and not is_hl:
            return {"trend": "bearish_divergence", "structure": "higher_highs_lower_lows",
                    "bos": False, "choch": choch_bearish,
                    "description": "Bearish divergence — price making HH but LL (weakening)"}
        elif not is_hh and is_hl:
            return {"trend": "bullish_divergence", "structure": "lower_highs_higher_lows",
                    "bos": False, "choch": choch_bullish,
                    "description": "Bullish divergence — price making LH but HL (basing)"}
    
    return {"trend": "ranging", "structure": "mixed", "bos": False, "choch": False,
            "description": "Mixed signals — ranging market"}


def get_market_structure_summary(symbol: str, analyzer) -> str:
    """
    Get a formatted market structure summary for the AI.
    
    Args:
        symbol: coin symbol e.g. "XRP-EUR"
        analyzer: MarketAnalyzer instance
        
    Returns:
        Formatted string like:
        🏗️ STRUCTURE (XRP): Bullish HH/HL
           Swing highs: €0.96 → €0.98 → €1.01 (ascending)
           Swing lows:  €0.91 → €0.93 → €0.95 (ascending)
           BOS ✅ (broke above €0.98 resistance)
    """
    try:
        df_1h = analyzer.get_candles_df(symbol, "1h", hours_back=120)
        if df_1h is None or len(df_1h) < 20:
            return ""
        
        closes = df_1h["close"].values.astype(float)
        highs = df_1h["high"].values.astype(float)
        lows = df_1h["low"].values.astype(float)
        
        mkt = classify_market_structure(closes, highs, lows, left_bars=3, right_bars=3)
        if mkt["trend"] == "unknown":
            return ""
        
        base = symbol.split("-")[0]
        emoji = {"bullish": "🟢", "bearish": "🔴", "ranging": "🟡"}.get(mkt["trend"], "⚪")
        
        lines = [f"{emoji} STRUCTURE ({base}): {mkt['description'][:60]}"]
        
        # Add swing points
        swing_highs, swing_lows = find_swing_points(closes, highs, lows, 3, 3)
        if swing_highs:
            recent_h = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            prices_h = [f"€{p[1]:.4f}" for p in recent_h]
            lines.append(f"   Highs: {' → '.join(prices_h)}")
        if swing_lows:
            recent_l = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            prices_l = [f"€{p[1]:.4f}" for p in recent_l]
            lines.append(f"   Lows:  {' → '.join(prices_l)}")
        
        if mkt["bos"]:
            lines.append(f"   💥 Break of Structure ✅")
        if mkt["choch"]:
            lines.append(f"   🔄 Change of Character — possible trend reversal")
        
        return "\n".join(lines)
    except Exception:
        return ""


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from market_analyzer import MarketAnalyzer
    from revolut_client import RevolutXClient
    client = RevolutXClient()
    analyzer = MarketAnalyzer(client)
    for sym in ["XRP-EUR", "BTC-EUR", "AVAX-EUR"]:
        print(get_market_structure_summary(sym, analyzer))
        print()