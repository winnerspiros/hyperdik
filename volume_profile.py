"""
Volume Profile — identifies high-volume nodes (HVNs), low-volume nodes (LVNs),
Point of Control (POC), and Value Area High/Low from OHLCV data.
These act as key support/resistance levels based on real trading activity.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24) -> dict:
    """
    Calculate Volume Profile from OHLCV data.
    
    Args:
        df: DataFrame with 'high', 'low', 'volume' columns
        num_bins: number of price bins to divide the range into
        
    Returns:
        dict with:
          - poc: Point of Control (price level with highest volume)
          - vah: Value Area High (70% of volume)
          - val: Value Area Low
          - hvns: list of high-volume node prices
          - lvns: list of low-volume node prices
          - bins: price -> volume dict
    """
    if df is None or len(df) < 10:
        return {"poc": 0, "vah": 0, "val": 0, "hvns": [], "lvns": [], "bins": {}}
    
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    
    price_min = np.min(lows)
    price_max = np.max(highs)
    price_range = price_max - price_min
    
    if price_range == 0:
        return {"poc": price_min, "vah": price_min, "val": price_min, 
                "hvns": [price_min], "lvns": [], "bins": {price_min: np.sum(volumes)}}
    
    bin_size = price_range / num_bins
    bins = {i: 0.0 for i in range(num_bins)}
    
    # Distribute each candle's volume across price bins it touched
    for i in range(len(highs)):
        if highs[i] == lows[i]:
            bin_idx = min(int((highs[i] - price_min) / bin_size), num_bins - 1)
            bins[bin_idx] += volumes[i]
        else:
            candle_range = highs[i] - lows[i]
            # Volume-weighted distribution across touched bins
            for b in range(num_bins):
                bin_low = price_min + b * bin_size
                bin_high = bin_low + bin_size
                overlap = min(highs[i], bin_high) - max(lows[i], bin_low)
                if overlap > 0:
                    bins[b] += volumes[i] * (overlap / candle_range)
    
    if not bins or max(bins.values()) == 0:
        return {"poc": 0, "vah": 0, "val": 0, "hvns": [], "lvns": [], "bins": {}}
    
    # Point of Control = bin with highest volume
    poc_bin = max(bins, key=bins.get)
    poc_price = price_min + (poc_bin + 0.5) * bin_size
    
    # Value Area = price levels containing 70% of total volume centered on POC
    total_vol = sum(bins.values())
    target_vol = total_vol * 0.70
    
    sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
    cum_vol = 0
    val_bins = set()
    for bin_idx, vol in sorted_bins:
        if cum_vol >= target_vol:
            break
        val_bins.add(bin_idx)
        cum_vol += vol
    
    if val_bins:
        vah_bin = max(val_bins)
        val_bin = min(val_bins)
        vah_price = price_min + (vah_bin + 1) * bin_size
        val_price = price_min + val_bin * bin_size
    else:
        vah_price = price_max
        val_price = price_min
    
    # Volume-weighted average = profile's value estimate
    vwap = sum((price_min + (b + 0.5) * bin_size) * vol for b, vol in bins.items()) / total_vol if total_vol > 0 else poc_price
    
    # Classify HVNs (>2x average bin volume) and LVNs (<0.3x average)
    avg_bin_vol = total_vol / num_bins
    hvns = sorted([price_min + (b + 0.5) * bin_size for b, v in bins.items() if v > avg_bin_vol * 2])
    lvns = sorted([price_min + (b + 0.5) * bin_size for b, v in bins.items() if v < avg_bin_vol * 0.3 and v > 0])
    
    return {
        "poc": round(poc_price, 6),
        "vah": round(vah_price, 6),
        "val": round(val_price, 6),
        "vwap": round(vwap, 6),
        "hvns": [round(p, 6) for p in hvns[:5]],
        "lvns": [round(p, 6) for p in lvns[:5]],
        "profile_spread": round((vah_price - val_price) / poc_price * 100, 2) if poc_price > 0 else 0,
    }


def get_volume_profile_summary(symbol: str, analyzer) -> str:
    """
    Get a formatted Volume Profile summary for the AI.
    
    Args:
        symbol: coin symbol e.g. "XRP-EUR"
        analyzer: MarketAnalyzer instance
        
    Returns:
        Formatted string like:
        📊 VOLUME PROFILE (XRP):
           POC: €0.9506 (highest volume)
           VA: €0.92 - €0.98 (70% of volume)
           Profile: 6.3% wide (narrow = high conviction)
           HVNs: €0.93, €0.95 (support/resistance zones)
    """
    try:
        df_4h = analyzer.get_candles_df(symbol, "4h", hours_back=168)  # 7 days
        df_1h = analyzer.get_candles_df(symbol, "1h", hours_back=72)  # 3 days
        
        profile = calculate_volume_profile(df_4h) if df_4h is not None and len(df_4h) >= 10 else {}
        if not profile or profile.get("poc", 0) == 0:
            profile = calculate_volume_profile(df_1h) if df_1h is not None and len(df_1h) >= 10 else {}
        
        if not profile or profile.get("poc", 0) == 0:
            return ""
        
        spread_label = "tight" if profile["profile_spread"] < 3 else ("normal" if profile["profile_spread"] < 8 else "wide")
        base = symbol.split("-")[0]
        
        lines = [f"📊 VOLUME PROFILE ({base}):"]
        lines.append(f"   POC: €{profile['poc']:.4f} (highest volume node)")
        lines.append(f"   VA: €{profile['val']:.4f} - €{profile['vah']:.4f} ({spread_label}, {profile['profile_spread']:.1f}% wide)")
        
        if profile.get("hvns"):
            lines.append(f"   HVNs: {', '.join(f'€{p:.4f}' for p in profile['hvns'][:3])} (support/resistance)")
        if profile.get("lvns"):
            lines.append(f"   LVNs: {', '.join(f'€{p:.4f}' for p in profile['lvns'][:3])} (low liquidity zones)")
        
        return "\n".join(lines)
    except Exception:
        return ""


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, ".")
    from market_analyzer import MarketAnalyzer
    from revolut_client import RevolutXClient
    client = RevolutXClient()
    analyzer = MarketAnalyzer(client)
    for sym in ["XRP-EUR", "BTC-EUR", "AVAX-EUR"]:
        print(get_volume_profile_summary(sym, analyzer))
        print()