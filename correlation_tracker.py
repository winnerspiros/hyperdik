"""
Correlation Tracker — tracks how coins move relative to each other.
Used for: portfolio diversification, hedging, spotting regime shifts.
"""
import os
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "correlations.json")


def _load_correlations() -> dict:
    """Load saved correlation matrix"""
    if not os.path.exists(DATA_PATH):
        return {"matrix": {}, "updated": None}
    try:
        with open(DATA_PATH) as f:
            return json.load(f)
    except:
        return {"matrix": {}, "updated": None}


def _save_correlations(data: dict):
    """Save correlation matrix"""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f)


def compute_rolling_correlations(price_history: Dict[str, List[float]], window: int = 24) -> Dict[str, Dict[str, float]]:
    """
    Compute rolling Pearson correlations between all coins.
    
    Args:
        price_history: dict of coin -> list of closing prices (same length, newest last)
        window: number of periods for rolling window
        
    Returns:
        dict of coin_a -> dict of coin_b -> correlation coefficient
    """
    coins = list(price_history.keys())
    if len(coins) < 2:
        return {}
    
    result = {}
    for i, coin_a in enumerate(coins):
        result[coin_a] = {}
        prices_a = np.array(price_history[coin_a][-window:], dtype=float)
        returns_a = np.diff(prices_a) / prices_a[:-1]
        
        for coin_b in coins[i+1:]:
            prices_b = np.array(price_history[coin_b][-window:], dtype=float)
            returns_b = np.diff(prices_b) / prices_b[:-1]
            
            # Ensure same length
            min_len = min(len(returns_a), len(returns_b))
            if min_len < 5:
                corr = 0
            else:
                corr = float(np.corrcoef(returns_a[-min_len:], returns_b[-min_len:])[0, 1])
                corr = round(corr, 3)
            
            result[coin_a][coin_b] = corr
    
    # Mirror matrix (coin_b -> coin_a)
    for coin_a in coins:
        for coin_b, corr in result.get(coin_a, {}).items():
            if coin_b not in result:
                result[coin_b] = {}
            result[coin_b][coin_a] = corr
    
    return result


def get_correlation_summary(coins: List[str], window: int = 24) -> str:
    """
    Get a formatted correlation summary for the AI.
    
    Args:
        coins: list of coin symbols to check
        window: rolling window size
        
    Returns:
        Formatted string like:
        "📊 CORRELATIONS (24h):
           AVAX-XRP: +0.85 (high) | AVAX-ADA: +0.72 (high)
           XRP-ADA: +0.68 (moderate)"
    """
    try:
        from market_analyzer import MarketAnalyzer
        from revolut_client import RevolutXClient
        
        client = RevolutXClient()
        analyzer = MarketAnalyzer(client)
        
        # Get closing prices for each coin (1h timeframe, last 48h)
        price_history = {}
        for coin in coins[:6]:  # Max 6 to keep it fast
            try:
                df = analyzer.get_candles_df(f"{coin}-EUR", "1h", hours_back=48)
                if df is not None and len(df) >= window + 5:
                    price_history[coin] = list(df["close"].values.astype(float))
            except:
                pass
        
        if len(price_history) < 2:
            return ""
        
        matrix = compute_rolling_correlations(price_history, window=min(window, len(next(iter(price_history.values()))) - 2))
        
        lines = []
        seen = set()
        for coin_a in sorted(matrix.keys()):
            for coin_b, corr in sorted(matrix[coin_a].items()):
                key = tuple(sorted([coin_a, coin_b]))
                if key in seen:
                    continue
                seen.add(key)
                corr = matrix[coin_a][coin_b]
                label = "high" if abs(corr) > 0.7 else ("moderate" if abs(corr) > 0.4 else "low")
                direction = "🟢" if corr > 0 else "🔴"
                lines.append(f"  {direction} {coin_a}-{coin_b}: {corr:+.2f} ({label})")
        
        if not lines:
            return ""
        
        return f"📊 CORRELATIONS ({window}h):\n" + "\n".join(lines[:10])
    
    except Exception:
        return ""


def get_diversification_advice(coins: List[str], holdings: Dict[str, float]) -> str:
    """
    Check if portfolio is over-concentrated in correlated coins.
    Returns advice string for the AI.
    """
    corr_text = get_correlation_summary(coins)
    if not corr_text:
        return ""
    
    # Check for pairs with >0.8 correlation (over-concentration risk)
    try:
        lines = corr_text.split("\n")
        high_corr_pairs = []
        for line in lines:
            if "high" in line and "+" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        corr_val = float(parts[1].split("(")[0].strip())
                        if corr_val > 0.8:
                            high_corr_pairs.append(line)
                    except:
                        pass
        
        advice = ""
        if high_corr_pairs:
            advice = "\n⚠️ OVER-CONCENTRATION RISK: " + ", ".join(
                p.split(":")[0].strip() for p in high_corr_pairs[:3]
            )
        
        return corr_text + advice
    except:
        return corr_text


if __name__ == "__main__":
    import sys
    coins = sys.argv[1:] if len(sys.argv) > 1 else ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
    print(get_correlation_summary(coins))
    print()
    print(get_diversification_advice(coins, {"AVAX": 10, "XRP": 5, "ADA": 3}))