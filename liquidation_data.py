"""
Liquidation Data — estimates forced liquidation activity from public data.
Binance and HyperLiquid endpoints blocked from our server region.
Alternative approach: detect liquidation events from volume/volatility anomalies.
"""
import time
import json
import logging
import urllib.request
from typing import Optional, Dict, List
from datetime import datetime, timezone
import numpy as np

log = logging.getLogger("liquidation_data")


def estimate_liquidations_from_volatility(closes: List[float], volumes: List[float],
                                          window: int = 24) -> Dict:
    """
    Estimate liquidation activity from price/volume anomalies.
    Large price drops on high volume = likely forced liquidations.
    
    Args:
        closes: list of closing prices (newest last)
        volumes: list of volumes (newest last)
        window: rolling window size
        
    Returns:
        dict with estimated liquidation data
    """
    if len(closes) < window + 5 or len(volumes) < window + 5:
        return {"estimated": False, "reason": "insufficient data"}
    
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)
    
    # Price changes
    returns = np.diff(closes) / closes[:-1]
    
    # Rolling stats
    vol_ratio = volumes[-1] / (np.mean(volumes[-window:-1]) + 1)
    recent_volatility = np.std(returns[-window:])
    latest_return = returns[-1] if len(returns) > 0 else 0
    
    # Detect liquidation-like events
    candles_liquidated = 0
    total_est_vol = 0
    
    for i in range(-min(window, len(returns)), 0):
        ret = returns[i]
        v = volumes[i]
        avg_v = np.mean(volumes[max(0, i-window):i]) + 1
        v_spike = v / avg_v
        
        # Liquidation signature: sharp drop + high volume
        if ret < -0.03 and v_spike > 2:  # 3%+ drop on 2x+ volume
            candles_liquidated += 1
            total_est_vol += v
    
    if candles_liquidated == 0:
        return {
            "estimated": True,
            "recent_liquidations": "none detected",
            "volatility_regime": "low" if recent_volatility < 0.01 else ("medium" if recent_volatility < 0.03 else "high"),
            "latest_candle_return": f"{latest_return*100:+.2f}%",
        }
    
    return {
        "estimated": True,
        "recent_liquidations": f"{candles_liquidated} candles with liquidation signature in last {window}",
        "estimated_total_volume": round(total_est_vol, 2),
        "volatility_regime": "high",
        "latest_candle_return": f"{latest_return*100:+.2f}%",
    }


def format_liquidation_for_ai(symbol: str = "BTC") -> str:
    """
    Get formatted liquidation context for the AI brain.
    Uses volatility estimation since direct API endpoints are blocked.
    """
    try:
        from market_analyzer import MarketAnalyzer
        from revolut_client import RevolutXClient
        client = RevolutXClient()
        analyzer = MarketAnalyzer(client)
        
        lines = [f"💧 LIQUIDATIONS:"]
        
        for sym in ["BTC", "ETH"]:
            try:
                df = analyzer.get_candles_df(f"{sym}-EUR", "1h", hours_back=48)
                if df is None or len(df) < 10:
                    continue
                closes = list(df["close"].values.astype(float))
                volumes = list(df["volume"].values.astype(float))
                est = estimate_liquidations_from_volatility(closes, volumes)
                if est.get("estimated"):
                    liq = est.get("recent_liquidations", "none")
                    vol_reg = est.get("volatility_regime", "?")
                    lines.append(f"  {sym}: {liq} (vol: {vol_reg})")
            except:
                pass
        
        # Also try CoinGecko global data for market-wide stress
        try:
            req = urllib.request.Request(
                "https://api.coingecko.com/api/v3/global",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            data = json.loads(urllib.request.urlopen(req, timeout=8).read())
            data = data.get("data", {})
            mcap_change = data.get("market_cap_change_percentage_24h_usd", 0)
            if mcap_change < -3:
                lines.append(f"  🌍 Market-wide: global mcap {mcap_change:+.1f}% (stress detected)")
            elif mcap_change < -1:
                lines.append(f"  🌍 Market-wide: global mcap {mcap_change:+.1f}% (mild selling)")
            else:
                lines.append(f"  🌍 Market-wide: global mcap {mcap_change:+.1f}% (calm)")
        except:
            pass
        
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(format_liquidation_for_ai("BTC"))
    print()
    print(format_liquidation_for_ai("ETH"))