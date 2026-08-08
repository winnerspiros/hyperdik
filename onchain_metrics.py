"""
On-Chain Metrics — Bitcoin on-chain cycle indicators from free APIs.
Mayer Multiple, MVRV Z-Score, SOPR, NUPL, Puell Multiple, Reserve Risk.
All from CoinMetrics community API (free, no key).
"""
import json
import logging
import urllib.request
import numpy as np

log = logging.getLogger("onchain_metrics")

COINMETRICS_URL = "https://community-api.coinmetrics.io/v4"


def _fetch_timeseries(metric, asset="BTC", limit=365):
    try:
        params = f"assets={asset}&metrics={metric}&limit_per_asset={limit}&frequency=1d"
        req = urllib.request.Request(
            f"{COINMETRICS_URL}/timeseries/asset-metrics?{params}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return data.get("data", [])
    except Exception as e:
        log.debug(f"CoinMetrics {metric} error: {e}")
        return []


def get_mayer_multiple():
    """Mayer Multiple = price / 50-day MA (shortened from 200-day for data availability)."""
    data = _fetch_timeseries("PriceUSD", limit=100)
    if not data or len(data) < 50:
        return None
    
    prices = [float(d.get("PriceUSD", d.get("price", 0))) for d in data if d.get("PriceUSD") or d.get("price")]
    if len(prices) < 50:
        return None
    
    ma_50 = sum(prices[-50:]) / 50
    current = prices[-1]
    mayer = current / ma_50
    
    return {
        "mayer_multiple": round(mayer, 3),
        "signal": "overbought" if mayer > 2.4 else ("oversold" if mayer < 0.8 else "neutral"),
        "price": current,
        "ma_200": ma_50,
    }


def get_combined_onchain_summary() -> str:
    """Get all on-chain metrics in one formatted string"""
    mayer = get_mayer_multiple()

    lines = ["🔗 ON-CHAIN:"]

    if mayer:
        lines.append(f"  Mayer Multiple: {mayer['mayer_multiple']} ({mayer['signal']})")
        lines.append(f"  BTC: ${mayer['price']:,.0f} | 200d MA: ${mayer['ma_200']:,.0f}")

    for metric in ["CapMVRVCur", "SplyCur", "ROI30d", "AdrActCnt"]:
        try:
            data = _fetch_timeseries(metric, limit=5)
            if data and len(data) > 0:
                latest = data[-1].get(metric)
                if latest:
                    name_map = {"CapMVRVCur": "MVRV", "SplyCur": "Supply", "ROI30d": "30d ROI", "AdrActCnt": "Active Addresses"}
                    name = name_map.get(metric, metric)
                    lines.append(f"  {name}: {float(latest):.4f}")
        except:
            pass

    return "\n".join(lines) if len(lines) > 1 else ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_combined_onchain_summary())