"""
Mempool Monitor — Bitcoin mempool data from mempool.space (free, no key).
Mempool pressure is a leading indicator for BTC volatility.
"""
import urllib.request
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("mempool_monitor")

MEMPOOL_API = "https://mempool.space/api"


def get_mempool_info() -> dict:
    """Get current mempool state: pending tx count, backlog size"""
    try:
        req = urllib.request.Request(f"{MEMPOOL_API}/mempool",
                                      headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=8).read())
        return {
            "count": data.get("count", 0),
            "vsize": data.get("vsize", 0),
            "fee_per_byte": round(data.get("total_fee", 0) / max(data.get("vsize", 1), 1), 2),
        }
    except Exception as e:
        log.debug(f"Mempool info failed: {e}")
        return {}


def get_fee_estimates() -> dict:
    """Get recommended fee rates in sat/vB"""
    try:
        req = urllib.request.Request(f"{MEMPOOL_API}/v1/fees/recommended",
                                      headers={"User-Agent": "Mozilla/5.0"})
        return json.loads(urllib.request.urlopen(req, timeout=8).read())
    except Exception as e:
        log.debug(f"Fee estimates failed: {e}")
        return {}


def format_mempool_for_ai() -> str:
    """Format mempool data for AI context"""
    mempool = get_mempool_info()
    fees = get_fee_estimates()
    
    if not mempool:
        return ""
    
    lines = ["⛏️ BITCOIN MEMPOOL:"]
    lines.append(f"  Pending tx: {mempool.get('count', 0):,}")
    lines.append(f"  Backlog: {mempool.get('vsize', 0)/1_000_000:.1f} MB")
    
    if fees:
        fast = fees.get("fastestFee", 0)
        hour = fees.get("hourFee", 0)
        econ = fees.get("economyFee", 0)
        
        # Interpret mempool pressure
        if fast > 100:
            lines.append(f"  🚨 NETWORK CONGESTION (fast fee {fast} sat/vB)")
        elif fast > 30:
            lines.append(f"  ⚠️ High traffic (fast fee {fast} sat/vB)")
        elif fast > 10:
            lines.append(f"  📊 Normal (fast fee {fast} sat/vB)")
        else:
            lines.append(f"  💤 Low activity (fast fee {fast} sat/vB)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(format_mempool_for_ai())