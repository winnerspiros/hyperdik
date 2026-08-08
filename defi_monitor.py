"""
DeFi Monitor — TVL, stablecoin mint/burn, protocol-level capital flows.
Free API from DefiLlama (no key needed).
Signals: rising TVL = bullish for ETH/alts, falling stablecoin supply = selling pressure.
"""
import json
import logging
import urllib.request
from datetime import datetime, timezone

log = logging.getLogger("defi_monitor")

DEFILLAMA_API = "https://api.llama.fi"
STABLECOIN_API = "https://stablecoins.llama.fi"


def _fetch_json(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception as e:
        log.debug(f"DefiLlama error: {e}")
        return None


def get_tvl_summary():
    """Get total DeFi TVL and chain breakdown"""
    data = _fetch_json(f"{DEFILLAMA_API}/v2/chains")
    if not data or not isinstance(data, list):
        return {}

    total = sum(c.get("tvl", 0) for c in data[:20])
    top_chains = [
        {"chain": c.get("name", "?"), "tvl": round(c.get("tvl", 0) / 1e9, 2)}
        for c in data[:5] if c.get("tvl", 0) > 0
    ]

    return {
        "total_tvl_billions": round(total / 1e9, 2),
        "top_chains": top_chains,
        "chain_count": len(data),
    }


def get_stablecoin_summary():
    """Get stablecoin supply data"""
    data = _fetch_json(f"{STABLECOIN_API}/stablecoins?includePrices=true")
    if not data:
        return {}

    assets = data.get("peggedAssets", [])
    total_supply = sum(
        a.get("circulating", {}).get("peggedUSD", 0)
        for a in assets if a.get("circulating", {})
    )

    top_stables = []
    for a in assets[:5]:
        circ = a.get("circulating", {}).get("peggedUSD", 0)
        if circ > 0:
            top_stables.append({
                "name": a.get("name", "?"),
                "symbol": a.get("symbol", "?"),
                "supply_billions": round(circ / 1e9, 2),
            })

    return {
        "total_stablecoin_supply_billions": round(total_supply / 1e9, 2),
        "top_stables": top_stables,
    }


def format_defi_for_ai() -> str:
    """Format DeFi data for AI context"""
    tvl = get_tvl_summary()
    stable = get_stablecoin_summary()

    if not tvl:
        return ""

    lines = ["🏗️ DEFI:"]
    lines.append(f"  TVL: ${tvl.get('total_tvl_billions', '?')}B across {tvl.get('chain_count', '?')} chains")
    for c in tvl.get("top_chains", [])[:3]:
        lines.append(f"    {c['chain']}: ${c['tvl']}B")

    if stable:
        supply = stable.get("total_stablecoin_supply_billions", 0)
        lines.append(f"  Stablecoins: ${supply}B total supply")
        for s in stable.get("top_stables", [])[:3]:
            lines.append(f"    {s['symbol']}: ${s['supply_billions']}B")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(format_defi_for_ai())