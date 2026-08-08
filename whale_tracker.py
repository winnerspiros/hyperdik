"""
Whale movement tracker — monitors large crypto transactions
Sources: Whale Alert website, blockchain explorers
"""
import urllib.request, ssl, json, re
from datetime import datetime

def get_whale_movements():
    """Get recent whale transaction data from free sources"""
    ctx = ssl.create_default_context()
    signals = []
    
    # 1. Check whale-alert.io alerts page
    try:
        req = urllib.request.Request(
            "https://r.jina.ai/https://whale-alert.io/alerts",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            content = resp.read().decode()
            txns = re.findall(r'(\d+[.,]?\d*)\s*(BTC|ETH|USDT|XRP|ADA|SOL|DOGE|LINK|AVAX|DOT)\s*(?:transferred|moved|sent)', content)
            for amount, coin in txns[:3]:
                try:
                    val = float(amount.replace(",", ""))
                    if val > 10:
                        signals.append(f"🐋 {val:.0f} {coin} moved")
                except: pass
    except: pass
    
    # 2. Etherscan large ETH transfers
    try:
        req = urllib.request.Request(
            "https://api.etherscan.io/api?module=account&action=txlist&address=0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18&sort=desc&limit=5",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "1":
                for tx in data.get("result", [])[:3]:
                    val = float(tx.get("value", 0)) / 1e18
                    if val > 100:
                        signals.append(f"🐋 {val:.0f} ETH moved to/from whale wallet")
    except: pass
    
    # 3. Binance BTC wallet movements
    try:
        req = urllib.request.Request(
            "https://blockchain.info/rawaddr/34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo?limit=2",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            data = json.loads(resp.read())
            for tx in data.get("txs", [])[:2]:
                total = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
                if total > 50:
                    signals.append(f"🐋 {total:.0f} BTC moved from Binance wallet")
    except: pass
    
    if not signals:
        signals.append("No whale movements in last cycle")
    
    return signals[:3]

if __name__ == "__main__":
    print(f"🐋 Whale movements at {datetime.now().strftime('%H:%M')}:")
    for s in get_whale_movements():
        print(f"  • {s}")