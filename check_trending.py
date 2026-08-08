"""Check trending coins and pump opportunities"""
import urllib.request, json

# CoinGecko trending
req = urllib.request.Request('https://api.coingecko.com/api/v3/search/trending', headers={'User-Agent':'Mozilla/5.0'})
d = json.loads(urllib.request.urlopen(req, timeout=10).read())
coins = d.get('coins', [])
print('TOP TRENDING ON COINGECKO:')
for item in coins[:15]:
    c = item.get('item', {})
    name = c.get('name', '?')
    symbol = c.get('symbol', '?')
    price_btc = c.get('data', {}).get('price', '?')
    market_cap = c.get('data', {}).get('market_cap', '?')
    print(f'  {name:20s} ({symbol:8s}) price_btc: {str(price_btc)[:12]} | cap: {str(market_cap)[:12]}')

# Check MEW specifically
req2 = urllib.request.Request('https://api.coingecko.com/api/v3/coins/mew', headers={'User-Agent':'Mozilla/5.0'})
try:
    mew = json.loads(urllib.request.urlopen(req2, timeout=10).read())
    md = mew.get('market_data', {})
    print(f'\nMEW: rank #{mew.get("market_cap_rank", "?")} | ${md.get("current_price",{}).get("usd",0):.6f} | 24h: {md.get("price_change_percentage_24h",0):+.2f}%')
    print(f'  Market cap: ${md.get("market_cap",{}).get("usd",0):,.0f}')
    print(f'  Volume 24h: ${md.get("total_volume",{}).get("usd",0):,.0f}')
    print(f'  All-time high: ${md.get("ath",{}).get("usd",0):.6f}')
except Exception as e:
    print(f'MEW lookup: {e}')

# Check which of these are on Revolut X
revolut_coins = ['BTC','ETH','SOL','XRP','ADA','DOT','LINK','AVAX','DOGE','APT','ATOM','ARB','OP','INJ','SUI','LTC','FIL','TRX','PEPE','BONK','WIF','NEAR','UNI','SEI','SHIB','TON','ENA','CRV','HYPE','MAGIC','PENDLE','1INCH']
print('\n--- REVOLUT X TRADABLE PUMP CANDIDATES ---')
for item in coins[:15]:
    c = item.get('item', {})
    sym = c.get('symbol', '').upper()
    if sym in revolut_coins:
        print(f'  {c.get("name","?"):20s} ({sym:8s}) ★ TRADABLE ON REVOLUT X')