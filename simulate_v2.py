#!/usr/bin/env python3
"""Simulation runner — fetches data, runs backtest, outputs results."""
import json, time, ssl, urllib.request, math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backtest_engine import BacktestEngine

def fetch(coin, interval='4h'):
    url = 'https://api.hyperliquid.xyz/info'
    payload = json.dumps({'type': 'candleSnapshot', 'req': {'coin': coin, 'interval': interval, 'startTime': 0, 'endTime': int(time.time()*1000)}}).encode()
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        data = json.loads(resp.read())
    candles = [{'t': c['t'], 'o': float(c['o']), 'h': float(c['h']), 'l': float(c['l']), 'c': float(c['c']), 'v': float(c['v'])} for c in data]
    candles.sort(key=lambda x: x['t'])
    return candles

def ema(data, n):
    k = 2/(n+1); out = [data[0]]
    for i in range(1, len(data)): out.append(data[i]*k + out[-1]*(1-k))
    return out

def sma(data, n):
    out = []
    for i in range(len(data)):
        start = max(0, i-n+1); count = i-start+1
        out.append(sum(data[start:i+1])/count)
    return out

def atr(highs, lows, closes, period=14):
    tr = [highs[0]-lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    return sma(tr, period)

def make_strategy(name, closes, highs, lows, vols):
    e20 = ema(closes, 20); e50 = ema(closes, 50); e200 = ema(closes, 200)
    a = atr(highs, lows, closes, 14); vma = sma(vols, 20)
    
    # Strategy state (mutable across backtest engine calls)
    state = {'pos': False, 'entry': 0, 'trail': 0}
    
    def strat(i, ohlcv):
        nonlocal state
        p = closes[i]
        if i < 200: return 0.0
        
        if not state['pos']:
            uptrend = e50[i] > e200[i] and closes[i] > e200[i]
            cross = e20[i] > e50[i] and e20[i-1] <= e50[i-1]
            vol_ok = vols[i] > vma[i] * 1.3
            if uptrend and cross and vol_ok:
                state['pos'] = True; state['entry'] = p; state['trail'] = p
                return 1.0
            return 0.0
        else:
            state['trail'] = max(state['trail'], p)
            stop = state['trail'] - 2.0 * a[i]
            trend_broke = p < e50[i] and e20[i] < e50[i]
            trail_hit = p < stop
            if trend_broke or trail_hit:
                state['pos'] = False; state['entry'] = 0; state['trail'] = 0
                return -1.0
            return 0.0
    
    return strat

# ── RUN ──
os.makedirs('data', exist_ok=True)

coins = ['BTC', 'ETH', 'SOL', 'AVAX', 'DYDX', 'DOGE', 'ADA']
all_results = []

for coin in coins:
    try:
        print(f'  Fetching {coin}...', end=' ', flush=True)
        raw = fetch(coin, '4h')
        print(f'{len(raw)} candles')
        
        closes = [c['c'] for c in raw]; highs = [c['h'] for c in raw]
        lows = [c['l'] for c in raw]; vols = [c['v'] for c in raw]
        
        ohlcv = [{'timestamp': c['t'], 'open': c['o'], 'high': c['h'],
                  'low': c['l'], 'close': c['c'], 'volume': c['v']} for c in raw]
        
        strat = make_strategy('trend_mom_atr', closes, highs, lows, vols)
        engine = BacktestEngine(initial_capital=1000, fee=0.001, position_pct=0.95)
        res = engine.run(ohlcv, strat, warmup=200)
        
        r = {
            'coin': coin, 'candles': len(raw),
            'return_pct': res['total_return_pct'],
            'sharpe': res['sharpe_ratio'],
            'max_dd': res['max_drawdown_pct'],
            'win_rate': res['win_rate'],
            'trades': res['num_trades'],
        }
        all_results.append(r)
        time.sleep(2)
    except Exception as e:
        print(f'FAILED: {e}')
        break

# Print results
print(f"\n{'='*65}")
print(f"{'COIN':<8} {'RETURN':>8} {'SHARPE':>7} {'MAX DD':>7} {'WIN%':>6} {'TRADES':>7}")
print(f"{'='*65}")
for r in sorted(all_results, key=lambda x: x['return_pct'], reverse=True):
    flag = '>' if r['return_pct'] > 0 else '<'
    print(f"{flag} {r['coin']:<6} {r['return_pct']:>+7.1f}% {r['sharpe']:>+6.2f} {r['max_dd']:>+6.1f}% {r['win_rate']:>5.0f}% {r['trades']:>6}")
winners = sum(1 for r in all_results if r['return_pct'] > 0)
avg = sum(r['return_pct'] for r in all_results)/len(all_results) if all_results else 0
print(f"{'='*65}")
print(f"  {winners}/{len(all_results)} profitable  |  avg return: {avg:+.1f}%")

with open('data/simulation_advanced.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved to data/simulation_advanced.json")
