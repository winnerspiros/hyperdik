import re
from collections import defaultdict

with open('/home/ubuntu/hyperliquid-trader/logs/hyperliquid_daemon.log') as f:
    lines = f.readlines()

trades = []
current_open = None

for line in lines:
    m = re.search(r'DIRECT OPEN: (\w+) (LONG|SHORT).*oid=(\d+).*— (.+)', line)
    if m:
        ts = ' '.join(line.split(' ')[:2])
        current_open = {
            'coin': m.group(1), 'dir': m.group(2), 'oid': m.group(3),
            'reason': m.group(4)[:60], 'open_ts': ts, 'peak_pnl': -99
        }
        continue
    
    if current_open:
        coin = current_open['coin']
        m = re.search(coin + r'.*peak_pnl=([+-]?\d+\.\d+)%', line)
        if m:
            pnl = float(m.group(1))
            if pnl > current_open.get('peak_pnl', -99):
                current_open['peak_pnl'] = pnl
    
    if current_open:
        coin = current_open['coin']
        if 'TRAIL' in line and 'hard-killing' in line and coin in line:
            current_open['exit'] = 'TRAIL'
            trades.append(current_open)
            current_open = None
            continue
        m = re.search(coin + r': ROI timeout.*gross=([-\d.]+%).*net=([-\d.]+%)', line)
        if m:
            current_open['exit'] = 'ROI'
            current_open['gross'] = float(m.group(1).replace('%',''))
            current_open['net'] = float(m.group(2).replace('%',''))
            trades.append(current_open)
            current_open = None
            continue
        m = re.search(coin + r': PEAK LOCK TIER \d.*peak \+([\d.]+)%', line)
        if m:
            current_open['peak_lock'] = float(m.group(1))
        m = re.search(coin + r': TIGHT TRAIL.*locking \+([\d.]+)%', line)
        if m:
            current_open['trail_lock'] = float(m.group(1))

# Classify
winners = []
losers = []
for t in trades:
    net = t.get('net', None)
    if net is not None and net > 0:
        winners.append(t)
    elif net is not None and net < 0:
        losers.append(t)
    elif t.get('peak_lock') or t.get('trail_lock'):
        winners.append(t)

winners.sort(key=lambda t: t.get('peak_pnl', -99), reverse=True)

print("=== GOOD TRADES (top 20) ===")
for t in winners[:20]:
    net = t.get('net', '?')
    peak = t.get('peak_pnl', -99)
    plk = t.get('peak_lock', '?')
    tlk = t.get('trail_lock', '?')
    ex = t.get('exit', '?')
    print(f"  {t.get('open_ts','?'):22s} {t['coin']:8s} {t['dir']:5s} peak={peak:+.2f}% net={net} lock={plk} trail={tlk} exit={ex}")

print()
print("=== PEAK PnL DISTRIBUTION ===")
peaks = [t.get('peak_pnl', -99) for t in trades if t.get('peak_pnl', -99) > -99]
buckets = defaultdict(int)
for p in peaks:
    if p < 0: buckets['<0%'] += 1
    elif p < 0.25: buckets['0-0.25%'] += 1
    elif p < 0.5: buckets['0.25-0.5%'] += 1
    elif p < 1.0: buckets['0.5-1.0%'] += 1
    elif p < 2.0: buckets['1.0-2.0%'] += 1
    elif p < 3.0: buckets['2.0-3.0%'] += 1
    else: buckets['3.0%+'] += 1
for k, v in sorted(buckets.items()):
    print(f"  {k}: {v}")

print()
print(f"Total: {len(trades)} trades, {len(winners)} winners, {len(losers)} losers")
if peaks:
    print(f"Max peak: {max(peaks):+.2f}%")
    print(f"Avg peak: {sum(peaks)/len(peaks):+.2f}%")

# Now show what killed the winners
print()
print("=== WINNER EXIT REASONS ===")
wexits = defaultdict(int)
for t in winners:
    wexits[t.get('exit','?')] += 1
print(f"  {dict(wexits)}")

# Show winners that had good peaks but got killed early
print()
print("=== WINNERS WITH PEAK > 0.5% BUT NET < PEAK (killed early) ===")
for t in winners:
    net = t.get('net')
    peak = t.get('peak_pnl', 0)
    if net is not None and peak > 0.5 and net < peak * 0.5:
        print(f"  {t['coin']:8s} peak={peak:+.2f}% net={net:+.2f}% lost {peak-net:.2f}% exit={t.get('exit','?')}")
