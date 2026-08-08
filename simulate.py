#!/usr/bin/env python3
"""
Simulation engine — fetches real Hyperliquid data, runs backtest,
produces PnL, win rate, Sharpe, drawdown. Self-contained.
"""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import BacktestEngine

# ── Data fetching with retry ──
def fetch_hl_candles(coin, interval='1h', limit=200, max_retries=3):
    """Fetch OHLCV from Hyperliquid REST. Retries on rate limit."""
    import urllib.request, ssl
    url = "https://api.hyperliquid.xyz/info"
    payload = json.dumps({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": 0, "endTime": int(time.time()*1000)}
    }).encode()
    
    for attempt in range(max_retries):
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read())
            candles = []
            for c in data:
                candles.append({
                    "timestamp": c["t"],
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                })
            candles.sort(key=lambda x: x["timestamp"])
            return candles
        except Exception as e:
            wait = (attempt + 1) * 3
            print(f"  {coin} attempt {attempt+1} failed ({type(e).__name__}), waiting {wait}s...")
            time.sleep(wait)
    return []

# ── Strategy: EMA crossover + RSI filter ──
def compute_ema(data, period):
    if len(data) < period:
        return []
    k = 2 / (period + 1)
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(data[i] * k + ema[-1] * (1 - k))
    return ema

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [50] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi = [50] * (period)  # warmup
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
        rs = avg_gain / max(avg_loss, 1e-10)
        rsi.append(100 - 100/(1+rs))
    return rsi

def run_simulation(coin='BTC', interval='1h', limit=500):
    print(f"\n{'='*60}")
    print(f"  SIMULATION: {coin} {interval} — {limit} candles")
    print(f"{'='*60}")
    
    print(f"  Fetching {coin} {interval} candles...")
    candles = fetch_hl_candles(coin, interval, limit)
    if not candles:
        print(f"  FAILED: no data for {coin}")
        return None
    print(f"  Got {len(candles)} candles ({candles[0]['timestamp']} → {candles[-1]['timestamp']})")
    
    closes = [c['close'] for c in candles]
    
    # Compute indicators
    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    rsi14 = compute_rsi(closes, 14)
    
    # Build strategy function
    def strategy_fn(i, ohlcv):
        if i < 21:
            return 0.0
        # Buy: EMA9 crosses above EMA21 AND RSI > 40 (not oversold)
        if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and rsi14[i] > 40:
            return 1.0
        # Sell: EMA9 crosses below EMA21 OR RSI > 80 (overbought)
        if (ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]) or rsi14[i] > 80:
            return -1.0
        return 0.0
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100.0, fee=0.001, position_pct=0.95)
    results = engine.run(candles, strategy_fn, warmup=50, verbose=False)
    
    # Print results
    print(f"\n  ── RESULTS ──")
    print(f"  Final capital:    ${results['final_capital']:.2f}")
    print(f"  Total return:     {results['total_return_pct']:+.2f}%")
    print(f"  Sharpe ratio:     {results['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:     {results['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:         {results['win_rate']:.1f}%")
    print(f"  Total trades:     {results['num_trades']}")
    
    # Trade log
    if results['closed_trades']:
        print(f"\n  ── TRADE LOG ──")
        for t in results['closed_trades'][-10:]:
            print(f"  {t.get('timestamp','?')}: {t.get('pnl_pct',0):+.2f}% PnL=${t.get('pnl',0):+.2f}")
    
    # Save results
    out = {
        "coin": coin,
        "interval": interval,
        "candles": len(candles),
        "results": {k: v for k, v in results.items() if k not in ('equity_curve', 'trades')},
        "trades_summary": [{"ts": t.get("timestamp"), "pnl_pct": t.get("pnl_pct", 0)} for t in results.get('closed_trades', [])]
    }
    with open(f"/home/ubuntu/hyperliquid-trader/data/sim_{coin}_{interval}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to data/sim_{coin}_{interval}.json")
    
    return results

if __name__ == "__main__":
    os.makedirs("/home/ubuntu/hyperliquid-trader/data", exist_ok=True)
    
    coins = ["BTC", "ETH", "SOL", "AVAX"]
    for coin in coins:
        run_simulation(coin, '1h', 500)
        time.sleep(2)  # Rate limit protection
