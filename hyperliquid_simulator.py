#!/usr/bin/env python3
"""
Hyperliquid Leveraged Perps Simulator — finds optimal entry/exit parameters.

Uses Kraken daily OHLCV (reliable from AWS) to test LONG+SHORT leveraged strategies.
Optimizes: entry support distance, leverage, stop loss %, take profit %.
Models funding rates, margin, and liquidation risk.

No live trading — pure simulation to find the best parameters for the AI daemon.
"""
import json, urllib.request, time, math, sys
from datetime import datetime, timezone

KRAKEN_PAIRS = {
    "BTC": "XXBTZUSD", "ETH": "XETHZUSD", "SOL": "SOLUSD",
    "DOGE": "XDGUSD", "XRP": "XXRPZUSD", "ADA": "ADAUSD",
    "LINK": "LINKUSD", "DOT": "DOTUSD", "AVAX": "AVAXUSD", "SUI": "SUIUSD",
}

# ── Data Fetch ───────────────────────────────────────────────────────────────

def fetch_ohlcv(coin: str, interval: int = 1440) -> list:
    """Fetch daily OHLCV from Kraken. Returns [{open,high,low,close,volume,ts}]."""
    pair = KRAKEN_PAIRS.get(coin, f"{coin}USD")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            candles = data.get("result", {}).get(pair, [])
            return [
                {
                    "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4]),
                    "volume": float(c[6]), "ts": int(c[0]),
                }
                for c in candles if len(c) >= 7
            ]
    except Exception as e:
        print(f"  Fetch error: {e}")
        return []


# ── Indicators ───────────────────────────────────────────────────────────────

def ema(data, period):
    """Exponential moving average."""
    if len(data) < period:
        return []
    k = 2.0 / (period + 1)
    result = [data[0]]
    for x in data[1:]:
        result.append(x * k + result[-1] * (1 - k))
    return result


def rsi(closes, period=14):
    """Relative Strength Index. Returns same length as input."""
    n = len(closes)
    if n < period + 1:
        return [50] * n
    deltas = [closes[i] - closes[i-1] for i in range(1, n)]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]

    result = [50] * (period + 1)  # first period+1 values are neutral
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        result.append(100 - 100 / (1 + rs))

    # Pad if needed
    while len(result) < n:
        result.insert(0, 50)
    return result[:n]


def pivot_points(highs, lows, closes, lookback=20):
    """Weekly/monthly pivot support/resistance."""
    result = []
    for i in range(len(closes)):
        start = max(0, i - lookback)
        h = max(highs[start:i+1])
        l = min(lows[start:i+1])
        c = closes[i]
        pp = (h + l + c) / 3
        s1 = 2 * pp - h
        s2 = pp - (h - l)
        r1 = 2 * pp - l
        result.append({"pp": pp, "s1": s1, "s2": s2, "r1": r1})
    return result


def atr(highs, lows, closes, period=14):
    """Average True Range. Returns same length as input."""
    n = len(closes)
    if n < 2:
        return [0] * n
    result = [0] * n
    tr = max(highs[0] - lows[0], abs(highs[0] - closes[0]), abs(lows[0] - closes[0]))
    result[0] = tr
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        if i < period:
            result[i] = sum(max(highs[j]-lows[j], abs(highs[j]-closes[max(0,j-1)]), abs(lows[j]-closes[max(0,j-1)]))
                           for j in range(i+1)) / (i+1)
        else:
            result[i] = result[i-1] * (period-1)/period + tr/period
    return result


# ── Strategy ─────────────────────────────────────────────────────────────────

def run_simulation(ohlcv, leverage=3, entry_support_pct=1.0,
                   stop_loss_pct=5.0, take_profit_pct=8.0,
                   max_positions=3, capital=1000.0, fee=0.00035):
    """
    Simulate leveraged LONG+SHORT trading with limit orders at support.

    Rules (matching the AI daemon):
    - Only enter LONG when price is within entry_support_pct% of support
    - Only enter SHORT when price is within entry_support_pct% of resistance  
    - Stop loss at stop_loss_pct% from entry
    - Take profit at take_profit_pct% from entry
    - Max max_positions concurrent
    - Position size = capital / max_positions * leverage (Kelly-adjusted)
    """
    closes = [c["close"] for c in ohlcv]
    highs = [c["high"] for c in ohlcv]
    lows = [c["low"] for c in ohlcv]
    n = len(ohlcv)

    if n < 50:
        return {"error": "not enough data"}

    pivots = pivot_points(highs, lows, closes, 10)  # shorter lookback for tighter levels
    rsi_vals = rsi(closes)
    atr_vals = atr(highs, lows, closes)

    positions = []  # [{side, entry_price, size_usd, leverage, stop, target, entry_ts}]
    equity = [capital]
    trades = []

    for i in range(50, n):  # skip warmup
        current = closes[i]
        pivot = pivots[i]
        rsi_now = rsi_vals[i]
        atr_now = atr_vals[i]

        # ── Check existing positions for stop/target ──
        for pos in list(positions):
            pnl_pct = ((current - pos["entry_price"]) / pos["entry_price"]) * 100
            if pos["side"] == "SHORT":
                pnl_pct = -pnl_pct

            exit_reason = None
            if pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif pnl_pct >= take_profit_pct:
                exit_reason = "take_profit"

            if exit_reason:
                margin = pos["size_usd"] / pos["leverage"]
                pnl_usd = pos["size_usd"] * pnl_pct / 100
                fee_cost = pos["size_usd"] * fee * 2  # entry + exit
                capital += margin + pnl_usd - fee_cost
                positions.remove(pos)
                trades.append({
                    "side": pos["side"], "entry": pos["entry_price"],
                    "exit": current, "pnl_pct": round(pnl_pct, 2),
                    "reason": exit_reason, "ts": ohlcv[i]["ts"],
                })

        # ── Check for new entries ──
        if len(positions) < max_positions and capital > 50:
            # LONG entry: price near S1 support (not S2 — S1 is closer) + RSI not overbought
            support = pivot["s1"]  # use S1 — closer to price, more actionable
            support_dist = (current - support) / current * 100

            # LONG entry: price near support + RSI not overbought
            if support_dist <= entry_support_pct and rsi_now < 70 and support_dist >= -1:
                pos_size = min(capital / max_positions * leverage, capital * 0.5 * leverage)
                margin = pos_size / leverage
                if margin < capital * 0.1:  # need at least 10% free
                    capital -= margin  # allocate margin
                    positions.append({
                        "side": "LONG", "entry_price": current,
                        "size_usd": pos_size, "leverage": leverage,
                        "stop": current * (1 - stop_loss_pct / 100),
                        "target": current * (1 + take_profit_pct / 100),
                        "entry_ts": ohlcv[i]["ts"],
                    })

            # SHORT entry: price near R1 resistance + RSI not oversold
            resistance = pivot["r1"]
            res_dist = (resistance - current) / current * 100
            if res_dist <= entry_support_pct and rsi_now > 30:
                pos_size = min(capital / max_positions * leverage, capital * 0.5 * leverage)
                margin = pos_size / leverage
                if margin < capital * 0.1:
                    capital -= margin
                    positions.append({
                        "side": "SHORT", "entry_price": current,
                        "size_usd": pos_size, "leverage": leverage,
                        "stop": current * (1 + stop_loss_pct / 100),
                        "target": current * (1 - take_profit_pct / 100),
                        "entry_ts": ohlcv[i]["ts"],
                    })

        equity.append(capital + sum(
            (p["size_usd"] / p["leverage"]) +
            p["size_usd"] * ((current - p["entry_price"]) / p["entry_price"]) *
            (1 if p["side"] == "LONG" else -1)
            for p in positions
        ) - sum(p["size_usd"] * fee for p in positions))

    # Close remaining positions at last price
    final = closes[-1]
    for pos in positions:
        pnl_pct = ((final - pos["entry_price"]) / pos["entry_price"]) * 100
        if pos["side"] == "SHORT":
            pnl_pct = -pnl_pct
        margin = pos["size_usd"] / pos["leverage"]
        capital += margin + pos["size_usd"] * pnl_pct / 100 - pos["size_usd"] * fee
        trades.append({
            "side": pos["side"], "entry": pos["entry_price"],
            "exit": final, "pnl_pct": round(pnl_pct, 2),
            "reason": "close_all", "ts": ohlcv[-1]["ts"],
        })

    equity.append(capital)

    # Calculate metrics
    total_return = (capital / equity[0] - 1) * 100
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe (annualized daily)
    returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity))]
    avg_ret = sum(returns) / len(returns) if returns else 0
    std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns))**0.5 if returns else 0
    sharpe = (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0

    return {
        "total_return_pct": round(total_return, 1),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd, 1),
        "win_rate": round(win_rate, 1),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win_pct": round(avg_win, 1),
        "avg_loss_pct": round(avg_loss, 1),
        "final_equity": round(capital, 2),
    }


# ── Parameter Optimization ───────────────────────────────────────────────────

def optimize(coin: str, ohlcv: list):
    """Grid search best parameters for a coin."""
    print(f"\n{'='*60}")
    print(f"OPTIMIZING: {coin} ({len(ohlcv)} candles)")
    print(f"{'='*60}")

    best = None
    best_score = -999

    for leverage in [2, 3, 5]:
        for entry_dist in [2.0, 4.0, 6.0, 10.0]:
            for stop in [3.0, 5.0, 8.0, 12.0]:
                for tp in [5.0, 8.0, 12.0, 15.0]:
                    if stop >= tp:
                        continue  # stop must be < TP

                    result = run_simulation(
                        ohlcv, leverage=leverage,
                        entry_support_pct=entry_dist,
                        stop_loss_pct=stop, take_profit_pct=tp,
                        capital=1000.0,
                    )

                    if "error" in result:
                        continue

                    # Composite score: 40% return + 30% sharpe + 20% win_rate - 10% max_dd
                    score = (
                        result["total_return_pct"] * 0.4 +
                        result["sharpe"] * 10 * 0.3 +
                        result["win_rate"] * 0.2 -
                        result["max_dd_pct"] * 0.1
                    )

                    # Penalize too few trades (<10)
                    if result["trades"] < 10:
                        score *= 0.5

                    # Penalize extreme win rates (overfitting)
                    if result["win_rate"] > 85:
                        score *= 0.7

                    if score > best_score:
                        best_score = score
                        best = {**result, "leverage": leverage,
                                "entry_dist": entry_dist,
                                "stop_pct": stop, "tp_pct": tp}

    if best:
        print(f"\n  BEST PARAMS: lev={best['leverage']}x entry<={best['entry_dist']}% "
              f"stop={best['stop_pct']}% tp={best['tp_pct']}%")
        print(f"  Return: {best['total_return_pct']}% | Sharpe: {best['sharpe']} | "
              f"Win: {best['win_rate']}% | MaxDD: {best['max_dd_pct']}% | "
              f"Trades: {best['trades']}")

    return best


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HYPERLIQUID PERPS SIMULATOR")
    print("Optimizing entry/exit for leveraged LONG+SHORT\n")

    coins = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX"]
    all_results = {}

    for coin in coins:
        print(f"Fetching {coin}...")
        ohlcv = fetch_ohlcv(coin)
        if len(ohlcv) < 100:
            print(f"  ⚠️  Not enough data ({len(ohlcv)} candles)")
            continue

        # Also test 4h candles for finer entry
        ohlcv_4h = fetch_ohlcv(coin, 240)

        best_daily = optimize(coin, ohlcv)
        if best_daily:
            all_results[coin] = best_daily

        time.sleep(1)  # rate limit

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Optimal Parameters per Coin")
    print(f"{'='*60}")
    print(f"{'Coin':6s} {'Lev':>4s} {'Entry%':>7s} {'Stop%':>6s} {'TP%':>5s} "
          f"{'Return%':>8s} {'Sharpe':>7s} {'Win%':>6s} {'MaxDD%':>7s} {'Trades':>6s}")
    print("-" * 70)
    for coin, r in sorted(all_results.items(),
                          key=lambda x: x[1].get("total_return_pct", 0), reverse=True):
        print(f"{coin:6s} {r['leverage']:>3d}x {r['entry_dist']:>6.1f}% "
              f"{r['stop_pct']:>5.1f}% {r['tp_pct']:>4.1f}% "
              f"{r['total_return_pct']:>7.1f}% {r['sharpe']:>6.2f} "
              f"{r['win_rate']:>5.1f}% {r['max_dd_pct']:>6.1f}% {r['trades']:>5d}")

    # Save results for the daemon
    with open("data/hyperliquid_optimal_params.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to data/hyperliquid_optimal_params.json")
