#!/usr/bin/env python3
"""
Fast simulation — tests the continuous prediction engine against
historical Hyperliquid candle data.

Validates: signal quality, regime-adaptive weights, win rate, Sharpe, drawdown.

Usage: python3 simulate_continuous.py
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))

from hyperliquid.info import Info
from hyperliquid.utils import constants

COINS = ["SOL", "BTC", "ETH", "DOGE", "AVAX", "LINK"]
FEE = 0.001  # 0.1% per trade


def fetch_history(coin: str, interval: str = "15m", hours: int = 168) -> list[dict]:
    """Fetch historical candles from Hyperliquid."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    end_ms = int(time.time() * 1000)
    n_candles = {"15m": 4 * hours, "1h": hours}[interval]
    interval_ms = {"15m": 900_000, "1h": 3_600_000}[interval]
    start_ms = end_ms - n_candles * interval_ms
    candles = info.candles_snapshot(coin, interval, start_ms, end_ms)
    if not isinstance(candles, list):
        return []
    # Normalize
    result = []
    for c in candles:
        result.append({
            "o": float(c.get("o", c.get("open", 0))),
            "h": float(c.get("h", c.get("high", 0))),
            "l": float(c.get("l", c.get("low", 0))),
            "c": float(c.get("c", c.get("close", 0))),
            "v": float(c.get("v", c.get("volume", 0))),
            "t": c.get("t", 0),
        })
    return result


def resample_to_1h(candles_15m: list[dict]) -> list[dict]:
    """Convert 15m candles to 1h candles."""
    result = []
    buf = []
    for c in candles_15m:
        buf.append(c)
        if len(buf) >= 4:
            result.append({
                "o": buf[0]["o"],
                "h": max(x["h"] for x in buf),
                "l": min(x["l"] for x in buf),
                "c": buf[-1]["c"],
                "v": sum(x["v"] for x in buf),
                "t": buf[0]["t"],
            })
            buf = []
    return result


def run_simulation(coin: str, candles_15m: list[dict], min_confidence: int = 15) -> dict:
    """Run continuous predictor at each candle step. Simple long-only or short."""
    from continuous_predictor import predict_continuous, format_prediction_compact
    
    candles_1h = resample_to_1h(candles_15m)
    warmup = 60  # Need enough candles for indicators
    
    capital = 100.0
    position = 0.0  # +long, -short, 0=flat
    entry_price = 0.0
    trades = []
    equity_curve = []
    signals = []
    
    for i in range(warmup, len(candles_15m) - 1):
        current = candles_15m[i]
        nxt = candles_15m[i + 1]
        mid = current["c"]
        next_mid = nxt["c"]
        
        # Slice candle windows
        c15m = candles_15m[max(0, i-200):i+1]
        
        # Find matching 1h candles
        t_ms = current["t"]
        c1h_idx = next((j for j, c in enumerate(candles_1h) if c["t"] >= t_ms - 3600000), 0)
        c1h = candles_1h[max(0, c1h_idx-100):c1h_idx+1] if c1h_idx < len(candles_1h) else candles_1h[-100:]
        
        try:
            cpred = predict_continuous(coin, mid,
                candles_1m=None, candles_5m=None,
                candles_15m=c15m, candles_1h=c1h,
                funding_rate=0.0001, ml_signal=None,
                hurst_H=0.5)
        except Exception:
            continue
        
        # Decision: trade based on prediction confidence and bias
        bias = cpred.overall_bias
        conf = cpred.overall_confidence
        
        if conf < min_confidence:
            # Not enough confidence — hold
            if position != 0:
                equity_curve.append(capital + position * mid)
            else:
                equity_curve.append(capital)
            signals.append("hold")
            continue
        
        # Determine side
        if bias == "bullish" and conf > 20:
            signal = "buy"
        elif bias == "bearish" and conf > 20:
            signal = "sell"
        else:
            signal = "hold"
        
        signals.append(signal)
        
        # Execute
        if signal == "buy" and position <= 0:
            # Close short if any, open long
            if position < 0:
                pnl = (entry_price - mid) / entry_price * abs(position) * mid
                capital += position * mid + (pnl - FEE * abs(position) * mid)
                trades.append({"type": "close_short", "entry": entry_price, "exit": mid,
                              "pnl": round(pnl, 3), "pnl_pct": round((entry_price - mid) / entry_price * 100, 2)})
                position = 0
            
            # Open long — use 20% of capital
            size = (capital * 0.2) / mid
            position = size
            entry_price = mid
            capital -= FEE * size * mid
            trades.append({"type": "long", "entry": mid, "size": round(size, 4)})
            
        elif signal == "sell" and position >= 0:
            # Close long if any, open short
            if position > 0:
                pnl = (mid - entry_price) * position
                capital += position * entry_price + (pnl - FEE * position * mid)
                trades.append({"type": "close_long", "entry": entry_price, "exit": mid,
                              "pnl": round(pnl, 3), "pnl_pct": round((mid - entry_price) / entry_price * 100, 2)})
                position = 0
            
            # Open short — use 20% of capital
            size = (capital * 0.2) / mid
            position = -size
            entry_price = mid
            capital -= FEE * size * mid
            trades.append({"type": "short", "entry": mid, "size": round(size, 4)})
        
        # Track equity
        equity = capital + position * mid if position > 0 else capital + position * (2 * entry_price - mid) if position < 0 else capital
        equity_curve.append(equity)
    
    # Close any open position at end
    if position != 0:
        final_price = candles_15m[-1]["c"]
        if position > 0:
            pnl = (final_price - entry_price) * position
        else:
            pnl = (entry_price - final_price) * abs(position)
        capital += pnl - FEE * abs(position) * final_price
        trades.append({"type": "close_final", "entry": entry_price, "exit": final_price,
                      "pnl": round(pnl, 3), "pnl_pct": round(pnl / (abs(position) * entry_price) * 100, 2)})
        position = 0
    
    # Compute metrics
    final_capital = capital
    total_return = (final_capital - 100) / 100 * 100
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    win_rate = len(wins) / max(1, len(wins) + len(losses)) * 100
    
    # Max drawdown
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0
        for e in equity_curve:
            peak = max(peak, e)
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)
    else:
        max_dd = 0
    
    # Sharpe
    if len(equity_curve) > 5:
        returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] for i in range(1, len(equity_curve))]
        mean_ret = sum(returns) / len(returns)
        if len(returns) > 1:
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(variance) if variance > 0 else 0.001
            sharpe = mean_ret / std_ret * math.sqrt(252 * 24 * 4)  # Annualized (15m bars)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Signal distribution
    buy_count = signals.count("buy")
    sell_count = signals.count("sell")
    hold_count = signals.count("hold")
    
    return {
        "coin": coin,
        "final_capital": round(final_capital, 2),
        "total_return_pct": round(total_return, 1),
        "num_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "signals": f"↑{buy_count} ↓{sell_count} →{hold_count}",
        "total_pnl": round(sum(t.get("pnl", 0) for t in trades), 2),
        "data_hours": round(len(candles_15m) / 4, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("CONTINUOUS PREDICTOR — 7-DAY SIMULATION (15m candles)")
    print("=" * 70)
    
    results = []
    for coin in COINS:
        print(f"\n{coin}: fetching 168h of 15m candles...", end=" ", flush=True)
        candles = fetch_history(coin, "15m", 168)
        if len(candles) < 100:
            print(f"ONLY {len(candles)} candles — skipping")
            continue
        print(f"{len(candles)} candles ({len(candles)//4:.0f} hours)")
        
        print(f"  Running simulation...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(coin, candles)
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")
        
        emoji = "🟢" if result["total_return_pct"] > 0 else "🔴"
        print(f"  {emoji} Return: {result['total_return_pct']:+.1f}% | "
              f"Sharpe: {result['sharpe']:+.2f} | "
              f"WR: {result['win_rate']:.0f}% | "
              f"DD: {result['max_drawdown']:.1f}% | "
              f"Trades: {result['num_trades']} | "
              f"Signals: {result['signals']}")
        
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    
    if results:
        avg_return = sum(r["total_return_pct"] for r in results) / len(results)
        avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
        avg_wr = sum(r["win_rate"] for r in results) / len(results)
        total_pnl = sum(r["total_pnl"] for r in results)
        
        for r in sorted(results, key=lambda x: x["total_return_pct"], reverse=True):
            emoji = "🟢" if r["total_return_pct"] > 0 else "🔴"
            print(f"  {emoji} {r['coin']:6s} | {r['total_return_pct']:+5.1f}% | "
                  f"Sharpe:{r['sharpe']:+5.2f} | WR:{r['win_rate']:4.0f}% | "
                  f"DD:{r['max_drawdown']:5.1f}% | Trades:{r['num_trades']:2d} | "
                  f"Sig:{r['signals']}")
        
        print(f"\n  AVERAGE: Return={avg_return:+.1f}% Sharpe={avg_sharpe:+.2f} WR={avg_wr:.0f}%")
        print(f"  TOTAL PnL: ${total_pnl:+.2f}")
