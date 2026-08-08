#!/usr/bin/env python3
"""
Full-system simulation & optimizer.

Simulates the complete signal → conviction → AI → sizing → trade pipeline
using historical Hyperliquid candle data. 

Finds optimal parameters: conviction threshold, position size, leverage, stops.

Usage: python3 simulate_full.py
"""
import sys, os, time, math, json
sys.path.insert(0, os.path.dirname(__file__))

from hyperliquid.info import Info
from hyperliquid.utils import constants

COINS = ["SOL", "BTC", "ETH", "DOGE"]
FEE = 0.001  # 0.1% per trade


def fetch_1h_history(coin: str, hours: int = 720) -> list[dict]:
    """Fetch 1h candles (30 days = 720 candles)."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - hours * 3_600_000
    candles = info.candles_snapshot(coin, "1h", start_ms, end_ms)
    if not isinstance(candles, list):
        return []
    result = []
    for c in candles:
        result.append({
            "o": float(c.get("o", 0)), "h": float(c.get("h", 0)),
            "l": float(c.get("l", 0)), "c": float(c.get("c", 0)),
            "v": float(c.get("v", 0)), "t": c.get("t", 0),
        })
    return result


def compute_atr(candles: list[dict], period: int = 14, idx: int = None) -> float:
    """ATR at given index."""
    if idx is None:
        idx = len(candles) - 1
    start = max(0, idx - period)
    trs = []
    for i in range(start + 1, idx + 1):
        h = candles[i]["h"]
        l = candles[i]["l"]
        prev_c = candles[i-1]["c"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else candles[idx]["c"] * 0.02


def compute_rsi(candles: list[dict], period: int = 14, idx: int = None) -> float:
    if idx is None:
        idx = len(candles) - 1
    if idx < period + 1:
        return 50
    gains = 0; losses = 0
    for i in range(idx - period, idx):
        change = candles[i+1]["c"] - candles[i]["c"]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_adx(candles: list[dict], period: int = 14, idx: int = None) -> float:
    """Simple ADX approximation."""
    if idx is None:
        idx = len(candles) - 1
    start = max(0, idx - period * 2)
    if idx - start < period:
        return 15
    trs = []; plus_dm = []; minus_dm = []
    for i in range(start + 1, idx + 1):
        h, l = candles[i]["h"], candles[i]["l"]
        prev_h, prev_l = candles[i-1]["h"], candles[i-1]["l"]
        prev_c = candles[i-1]["c"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        up = h - prev_h if h > prev_h else 0
        down = prev_l - l if l < prev_l else 0
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
    if len(trs) < period:
        return 15
    atr_val = sum(trs[-period:]) / period
    plus_di = (sum(plus_dm[-period:]) / period) / atr_val * 100 if atr_val > 0 else 0
    minus_di = (sum(minus_dm[-period:]) / period) / atr_val * 100 if atr_val > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    return dx


def detect_regime(candles: list[dict], idx: int) -> str:
    """Classify market regime from candle data."""
    if idx < 20:
        return "sideways"
    adx = compute_adx(candles, 14, idx)
    rsi = compute_rsi(candles, 14, idx)
    recent = candles[max(0, idx-10):idx+1]
    closes = [c["c"] for c in recent]
    trend = closes[-1] - closes[0]
    trend_pct = (trend / closes[0]) * 100
    
    vols = [c["v"] for c in recent]
    avg_vol = sum(vols) / len(vols) if vols else 0
    prev_vols = [c["v"] for c in candles[max(0, idx-30):max(0, idx-10)]]
    prev_avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else avg_vol
    vol_ratio = avg_vol / prev_avg_vol if prev_avg_vol > 0 else 1
    
    if adx > 30 and abs(trend_pct) > 3 and vol_ratio > 1.5:
        if trend > 0:
            return "trending_up"
        else:
            return "trending_down"
    elif adx > 25:
        if trend > 0:
            return "trending_up"
        else:
            return "trending_down"
    elif adx < 15:
        return "sideways"
    elif vol_ratio > 2.0:
        return "high_vol"
    else:
        return "sideways"


def compute_signal_score(candles: list[dict], idx: int, regime: str) -> dict:
    """Compute multi-factor signal at a given candle index."""
    mid = candles[idx]["c"]
    rsi = compute_rsi(candles, 14, idx)
    atr = compute_atr(candles, 14, idx)
    adx = compute_adx(candles, 14, idx)
    
    # Short-term momentum
    if idx >= 10:
        short_ma = sum(c["c"] for c in candles[idx-10:idx+1]) / 11
        momentum_up = mid > short_ma
    else:
        momentum_up = False
    
    # RSI signal
    rsi_bullish = rsi < 35  # Oversold → potential bounce
    rsi_bearish = rsi > 65  # Overbought → potential drop
    
    # Price vs moving averages
    if idx >= 50:
        ma50 = sum(c["c"] for c in candles[idx-50:idx+1]) / 51
        above_ma50 = mid > ma50
    else:
        above_ma50 = None
    
    # Determine side and confidence
    score = {"side": None, "confidence": 0, "reason": ""}
    
    if regime in ("trending_up",):
        if momentum_up or rsi_bullish:
            score["side"] = "BUY"
            score["confidence"] = 0.55
            score["reason"] = "trending_up:momentum"
        elif rsi_bearish:
            score["side"] = None  # Wait for pullback
            score["confidence"] = 0.35
            score["reason"] = "trending_up:wait_pullback"
    
    elif regime in ("trending_down",):
        if not momentum_up or rsi_bearish:
            score["side"] = "SELL"
            score["confidence"] = 0.55
            score["reason"] = "trending_down:momentum"
        elif rsi_bullish:
            score["side"] = None
            score["confidence"] = 0.35
            score["reason"] = "trending_down:wait_bounce"
    
    elif regime in ("high_vol",):
        if rsi_bullish and momentum_up:
            score["side"] = "BUY"
            score["confidence"] = 0.60
            score["reason"] = "high_vol:oversold_bounce"
        elif rsi_bearish and not momentum_up:
            score["side"] = "SELL"
            score["confidence"] = 0.60
            score["reason"] = "high_vol:overbought_drop"
    
    else:  # sideways
        if rsi_bullish and momentum_up and above_ma50 is False:
            score["side"] = "BUY"
            score["confidence"] = 0.50
            score["reason"] = "sideways:oversold"
        elif rsi_bearish and not momentum_up and above_ma50:
            score["side"] = "SELL"
            score["confidence"] = 0.50
            score["reason"] = "sideways:overbought"
    
    return score


def score_conviction_sim(signal_conf: float, regime: str, adx: float, rsi: float, vol_ratio: float) -> float:
    """
    Simulate the daemon's score_conviction function.
    Returns 0-100 conviction score.
    """
    score = 0.0
    
    # 1. Signal Quality (0-30)
    score += min(30, signal_conf * 30)
    
    # 2. Regime Clarity (0-20)
    rc_map = {"trending_up": 20, "trending_down": 20, "high_vol": 10, "sideways": 12}
    score += rc_map.get(regime, 12)
    
    # 3. Volume Confirmation (0-15)
    score += min(15, vol_ratio * 10)
    
    # 4. ADX Strength (0-15)
    score += min(15, adx * 0.5)
    
    # 5. RSI Sweet Spot (0-20): bonus for RSI in 30-70 range
    rsi_dist = min(abs(rsi - 50), 50)
    score += max(0, 20 - rsi_dist * 0.4)
    
    return min(100, score)


def simulate_coin(coin: str, candles: list[dict], 
                  min_conviction: int = 65,
                  max_position_pct: float = 0.08,
                  max_leverage: int = 3,
                  risk_per_trade_pct: float = 0.015,
                  stop_atr_mult: float = 2.0,
                  tp_r_multiple: float = 2.0,
                  ) -> dict:
    """
    Run full simulation on one coin with given parameters.
    """
    warmup = 50
    capital = 100.0
    position = 0.0  # +long, -short
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    trades = []
    equity_curve = [capital]
    
    for i in range(warmup, len(candles) - 1):
        current = candles[i]
        nxt = candles[i + 1]
        price = current["c"]
        next_price = nxt["c"]
        
        # Check stop/tp on existing position
        if position > 0:  # Long
            if price <= stop_price or price >= tp_price:
                exit_px = stop_price if price <= stop_price else tp_price
                pnl = (exit_px - entry_price) * position
                capital += position * entry_price + pnl
                capital -= FEE * position * exit_px
                trades.append({"entry": entry_price, "exit": exit_px, "pnl": round(pnl, 2),
                              "pnl_pct": round((exit_px - entry_price) / entry_price * 100, 2),
                              "type": "long"})
                position = 0; stop_price = 0; tp_price = 0
                
        elif position < 0:  # Short
            if price >= stop_price or price <= tp_price:
                exit_px = stop_price if price >= stop_price else tp_price
                pnl = (entry_price - exit_px) * abs(position)
                capital += abs(position) * entry_price + pnl
                capital -= FEE * abs(position) * exit_px
                trades.append({"entry": entry_price, "exit": exit_px, "pnl": round(pnl, 2),
                              "pnl_pct": round((entry_price - exit_px) / entry_price * 100, 2),
                              "type": "short"})
                position = 0; stop_price = 0; tp_price = 0
        
        # Compute regime
        regime = detect_regime(candles, i)
        
        # Compute signal
        sig = compute_signal_score(candles, i, regime)
        
        if sig["side"] is None:
            equity_curve.append(capital + position * price if position > 0 
                              else capital + position * (2 * entry_price - price) if position < 0 
                              else capital)
            continue
        
        # Compute conviction
        adx = compute_adx(candles, 14, i)
        rsi = compute_rsi(candles, 14, i)
        vols = [c["v"] for c in candles[max(0, i-10):i+1]]
        avg_vol = sum(vols) / len(vols) if vols else 0
        global_vols = [c["v"] for c in candles[max(0, i-50):i+1]]
        global_avg = sum(global_vols) / len(global_vols) if global_vols else 0.001
        vol_ratio = avg_vol / global_avg
        
        conviction = score_conviction_sim(sig["confidence"], regime, adx, rsi, vol_ratio)
        
        if conviction < min_conviction:
            equity_curve.append(capital + position * price if position > 0
                              else capital + position * (2 * entry_price - price) if position < 0 
                              else capital)
            continue
        
        # Only enter if flat
        if position != 0:
            equity_curve.append(capital + position * price if position > 0
                              else capital + position * (2 * entry_price - price) if position < 0
                              else capital)
            continue
        
        # Sizing
        pos_size = capital * max_position_pct
        position = pos_size / price if sig["side"] == "BUY" else -pos_size / price
        entry_price = price
        capital -= FEE * pos_size
        
        # ATR-based stops and targets
        atr = compute_atr(candles, 14, i)
        stop_dist = atr * stop_atr_mult
        tp_dist = stop_dist * tp_r_multiple
        
        if sig["side"] == "BUY":
            stop_price = price - stop_dist
            tp_price = price + tp_dist
        else:
            stop_price = price + stop_dist
            tp_price = price - tp_dist
        
        equity_curve.append(capital + position * price)
    
    # Close any final position
    final_price = candles[-1]["c"]
    if position > 0:
        pnl = (final_price - entry_price) * position
        capital += position * entry_price + pnl
        trades.append({"entry": entry_price, "exit": final_price, "pnl": round(pnl, 2),
                      "pnl_pct": round((final_price - entry_price) / entry_price * 100, 2), "type": "final_long"})
    elif position < 0:
        pnl = (entry_price - final_price) * abs(position)
        capital += abs(position) * entry_price + pnl
        trades.append({"entry": entry_price, "exit": final_price, "pnl": round(pnl, 2),
                      "pnl_pct": round((entry_price - final_price) / entry_price * 100, 2), "type": "final_short"})
    
    # Metrics
    final_capital = capital
    total_return = (final_capital - 100) / 100 * 100
    pnl_trades = [t for t in trades if t["type"] not in ("final_long", "final_short")]
    wins = [t for t in pnl_trades if t["pnl"] > 0]
    win_rate = len(wins) / max(1, len(pnl_trades)) * 100
    
    if equity_curve and len(equity_curve) > 1:
        peak = equity_curve[0]
        max_dd = 0
        for e in equity_curve:
            peak = max(peak, e)
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)
    else:
        max_dd = 0
    
    # Sharpe
    if len(equity_curve) > 10:
        rets = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] for i in range(1, len(equity_curve))]
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        std_r = math.sqrt(var) if var > 0 else 0.001
        sharpe = mean_r / std_r * math.sqrt(365 * 24)  # Annualized (1h bars)
    else:
        sharpe = 0
    
    return {
        "coin": coin,
        "total_return_pct": round(total_return, 2),
        "num_trades": len(pnl_trades),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 1),
        "final_capital": round(final_capital, 2),
        "total_pnl": round(sum(t["pnl"] for t in pnl_trades), 2),
        "avg_win": round(sum(t["pnl_pct"] for t in wins) / max(1, len(wins)), 2),
        "avg_loss": round(sum(t["pnl_pct"] for t in pnl_trades if t["pnl"] <= 0) / max(1, len(pnl_trades) - len(wins)), 2),
        "data_hours": len(candles),
    }


# ── Grid Search Optimizer ────────────────────────────────────────────────────
def grid_search(coin: str, candles: list[dict]):
    """Find optimal parameters via grid search."""
    best_result = None
    best_return = -999
    
    for min_conv in [55, 60, 62, 65, 68, 70]:
        for max_pos in [0.05, 0.08, 0.10]:
            for max_lev in [2, 3]:
                for stop_atr in [1.5, 2.0, 2.5]:
                    for tp_r in [1.5, 2.0, 2.5]:
                        result = simulate_coin(coin, candles,
                            min_conviction=min_conv,
                            max_position_pct=max_pos,
                            max_leverage=max_lev,
                            stop_atr_mult=stop_atr,
                            tp_r_multiple=tp_r)
                        
                        # Score: return + sharpe penalty for low WR
                        score = result["total_return_pct"] + result["sharpe"] * 5
                        if result["win_rate"] < 35:
                            score -= 10  # Penalty for terrible WR
                        if result["num_trades"] < 5:
                            score -= 20  # Penalty for too few trades
                        
                        if score > best_return:
                            best_return = score
                            best_result = {**result, "params": {
                                "min_conviction": min_conv, "max_position_pct": max_pos,
                                "max_leverage": max_lev, "stop_atr": stop_atr, "tp_r": tp_r,
                            }}
    
    return best_result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 75)
    print("FULL-SYSTEM SIMULATION + OPTIMIZER — 30 DAYS (1h candles)")
    print("=" * 75)
    
    all_results = []
    optimal_params = {}
    
    for coin in COINS:
        print(f"\n{coin}: fetching 720h of 1h candles...", end=" ", flush=True)
        candles = fetch_1h_history(coin, 720)
        if len(candles) < 200:
            print(f"ONLY {len(candles)} — skipping")
            continue
        print(f"{len(candles)} candles (~{len(candles)//24}d)")
        
        print(f"  Grid search...", end=" ", flush=True)
        t0 = time.time()
        best = grid_search(coin, candles)
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s ({elapsed/len(COINS):.0f}s/coin)")
        
        if best:
            all_results.append(best)
            optimal_params[coin] = best["params"]
            
            emoji = "🟢" if best["total_return_pct"] > 0 else "🔴"
            print(f"  {emoji} Optimal: Ret={best['total_return_pct']:+.1f}% "
                  f"Sharpe={best['sharpe']:+.2f} WR={best['win_rate']:.0f}% "
                  f"DD={best['max_drawdown']:.1f}% Trades={best['num_trades']}")
            print(f"       Params: conv≥{best['params']['min_conviction']} "
                  f"pos={best['params']['max_position_pct']*100:.0f}% "
                  f"lev={best['params']['max_leverage']}x "
                  f"stop={best['params']['stop_atr']}xATR "
                  f"tp={best['params']['tp_r']}R")
    
    # Summary
    if all_results:
        print("\n" + "=" * 75)
        print("AGGREGATE RESULTS")
        print("=" * 75)
        avg_ret = sum(r["total_return_pct"] for r in all_results) / len(all_results)
        avg_sharpe = sum(r["sharpe"] for r in all_results) / len(all_results)
        avg_wr = sum(r["win_rate"] for r in all_results) / len(all_results)
        total_pnl = sum(r["total_pnl"] for r in all_results)
        
        for r in sorted(all_results, key=lambda x: x["total_return_pct"], reverse=True):
            p = r["params"]
            emoji = "🟢" if r["total_return_pct"] > 0 else "🔴"
            print(f"  {emoji} {r['coin']:6s} | Ret:{r['total_return_pct']:+6.1f}% "
                  f"Sharpe:{r['sharpe']:+5.2f} WR:{r['win_rate']:4.0f}% "
                  f"DD:{r['max_drawdown']:5.1f}% T:{r['num_trades']:2d} "
                  f"[c≥{p['min_conviction']} {p['max_position_pct']*100:.0f}% {p['max_leverage']}x]")
        
        print(f"\n  AVERAGE: Ret={avg_ret:+.1f}% Sharpe={avg_sharpe:+.2f} WR={avg_wr:.0f}%")
        print(f"  TOTAL PnL: ${total_pnl:+.2f}")
        
        # Consensus optimal params
        if len(optimal_params) >= 2:
            consensus_conv = round(sum(p["min_conviction"] for p in optimal_params.values()) / len(optimal_params))
            consensus_pos = sum(p["max_position_pct"] for p in optimal_params.values()) / len(optimal_params)
            consensus_lev = round(sum(p["max_leverage"] for p in optimal_params.values()) / len(optimal_params))
            consensus_stop = sum(p["stop_atr"] for p in optimal_params.values()) / len(optimal_params)
            consensus_tp = sum(p["tp_r"] for p in optimal_params.values()) / len(optimal_params)
            
            print(f"\n  ★ CONSENSUS OPTIMAL PARAMS:")
            print(f"    min_conviction = {consensus_conv}")
            print(f"    max_position_pct = {consensus_pos:.2f}")
            print(f"    max_leverage = {consensus_lev}")
            print(f"    stop_atr_mult = {consensus_stop:.1f}")
            print(f"    tp_r_multiple = {consensus_tp:.1f}")
            
            print(f"\n  📋 Apply to daemon:")
            print(f"    ai_decider.py: set tier_config['micro'] = {{")
            print(f"      'max_pos_pct': {consensus_pos:.2f},")
            print(f"      'max_lev': {consensus_lev},")
            print(f"      'risk_pct': 0.015")
            print(f"    }}")
            print(f"    hyperliquid_daemon.py: min_conviction = {consensus_conv}")
