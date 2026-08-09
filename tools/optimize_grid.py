#!/usr/bin/env python3
"""Grid-search optimizer for 5-pillar composite strategy on real HL data.
Tests 200+ parameter combinations, finds optimal settings per coin and globally."""

import json, time, urllib.request, sys
import numpy as np
from collections import defaultdict

def fetch_data(coin="BTC"):
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps({"type":"candleSnapshot","req":{"coin":coin,"interval":"1h","startTime":0,"endTime":9999999999999}}).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def run_one(coin_data, df, params):
    """Run single parameter combo. Returns (pnl, wr, pf, dd, sharpe, trades)."""
    min_comp = params["min_comp"]
    stop_atr = params["stop_atr"]
    tp_atr = params["tp_atr"]
    pos_pct = params["pos_pct"]
    max_pos = params["max_pos"]
    step = params.get("step", 4)
    eq = params.get("equity", 10000)
    
    n = len(df)
    idxs = list(range(200, n, step))
    
    pos_list = []
    exits = []
    entries = []
    eq_curve = [eq]
    max_eq = eq
    
    for i in idxs:
        w = df.iloc[i-200:i+1]
        lo = float(df.iloc[i]["close"])
        hi_c = float(df.iloc[i]["high"])
        lo_c = float(df.iloc[i]["low"])
        
        # EXITS
        surviving = []
        for p in pos_list:
            ent = p["entry"]; st = p["stop"]; tp = p["tp"]
            nl = p["notional"]; L = p["long"]
            x = False
            if L and lo_c <= st:
                eq += (st-ent)/ent*nl; exits.append({"pnl":(st-ent)/ent*nl,"type":"stop"}); x=True
            elif not L and hi_c >= st:
                eq += (ent-st)/ent*nl; exits.append({"pnl":(ent-st)/ent*nl,"type":"stop"}); x=True
            elif L and hi_c >= tp:
                eq += (tp-ent)/ent*nl; exits.append({"pnl":(tp-ent)/ent*nl,"type":"tp"}); x=True
            elif not L and lo_c <= tp:
                eq += (ent-tp)/ent*nl; exits.append({"pnl":(ent-tp)/ent*nl,"type":"tp"}); x=True
            if not x:
                surviving.append(p)
        pos_list = surviving
        max_eq = max(max_eq, eq)
        
        # Composite signal
        from hyperliquid_strategy import compute_composite
        comp = compute_composite(w)
        
        if abs(comp) < min_comp:
            eq_curve.append(eq)
            continue
        if len(pos_list) >= max_pos:
            eq_curve.append(eq)
            continue
        
        side = "BUY" if comp > 0 else "SELL"
        atr = float(w.iloc[-1].get("atr14", lo*0.015)) or lo*0.015
        
        notional = eq * pos_pct
        sz = notional / lo if lo > 0 else 0
        
        if side == "BUY":
            stop = lo - atr * stop_atr
            tp = lo + atr * tp_atr
        else:
            stop = lo + atr * stop_atr
            tp = lo - atr * tp_atr
        
        if stop <= 0 or notional < 5:
            eq_curve.append(eq)
            continue
        
        pos_list.append({"long": side=="BUY","entry":lo,"stop":stop,"tp":tp,
                        "size":sz,"notional":notional})
        entries.append({"side": side})
        eq_curve.append(eq)
    
    # Close remaining
    lc = float(df.iloc[-1]["close"])
    for p in pos_list:
        pnl = (lc-p["entry"])/p["entry"]*p["notional"] if p["long"] else (p["entry"]-lc)/p["entry"]*p["notional"]
        exits.append({"pnl": pnl, "type": "final"})
        eq += pnl
    
    # Metrics
    ep = [e["pnl"] for e in exits]
    if not ep:
        return 0, 0, 0, 0, 0, 0
    
    total_pnl = sum(ep)
    wins = [p for p in ep if p > 0]
    losses = [p for p in ep if p < 0]
    wr = len(wins)/len(ep)
    pf = sum(wins)/abs(sum(losses)) if losses and sum(losses)!=0 else 0
    dd = max((max_eq - min(eq_curve))/max_eq, 0)*100 if eq_curve else 0
    
    if len(ep) > 1:
        mu = np.mean(ep); si = np.std(ep)
        sh = (mu/si)*np.sqrt(8760) if si>0 else 0
    else:
        sh = 0
    
    # Composite score: weight PnL + PF + WR, penalize DD
    score = total_pnl * 0.4 + pf * 100 * 0.3 + wr * 100 * 0.2 - dd * 0.1
    return total_pnl, wr, pf, dd, sh, len(ep), score


def grid_search(coin="BTC"):
    """Run grid search across parameter space."""
    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {coin}")
    print(f"{'='*60}")
    
    data = fetch_data(coin)
    from hyperliquid_strategy import candles_to_frame, add_basic_indicators
    df = add_basic_indicators(candles_to_frame(data))
    print(f"  {len(df)} candles, pre-computed indicators")
    
    # Parameter grid
    grid = {
        "min_comp": [0.06, 0.08, 0.10, 0.12, 0.15],
        "stop_atr": [1.0, 1.5, 2.0, 2.5],
        "tp_atr": [1.5, 2.0, 2.5, 3.0],
        "pos_pct": [0.03, 0.05, 0.08],
        "max_pos": [1, 2],
    }
    
    total = 1
    for v in grid.values():
        total *= len(v)
    
    results = []
    tested = 0
    t0 = time.time()
    
    for min_comp in grid["min_comp"]:
        for stop_atr in grid["stop_atr"]:
            for tp_atr in grid["tp_atr"]:
                for pos_pct in grid["pos_pct"]:
                    for max_pos in grid["max_pos"]:
                        params = {
                            "min_comp": min_comp, "stop_atr": stop_atr,
                            "tp_atr": tp_atr, "pos_pct": pos_pct,
                            "max_pos": max_pos, "step": 6,  # faster: every 6th candle
                        }
                        pnl, wr, pf, dd, sh, trades, score = run_one(data, df, params)
                        results.append({
                            **params, "pnl": round(pnl,2), "wr": round(wr,3),
                            "pf": round(pf,3), "dd": round(dd,2), "sharpe": round(sh,3),
                            "trades": trades, "score": round(score,2),
                        })
                        tested += 1
                        if tested % 40 == 0:
                            elapsed = time.time() - t0
                            print(f"  {tested}/{total} ({elapsed:.0f}s) — best so far: score={max(r['score'] for r in results):.1f}")
    
    # Sort by score
    results.sort(key=lambda r: r["score"], reverse=True)
    
    print(f"\n  TOP 10 PARAMETER COMBOS:")
    print(f"  {'Rank':<5} {'Score':<8} {'PnL':<10} {'WR':<8} {'PF':<8} {'DD':<8} {'Tr':<5} {'min_comp':<10} {'stop':<8} {'tp':<8} {'pos':<8} {'max_p':<6}")
    print(f"  {'-'*90}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:<5} {r['score']:<8.1f} ${r['pnl']:<9.2f} {r['wr']*100:<7.1f}% {r['pf']:<8.2f} {r['dd']:<7.1f}% {r['trades']:<5} "
              f"{r['min_comp']:<10.2f} {r['stop_atr']:<7.1f}x {r['tp_atr']:<7.1f}x {r['pos_pct']*100:<7.0f}% {r['max_pos']:<6}")
    
    return results


if __name__ == "__main__":
    import sys
    coins = sys.argv[1:] if len(sys.argv)>1 else ["BTC","ETH","SOL"]
    
    all_results = {}
    for c in coins:
        all_results[c] = grid_search(c)
    
    # Best params per coin
    print(f"\n{'='*60}")
    print(f"OPTIMAL PARAMETERS PER COIN")
    print(f"{'='*60}")
    for c, results in all_results.items():
        best = results[0]
        print(f"  {c}: comp>{best['min_comp']:.2f} stop={best['stop_atr']:.1f}x tp={best['tp_atr']:.1f}x "
              f"pos={best['pos_pct']*100:.0f}% max={best['max_pos']}p → "
              f"PnL=${best['pnl']:+.0f} WR={best['wr']*100:.0f}% PF={best['pf']:.2f} DD={best['dd']:.1f}% "
              f"score={best['score']:.1f}")
