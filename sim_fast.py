#!/usr/bin/env python3
"""Fast simulation — real HL data, 5-pillar composite signal, ~3s per coin."""
import json, time, urllib.request, sys
import numpy as np
from collections import defaultdict

def run(coin="BTC", eq=10000, lev=3, step=4, min_comp=0.12, max_pos=2):
    # Fetch
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps({"type":"candleSnapshot","req":{"coin":coin,"interval":"1h","startTime":0,"endTime":9999999999999}}).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())

    # Build DataFrame with basic indicators
    import pandas as pd
    from hyperliquid_strategy import candles_to_frame, add_basic_indicators, compute_composite, detect_regime
    df = add_basic_indicators(candles_to_frame(data))
    n = len(df)

    idxs = list(range(200, n, step))
    pos_list = []  # [{long,entry,stop,tp,size,notional,regime,entry_i}]
    exits = []     # [{pnl,type,idx}]
    entries = []   # [{side,regime,composite,entry_i}]
    eq_curve = [eq]
    max_eq = eq
    blocked_count = {"weak_signal": 0, "max_pos": 0, "tiny_size": 0}

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
                eq += (st-ent)/ent*nl; exits.append({"pnl":(st-ent)/ent*nl,"type":"stop","idx":i}); x=True
            elif not L and hi_c >= st:
                eq += (ent-st)/ent*nl; exits.append({"pnl":(ent-st)/ent*nl,"type":"stop","idx":i}); x=True
            elif L and hi_c >= tp:
                eq += (tp-ent)/ent*nl; exits.append({"pnl":(tp-ent)/ent*nl,"type":"tp","idx":i}); x=True
            elif not L and lo_c <= tp:
                eq += (ent-tp)/ent*nl; exits.append({"pnl":(ent-tp)/ent*nl,"type":"tp","idx":i}); x=True
            if not x:
                surviving.append(p)
        pos_list = surviving
        max_eq = max(max_eq, eq)

        # ENTRIES
        comp = compute_composite(w)
        reg, _ = detect_regime(w)
        if abs(comp) < min_comp:
            blocked_count["weak_signal"] += 1
            eq_curve.append(eq)
            continue
        if len(pos_list) >= max_pos:
            blocked_count["max_pos"] += 1
            eq_curve.append(eq)
            continue

        side = "BUY" if comp > 0 else "SELL"
        atr = float(w.iloc[-1].get("atr14", lo*0.015)) or lo*0.015

        notional = eq * 0.05
        sz = notional / lo
        if side == "BUY":
            stop = lo - atr * 1.5
            tp = lo + atr * 2.5
        else:
            stop = lo + atr * 1.5
            tp = lo - atr * 2.5

        if stop <= 0 or notional < 5:
            blocked_count["tiny_size"] += 1
            eq_curve.append(eq)
            continue

        pos_list.append({"long": side=="BUY","entry":lo,"stop":stop,"tp":tp,
                        "size":sz,"notional":notional,"regime":reg.value,"entry_i":i})
        entries.append({"side": side, "regime": reg.value, "composite": comp, "entry_i": i})
        eq_curve.append(eq)

    # Close remaining
    lc = float(df.iloc[-1]["close"])
    for p in pos_list:
        pnl = (lc-p["entry"])/p["entry"]*p["notional"] if p["long"] else (p["entry"]-lc)/p["entry"]*p["notional"]
        exits.append({"pnl": pnl, "type": "final", "idx": n-1})
        eq += pnl
    eq_curve.append(eq)

    # === METRICS ===
    ep = [e["pnl"] for e in exits]
    wins = [p for p in ep if p > 0]
    losses = [p for p in ep if p < 0]
    total_pnl = sum(ep)
    wr = len(wins)/len(ep) if ep else 0
    pf = sum(wins)/abs(sum(losses)) if losses and sum(losses)!=0 else 0
    dd = max((max_eq - min(eq_curve))/max_eq, 0)*100

    if len(ep) > 1:
        mu = np.mean(ep); si = np.std(ep)
        sh = (mu/si)*np.sqrt(8760) if si>0 else 0
    else:
        sh = 0

    # Regime stats
    reg_stats = defaultdict(lambda: {"cnt":0,"pnl":0,"w":0})
    for j, ex in enumerate(exits):
        ei = ex.get("idx", 0)
        best = None
        for en in entries:
            if en["entry_i"] < ei or best is None:
                best = en
        reg = best["regime"] if best else "unknown"
        reg_stats[reg]["cnt"] += 1
        reg_stats[reg]["pnl"] += ex["pnl"]
        if ex["pnl"] > 0:
            reg_stats[reg]["w"] += 1

    return {
        "coin": coin, "candles": n, "tested": len(idxs),
        "entries": len(entries), "exits": len(exits),
        "win_rate": round(wr,3), "profit_factor": round(pf,3),
        "total_pnl": round(total_pnl,2), "return_pct": round(total_pnl/eq*100 if hasattr(locals(),'eq') else 0,2),
        "max_drawdown_pct": round(dd,2), "sharpe": round(sh,3),
        "final_equity": round(eq,2),
        "avg_win": round(np.mean(wins),2) if wins else 0,
        "avg_loss": round(abs(np.mean(losses)),2) if losses else 0,
        "regimes": {k: {"trades":v["cnt"],"pnl":round(v["pnl"],2),
                        "wr":round(v["w"]/v["cnt"]*100,1) if v["cnt"] else 0}
                    for k,v in reg_stats.items()},
        "blocked": blocked_count,
    }


if __name__ == "__main__":
    coins = sys.argv[1:] if len(sys.argv)>1 else ["BTC","ETH","SOL"]
    t0 = time.time()
    results = {}
    
    for c in coins:
        r = run(c)
        results[c] = r
        print(f"\n{'─'*40}")
        print(f"{c}: {r['candles']} candles, {r['tested']} tested")
        print(f"{'─'*40}")
        print(f"  Entries:{r['entries']}  Exits:{r['exits']}  WR:{r['win_rate']*100:.1f}%")
        print(f"  PnL: ${r['total_pnl']:+.2f} ({r['return_pct']:+.2f}%)  PF:{r['profit_factor']:.2f}")
        print(f"  MaxDD:{r['max_drawdown_pct']:.1f}%  Sharpe:{r['sharpe']:.3f}")
        print(f"  AvgWin:${r['avg_win']:+.2f}  AvgLoss:${r['avg_loss']:+.2f}")
        print(f"  Final: ${r['final_equity']:,.2f}")
        if r.get("regimes"):
            print(f"  Regimes:")
            for reg,st in r["regimes"].items():
                print(f"    {reg}: {st['trades']}t ${st['pnl']:+.0f} {st['wr']:.0f}%WR")
        print(f"  Blocked: {r['blocked']}")

    tot_pnl = sum(v["total_pnl"] for v in results.values())
    tot_trades = sum(v["exits"] for v in results.values())
    avg_wr = sum(v["win_rate"] for v in results.values())/len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({time.time()-t0:.1f}s) — {len(coins)} coins, {5000*len(coins)} candles")
    print(f"  Total PnL: ${tot_pnl:+.2f}")
    print(f"  Total Trades: {tot_trades}")
    print(f"  Avg Win Rate: {avg_wr*100:.1f}%")
