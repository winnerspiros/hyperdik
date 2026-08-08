#!/usr/bin/env python3
"""
Hyperliquid Backtest Engine v3 — Runs the full 5-pillar composite + ensemble
strategy through real Hyperliquid candle data with simulated trading.

Measures: Win rate, profit factor, max drawdown, Sharpe, regime performance,
conviction tier performance, and per-coin breakdown.

Usage:
  python3 hyperliquid_backtest.py           # Run full simulation
  python3 hyperliquid_backtest.py --coin BTC # Single coin
  python3 hyperliquid_backtest.py --quick    # Quick test (last 500 candles)
"""

import json
import os
import sys
import time
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import urllib.request
import pandas as pd
import numpy as np

from hyperliquid_strategy import (
    generate_master_signal, compute_composite, detect_regime,
    candles_to_frame, add_basic_indicators,
    MasterSignal, MarketRegime, SignalSide, PillarWeights,
    score_conviction, ConvictionTier, get_tier_sizing,
    MIN_CONVICTION_SCORE,
)
from hyperliquid_risk import (
    HSLState, ExposureState, CooldownState,
    full_risk_check, is_funding_window,
    compute_portfolio_heat, is_portfolio_heat_safe,
)
from hyperliquid_execution import (
    build_exit_plan, get_regime_stop_mult,
)


# ============================================================
# Data Fetch
# ============================================================

def fetch_hl_candles(coin: str, interval: str = "1h") -> list[dict]:
    """Fetch candles from Hyperliquid API."""
    payload = json.dumps({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": 0, "endTime": 9999999999999}
    }).encode()

    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    return data if isinstance(data, list) else []


# ============================================================
# Simulation Engine
# ============================================================

class BacktestEngine:
    """Walk-forward backtest through real candle data."""

    def __init__(self, equity: float = 10000.0, leverage: int = 3,
                 max_positions: int = 3):
        self.initial_equity = equity
        self.equity = equity
        self.leverage = leverage
        self.max_positions = max_positions
        self.positions: dict[str, dict] = {}  # coin → {entry, size, side, stop, tps}
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.cooldown = CooldownState(cooldown_seconds=1)  # 1 candle cooldown
        self.hsl = HSLState()
        self.exposure = ExposureState()

        # Stats
        self.signals_generated = 0
        self.trades_blocked = defaultdict(int)
        self.regime_performance: dict[str, list[dict]] = defaultdict(list)
        self.tier_performance: dict[str, list[dict]] = defaultdict(list)

    def step(self, coin: str, candles: list[dict], btc_candles: list[dict] | None = None,
             timestamp: int = 0) -> dict | None:
        """Process one candle step. Returns trade dict if executed."""

        if len(candles) < 55:
            return None

        # Generate signal
        sig = generate_master_signal(
            symbol=coin,
            candles=candles,
            trades=self.trades[-20:] if self.trades else None,
            btc_candles=btc_candles,
            equity=self.equity,
        )
        self.signals_generated += 1

        if sig.side not in ("BUY", "SELL"):
            self.trades_blocked["no_signal"] += 1
            return None

        # Already have position?
        if coin in self.positions:
            self.trades_blocked["position_exists"] += 1
            return None

        # Conviction scoring
        conviction_score, conviction_tier, breakdown = score_conviction(
            sig, sig.composite_score, sig.regime,
            bb_width=0.03,  # placeholder
        )

        if conviction_tier == ConvictionTier.NO_TRADE:
            self.trades_blocked["low_conviction"] += 1
            return None

        pos_pct, risk_pct, stop_atr = get_tier_sizing(conviction_tier)

        # Entry/stop calculation
        close = float(candles[-1].get("c", 0))
        atr = sig.atr or (close * 0.015)
        entry_price = sig.entry_price or close
        stop_price = sig.stop_price or (entry_price * 0.97)

        # Position size (fraction of equity at risk_pct)
        notional = self.equity * pos_pct
        size_units = notional / entry_price if entry_price > 0 else 0

        if size_units <= 0 or notional < 5:
            self.trades_blocked["too_small"] += 1
            return None

        # Simulate trade
        trade = {
            "coin": coin,
            "side": sig.side,
            "entry_price": entry_price,
            "size_units": size_units,
            "notional": notional,
            "stop_price": stop_price,
            "leverage": self.leverage,
            "regime": sig.regime.value,
            "conviction_score": conviction_score,
            "conviction_tier": conviction_tier,
            "entry_time": timestamp,
            "confidence": sig.confidence,
            "composite": sig.composite_score,
            "reason": sig.reason[:80],
        }

        # Multi-tier TP levels
        from hyperliquid_execution import calculate_tp_levels, TierConfig
        tps = calculate_tp_levels(
            sig.side == "BUY", entry_price, stop_price,
            self.leverage, atr,
            [TierConfig(0.333, 0.30, "TP1"), TierConfig(0.333, 0.50, "TP2"),
             TierConfig(0.334, 1.20, "TP3")]
        )
        trade["tp_levels"] = [{"price": t["price"], "fraction": t["fraction"]} for t in tps]

        self.positions[coin] = trade
        self.cooldown.record(coin)

        return trade

    def check_exits(self, coin: str, candle: dict) -> dict | None:
        """Check if position should be exited. Returns single exit or None."""
        if coin not in self.positions:
            return None

        pos = self.positions[coin]
        high = float(candle.get("h", 0))
        low = float(candle.get("l", 0))
        close = float(candle.get("c", 0))

        is_long = pos["side"] == "BUY"
        entry = pos["entry_price"]
        stop = pos["stop_price"]
        size = pos["size_units"]
        notional = pos["notional"]

        # Check stop loss first
        if is_long and low <= stop:
            pnl = (stop - entry) / entry * notional
            del self.positions[coin]
            return {"type": "stop_loss", "price": stop, "pnl": pnl, "coin": coin}
        elif not is_long and high >= stop:
            pnl = (entry - stop) / entry * notional
            del self.positions[coin]
            return {"type": "stop_loss", "price": stop, "pnl": pnl, "coin": coin}

        # Check TP — pick the BEST TP that was hit
        best_tp = None
        for tp in pos.get("tp_levels", []):
            tp_price = tp["price"]
            hit = (is_long and high >= tp_price) or (not is_long and low <= tp_price)
            if hit:
                if best_tp is None:
                    best_tp = tp
                elif is_long and tp_price > best_tp["price"]:
                    best_tp = tp
                elif not is_long and tp_price < best_tp["price"]:
                    best_tp = tp

        if best_tp:
            pnl_pct = (best_tp["price"] - entry) / entry if is_long else (entry - best_tp["price"]) / entry
            pnl = pnl_pct * notional
            del self.positions[coin]
            return {"type": "take_profit", "price": best_tp["price"], "pnl": pnl, "coin": coin}

        return None

    def run(self, coin: str, candles: list[dict], btc_candles: list[dict] | None = None,
            warmup: int = 100) -> dict:
        """Run full backtest on a coin's candle data."""
        self.equity = self.initial_equity
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_equity]
        self.hsl = HSLState()
        self.hsl.update(self.equity)
        self.exposure = ExposureState()

        all_trades = []

        for i in range(warmup, len(candles)):
            window = candles[i - 200:i + 1] if i >= 200 else candles[:i + 1]
            btc_window = btc_candles[i - 200:i + 1] if btc_candles and i >= 200 else None

            candle = candles[i]
            ts = candle.get("t", 0)

            # Check exits first
            for coin_name in list(self.positions.keys()):
                exit_trade = self.check_exits(coin_name, candle)
                if exit_trade:
                    exit_trade["exit_time"] = ts
                    self.equity += exit_trade["pnl"]
                    all_trades.append(exit_trade)
                    self.hsl.update(self.equity)

            # Try to enter
            if len(self.positions) < self.max_positions:
                trade = self.step(coin, window, btc_window, ts)
                if trade:
                    all_trades.append(trade)

            self.equity_curve.append(self.equity)

        return self._compute_metrics(coin, all_trades)

    def _compute_metrics(self, coin: str, trades: list[dict]) -> dict:
        """Compute performance metrics."""
        if not trades:
            return {"coin": coin, "trades": 0, "win_rate": 0, "profit_factor": 0,
                    "total_pnl": 0, "max_drawdown": 0, "sharpe": 0}

        # Split trades
        entries = [t for t in trades if "entry_price" in t and "side" in t]
        exits = [t for t in trades if "type" in t]

        pnls = [e["pnl"] for e in exits]
        total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0

        # Drawdown
        cum = np.cumsum(pnls) + self.initial_equity if pnls else np.array([self.initial_equity])
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        # Sharpe (annualized, hourly)
        if len(pnls) > 1:
            mu = np.mean(pnls)
            sigma = np.std(pnls)
            sharpe = (mu / sigma) * np.sqrt(8760) if sigma > 0 else 0
        else:
            sharpe = 0

        # Regime breakdown
        regime_stats = {}
        for t in entries:
            reg = t.get("regime", "unknown")
            if reg not in regime_stats:
                regime_stats[reg] = {"signals": 0, "trades": 0}
            regime_stats[reg]["signals"] += 1

        for e in exits:
            # Match exit to entry (simplified: use coin)
            entry = next((t for t in entries if t["coin"] == e["coin"]), None)
            if entry:
                reg = entry.get("regime", "unknown")
                if reg in regime_stats:
                    regime_stats[reg]["trades"] += 1

        return {
            "coin": coin,
            "signals": self.signals_generated,
            "entries": len(entries),
            "exits": len(exits),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 3),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 3),
            "return_pct": round(total_pnl / self.initial_equity * 100, 2),
            "final_equity": round(self.equity, 2),
            "equity_curve": self.equity_curve[-100:],  # last 100 points
            "regime_stats": regime_stats,
            "blocked_reasons": dict(self.trades_blocked),
        }


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--coin", default=None, help="Single coin to test")
    p.add_argument("--quick", action="store_true", help="Quick test (last 500 candles)")
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--leverage", type=int, default=3)
    args = p.parse_args()

    coins = [args.coin] if args.coin else ["BTC", "ETH", "SOL"]
    results = {}

    print("=" * 60)
    print(f"HYPERLIQUID BACKTEST v3 — {len(coins)} coins, {args.leverage}x leverage")
    print(f"  Equity: ${args.equity:,.0f}")
    print(f"  Using REAL Hyperliquid 1h candle data")
    print("=" * 60)

    for coin in coins:
        print(f"\n🔄 Fetching {coin} data...")
        candles = fetch_hl_candles(coin, "1h")
        print(f"   {len(candles)} candles loaded")

        # Quick mode: last 500 only
        if args.quick:
            candles = candles[-500:]
            print(f"   Quick mode: using last {len(candles)}")

        btc_candles = None
        if coin != "BTC":
            print(f"   Fetching BTC reference data...")
            btc_candles = fetch_hl_candles("BTC", "1h")

        print(f"   Running simulation...")
        engine = BacktestEngine(equity=args.equity, leverage=args.leverage)
        result = engine.run(coin, candles, btc_candles)
        results[coin] = result

        print(f"\n{'─' * 40}")
        print(f"RESULTS: {coin}")
        print(f"{'─' * 40}")
        print(f"  Signals generated: {result['signals']}")
        print(f"  Entries: {result['entries']} | Exits: {result['exits']}")
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")
        print(f"  Total PnL: ${result['total_pnl']:+.2f}")
        print(f"  Return: {result['return_pct']:+.2f}%")
        print(f"  Max Drawdown: {result['max_drawdown_pct']:.1f}%")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        print(f"  Final Equity: ${result['final_equity']:,.2f}")
        print(f"  Avg Win: ${result['avg_win']:+.2f} | Avg Loss: ${result['avg_loss']:+.2f}")
        if result.get("regime_stats"):
            print(f"  Regime breakdown:")
            for reg, stats in result["regime_stats"].items():
                print(f"    {reg}: {stats['signals']} signals, {stats['trades']} trades")
        if result.get("blocked_reasons"):
            print(f"  Blocked reasons:")
            for reason, count in sorted(result["blocked_reasons"].items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")

    # Summary
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"PORTFOLIO SUMMARY")
        print(f"{'=' * 60}")
        total_pnl = sum(r["total_pnl"] for r in results.values())
        total_trades = sum(r["exits"] for r in results.values())
        avg_wr = sum(r["win_rate"] for r in results.values()) / len(results)
        print(f"  Total PnL: ${total_pnl:+.2f}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Avg Win Rate: {avg_wr*100:.1f}%")
        print(f"  Return: {total_pnl/args.equity*100:+.2f}%")

    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/backtest_results.json", "w") as f:
        # Remove equity curves for file size
        clean = {}
        for k, v in results.items():
            clean[k] = {kk: vv for kk, vv in v.items() if kk != "equity_curve"}
        json.dump(clean, f, indent=2, default=str)
    print(f"\n📁 Results saved: data/backtest_results.json")


if __name__ == "__main__":
    main()
