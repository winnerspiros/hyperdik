"""
Hyperliquid Self-Learning Engine — learns from every trade to improve future decisions.

HOW IT WORKS:
  1. Records every trade outcome: entry price, exit price, PnL, signal scores
  2. Analyzes which signal combinations predict wins vs losses
  3. Dynamically adjusts per-coin strategy weights and conviction thresholds
  4. Feeds back into the daemon's signal pipeline as "learned adjustments"

NO SQLite dependency — uses simple JSON files for persistence.
"""

import json
import os
import time
import math
from dataclasses import dataclass, field
from typing import Optional


TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "hyperliquid_trades.json")
LEARNED_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "learned_weights.json")


@dataclass
class TradeRecord:
    """One completed trade for learning."""
    coin: str
    side: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    pnl: float  # USD
    pnl_pct: float  # % return
    entry_time: float
    exit_time: float
    regime: str  # trending_up/down/sideways/volatile
    conviction: float  # 0-100
    composite_score: float
    leverage: float
    # Signal component scores at entry
    trend_score: float = 0
    momentum_score: float = 0
    volume_score: float = 0
    btc_corr_score: float = 0
    funding_score: float = 0
    reason: str = ""


def record_trade(trade: TradeRecord) -> None:
    """Append a completed trade to the learning log."""
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    trades = []
    if os.path.exists(TRADE_LOG_PATH):
        try:
            with open(TRADE_LOG_PATH) as f:
                trades = json.load(f)
        except Exception:
            trades = []

    trades.append({
        "coin": trade.coin,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl": round(trade.pnl, 4),
        "pnl_pct": round(trade.pnl_pct, 4),
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "regime": trade.regime,
        "conviction": trade.conviction,
        "composite_score": trade.composite_score,
        "leverage": trade.leverage,
        "trend_score": trade.trend_score,
        "momentum_score": trade.momentum_score,
        "volume_score": trade.volume_score,
        "btc_corr_score": trade.btc_corr_score,
        "funding_score": trade.funding_score,
        "reason": trade.reason,
    })

    # Keep last 500 trades max
    if len(trades) > 500:
        trades = trades[-500:]

    with open(TRADE_LOG_PATH, "w") as f:
        json.dump(trades, f, indent=2)


def analyze_and_learn() -> dict:
    """
    Analyze all past trades and generate learned adjustments.
    Returns dict with per-coin weight adjustments and conviction modifiers.
    """
    if not os.path.exists(TRADE_LOG_PATH):
        return {"ready": False, "total_trades": 0}

    try:
        with open(TRADE_LOG_PATH) as f:
            trades = json.load(f)
    except Exception:
        return {"ready": False, "total_trades": 0}

    if len(trades) < 5:
        return {"ready": False, "total_trades": len(trades)}

    # ── Per-coin analysis ──
    coin_stats = {}
    for t in trades:
        coin = t["coin"]
        if coin not in coin_stats:
            coin_stats[coin] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        coin_stats[coin]["total_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            coin_stats[coin]["wins"] += 1
        else:
            coin_stats[coin]["losses"] += 1

    # ── Per-regime analysis ──
    regime_stats = {}
    for t in trades:
        regime = t["regime"]
        if regime not in regime_stats:
            regime_stats[regime] = {"wins": 0, "total": 0}
        regime_stats[regime]["total"] += 1
        if t["pnl"] > 0:
            regime_stats[regime]["wins"] += 1

    # ── Signal weight analysis ──
    # Which signals correlate with wins?
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]

    signal_keys = ["trend_score", "momentum_score", "volume_score", "btc_corr_score", "funding_score"]
    signal_adjustments = {}
    for key in signal_keys:
        win_avg = sum(t.get(key, 0) for t in winning_trades) / max(len(winning_trades), 1)
        lose_avg = sum(t.get(key, 0) for t in losing_trades) / max(len(losing_trades), 1)
        diff = win_avg - lose_avg
        # Positive diff = higher scores on this signal → wins → increase weight
        # Negative diff = higher scores → losses → decrease weight
        signal_adjustments[key] = round(diff, 4)

    # ── Conviction threshold analysis ──
    # What conviction level predicts wins?
    win_convs = [t["conviction"] for t in winning_trades if t.get("conviction", 0) > 0]
    lose_convs = [t["conviction"] for t in losing_trades if t.get("conviction", 0) > 0]
    win_avg_conv = sum(win_convs) / max(len(win_convs), 1)
    lose_avg_conv = sum(lose_convs) / max(len(lose_convs), 1)

    # Suggested conviction adjustment
    if win_avg_conv > lose_avg_conv:
        suggested_min_conviction = max(50, lose_avg_conv + 5)
    else:
        suggested_min_conviction = 55  # default

    # ── Build learned weights ──
    learned = {
        "updated_at": time.time(),
        "total_trades": len(trades),
        "win_rate": round(len(winning_trades) / len(trades) * 100, 1),
        "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        "per_coin": {},
        "per_regime": {},
        "signal_adjustments": signal_adjustments,
        "suggested_min_conviction": suggested_min_conviction,
        "avoid_coins": [],
        "prefer_coins": [],
    }

    for coin, stats in coin_stats.items():
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total * 100 if total > 0 else 0
        learned["per_coin"][coin] = {
            "win_rate": round(wr, 1),
            "total_pnl": round(stats["total_pnl"], 2),
            "total_trades": total,
        }
        if total >= 3:
            if wr < 30:
                learned["avoid_coins"].append(coin)
            elif wr > 60:
                learned["prefer_coins"].append(coin)

    for regime, stats in regime_stats.items():
        wr = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
        learned["per_regime"][regime] = {
            "win_rate": round(wr, 1),
            "total": stats["total"],
        }

    # Save learned weights
    os.makedirs(os.path.dirname(LEARNED_WEIGHTS_PATH), exist_ok=True)
    with open(LEARNED_WEIGHTS_PATH, "w") as f:
        json.dump(learned, f, indent=2)

    learned["ready"] = True
    return learned


def get_learned_context() -> str:
    """
    Generate AI prompt context from learned patterns.
    Feed this into the daemon's AI context.
    """
    learned = analyze_and_learn()
    if not learned.get("ready"):
        return ""

    lines = ["📚 SELF-LEARNED PATTERNS:"]
    lines.append(f"  Total: {learned['total_trades']} trades | WR: {learned['win_rate']:.0f}% | PnL: ${learned['total_pnl']:+.2f}")

    if learned.get("avoid_coins"):
        lines.append(f"  🚫 AVOID: {', '.join(learned['avoid_coins'])} (WR < 30%)")
    if learned.get("prefer_coins"):
        lines.append(f"  ✅ PREFER: {', '.join(learned['prefer_coins'])} (WR > 60%)")

    lines.append(f"  Conviction floor: {learned.get('suggested_min_conviction', 55):.0f}/100")

    # Signal adjustments
    adj = learned.get("signal_adjustments", {})
    if adj:
        adjustments_str = []
        for key, val in adj.items():
            label = key.replace("_score", "").replace("_", " ").title()
            if abs(val) > 0.05:
                direction = "↑boost" if val > 0 else "↓reduce"
                adjustments_str.append(f"{label}:{direction}")
        if adjustments_str:
            lines.append(f"  Signal weights: {', '.join(adjustments_str)}")

    # Per-regime
    regimes = learned.get("per_regime", {})
    regime_lines = []
    for regime, stats in regimes.items():
        if stats["total"] >= 3:
            regime_lines.append(f"{regime}:{stats['win_rate']:.0f}%WR")
    if regime_lines:
        lines.append(f"  Regimes: {', '.join(regime_lines)}")

    return "\n".join(lines)


def get_learned_weights() -> dict:
    """Load learned weights from disk (fast, no analysis)."""
    if os.path.exists(LEARNED_WEIGHTS_PATH):
        try:
            with open(LEARNED_WEIGHTS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"ready": False}


# ── Self-test ──
if __name__ == "__main__":
    # Create some fake trades for testing
    import random
    random.seed(42)

    for i in range(20):
        win = random.random() > 0.55
        pnl = random.uniform(0.5, 3.0) if win else random.uniform(-3.0, -0.5)
        entry = random.uniform(50, 100)
        record_trade(TradeRecord(
            coin=random.choice(["BTC", "ETH", "SOL", "AVAX", "DOT"]),
            side="LONG",
            entry_price=entry,
            exit_price=entry * (1 + pnl / 100),
            pnl=pnl,
            pnl_pct=pnl,
            entry_time=time.time() - random.uniform(0, 86400),
            exit_time=time.time(),
            regime=random.choice(["trending_up", "sideways", "volatile"]),
            conviction=random.uniform(55, 85),
            composite_score=random.uniform(-1, 1),
            leverage=3,
            trend_score=random.uniform(0, 10),
            momentum_score=random.uniform(0, 10),
            volume_score=random.uniform(0, 10),
            reason="test trade",
        ))

    context = get_learned_context()
    print(context)
    print(f"\nLearned weights saved to: {LEARNED_WEIGHTS_PATH}")
