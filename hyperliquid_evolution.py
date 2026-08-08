#!/usr/bin/env python3
"""
Hyperliquid Evolution Engine — LLM-based parameter optimization via
segmented backtest reflection (Moss-style evolution loop).

FLOW:
  1. Collect trade performance data in weekly segments
  2. Compute per-segment metrics (WR, PF, avg win/loss, drawdown)
  3. LLM reads evolution_log, applies 7 Reflection Principles
  4. LLM produces micro-adjustments (±10% per param, max ±30% drift)
  5. Rerun with new params; repeat weekly

PERSONALITY PARAMS (locked): weights, leverage, bias, rolling
TACTICAL PARAMS (adjustable): stop/TP multiples, threshold pcts, EMA spans

Built from:
  - moss-trade-bot-skills — 7 Reflection Principles, personality/tactical split
  - master-confluence — strategy decay, shadow signaling
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Parameter Schemas
# ============================================================

@dataclass
class PersonalityParams:
    """Locked personality params — define the strategy's core identity."""
    trend_weight: float = 0.30
    momentum_weight: float = 0.25
    mean_reversion_weight: float = 0.15
    volume_weight: float = 0.15
    volatility_weight: float = 0.15
    long_bias: float = 0.5
    base_leverage: int = 3
    max_positions: int = 5
    rolling_enabled: bool = False
    rolling_trigger_pct: float = 0.30
    rolling_reinvest_pct: float = 0.80
    rolling_max_times: int = 3

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TacticalParams:
    """Adjustable tactical params — optimized by evolution loop.
    Bounded: ±30% drift from initial values."""
    # Stops & Targets
    stop_atr_mult: float = 1.8
    tp1_r_mult: float = 0.30
    tp2_r_mult: float = 0.50
    tp3_r_mult: float = 1.20
    trail_atr_mult: float = 2.0

    # Entry thresholds
    rsi_buy_max: float = 45.0
    rsi_sell_min: float = 55.0
    bb_pct_buy: float = 0.35
    bb_pct_sell: float = 0.65
    adx_trend_min: float = 25.0
    adx_sideways_max: float = 30.0

    # Risk
    max_risk_pct: float = 0.02
    cooldown_seconds: float = 1800.0
    unstuck_age_seconds: float = 7200.0
    unstuck_close_pct: float = 0.50

    # Volatility
    vol_scalar_min: float = 0.5
    vol_scalar_max: float = 1.5
    target_atr_pct: float = 0.01

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TacticalParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# Evolution State
# ============================================================

@dataclass
class SegmentMetrics:
    """Performance metrics for a single time segment."""
    segment_id: str
    start_time: str
    end_time: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown_pct: float
    sharpe: float
    long_trades: int
    short_trades: int
    exit_reasons: dict[str, int] = field(default_factory=dict)
    market_context: str = ""


@dataclass
class EvolutionState:
    """Persistent evolution state."""
    version: int = 0
    personality: PersonalityParams = field(default_factory=PersonalityParams)
    tactical: TacticalParams = field(default_factory=TacticalParams)
    initial_tactical: dict = field(default_factory=dict)
    segments: list[SegmentMetrics] = field(default_factory=list)
    last_evolution: str = ""
    adjustments: list[dict] = field(default_factory=list)
    rounds_since_change: int = 0

    def save(self, path: str = "data/evolution_state.json") -> None:
        """Persist evolution state."""
        data = {
            "version": self.version,
            "personality": self.personality.to_dict(),
            "tactical": self.tactical.to_dict(),
            "initial_tactical": self.initial_tactical,
            "segments": [s.__dict__ for s in self.segments[-20:]],  # Keep last 20
            "last_evolution": self.last_evolution,
            "adjustments": self.adjustments[-50:],  # Keep last 50
            "rounds_since_change": self.rounds_since_change,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = "data/evolution_state.json") -> "EvolutionState":
        """Load evolution state from disk."""
        if not os.path.exists(path):
            state = cls()
            state.initial_tactical = state.tactical.to_dict()
            return state

        with open(path) as f:
            data = json.load(f)

        state = cls()
        state.version = data.get("version", 0) + 1
        state.personality = PersonalityParams.from_dict(data.get("personality", {}))
        state.tactical = TacticalParams.from_dict(data.get("tactical", {}))
        state.initial_tactical = data.get("initial_tactical", state.tactical.to_dict())
        state.adjustments = data.get("adjustments", [])
        state.last_evolution = data.get("last_evolution", "")
        state.rounds_since_change = data.get("rounds_since_change", 0)
        return state


# ============================================================
# 7 Reflection Principles Prompt
# ============================================================

REFLECTION_PROMPT = """You are a trading strategy optimizer. Review the evolution log
below and suggest micro-adjustments to tactical parameters.

APPLY THESE 7 PRINCIPLES:

P1: BIG PICTURE FIRST
  - If cumulative return is positive, don't overreact to individual bad segments.
  - Focus on trends across 3+ segments, not single-segment swings.

P2: ANALYZE WINNING TRADES
  - Why did winning segments work? What was the market context?
  - Don't change what's working.

P3: ANALYZE LOSING TRADES
  - Were stops too tight? Wrong direction? Bad entries?
  - Identify the ROOT CAUSE, not just "it lost money".

P4: IDENTIFY EXACT PARAMETER
  - NEVER suggest vague changes like "tighten stops".
  - Always specify: "stop_atr_mult: 1.8 → 1.9" or "tp1_r_mult: 0.30 → 0.25"
  - Each suggestion must map to one specific parameter name.

P5: MAX ±10% PER PARAMETER PER ROUND
  - Each adjustment is at most 10% of the current value.
  - Small, safe steps — evolution, not revolution.

P6: INERTIA — don't change if <2 segments have passed
  - If a parameter was changed recently, let it run for at least 2 segments
    before changing again.

P7: ADAPTATION MANDATE
  - Must suggest at least one adjustment every 3 rounds.
  - Even if things are going well, find ONE small improvement.

CONSTRAINTS:
  - Personality params (weights, leverage, bias) are LOCKED. Do NOT suggest changes.
  - Tactical params can drift at most ±30% from initial values.
  - Initial tactical values: {initial_tactical}

CURRENT TACTICAL PARAMS:
{current_tactical}

RECENT ADJUSTMENTS:
{recent_adjustments}

EVOLUTION LOG:
{evolution_log}

OUTPUT: JSON array of adjustments, each with:
  {{"parameter": "name", "current": value, "suggested": value, "reason": "P#: ..."}}

Only output the JSON array, nothing else."""


def build_evolution_log(segments: list[SegmentMetrics]) -> str:
    """Build structured evolution log for LLM consumption."""
    lines = []
    cumulative_pnl = 0.0
    cumulative_trades = 0

    for seg in segments:
        cumulative_pnl += seg.total_pnl
        cumulative_trades += seg.trades

        lines.append(f"\n### Segment {seg.segment_id} ({seg.start_time} → {seg.end_time})")
        lines.append(f"  Market: {seg.market_context}")
        lines.append(f"  Trades: {seg.trades} (L:{seg.long_trades} S:{seg.short_trades})")
        lines.append(f"  Win Rate: {seg.win_rate:.1%} | PF: {seg.profit_factor:.2f}")
        lines.append(f"  PnL: ${seg.total_pnl:+.2f} | Cum: ${cumulative_pnl:+.2f}")
        lines.append(f"  Avg Win: ${seg.avg_win:+.2f} | Avg Loss: ${seg.avg_loss:+.2f}")
        lines.append(f"  Max DD: {seg.max_drawdown_pct:.1f}% | Sharpe: {seg.sharpe:.2f}")
        lines.append(f"  Exits: {seg.exit_reasons} | Cum Trades: {cumulative_trades}")

    return "\n".join(lines)


def evolve_parameters(segments: list[SegmentMetrics], state: EvolutionState) -> list[dict]:
    """Run LLM-based parameter evolution.

    Returns list of adjustments made.
    """
    # Check if we should evolve
    if len(segments) < 2:
        return []

    if state.rounds_since_change < 2 and state.adjustments:
        # Inertia: wait for at least 2 more segments
        return []

    # Build evolution log
    evo_log = build_evolution_log(segments)

    # Build prompt
    prompt = REFLECTION_PROMPT.format(
        initial_tactical=json.dumps(state.initial_tactical, indent=2),
        current_tactical=json.dumps(state.tactical.to_dict(), indent=2),
        recent_adjustments=json.dumps(state.adjustments[-10:], indent=2),
        evolution_log=evo_log,
    )

    # NOTE: In production, this sends to the LLM. Here we return the prompt
    # for the daemon to handle via its own AI infrastructure.
    return [{
        "type": "evolution_request",
        "prompt": prompt,
        "segments_analyzed": len(segments),
        "current_params": state.tactical.to_dict(),
        "initial_params": state.initial_tactical,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]


def apply_adjustments(adjustments: list[dict], state: EvolutionState) -> list[str]:
    """Apply LLM-suggested adjustments to tactical params with drift guard."""
    tactical_dict = state.tactical.to_dict()
    applied = []
    initial = state.initial_tactical

    for adj in adjustments:
        param = adj.get("parameter", "")
        suggested = adj.get("suggested")

        if param not in tactical_dict:
            applied.append(f"SKIP:{param}:unknown")
            continue

        current = tactical_dict[param]
        initial_val = initial.get(param, current)

        # ±10% max change per round
        change_pct = abs(suggested - current) / abs(current) if current != 0 else 1.0
        if change_pct > 0.10:
            # Clamp to ±10%
            direction = 1 if suggested > current else -1
            suggested = current * (1 + direction * 0.10)
            applied.append(f"CLAMP:{param}:{current:.4f}→{suggested:.4f}(±10% cap)")
        else:
            applied.append(f"OK:{param}:{current:.4f}→{suggested:.4f}")

        # ±30% total drift guard
        drift = abs(suggested - initial_val) / abs(initial_val) if initial_val != 0 else 0
        if drift > 0.30:
            direction = 1 if suggested > initial_val else -1
            suggested = initial_val * (1 + direction * 0.30)
            applied[-1] = f"DRIFT_GUARD:{param}:{current:.4f}→{suggested:.4f}(±30% cap)"

        tactical_dict[param] = suggested

    state.tactical = TacticalParams.from_dict(tactical_dict)
    state.adjustments.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changes": applied,
    })

    if applied:
        state.rounds_since_change = 0
    else:
        state.rounds_since_change += 1

    state.version += 1
    return applied


# ============================================================
# Segment Builder — create from trade history
# ============================================================

def build_segment(
    segment_id: str,
    trades: list[dict],
    start_time: str,
    end_time: str,
    market_context: str = "",
) -> SegmentMetrics:
    """Build segment metrics from a list of trades."""
    if not trades:
        return SegmentMetrics(
            segment_id=segment_id, start_time=start_time, end_time=end_time,
            trades=0, wins=0, losses=0, win_rate=0, profit_factor=0,
            total_pnl=0, avg_win=0, avg_loss=0, max_drawdown_pct=0,
            sharpe=0, long_trades=0, short_trades=0, market_context=market_context,
        )

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(w["pnl"] for w in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(l["pnl"] for l in losses) / len(losses)) if losses else 0

    profit_factor = (sum(w["pnl"] for w in wins) / abs(sum(l["pnl"] for l in losses))
                     if losses and sum(l["pnl"] for l in losses) != 0 else 0)

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.get("pnl", 0)
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    # Sharpe (simplified)
    pnls = [t.get("pnl", 0) for t in trades]
    import statistics
    avg_pnl = statistics.mean(pnls) if pnls else 0
    std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1
    sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0

    # Exit reasons
    from collections import Counter
    exit_reasons = dict(Counter(t.get("exit_reason", "unknown") for t in trades))

    long_trades = sum(1 for t in trades if t.get("side") == "BUY")
    short_trades = sum(1 for t in trades if t.get("side") == "SELL")

    return SegmentMetrics(
        segment_id=segment_id,
        start_time=start_time,
        end_time=end_time,
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        long_trades=long_trades,
        short_trades=short_trades,
        exit_reasons=exit_reasons,
        market_context=market_context,
    )
