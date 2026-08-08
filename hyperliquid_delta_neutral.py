#!/usr/bin/env python3
"""
Hyperliquid Delta-Neutral Funding Arb v1 — Separate strategy that opens
opposing positions to harvest funding rate spreads.

Strategy: Long spot (no funding) + Short perp (collect funding) on same asset.
Both legs cancel price exposure; collect the funding rate difference.

Also supports: Cross-exchange arb (Hyperliquid + external exchange) when
funding spreads are extreme enough to cover trading friction.

Built from:
  - vooi-funding-bot-example — delta-neutral funding arb architecture
  - djienne/DELTA_NEUTRAL_HYPERLIQUID_PERP_SPOT — spot-long + perp-short
  - teslashibe/permafrost — institutional-grade delta-neutral protocol
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# Config
# ============================================================

ARB_STATE_PATH = "data/delta_neutral_state.json"

# Minimum annualized funding rate to enter arb (%)
MIN_ARB_APR = 50.0

# Minimum hold time (seconds) — avoid churn
MIN_HOLD_SECONDS = 3600  # 1 hour

# Trading friction (fees + slippage as % of notional)
TOTAL_FRICTION_PCT = 0.10  # 0.035% taker × 2 sides + slippage on both

# Maximum per-asset exposure as fraction of equity
MAX_EXPOSURE_PCT = 0.20

# Minimum expected return after friction (as % of notional)
MIN_NET_RETURN_PCT = 0.05  # Must make at least 0.05% after costs

# Cooldown between exits and re-entries (seconds)
EXIT_COOLDOWN = 7200  # 2 hours


# ============================================================
# State
# ============================================================

@dataclass
class ArbPosition:
    """A delta-neutral funding arb position."""
    symbol: str
    spot_size: float      # Long spot (units)
    perp_size: float      # Short perp (units)
    entry_price: float
    entry_time: float
    perp_entry_price: float
    spot_entry_price: float
    funding_collected: float = 0.0  # Total funding collected
    total_fees: float = 0.0
    active: bool = True

    @property
    def notional(self) -> float:
        return self.perp_size * self.entry_price

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "ArbPosition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ArbState:
    """Global delta-neutral arb state."""
    positions: dict[str, ArbPosition] = field(default_factory=dict)
    total_funding_collected: float = 0.0
    total_fees_paid: float = 0.0
    exit_cooldowns: dict[str, float] = field(default_factory=dict)
    equity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "total_funding_collected": self.total_funding_collected,
            "total_fees_paid": self.total_fees_paid,
            "exit_cooldowns": self.exit_cooldowns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArbState":
        state = cls()
        state.total_funding_collected = d.get("total_funding_collected", 0)
        state.total_fees_paid = d.get("total_fees_paid", 0)
        state.exit_cooldowns = d.get("exit_cooldowns", {})
        state.positions = {
            k: ArbPosition.from_dict(v) for k, v in d.get("positions", {}).items()
        }
        return state


def load_arb_state() -> ArbState:
    """Load arb state from disk."""
    if os.path.exists(ARB_STATE_PATH):
        with open(ARB_STATE_PATH) as f:
            return ArbState.from_dict(json.load(f))
    return ArbState()


def save_arb_state(state: ArbState) -> None:
    """Save arb state to disk."""
    os.makedirs(os.path.dirname(ARB_STATE_PATH), exist_ok=True)
    with open(ARB_STATE_PATH, "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)


def calculate_arb_return(funding_rate: float, hold_seconds: float) -> float:
    """Calculate expected return from funding rate over hold period.
    funding_rate: hourly rate as decimal (e.g., 0.0001 = 0.01%/hr)
    hold_seconds: expected hold time
    Returns: return as fraction of notional.
    """
    hourly_rate = abs(funding_rate)
    hours = hold_seconds / 3600
    gross_return = hourly_rate * hours
    net_return = gross_return * 100 - TOTAL_FRICTION_PCT  # Convert to %
    return net_return


def is_arb_viable(
    symbol: str,
    funding_rate: float,
    mark_price: float,
    equity: float,
    state: ArbState,
    funding_history: list[dict] | None = None,
) -> tuple[bool, str, float]:
    """Check if delta-neutral arb is viable.

    Returns (viable, reason, expected_return_pct).
    """
    # Check cooldown
    if symbol in state.exit_cooldowns:
        if time.time() - state.exit_cooldowns[symbol] < EXIT_COOLDOWN:
            return False, "cooldown", 0.0

    # Check already active
    if symbol in state.positions and state.positions[symbol].active:
        return False, "already_active", 0.0

    # Check funding rate direction and magnitude
    annual_rate = abs(funding_rate) * 24 * 365 * 100
    if annual_rate < MIN_ARB_APR:
        return False, f"low_apr:{annual_rate:.0f}%", 0.0

    # Check if funding is consistent (not just a spike)
    if funding_history:
        recent = [f for f in funding_history if time.time() - f.get("time", 0) < 3600 * 4]
        if len(recent) > 1:
            same_sign = all(
                (f.get("rate", 0) > 0) == (funding_rate > 0) for f in recent
            )
            if not same_sign:
                return False, "funding_flipping", 0.0

    # Expected return
    expected_return = calculate_arb_return(funding_rate, MIN_HOLD_SECONDS)
    if expected_return < MIN_NET_RETURN_PCT:
        return False, f"low_return:{expected_return:.2f}%", expected_return

    # Position size check
    max_notional = equity * MAX_EXPOSURE_PCT
    if max_notional < 10.0:  # Hyperliquid minimum
        return False, "notional_too_small", expected_return

    return True, "viable", expected_return


def should_exit_arb(
    symbol: str,
    funding_rate: float,
    state: ArbState,
) -> tuple[bool, str]:
    """Check if an arb position should be exited.

    Exit conditions:
    1. Funding rate normalized (below threshold)
    2. Funding rate flipped direction
    3. Position has been open long enough and funding collected
    """
    pos = state.positions.get(symbol)
    if not pos or not pos.active:
        return False, ""

    annual_rate = abs(funding_rate) * 24 * 365 * 100
    hold_hours = (time.time() - pos.entry_time) / 3600

    # Exit if funding normalized
    if annual_rate < MIN_ARB_APR * 0.5:  # Below 50% of entry threshold
        return True, f"funding_normalized:{annual_rate:.0f}%_apr"

    # Exit if already held long enough AND funding declining
    if hold_hours > 4 and annual_rate < MIN_ARB_APR:
        return True, f"funding_declining:{annual_rate:.0f}%_apr"

    # Exit if funding flipped direction
    if (funding_rate > 0 and pos.perp_size > 0) or (funding_rate < 0 and pos.perp_size < 0):
        return True, "funding_flipped"

    return False, ""


def get_arb_context(state: ArbState) -> str:
    """Build AI context string of active arb positions."""
    active = {k: v for k, v in state.positions.items() if v.active}
    if not active:
        return "No active delta-neutral arb positions"

    lines = ["DELTA-NEUTRAL FUNDING ARB (active positions):"]
    for sym, pos in active.items():
        hours = (time.time() - pos.entry_time) / 3600
        lines.append(
            f"  {sym}: spot={pos.spot_size:.4f}u perp={pos.perp_size:.4f}u "
            f"entry=${pos.entry_price:,.2f} "
            f"funding_collected=${pos.funding_collected:+.2f} "
            f"fees=${pos.total_fees:.2f} "
            f"held={hours:.1f}h"
        )
    lines.append(f"  TOTAL collected: ${state.total_funding_collected:+.2f}")
    lines.append(f"  TOTAL fees: ${state.total_fees_paid:.2f}")
    return "\n".join(lines)


# ============================================================
# Opportunity Scanner (runs periodically)
# ============================================================

def scan_arb_opportunities(
    funding_rates: dict[str, float],  # symbol → hourly rate
    marks: dict[str, float],
    equity: float,
    state: ArbState,
    funding_histories: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Scan all assets for viable arb opportunities.

    Returns list of {symbol, side, rate, annual_apr, expected_return, notional}.
    """
    opportunities = []
    funding_histories = funding_histories or {}

    for symbol, rate in funding_rates.items():
        mark_price = marks.get(symbol, 0)
        if mark_price <= 0:
            continue

        viable, reason, expected_ret = is_arb_viable(
            symbol, rate, mark_price, equity, state,
            funding_histories.get(symbol),
        )

        if viable:
            # Positive funding = longs pay shorts → short perp, long spot
            # Negative funding = shorts pay longs → long perp, short spot
            side = "SELL" if rate > 0 else "BUY"
            max_notional = equity * MAX_EXPOSURE_PCT
            annual_apr = abs(rate) * 24 * 365 * 100

            opportunities.append({
                "symbol": symbol,
                "perp_side": side,
                "funding_rate": rate,
                "annual_apr": round(annual_apr, 1),
                "expected_return_pct": round(expected_ret, 2),
                "max_notional": max_notional,
                "mark_price": mark_price,
            })

    opportunities.sort(key=lambda x: x["expected_return_pct"], reverse=True)
    return opportunities
