#!/usr/bin/env python3
"""
Hyperliquid Execution Engine v3 — Multi-tier take profit, break-even stops,
trailing stops, balanced TP/SL groups, fee-aware sizing.

Patterns from:
  - master-confluence execution.py — multi-tier TP, break-even, trailing
  - OctoBot — fee-aware mirror orders, balanced TP/SL groups
  - passivbot — dynamic threshold scaling
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any


# ============================================================
# Data Classes
# ============================================================

@dataclass
class TierConfig:
    """Take-profit tier configuration."""
    fraction: float         # Size fraction (e.g., 0.33 = 33%)
    r_multiple: float       # Risk multiple (e.g., 0.30 = 30% of stop distance)
    description: str = ""


@dataclass
class MultiTierTP:
    """Multi-tier take profit configuration."""
    tiers: list[TierConfig] = field(default_factory=lambda: [
        TierConfig(0.333, 1.00, "TP1_1R"),      # 33% at 1× stop distance
        TierConfig(0.333, 2.00, "TP2_2R"),      # 33% at 2× stop distance  
        TierConfig(0.334, 3.50, "TP3_runner"),  # 34% at 3.5× stop distance
    ])
    break_even_after_tier: int = 1  # Move SL to BE after this tier fills
    trail_after_tier: int = 2       # Start trailing after this tier fills
    trail_atr_mult: float = 2.0     # ATR multiplier for trailing stop


@dataclass
class ExitPlan:
    """Complete exit plan for a position."""
    symbol: str
    is_long: bool
    entry_price: float
    total_size: float
    atr: float
    leverage: int
    stop_loss: float
    tp_levels: list[dict]  # [{size, price, tier_name}]
    break_even_enabled: bool = True
    trail_enabled: bool = True
    trail_atr: float = 2.0
    price_decimals: int = 2  # for display formatting
    sz_decimals: int = 6      # for size display


# ============================================================
# Price/Size Rounding (Hyperliquid-specific)
# ============================================================

def round_size(sz_decimals: int, size: float) -> float:
    """Round position size to Hyperliquid's szDecimals."""
    quantum = 10 ** (-sz_decimals)
    return math.floor(size / quantum) * quantum


def round_price(price_decimals: int, price: float, is_buy: bool = True) -> float:
    """Round price to the appropriate tick size (direction-aware)."""
    if price <= 0:
        return 0.0  # Guard against zero/negative — caller must handle
    quantum = 10 ** (-price_decimals)
    if is_buy:
        return math.ceil(price / quantum) * quantum
    else:
        return math.floor(price / quantum) * quantum


# ============================================================
# Multi-Tier TP Calculation
# ============================================================

def calculate_tp_levels(
    is_long: bool,
    entry_price: float,
    stop_price: float,
    leverage: int,
    atr: float,
    tier_config: list[TierConfig] | None = None,
    price_decimals: int = 2,
) -> list[dict]:
    """Calculate multi-tier take profit levels.

    Returns list of {size_fraction, price, r_multiple, tier_name}.
    """
    tier_config = tier_config or MultiTierTP().tiers
    levels = []
    remaining_fraction = 1.0

    for tier in tier_config:
        if remaining_fraction <= 0:
            break

        # Calculate TP price from R-multiple
        if is_long:
            tp_price = entry_price + (entry_price - stop_price) * tier.r_multiple
        else:
            tp_price = entry_price - (stop_price - entry_price) * tier.r_multiple

        # Also anchor to ATR for reasonability — use the WIDER of the two
        # (was min(), which capped TPs at near-zero for low-ATR coins like SUI)
        if is_long:
            atr_tp = entry_price + max(atr * tier.r_multiple * 2, entry_price * 0.01)
            tp_price = max(tp_price, atr_tp)  # Take the wider target
        else:
            atr_tp = entry_price - max(atr * tier.r_multiple * 2, entry_price * 0.01)
            tp_price = min(tp_price, atr_tp)

        levels.append({
            "fraction": tier.fraction,
            "price": round(tp_price, price_decimals),
            "r_multiple": tier.r_multiple,
            "tier_name": tier.description,
        })

        remaining_fraction -= tier.fraction

    return levels


def build_exit_plan(
    symbol: str,
    is_long: bool,
    entry_price: float,
    total_size: float,
    stop_price: float,
    atr: float,
    leverage: int,
    sz_decimals: int,
    price_decimals: int = 2,
    regime: str = "sideways",
    tier_config: list[TierConfig] | None = None,
    break_even_after: int = 1,
    trail_after: int = 2,
    trail_atr_mult: float = 2.0,
    ai_target_pct: float = 0.0,  # AI's predicted target % (0 = use default tiers)
) -> ExitPlan:
    """Build a complete exit plan for a new position, with regime-adjusted stops."""
    adjusted_stop = stop_price

    # If AI has a target, create TP tiers around it
    if ai_target_pct > 0:
        ai_target_price = entry_price * (1 + ai_target_pct / 100) if is_long else entry_price * (1 - ai_target_pct / 100)
        tier_config = [
            TierConfig(0.40, ai_target_pct / (abs(entry_price - stop_price) / entry_price * 100) * 0.6 if stop_price > 0 else 1.0, "AI TP 40%"),
            TierConfig(0.35, ai_target_pct / (abs(entry_price - stop_price) / entry_price * 100) * 0.85 if stop_price > 0 else 2.0, "AI TP 35%"),
            TierConfig(0.25, ai_target_pct / (abs(entry_price - stop_price) / entry_price * 100) * 1.1 if stop_price > 0 else 3.0, "AI runner 25%"),
        ]

    tp_levels = calculate_tp_levels(is_long, entry_price, adjusted_stop, leverage,
                                     atr, tier_config, price_decimals=price_decimals)

    # Assign actual sizes to each TP level
    sized_levels = []
    cumulative = 0.0
    for level in tp_levels:
        tier_size = round_size(sz_decimals, total_size * level["fraction"])
        if tier_size <= 0:
            continue
        tier_price = round_price(price_decimals, level["price"], is_buy=not is_long)
        sized_levels.append({
            "size": tier_size,
            "price": tier_price,
            "tier_name": level["tier_name"],
            "r_multiple": level["r_multiple"],
        })
        cumulative += tier_size

    # Any remaining size goes to last tier
    remaining = total_size - cumulative
    if remaining > 0 and sized_levels:
        sized_levels[-1]["size"] = round_size(sz_decimals, sized_levels[-1]["size"] + remaining)

    return ExitPlan(
        symbol=symbol,
        is_long=is_long,
        entry_price=entry_price,
        total_size=total_size,
        atr=atr,
        leverage=leverage,
        stop_loss=round_price(price_decimals, adjusted_stop, is_buy=not is_long),
        tp_levels=sized_levels,
        break_even_enabled=break_even_after > 0,
        trail_enabled=trail_after > 0,
        trail_atr=trail_atr_mult,
        price_decimals=price_decimals,
        sz_decimals=sz_decimals,
    )


# ============================================================
# Trailing Stop Logic
# ============================================================

@dataclass
class TrailState:
    """Tracks trailing stop state per position."""
    symbol: str
    is_long: bool
    entry_price: float
    highest_price: float = 0.0   # highest since entry (long) / lowest (short)
    current_stop: float = 0.0
    trail_distance: float = 0.0
    activated: bool = False
    activation_pct: float = 0.01  # 1% move triggers trail (was 2% — missed small coin peaks)


def update_trail(
    trail: TrailState,
    current_price: float,
    atr: float,
    trail_atr_mult: float = 2.0,
    activation_pct: float = None,
) -> float | None:
    """Update trailing stop. Returns new stop price if changed, None if unchanged."""
    # Compute dynamic activation: 2x ATR as % of price, clamped 0.1%-1%
    if activation_pct is None:
        activation_pct = min(0.01, max(0.001, atr / current_price * 2.0)) if current_price > 0 else 0.01
    trail.activation_pct = activation_pct
    
    # Track high/low water mark
    if trail.is_long:
        trail.highest_price = max(trail.highest_price, current_price)
        if not trail.activated:
            move_pct = (current_price - trail.entry_price) / trail.entry_price
            if move_pct >= activation_pct:
                trail.activated = True
                trail.trail_distance = atr * trail_atr_mult

        if trail.activated:
            new_stop = trail.highest_price - trail.trail_distance
            if new_stop > trail.current_stop:
                trail.current_stop = new_stop
                return new_stop
    else:
        trail.highest_price = min(trail.highest_price, current_price) if trail.highest_price > 0 else current_price
        if not trail.activated:
            move_pct = (trail.entry_price - current_price) / trail.entry_price
            if move_pct >= activation_pct:
                trail.activated = True
                trail.trail_distance = atr * trail_atr_mult

        if trail.activated:
            new_stop = trail.highest_price + trail.trail_distance
            if new_stop < trail.current_stop:
                trail.current_stop = new_stop
                return new_stop

    return None


# ============================================================
# Dynamic ATR Multiplier (volatility-regime adaptive)
# Research: Chandelier Exit with dynamic multiplier is #1 for leveraged crypto.
# Low vol regime (ATR at 50-period low) → tighter trail (2.5x)
# Extreme vol regime (ATR at 50-period high) → wide trail (4.5x)
# Formula: multiplier = base * (current_ATR / avg_ATR_50) → auto-scales
# ============================================================

def chandelier_atr_mult(atr: float, avg_atr_50: float = None, base_mult: float = 3.0) -> float:
    """Dynamic Chandelier Exit multiplier based on volatility regime.
    
    Research-backed: ATR(14)/SMA(ATR(14),50) ratio determines regime.
    Low < 0.85 → 2.5x | Normal 0.85-1.15 → 3.0x | High 1.15-1.5 → 3.5x | Extreme >1.5 → 4.5x
    
    Falls back to base_mult if avg_atr_50 unavailable.
    """
    if avg_atr_50 is None or avg_atr_50 <= 0:
        return base_mult
    
    vr = atr / avg_atr_50  # volatility ratio
    if vr > 1.5:
        return 4.5  # Extreme — survival mode, wide stop
    elif vr > 1.15:
        return 3.5  # High volatility
    elif vr > 0.85:
        return 3.0  # Normal
    else:
        return 2.5  # Low volatility — tight stop


# ============================================================
# Dynamic Distance Scaling (passivbot)
# ============================================================

def dynamic_threshold(
    base_pct: float,
    wallet_exposure_ratio: float,
    we_weight: float = 2.0,
    volatility_ema: float | None = None,
    vol_weight: float = 10.0,
) -> float:
    """Scale entry threshold dynamically based on exposure and volatility.

    Higher exposure → wider threshold (fewer entries)
    Higher volatility → wider threshold
    """
    scaled = base_pct * (1.0 + wallet_exposure_ratio * we_weight)
    if volatility_ema is not None and volatility_ema > 0:
        scaled *= (1.0 + volatility_ema * vol_weight)
    return scaled


# ============================================================
# Regime-Adjusted Stop & TP Multipliers (alphastrike)
# ============================================================

REGIME_STOP_MULTIPLIERS: dict[str, float] = {
    "trending_up": 1.25,     # Wider stops in trends — give room
    "trending_down": 1.25,
    "sideways": 0.8,          # Tighter stops in chop — quick exits
    "high_vol": 2.0,          # Much wider in high vol — avoid whipsaw
    "crisis": 2.0,
}


def get_regime_stop_mult(regime: str) -> float:
    """Get regime-adjusted stop loss multiplier.

    trending → 1.25x (wider, let trend develop)
    sideways → 0.80x (tighter, quick exits in chop)
    high_vol/crisis → 2.0x (wider, accommodate swings)
    """
    norm = regime.lower().strip()
    return REGIME_STOP_MULTIPLIERS.get(norm, 1.0)


# ============================================================
# Fee-Aware Sizing
# ============================================================

def fee_aware_size(raw_size: float, fee_rate: float = 0.00035) -> float:
    """Adjust size for taker fees (0.035% on Hyperliquid)."""
    return raw_size * (1.0 - fee_rate)


# ============================================================
# Risk Per Trade (max loss should be ≤ 2% of equity)
# ============================================================

def max_size_from_risk(
    equity: float,
    entry_price: float,
    stop_price: float,
    leverage: int,
    max_risk_pct: float = 0.02,
) -> float:
    """Calculate max position size such that stop-loss hit = max_risk_pct of equity."""
    if equity <= 0 or entry_price <= 0:
        return 0.0

    risk_usd = equity * max_risk_pct
    stop_distance_pct = abs(entry_price - stop_price) / entry_price

    if stop_distance_pct <= 0:
        return 0.0

    # Position value where stop-hit = risk_usd
    position_usd = risk_usd / stop_distance_pct

    # Size in units
    size = position_usd / entry_price

    # Leverage cap
    max_size = (equity * leverage) / entry_price

    return min(size, max_size)


# ============================================================
# Order Group Helper (for daemon integration)
# ============================================================

def describe_exit_plan(plan: ExitPlan) -> str:
    """Human-readable exit plan for logging."""
    px = plan.price_decimals
    sz = plan.sz_decimals
    lines = [f"Exit Plan for {plan.symbol} {'LONG' if plan.is_long else 'SHORT'}"]
    lines.append(f"  Entry: ${plan.entry_price:,.{px}f}")
    lines.append(f"  Stop:  ${plan.stop_loss:,.{px}f} (risk: ${abs(plan.entry_price - plan.stop_loss):,.{px}f})")
    for tp in plan.tp_levels:
        pnl_pct = (tp["price"] - plan.entry_price) / plan.entry_price * 100
        if not plan.is_long:
            pnl_pct = -pnl_pct
        lines.append(f"  TP{tp['tier_name'].replace('TP','')}: {tp['size']:.{min(sz,4)}f}u @ ${tp['price']:,.{px}f} ({pnl_pct:+.2f}%)")
    lines.append(f"  BE after TP1: {plan.break_even_enabled} | Trail after TP2: {plan.trail_enabled}")
    return "\n".join(lines)
