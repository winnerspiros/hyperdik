#!/usr/bin/env python3
"""
Hyperliquid Risk Engine v3 — Dual-layer WEL/TWEL, Equity Hard Stop (HSL),
Unstucking, Kelly sizing, correlation group caps, volatility-targeted sizing.

Built from research across:
  - passivbot (enarjord) — WEL/TWEL dual-layer, unstucking, HSL 4-tier
  - master-confluence (Enki1444) — Kelly criterion, correlation groups
  - ref-perp-bot (CShear) — signal cooldown, dead man's switch
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hyperliquid_strategy import (get_correlation_group, get_correlation_members,
                                   kelly_fraction, MarketRegime)


# ============================================================
# HSL — Equity Hard Stop (4-tier color system)
# ============================================================

class HSLTier(str, Enum):
    """4-tier equity drawdown protection."""
    GREEN = "green"       # Normal trading — all actions allowed
    YELLOW = "yellow"     # Position-only — no new entries, manage existing
    ORANGE = "orange"     # Close-only — exit all positions ASAP
    RED = "red"           # PANIC — emergency close all, stop bot


@dataclass
class HSLState:
    """Equity Hard Stop state tracking."""
    tier: HSLTier = HSLTier.GREEN
    peak_equity: float = 0.0
    drawdown_ema: float = 0.0
    current_drawdown_pct: float = 0.0
    red_threshold_pct: float = 0.30    # 30% drawdown = RED
    orange_threshold_pct: float = 0.20  # 20% drawdown = ORANGE
    yellow_threshold_pct: float = 0.10  # 10% drawdown = YELLOW
    ema_alpha: float = 0.05            # EMA smoothing for drawdown
    consecutive_red_checks: int = 0
    last_check: float = 0.0

    def update(self, equity: float) -> None:
        """Update HSL state with current equity."""
        self.peak_equity = max(self.peak_equity, equity)
        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # EMA-smoothed drawdown
        self.drawdown_ema = (self.ema_alpha * current_dd +
                             (1 - self.ema_alpha) * self.drawdown_ema)
        self.current_drawdown_pct = self.drawdown_ema * 100

        # Tier classification
        if self.drawdown_ema >= self.red_threshold_pct:
            self.consecutive_red_checks += 1
            if self.consecutive_red_checks >= 2:
                self.tier = HSLTier.RED
        elif self.drawdown_ema >= self.orange_threshold_pct:
            self.tier = HSLTier.ORANGE
            self.consecutive_red_checks = 0
        elif self.drawdown_ema >= self.yellow_threshold_pct:
            self.tier = HSLTier.YELLOW
            self.consecutive_red_checks = 0
        else:
            self.tier = HSLTier.GREEN
            self.consecutive_red_checks = 0

        self.last_check = time.time()


# ============================================================
# WEL/TWEL — Wallet/Total Wallet Exposure Limits
# ============================================================

@dataclass
class ExposureLimits:
    """Dual-layer exposure limits. Scales with account size."""

    @staticmethod
    def for_equity(equity: float) -> "ExposureLimits":
        """Return appropriate limits for account size."""
        if equity < 50:
            return ExposureLimits(
                wel_limit=0.80,          # 80% per position on micro (6x lev needs room)
                twel_limit=1.00,         # 100% — full account can be deployed
                group_limit=0.60,
            )
        elif equity < 200:
            return ExposureLimits(
                wel_limit=0.60,          # 60% per position (6x lev: 10% margin = 60% notional)
                twel_limit=0.80,         # was 0.50 — need room for multiple positions
                group_limit=0.50,        # was 0.35
            )
        elif equity < 1000:
            return ExposureLimits(
                wel_limit=0.20,
                twel_limit=0.35,   # Small: room for 3-4 positions
                group_limit=0.30,
            )
        else:
            return ExposureLimits()  # Default: conservative

    # Per-position
    wel_limit: float = 0.20          # Max 20% of equity per position
    wel_trigger: float = 1.0         # Allow full wel_limit
    # Aggregate
    twel_limit: float = 0.30         # Max 30% total exposure
    twel_trigger: float = 0.90       # Warn at 90% of TWEL
    # Per-group
    group_limit: float = 0.25        # Max 25% per correlation group
    # Min/max
    min_position_usd: float = 7.0   # Aug 7: was 8.5→9.5 — $42 equity at 20% cap = $8.48 notional, can't clear 8.50
    max_position_usd: float = 500.0  # Hard cap


@dataclass
class ExposureState:
    """Current exposure tracking."""
    limits: ExposureLimits = field(default_factory=ExposureLimits)
    equity: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)  # symbol → notional_usd
    group_exposure: dict[str, float] = field(default_factory=dict)

    @property
    def total_exposure(self) -> float:
        return sum(self.positions.values())

    @property
    def wel_used_pct(self) -> float:
        return self.total_exposure / self.equity if self.equity > 0 else 0.0

    def update(self, equity: float, positions: dict[str, float]) -> None:
        """Update exposure from current positions."""
        self.equity = equity
        self.positions = {k.upper(): v for k, v in positions.items()}
        self.group_exposure = {}
        for sym, notional in self.positions.items():
            group = get_correlation_group(sym)
            if group:
                self.group_exposure[group] = self.group_exposure.get(group, 0.0) + notional


def check_position_allowed(
    symbol: str,
    notional_usd: float,
    exposure: ExposureState,
    open_positions: dict[str, dict] | None = None,
) -> tuple[bool, str]:
    """Check if a new position passes all exposure limits.

    Returns (allowed, reason).
    """
    limits = exposure.limits
    sym = symbol.upper()
    equity = exposure.equity or 1.0  # Guard against zero equity
    open_positions = open_positions or {}

    # Check minimum
    if notional_usd < limits.min_position_usd:
        return False, f"below_min:{notional_usd:.2f}<{limits.min_position_usd}"

    # Check maximum
    if notional_usd > limits.max_position_usd:
        return False, f"above_max:{notional_usd:.2f}>{limits.max_position_usd}"

    # Already have a position?
    if sym in open_positions:
        return False, f"position_exists:{sym}"

    # Per-position WEL check
    current_name_exp = exposure.positions.get(sym, 0)
    new_name_exp = current_name_exp + notional_usd
    if new_name_exp > equity * limits.wel_limit * limits.wel_trigger + 0.01:  # 1¢ epsilon for float rounding
        return False, f"wel_exceeded:{sym}={new_name_exp/equity*100:.1f}%"

    # Total exposure TWEL check
    new_total = exposure.total_exposure + notional_usd
    if new_total > equity * limits.twel_limit:
        return False, f"twel_exceeded:{new_total/equity*100:.1f}%"

    # Group exposure check
    group = get_correlation_group(sym)
    if group:
        current_group = exposure.group_exposure.get(group, 0)
        new_group = current_group + notional_usd
        if new_group > equity * limits.group_limit:
            return False, f"group_exceeded:{group}={new_group/equity*100:.1f}%"

    return True, "allowed"


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    leverage: int,
    kelly: float,
    atr: float = 0.0,
    max_risk_pct: float = 0.02,
    vol_scale: bool = True,
    max_position_pct: float = 0.20,
    _MIN_POS_USD: float = 7.0,  # mirrors ExposureLimits.min_position_usd
) -> float:
    """Calculate position size with volatility-targeted scaling.

    Uses fractional Kelly × confidence, bounded by max risk per trade AND
    max position percentage of equity. Critical for small accounts where
    risk_usd / stop_distance can produce oversized notional values.

    Args:
        max_position_pct: Maximum position size as fraction of equity (default 0.20 = 20%).
            This is the user's hard cap — position notional must never exceed this.
    """
    if equity <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0.0

    # Base: risk % of equity
    risk_usd = equity * min(kelly, max_risk_pct)

    # Volatility-targeted scaling
    if vol_scale and atr > 0 and entry_price > 0:
        atr_pct = atr / entry_price
        # Target 1% ATR = normal size, 2% ATR = half, 0.5% ATR = 1.5x
        vol_scalar = max(0.5, min(1.5, 0.01 / max(atr_pct, 0.001)))
        risk_usd *= vol_scalar

    # ── Half-Kelly drawdown scaling: prevent Kelly death spiral ──
    # Research shows Half-Kelly produced 34% higher returns over 6 months
    # As drawdown grows, position size shrinks gracefully
    # SKIP for micro accounts ($50-150): half-kelly on already-conservative sizing → zero_size
    if equity >= 150:
        half_kelly = min(kelly * 0.5, max_risk_pct * 0.5)
        risk_usd = min(risk_usd, equity * half_kelly)

    # Stop distance
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0

    size = risk_usd / stop_distance

    # Leverage check (capped by exchange leverage)
    max_size_leverage = (equity * leverage) / entry_price
    size = min(size, max_size_leverage)

    # CRITICAL: Cap notional by max position % of equity
    # Without this, small accounts produce oversized nominals because
    # risk_usd / stop_distance ignores total position value.
    # E.g., $75 account with 2% risk ($1.50) and 3% stop on $13 coin
    # → size = $1.50 / ($13*0.03) = 3.85 → notional = $50 (66% of equity!)
    # Aug 8: micro-account floor — cap must never produce notional < min_position_usd
    # Otherwise the downstream check_position_allowed rejects the trade we just sized.
    max_notional = equity * max_position_pct
    # For micro accounts (<$100): ensure cap is at least 5% above minimum to avoid boundary collision
    _min_notional_floor = _MIN_POS_USD * 1.05 if equity < 100 else 0
    max_notional = max(max_notional, _min_notional_floor)
    max_size_position = max_notional / entry_price
    size = min(size, max_size_position)

    notional = size * entry_price
    # Final guard: if cap squeezed us below minimum, bump to minimum
    if notional < _MIN_POS_USD:
        notional = _MIN_POS_USD
        size = notional / entry_price

    # Min notional check (Hyperliquid requires $10 minimum)
    # Micro accounts ($50-100): relax to $5 — position sizing already conservative
    _min_notional = 5.0 if equity < 150 else 10.0
    if notional < _min_notional:
        return 0.0

    return max(0.0, size)


def get_leverage_for_regime(regime: str, base_leverage: int = 3) -> int:
    "User directive: use base_leverage as-is. No regime caps."
    return base_leverage


# ============================================================
# UNSTUCKING — Auto-close stale positions
# ============================================================

@dataclass
class UnstuckState:
    """Tracks unstucking for each position."""
    entries: dict[str, dict] = field(default_factory=dict)  # symbol → {entry_time, entry_price, unstuck_count}

    def register(self, symbol: str, entry_price: float) -> None:
        """Register a new position for unstuck tracking."""
        sym = symbol.upper()
        self.entries[sym] = {
            "entry_time": time.time(),
            "entry_price": entry_price,
            "unstuck_count": 0,
        }

    def remove(self, symbol: str) -> None:
        """Remove a closed position."""
        self.entries.pop(symbol.upper(), None)


def check_unstuck(
    symbol: str,
    current_price: float,
    position_size: float,
    unstuck_state: UnstuckState,
    max_hold_seconds: float = 7200.0,  # 2 hours
    unstuck_close_pct: float = 0.5,     # Close 50% per unstuck step
    loss_allowance_pct: float = 0.01,   # Max 1% of equity lost to unstucking
    equity: float = 10000.0,
    unstuck_min_age: float = 900.0,     # Min 15 min before unstucking
) -> tuple[bool, float, str]:
    """Check if a position should be unstuck.

    Returns (should_unstuck, close_fraction, reason).
    """
    sym = symbol.upper()
    entry = unstuck_state.entries.get(sym)
    if not entry:
        return False, 0.0, ""

    age = time.time() - entry["entry_time"]
    if age < unstuck_min_age:
        return False, 0.0, ""

    # Check if position is stagnant (small PnL, long time)
    pnl_pct = (current_price - entry["entry_price"]) / entry["entry_price"] * 100
    abs_pnl = abs(pnl_pct)

    # Unstuck if: held > max_hold AND PnL is flat (<2%)
    if age > max_hold_seconds and abs_pnl < 2.0:
        close_frac = unstuck_close_pct
        reason = f"unstuck_age:{age/3600:.1f}h_pnl:{pnl_pct:+.2f}%"

        # Check loss allowance
        if pnl_pct < 0:
            loss_usd = abs(pnl_pct / 100 * position_size * entry["entry_price"])
            max_loss = equity * loss_allowance_pct
            if loss_usd > max_loss:
                return False, 0.0, f"unstuck_loss_limit:{loss_usd:.2f}>{max_loss:.2f}"

        return True, close_frac, reason

    return False, 0.0, ""


# ============================================================
# SIGNAL COOLDOWN
# ============================================================

@dataclass
class CooldownState:
    """Prevents overtrading same coin."""
    last_signal: dict[str, float] = field(default_factory=dict)
    cooldown_seconds: float = 1800.0  # 30 min default

    def can_trade(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal."""
        sym = symbol.upper()
        if sym not in self.last_signal:
            return True
        return time.time() - self.last_signal[sym] >= self.cooldown_seconds

    def record(self, symbol: str) -> None:
        """Record a signal for cooldown."""
        self.last_signal[symbol.upper()] = time.time()

    def remaining(self, symbol: str) -> float:
        """Seconds remaining in cooldown."""
        sym = symbol.upper()
        if sym not in self.last_signal:
            return 0.0
        elapsed = time.time() - self.last_signal[sym]
        return max(0.0, self.cooldown_seconds - elapsed)


# ============================================================
# FUNDING WINDOW CHECK
# ============================================================

def is_funding_window(avoidance_minutes: int = 2) -> bool:
    """Check if we're in the funding avoidance window.
    Hyperliquid funding settles hourly — avoid entries near the turn.
    15-second window: blocks only the final seconds of each hour.
    The actual risk period is ~10-30s around settlement, not minutes."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return now.minute >= 59 and now.second >= 45  # Last 15 seconds of each hour


# ============================================================
# VOLATILITY-TARGETED MULTIPLIER
# ============================================================

def volatility_multiplier(atr: float, entry_price: float,
                          target_atr_pct: float = 0.01) -> float:
    """Calculate size multiplier based on ATR volatility.
    High vol → smaller position, low vol → larger position.
    Returns 0.5-1.5x multiplier."""
    if atr <= 0 or entry_price <= 0:
        return 1.0
    atr_pct = atr / entry_price
    scalar = target_atr_pct / max(atr_pct, 0.001)
    return max(0.5, min(1.5, scalar))


# ============================================================
# PORTFOLIO HEAT — Medallion-style correlation-adjusted exposure
# ============================================================

# Default BTC correlations (1.0 = perfectly correlated to BTC)
DEFAULT_BTC_CORRELATIONS: dict[str, float] = {
    "BTC": 1.00,
    "ETH": 0.85,
    "SOL": 0.75,
    "BNB": 0.82,
    "XRP": 0.78,
    "AVAX": 0.74,
    "DOGE": 0.68,
    "LINK": 0.72,
    "ADA": 0.76,
    "DOT": 0.80,
    "SUI": 0.65,
    "HYPE": 0.55,
    "RNDR": 0.60,
    "TAO": 0.58,
    "NEAR": 0.70,
    "TON": 0.62,
}

MAX_PORTFOLIO_HEAT = 0.60   # Max 60% correlation-adjusted exposure
MAX_SINGLE_POSITION = 0.25  # Max 25% per single position ($18.75 at $75 equity)
MAX_LEVERAGED_HEAT = 0.45   # MAX 45% when using leverage > 2x

def get_heat_limits(equity: float) -> tuple[float, float]:
    """Return (max_portfolio_heat, max_leveraged_heat) for account size."""
    if equity < 200:
        return (1.00, 1.00)  # Micro: no heat restrictions — user directive
    elif equity < 1000:
        return (0.45, 0.20)  # Small
    return (MAX_PORTFOLIO_HEAT, MAX_LEVERAGED_HEAT)  # Default


def compute_portfolio_heat(
    positions: dict[str, float],  # symbol → notional_usd
    equity: float,
    correlations: dict[str, float] | None = None,
) -> float:
    """Compute Medallion-style portfolio heat (correlation-adjusted exposure).

    Heat = sum(sqrt(size_i * size_j * corr_ij)) / equity

    This accounts for the fact that correlated positions are riskier
    than they appear individually.
    """
    if equity <= 0 or not positions:
        return 0.0

    corr = correlations or DEFAULT_BTC_CORRELATIONS
    symbols = list(positions.keys())
    heat = 0.0

    for i in range(len(symbols)):
        for j in range(len(symbols)):
            si = positions[symbols[i]]
            sj = positions[symbols[j]]
            ci = corr.get(symbols[i].upper(), 0.7)
            cj = corr.get(symbols[j].upper(), 0.7)
            corr_ij = min(ci, cj)  # Conservative: use minimum correlation
            heat += (si * sj * corr_ij) ** 0.5

    return heat / equity


def is_portfolio_heat_safe(
    positions: dict[str, float],
    equity: float,
    new_symbol: str | None = None,
    new_notional: float = 0.0,
    max_leverage: int = 1,
) -> tuple[bool, float, str]:
    """Check if adding a position would exceed portfolio heat limits.

    With leverage, use MAX_LEVERAGED_HEAT (15%) instead of MAX_PORTFOLIO_HEAT (35%).

    Returns (safe, current_heat_pct, reason).
    """
    max_heat = MAX_LEVERAGED_HEAT if max_leverage > 2 else MAX_PORTFOLIO_HEAT
    # ── Dynamic limits for micro accounts ──
    _ph, _lh = get_heat_limits(equity)
    if max_leverage > 2:
        max_heat = _lh
    else:
        max_heat = _ph

    if new_symbol and new_notional > 0:
        test_positions = dict(positions)
        test_positions[new_symbol.upper()] = test_positions.get(new_symbol.upper(), 0) + new_notional
    else:
        test_positions = positions

    heat = compute_portfolio_heat(test_positions, equity)

    if heat > max_heat:
        return False, heat, f"heat_exceeded:{heat*100:.1f}%>{max_heat*100:.0f}%"

    return True, heat, "heat_ok"


def compute_max_new_position(
    equity: float,
    current_heat: float,
    correlation_to_btc: float = 0.7,
    max_heat: float | None = None,
    leverage: int = 1,
) -> float:
    """Compute maximum new position notional given current portfolio heat.

    Uses the Medallion principle: never let correlation-adjusted exposure
    exceed the max heat budget.
    """
    max_h = max_heat or (MAX_LEVERAGED_HEAT if leverage > 2 else MAX_PORTFOLIO_HEAT)
    # ── Dynamic limits for micro accounts ──
    _ph, _lh = get_heat_limits(equity)
    if max_h == MAX_LEVERAGED_HEAT:
        max_h = _lh
    elif max_h == MAX_PORTFOLIO_HEAT:
        max_h = _ph
    remaining_heat = max_h - current_heat
    if remaining_heat <= 0:
        return 0.0

    # Simplified: max_new = remaining_heat * equity / sqrt(correlation)
    max_notional = remaining_heat * equity / (correlation_to_btc ** 0.5)

    # Also cap at max single position
    single_cap = equity * MAX_SINGLE_POSITION

    return min(max_notional, single_cap)

@dataclass
class RiskCheck:
    """Complete risk check result for a potential trade."""
    allowed: bool
    reason: str
    size_units: float
    notional_usd: float
    leverage: int
    kelly: float
    vol_multiplier: float
    hsl_tier: HSLTier
    exposure_pct: float


def full_risk_check(
    symbol: str,
    entry_price: float,
    stop_price: float,
    base_leverage: int,
    regime: str,
    atr: float,
    equity: float,
    open_positions: dict[str, dict],
    hsl_state: HSLState,
    exposure_state: ExposureState,
    cooldown_state: CooldownState,
    kelly: float = 0.01,
    max_risk_pct: float = 0.02,
) -> RiskCheck:
    """Complete risk assessment for a potential trade.

    Returns a RiskCheck with all sizing and gating decisions.
    """
    sym = symbol.upper()

    # 1. HSL check
    if hsl_state.tier == HSLTier.RED:
        return RiskCheck(False, "hsl_red", 0, 0, 1, 0, 1.0, hsl_state.tier, exposure_state.wel_used_pct)
    if hsl_state.tier == HSLTier.ORANGE:
        return RiskCheck(False, "hsl_orange_close_only", 0, 0, 1, 0, 1.0, hsl_state.tier, exposure_state.wel_used_pct)
    if hsl_state.tier == HSLTier.YELLOW:
        # Yellow: no new positions
        if sym not in open_positions:
            return RiskCheck(False, "hsl_yellow_no_new", 0, 0, 1, 0, 1.0, hsl_state.tier, exposure_state.wel_used_pct)

    # 2. Cooldown check
    if not cooldown_state.can_trade(sym):
        rem = cooldown_state.remaining(sym)
        return RiskCheck(False, f"cooldown:{rem:.0f}s", 0, 0, 1, 0, 1.0, hsl_state.tier, exposure_state.wel_used_pct)

    # 3. Funding window check
    if is_funding_window():
        return RiskCheck(False, "funding_window", 0, 0, 1, 0, 1.0, hsl_state.tier, exposure_state.wel_used_pct)

    # 4. Leverage by regime
    leverage = get_leverage_for_regime(regime, base_leverage)

    # 5. Volatility multiplier
    vol_mult = volatility_multiplier(atr, entry_price)

    # 6. Position size
    adj_kelly = kelly * vol_mult
    size_units = calculate_position_size(
        equity, entry_price, stop_price, leverage,
        adj_kelly, atr, max_risk_pct, vol_scale=True,
    )

    if size_units <= 0:
        return RiskCheck(False, "zero_size", 0, 0, leverage, adj_kelly, vol_mult, hsl_state.tier, exposure_state.wel_used_pct)

    notional = size_units * entry_price

    # 7. Exposure limits
    allowed, reason = check_position_allowed(sym, notional, exposure_state, open_positions)
    if not allowed:
        return RiskCheck(False, reason, size_units, notional, leverage, adj_kelly, vol_mult, hsl_state.tier, exposure_state.wel_used_pct)

    return RiskCheck(
        True, "ok", size_units, notional, leverage,
        adj_kelly, vol_mult, hsl_state.tier, exposure_state.wel_used_pct,
    )
