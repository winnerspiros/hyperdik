#!/usr/bin/env python3
"""
EV Gate + Portfolio Heat — adopted from Harper (balsimpson/virtual-investor).

Core principles:
- Every trade must have positive expected value after costs.
- Net reward/risk must be >= 1.5.
- Position size derives from invalidation distance, not flat %.
- Total portfolio heat (sum of distance-to-invalidation) capped at 5% NAV.

This replaces blind AI confidence gating with actual math.
Before: "AI says 80% BUY → enter"
After:  "AI says 80% BUY, target=2%, stop=1%, costs=0.3% → EV=-0.10% → BLOCK"
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EVResult:
    """Result of expected-value gate check."""
    passed: bool
    reward_pct: float       # raw reward (target - entry) / entry * 100
    risk_pct: float         # raw risk (entry - stop) / entry * 100
    net_reward_pct: float   # reward after costs
    net_risk_pct: float     # risk after costs
    reward_risk_ratio: float  # net_reward / net_risk
    expected_return_pct: float  # probability-weighted EV after costs
    confidence: float       # AI confidence used
    cost_pct: float         # estimated round-trip cost
    block_reason: str = ""


# ── Default cost model (crypto: higher than equities) ──
DEFAULT_FEE_BPS = 5.0      # 0.05% per leg (taker fee)
DEFAULT_SLIPPAGE_BPS = 10.0  # 0.10% slippage per leg
DEFAULT_TOTAL_COST_BPS = (DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2  # round-trip
DEFAULT_TOTAL_COST_PCT = DEFAULT_TOTAL_COST_BPS / 100  # 0.30%

# ── Risk limits ──
DEFAULT_MIN_REWARD_RISK = 0.55  # Was 0.9 — small-caps need wider stops vs tight AI targets. 0.55 = 1.5% tgt / 2.7% stop still passes.
DEFAULT_RISK_PER_TRADE_PCT = 1.0   # max 1% of NAV lost at invalidation
DEFAULT_MAX_PORTFOLIO_HEAT_PCT = 5.0  # max 5% of NAV in total heat


def check_ev_gate(
    direction: str,           # "BUY" or "SELL" (LONG/SHORT mapped)
    entry_price: float,
    target_price: float,
    stop_price: float,
    confidence: float,        # 0-100 (AI confidence %)
    cost_pct: float = DEFAULT_TOTAL_COST_PCT,
    min_reward_risk: float = DEFAULT_MIN_REWARD_RISK,
) -> EVResult:
    """
    Expected-value gate. Returns EVResult with pass/fail.

    For LONG:  reward = target - entry, risk = entry - stop
    For SHORT: reward = entry - target, risk = stop - entry

    expected_return = confidence × reward% − (1−confidence) × risk% − costs
    Must be positive AND net_reward/risk >= min_reward_risk.

    HARPER LESSON: AI confidence is meaningless without target/stop math.
    ASTER: AI=80% SHORT, target=2.35%, stop=1.87%, costs=0.30%
      → EV = 0.80×2.35 − 0.20×1.87 − 0.30 = 1.88−0.37−0.30 = +1.21% ✓
      BUT the direction was WRONG (price went UP).
      EV gate doesn't fix direction — ML/Unified gates handle that.
    MET: AI=82% LONG, target=2.88%, stop=1.91%, costs=0.30%
      → EV = 0.82×2.88 − 0.18×1.91 − 0.30 = 2.36−0.34−0.30 = +1.72% ✓
      Again EV passed, direction was wrong.
      EV gate is the FINAL check after direction gates pass.
    """
    # Normalize direction
    is_long = direction.upper() in ("BUY", "LONG")

    if is_long:
        reward_pct = ((target_price - entry_price) / entry_price) * 100
        risk_pct = ((entry_price - stop_price) / entry_price) * 100
    else:
        reward_pct = ((entry_price - target_price) / entry_price) * 100
        risk_pct = ((stop_price - entry_price) / entry_price) * 100

    # Sanity: reward and risk must be positive
    if reward_pct <= 0:
        return EVResult(
            passed=False, reward_pct=reward_pct, risk_pct=risk_pct,
            net_reward_pct=0, net_risk_pct=0, reward_risk_ratio=0,
            expected_return_pct=0, confidence=confidence, cost_pct=cost_pct,
            block_reason=f"reward={reward_pct:+.2f}% must be positive (target behind entry)"
        )
    if risk_pct <= 0:
        return EVResult(
            passed=False, reward_pct=reward_pct, risk_pct=risk_pct,
            net_reward_pct=0, net_risk_pct=0, reward_risk_ratio=0,
            expected_return_pct=0, confidence=confidence, cost_pct=cost_pct,
            block_reason=f"risk={risk_pct:+.2f}% must be positive (stop behind entry)"
        )

    # Net: subtract costs
    net_reward_pct = reward_pct - cost_pct
    net_risk_pct = risk_pct + cost_pct

    # Reward/risk ratio
    rr_ratio = net_reward_pct / net_risk_pct if net_risk_pct > 0 else 0.0
    if rr_ratio < min_reward_risk:
        return EVResult(
            passed=False, reward_pct=reward_pct, risk_pct=risk_pct,
            net_reward_pct=net_reward_pct, net_risk_pct=net_risk_pct,
            reward_risk_ratio=rr_ratio, expected_return_pct=0,
            confidence=confidence, cost_pct=cost_pct,
            block_reason=f"reward/risk={rr_ratio:.2f} below minimum {min_reward_risk} (target too close or stop too wide)"
        )

    # Expected value
    prob = confidence / 100.0
    expected_return_pct = (prob * reward_pct) - ((1 - prob) * risk_pct) - cost_pct

    if expected_return_pct <= 0:
        return EVResult(
            passed=False, reward_pct=reward_pct, risk_pct=risk_pct,
            net_reward_pct=net_reward_pct, net_risk_pct=net_risk_pct,
            reward_risk_ratio=rr_ratio, expected_return_pct=expected_return_pct,
            confidence=confidence, cost_pct=cost_pct,
            block_reason=f"EV={expected_return_pct:+.2f}% — negative after costs (conf={confidence:.0f}% too low for this R:R)"
        )

    return EVResult(
        passed=True, reward_pct=reward_pct, risk_pct=risk_pct,
        net_reward_pct=net_reward_pct, net_risk_pct=net_risk_pct,
        reward_risk_ratio=rr_ratio, expected_return_pct=expected_return_pct,
        confidence=confidence, cost_pct=cost_pct,
    )


def size_from_invalidation(
    nav: float,
    entry_price: float,
    invalidation_price: float,
    direction: str,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    gap_buffer_pct: float = 0.005,   # 0.5% gap buffer
    max_position_weight: float = 0.20,  # max 20% of NAV per position
) -> tuple[float, float]:
    """
    Size position from invalidation distance (Harper method).

    risk_per_share = |entry - invalidation| + entry * gap_buffer
    max_shares = floor(NAV * risk_per_trade_pct / risk_per_share)
    position_notional = shares * entry_price (capped at max_position_weight * NAV)

    Returns (shares, notional_usd).
    This is the opposite of our current approach (which picks position % first).
    """
    is_long = direction.upper() in ("BUY", "LONG")
    if is_long:
        invalidation_distance = abs(entry_price - invalidation_price)
    else:
        invalidation_distance = abs(invalidation_price - entry_price)

    risk_per_share = invalidation_distance + (entry_price * gap_buffer_pct)

    if risk_per_share <= 0:
        return (0.0, 0.0)

    max_loss_usd = nav * (risk_per_trade_pct / 100.0)
    shares = max_loss_usd / risk_per_share
    notional = shares * entry_price

    # Cap at max position weight
    max_notional = nav * max_position_weight
    if notional > max_notional:
        notional = max_notional
        shares = notional / entry_price

    return (shares, notional)


def portfolio_heat(
    positions: list[dict],  # [{"entry_price": X, "stop_price": Y, "size_notional": Z}, ...]
    nav: float,
    gap_buffer_pct: float = 0.005,
) -> float:
    """
    Total portfolio heat: sum of distance-to-invalidation for all positions.

    heat = sum(size_notional × (|price − invalidation| / price + gap_buffer))
    Returns heat as fraction of NAV.

    HARPER LESSON: aggregate risk matters. Two positions each at 2% risk
    that are correlated = 4% real heat. Cap at 5%.
    """
    if nav <= 0:
        return 1.0  # effectively blocked

    total_heat = 0.0
    for pos in positions:
        entry = pos.get("entry_price", 0)
        stop = pos.get("stop_price", 0)
        notional = pos.get("size_notional", 0)
        if entry <= 0 or notional <= 0:
            continue
        distance_pct = abs(entry - stop) / entry
        heat_contribution = notional * (distance_pct + gap_buffer_pct)
        total_heat += heat_contribution

    return total_heat / nav


# ── Brier score for AI direction calibration ──
def brier_score(forecasts: list[tuple[float, bool]]) -> float:
    """
    Brier score for binary directional forecasts.

    forecasts: list of (confidence_0_to_1, was_correct_bool)
    Lower is better. 0.25 = random guessing. 0.0 = perfect.

    HARPER LESSON: score the AI's directional calls separately from P&L.
    A correct DOWN call that lost 0.2% to fees is still a good signal.
    """
    if not forecasts:
        return 0.0
    return sum((conf - int(correct)) ** 2 for conf, correct in forecasts) / len(forecasts)


def brier_skill(forecasts: list[tuple[float, bool]]) -> Optional[float]:
    """
    Brier skill score vs base-rate forecast.
    Requires >= 5 forecasts. Returns None if insufficient data.
    """
    if len(forecasts) < 5:
        return None
    model_brier = brier_score(forecasts)
    event_rate = sum(1 for _, c in forecasts if c) / len(forecasts)
    base_brier = event_rate * (1 - event_rate)  # variance of binary outcome
    if base_brier == 0:
        return None
    return 1.0 - (model_brier / base_brier)


# ── CLI for testing ──
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: ev_gate.py <direction> <entry> <target> <stop> [confidence] [cost_pct]")
        print("Example: ev_gate.py BUY 0.1783 0.1834 0.1749 82 0.30")
        sys.exit(1)

    direction = sys.argv[1]
    entry = float(sys.argv[2])
    target = float(sys.argv[3])
    stop = float(sys.argv[4])
    confidence = float(sys.argv[5]) if len(sys.argv) > 5 else 80.0
    cost_pct = float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_TOTAL_COST_PCT

    result = check_ev_gate(direction, entry, target, stop, confidence, cost_pct)
    print(f"Direction: {direction}")
    print(f"Entry: ${entry:.4f}  Target: ${target:.4f}  Stop: ${stop:.4f}")
    print(f"Confidence: {confidence:.0f}%  Costs: {cost_pct:.2f}%")
    print(f"Reward: {result.reward_pct:+.2f}%  Risk: {result.risk_pct:+.2f}%")
    print(f"Net Reward: {result.net_reward_pct:+.2f}%  Net Risk: {result.net_risk_pct:+.2f}%")
    print(f"Reward/Risk: {result.reward_risk_ratio:.2f}  EV: {result.expected_return_pct:+.3f}%")
    print(f"Verdict: {'PASS' if result.passed else 'BLOCK'} — {result.block_reason}")

    # Test MET trade
    print("\n── MET Jul 29 test ──")
    met = check_ev_gate("BUY", 0.1783, 0.1834, 0.1749, 82, 0.30)
    print(f"EV: {met.expected_return_pct:+.3f}%  R:R={met.reward_risk_ratio:.2f}  {'PASS' if met.passed else 'BLOCK'}")

    # Test ASTER trade
    print("\n── ASTER Jul 29 test ──")
    aster = check_ev_gate("SELL", 0.6022, 0.5880, 0.6135, 80, 0.30)
    print(f"EV: {aster.expected_return_pct:+.3f}%  R:R={aster.reward_risk_ratio:.2f}  {'PASS' if aster.passed else 'BLOCK'}")

    # Test with too-wide stop
    print("\n── Bad trade: wide stop ──")
    bad = check_ev_gate("BUY", 100.0, 103.0, 95.0, 80, 0.30)
    print(f"EV: {bad.expected_return_pct:+.3f}%  R:R={bad.reward_risk_ratio:.2f}  {'PASS' if bad.passed else 'BLOCK'} — {bad.block_reason}")
