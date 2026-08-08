"""
Kelly Criterion Position Sizing — optimal bet sizing for crypto trading.

Mathematical foundation:
  f* = (p * b - q) / b   where:
    p = win probability (win_rate)
    q = loss probability (1 - p)
    b = win/loss ratio (avg_win / avg_loss)

Also provides fractional Kelly (half/quarter) for safety.
Based on well-known Quantopian/Investopedia implementations.

Reference:
  - https://en.wikipedia.org/wiki/Kelly_criterion
  - https://www.investopedia.com/articles/trading/04/091504.asp
  - Ralph Vince, "The Mathematics of Money Management" (1992)
"""
import math
from typing import Optional


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate the optimal Kelly fraction for position sizing.
    
    Args:
        win_rate: Probability of winning (0.0–1.0). E.g., 0.55 means 55% win rate.
        avg_win: Average profit on winning trades (as a ratio, e.g., 1.02 = 2% gain).
        avg_loss: Average loss on losing trades (as a positive ratio, e.g., 1.01 = 1% loss).
    
    Returns:
        Optimal fraction of capital to risk per trade (0.0–1.0).
        Returns 0.0 for non-positive edge trades.
        Returns the fraction capped at 0.25 (full Kelly can be aggressive).
    
    Example:
        kelly_fraction(0.55, 1.02, 1.01)
        # => ~0.14 (risk ~14% of capital per trade)
    """
    if not (0 < win_rate < 1):
        return 0.0  # No edge or degenerate input
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    
    # Win/loss ratio (odds ratio)
    b = avg_win / avg_loss
    
    # Kelly formula: f* = (p * b - q) / b
    # where p = win_rate, q = 1 - win_rate, b = win/loss ratio
    q = 1.0 - win_rate
    f_star = (win_rate * b - q) / b
    
    # Cap at 25% of capital (full Kelly can be dangerously large)
    f_star = max(0.0, min(f_star, 0.25))
    
    return round(f_star, 4)


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """
    Calculate fractional Kelly for safer position sizing.
    
    Args:
        win_rate: Probability of winning (0.0–1.0)
        avg_win: Average profit on winning trades (as ratio)
        avg_loss: Average loss on losing trades (as positive ratio)
        fraction: Fraction of full Kelly to use (0.25 = quarter, 0.5 = half)
    
    Returns:
        Fraction of capital to risk per trade (0.0–1.0)
    
    Example:
        fractional_kelly(0.55, 1.02, 1.01, fraction=0.25)  # quarter Kelly
        # => ~0.035 (risk ~3.5% of capital)
    """
    full = kelly_fraction(win_rate, avg_win, avg_loss)
    return round(full * fraction, 4)


def optimal_f_from_trades(
    trades: list,
    fraction: float = 0.5,
    max_f: float = 0.25,
) -> float:
    """
    Calculate Kelly-optimal position size from actual trade history.
    
    Args:
        trades: List of trade dicts with 'pnl_pct' key (profit/loss % as decimals,
                e.g., 0.02 = 2% gain, -0.01 = 1% loss)
        fraction: Fractional Kelly multiplier (default 0.5)
        max_f: Maximum position fraction cap (default 0.25)
    
    Returns:
        Optimal fraction of capital to risk per trade
    
    Example:
        trades = [{'pnl_pct': 0.02}, {'pnl_pct': -0.01}, {'pnl_pct': 0.03}]
        optimal_f_from_trades(trades)  # half Kelly from trade history
    """
    if not trades or len(trades) < 5:
        return 0.0  # Not enough data
    
    wins = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [abs(t["pnl_pct"]) for t in trades if t.get("pnl_pct", 0) < 0]
    
    if not wins or not losses:
        return 0.0
    
    win_rate = len(wins) / len(trades)
    avg_win = 1.0 + (sum(wins) / len(wins)) if wins else 1.0
    avg_loss = 1.0 + (sum(losses) / len(losses)) if losses else 1.0
    
    f = fractional_kelly(win_rate, avg_win, avg_loss, fraction)
    return min(f, max_f)


def position_size_from_kelly(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
    max_risk_pct: float = 0.25,
    min_size: float = 1.0,
) -> tuple:
    """
    Calculate actual USD/EUR position size from Kelly Criterion.
    
    Args:
        capital: Available capital in USD/EUR
        win_rate: Win rate (0.0–1.0)
        avg_win: Average winning trade return as ratio (e.g., 1.02)
        avg_loss: Average losing trade loss as ratio (e.g., 1.01)
        fraction: Fractional Kelly (0.5 = half, 0.25 = quarter)
        max_risk_pct: Maximum fraction of capital to risk (cap)
        min_size: Minimum trade size
    
    Returns:
        (position_size_in_currency, risk_fraction, label)
    
    Example:
        position_size_from_kelly(1000, 0.55, 1.02, 1.01, fraction=0.25)
        # => (35.0, 0.035, "quarter_kelly")
    """
    f = fractional_kelly(win_rate, avg_win, avg_loss, fraction)
    f_capped = min(f, max_risk_pct)
    size = capital * f_capped
    
    size = max(min_size, round(size, 2))
    
    labels = {1.0: "full_kelly", 0.5: "half_kelly", 0.25: "quarter_kelly"}
    label = labels.get(fraction, f"fraction_{fraction}")
    
    return (size, f_capped, label)


def estimate_kelly_from_db(db_stats: dict, fraction: float = 0.5) -> tuple:
    """
    Estimate Kelly-optimal position size from DB strategy stats.
    
    Args:
        db_stats: dict from trader_db.get_strategy_performance()
                  Must have: win_rate, avg_win, avg_loss
        fraction: Fractional Kelly multiplier
    
    Returns:
        (risk_fraction, label, stats_dict)
    """
    if not db_stats:
        return (0.0, "no_stats", {})
    
    win_rate = db_stats.get("win_rate", 0)
    # avg_win and avg_loss are stored as absolute PnL values
    # Convert to ratios: avg_win / entry_price
    avg_win_abs = db_stats.get("avg_win", 0)
    avg_loss_abs = abs(db_stats.get("avg_loss", 0)) if db_stats.get("avg_loss") else 1
    
    if avg_loss_abs < 0.01 or avg_win_abs < 0.01:
        return (0.0, "insufficient_data", db_stats)
    
    # Convert absolute PnL to ratios (assumes typical ~€1 position entry)
    # This is a rough conversion. For precision, pass explicit win/loss ratios.
    avg_win_ratio = 1.0 + (avg_win_abs / 10) if avg_win_abs > 0 else 1.01
    avg_loss_ratio = 1.0 + (avg_loss_abs / 10) if avg_loss_abs > 0 else 1.01
    
    f = fractional_kelly(win_rate, avg_win_ratio, avg_loss_ratio, fraction)
    
    labels = {1.0: "full_kelly", 0.5: "half_kelly", 0.25: "quarter_kelly"}
    label = labels.get(fraction, f"fraction_{fraction}")
    
    return (f, label, db_stats)


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Kelly Criterion Position Sizing ===\n")
    
    # Scenario: 55% win rate, 2% avg win, 1% avg loss
    wr, aw, al = 0.55, 1.02, 1.01
    
    print(f"Win rate: {wr:.0%}, Avg win: {aw-1:.1%}, Avg loss: {al-1:.1%}")
    print(f"  Full Kelly:     {kelly_fraction(wr, aw, al):.2%}")
    print(f"  Half Kelly:     {fractional_kelly(wr, aw, al, 0.5):.2%}")
    print(f"  Quarter Kelly:  {fractional_kelly(wr, aw, al, 0.25):.2%}")
    print()
    
    capital = 1000.0
    size, risk, label = position_size_from_kelly(capital, wr, aw, al, fraction=0.25)
    print(f"With €{capital:.0f} capital:")
    print(f"  Position size: €{size:.2f} ({risk:.2%} of capital) — {label}")
    
    # From trade history
    trades = [
        {"pnl_pct": 0.021}, {"pnl_pct": -0.009}, {"pnl_pct": 0.018},
        {"pnl_pct": 0.032}, {"pnl_pct": -0.011}, {"pnl_pct": -0.008},
        {"pnl_pct": 0.025}, {"pnl_pct": 0.015}, {"pnl_pct": -0.012},
        {"pnl_pct": 0.022},
    ]
    f_hist = optimal_f_from_trades(trades, fraction=0.5)
    print(f"\nFrom 10-trade history:")
    print(f"  Optimal half-Kelly: {f_hist:.2%}")
