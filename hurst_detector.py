#!/usr/bin/env python3
"""
Hurst Exponent Regime Detector

Computes Hurst exponent H via R/S analysis on rolling windows.
Tells the system whether price is:
  - Trending (H > 0.55): use momentum/trend-following strategies
  - Random (0.45 <= H <= 0.55): market noise, reduce size
  - Mean-reverting (H < 0.45): use mean-reversion strategies

Used by: funding sniper, scalper, continuous predictor, AI validator
Paper: RAVEN (arxiv.org/abs/2606.24062) — regime-adaptive context windows
"""
import math
from typing import Optional

def hurst_exponent(prices: list[float], max_lag: int = None) -> float:
    """
    Compute Hurst exponent via R/S (rescaled range) analysis.

    H > 0.5: trending (momentum persists)
    H = 0.5: random walk (Brownian motion)
    H < 0.5: mean-reverting (anti-persistent)

    Args:
        prices: List of closing prices, most recent last
        max_lag: Maximum lag for R/S calculation (default: min(50, len(prices)//2))

    Returns:
        Hurst exponent H (0.0 to 1.0). Returns 0.5 if insufficient data.
    """
    n = len(prices)
    if n < 20:
        return 0.5  # Insufficient data — assume random walk

    if max_lag is None:
        max_lag = min(50, n // 2)

    # Use log returns for stationarity
    returns = []
    for i in range(1, n):
        if prices[i-1] > 0:
            returns.append(math.log(prices[i] / prices[i-1]))
        else:
            returns.append(0)

    if len(returns) < 20:
        return 0.5

    # R/S analysis over different lags
    lags = []
    rs_values = []

    lag_min = max(10, n // 20)
    lag_step = max(1, (max_lag - lag_min) // 10)
    if lag_step < 1:
        lag_step = 1

    for lag in range(lag_min, max_lag + 1, lag_step):
        if lag < 2:
            continue

        # Split returns into non-overlapping chunks of size 'lag'
        chunks = len(returns) // lag
        if chunks < 2:
            continue

        rs_sum = 0.0
        valid_chunks = 0

        for c in range(chunks):
            chunk = returns[c * lag : (c + 1) * lag]
            if len(chunk) < 2:
                continue

            # Mean of chunk
            mean = sum(chunk) / len(chunk)

            # Deviations from mean
            dev = [x - mean for x in chunk]

            # Cumulative sum of deviations
            cum = []
            running = 0
            for d in dev:
                running += d
                cum.append(running)

            # Range R = max(cum) - min(cum)
            r = max(cum) - min(cum)

            # Standard deviation S
            variance = sum(d * d for d in dev) / len(dev)
            s = math.sqrt(variance)

            if s > 0:
                rs_sum += r / s
                valid_chunks += 1

        if valid_chunks > 0:
            lags.append(math.log(lag))
            rs_values.append(math.log(rs_sum / valid_chunks))

    if len(lags) < 3:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    n_points = len(lags)
    sum_x = sum(lags)
    sum_y = sum(rs_values)
    sum_xy = sum(x * y for x, y in zip(lags, rs_values))
    sum_x2 = sum(x * x for x in lags)

    denominator = n_points * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return 0.5

    H = (n_points * sum_xy - sum_x * sum_y) / denominator

    # Clamp to valid range
    return max(0.0, min(1.0, H))


def hurst_regime(H: float) -> str:
    """
    Classify Hurst exponent into actionable regime.

    Returns one of: "trending", "random", "mean_reverting"
    """
    if H > 0.55:
        return "trending"
    elif H < 0.45:
        return "mean_reverting"
    else:
        return "random"


def hurst_confidence(H: float) -> float:
    """
    How confident are we in the regime classification? (0-100)
    Further from 0.5 = more confident.
    """
    return min(100, abs(H - 0.5) * 200)


def should_mean_revert(H: float, min_confidence: float = 40) -> bool:
    """Should mean-reversion strategies be active?"""
    return H < 0.45 and hurst_confidence(H) >= min_confidence


def should_trend_follow(H: float, min_confidence: float = 40) -> bool:
    """Should trend-following strategies be active?"""
    return H > 0.55 and hurst_confidence(H) >= min_confidence


def get_regime_for_weights(H: float, default_regime: str = "sideways") -> str:
    """
    Convert Hurst exponent to a regime string for RAVEN adaptive weighting.

    H > 0.55 → "trending_up" or "trending_down" (need price direction to decide)
    H < 0.45 → "ranging" (mean-reverting → oscillating)
    else → default_regime
    """
    if H > 0.60:
        return "trending_up"  # Strong trend — direction determined elsewhere
    elif H > 0.55:
        return default_regime  # Mild trend — use the passed regime
    elif H < 0.40:
        return "ranging"  # Strong mean reversion
    elif H < 0.45:
        return "ranging"  # Mild mean reversion
    else:
        return default_regime


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)

    # Generate trending price series
    trend_prices = [100.0]
    for i in range(200):
        trend_prices.append(trend_prices[-1] + random.uniform(-0.2, 0.5))

    # Generate mean-reverting price series (Ornstein-Uhlenbeck)
    mean = 100.0
    mr_prices = [mean + random.uniform(-5, 5)]
    for i in range(200):
        next_p = mr_prices[-1] + 0.3 * (mean - mr_prices[-1]) + random.uniform(-0.5, 0.5)
        mr_prices.append(next_p)

    # Generate random walk
    rw_prices = [100.0]
    for i in range(200):
        rw_prices.append(rw_prices[-1] + random.uniform(-0.5, 0.5))

    H_trend = hurst_exponent(trend_prices)
    H_mr = hurst_exponent(mr_prices)
    H_rw = hurst_exponent(rw_prices)

    print(f"Trending series:     H = {H_trend:.3f} → {hurst_regime(H_trend)} ({hurst_confidence(H_trend):.0f}%)")
    print(f"Mean-reverting:      H = {H_mr:.3f} → {hurst_regime(H_mr)} ({hurst_confidence(H_mr):.0f}%)")
    print(f"Random walk:         H = {H_rw:.3f} → {hurst_regime(H_rw)} ({hurst_confidence(H_rw):.0f}%)")

    assert H_trend > 0.55, f"Expected trending H > 0.55, got {H_trend:.3f}"
    assert H_mr < 0.45, f"Expected mean-reverting H < 0.45, got {H_mr:.3f}"
    print("\n✓ Hurst exponent detector ready")
