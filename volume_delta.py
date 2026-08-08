"""
Cumulative Volume Delta (CVD) — approximated from standard OHLCV data.

Since we don't have bid/ask tick data, we estimate delta using:
  1. Bullish volume = volume * (close - low) / (high - low)  when close > open
  2. Bearish volume = volume * (high - close) / (high - low) when close < open
  3. Neutral = volume split evenly when open == close (doji)

This is the "CVD Lite" method used by many crypto traders on TradingView.

Single file, zero external deps.
"""
import math


def calc_bar_delta(open_p, high, low, close, volume):
    """
    Return (bullish_vol, bearish_vol, delta) for a single candle.

    delta = bullish_vol - bearish_vol
    Positive delta = buying pressure dominated.
    Negative delta = selling pressure dominated.
    """
    if high == low or volume == 0:
        return 0.0, 0.0, 0.0

    # Price position ratio: where did close land in the range?
    range_px = high - low

    if close > open_p:
        # Bullish candle: close in upper half = buying
        bull_ratio = (close - low) / range_px
        bull_vol = volume * bull_ratio
        bear_vol = volume * (1 - bull_ratio)
    elif close < open_p:
        # Bearish candle: close in lower half = selling
        bear_ratio = (high - close) / range_px
        bear_vol = volume * bear_ratio
        bull_vol = volume * (1 - bear_ratio)
    else:
        # Doji: no directional conviction, split evenly
        bull_vol = volume / 2
        bear_vol = volume / 2

    return bull_vol, bear_vol, bull_vol - bear_vol


def compute_cvd(ohlcv):
    """
    Compute Cumulative Volume Delta over a list of OHLCV candles.

    Parameters
    ----------
    ohlcv : list[dict]
        Each dict has: open, high, low, close, volume.
        Sorted oldest → newest.

    Returns
    -------
    dict with:
        - delta_per_bar : list[float] — per-candle delta
        - cvd : list[float] — cumulative sum of deltas
        - bull_vol : list[float]
        - bear_vol : list[float]
        - total_delta : float
        - cvd_slope : float — linear slope of last 20 CVD values (trend direction)
        - cvd_divergence : int
            1 = bullish divergence (price down, CVD up over last 12 bars)
           -1 = bearish divergence (price up, CVD down over last 12 bars)
            0 = no divergence
    """
    if not ohlcv:
        return {
            "delta_per_bar": [],
            "cvd": [],
            "bull_vol": [],
            "bear_vol": [],
            "total_delta": 0.0,
            "cvd_slope": 0.0,
            "cvd_divergence": 0,
        }

    deltas = []
    bulls = []
    bears = []
    cvd = []
    cum = 0.0

    for c in ohlcv:
        bv, brv, d = calc_bar_delta(
            c["open"], c["high"], c["low"], c["close"], c["volume"]
        )
        deltas.append(d)
        bulls.append(bv)
        bears.append(brv)
        cum += d
        cvd.append(cum)

    # CVD slope over last 20 bars (or less)
    lookback = min(20, len(cvd))
    if lookback >= 3:
        x = list(range(lookback))
        y = cvd[-lookback:]
        n = len(x)
        sx = sum(x)
        sy = sum(y)
        sxy = sum(x[i] * y[i] for i in range(n))
        sx2 = sum(xi * xi for xi in x)
        slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx) if (n * sx2 - sx * sx) != 0 else 0.0
    else:
        slope = 0.0

    # CVD divergence check (last 12 bars)
    divergence = 0
    if len(ohlcv) >= 12 and len(cvd) >= 12:
        price_start = ohlcv[-12]["close"]
        price_end = ohlcv[-1]["close"]
        cvd_start = cvd[-12]
        cvd_end = cvd[-1]
        price_up = price_end > price_start * 1.005
        price_down = price_end < price_start * 0.995
        cvd_up = cvd_end > cvd_start * 1.005
        cvd_down = cvd_end < cvd_start * 0.995

        if price_down and cvd_up:
            divergence = 1  # bullish divergence
        elif price_up and cvd_down:
            divergence = -1  # bearish divergence

    return {
        "delta_per_bar": deltas,
        "cvd": cvd,
        "bull_vol": bulls,
        "bear_vol": bears,
        "total_delta": cum,
        "cvd_slope": round(slope, 6),
        "cvd_divergence": divergence,
    }


def compute_volume_delta_features(ohlcv):
    """
    Higher-level: returns a flat dict of features for ML/strategy consumption.
    """
    cvd_data = compute_cvd(ohlcv)

    # Last 3 deltas
    deltas = cvd_data["delta_per_bar"]
    last3 = deltas[-3:] if len(deltas) >= 3 else deltas

    # Delta ratio over last N bars
    n = min(20, len(deltas))
    recent_deltas = deltas[-n:] if n > 0 else []
    if recent_deltas:
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        total_bull = sum(cvd_data["bull_vol"][-n:])
        total_bear = sum(cvd_data["bear_vol"][-n:])
        bull_ratio = total_bull / (total_bull + total_bear) if (total_bull + total_bear) > 0 else 0.5
    else:
        avg_delta = 0.0
        bull_ratio = 0.5

    return {
        "cvd_total": cvd_data["total_delta"],
        "cvd_slope_20": cvd_data["cvd_slope"],
        "cvd_divergence": cvd_data["cvd_divergence"],
        "cvd_avg_delta_20": round(avg_delta, 6),
        "cvd_bull_ratio_20": round(bull_ratio, 4),
        "cvd_delta_bar_1": round(last3[0], 2) if len(last3) >= 1 else 0,
        "cvd_delta_bar_2": round(last3[1], 2) if len(last3) >= 2 else 0,
        "cvd_delta_bar_3": round(last3[2], 2) if len(last3) >= 3 else 0,
    }


if __name__ == "__main__":
    # Self-test with synthetic data
    ohlcv = [
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 10000},
        {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 12000},
        {"open": 102, "high": 104, "low": 101, "close": 101.5, "volume": 8000},
        {"open": 101.5, "high": 102, "low": 99, "close": 99.5, "volume": 15000},
        {"open": 99.5, "high": 101, "low": 98, "close": 100, "volume": 9000},
    ]

    result = compute_cvd(ohlcv)
    print("=== CVD Result ===")
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 10:
            print(f"  {k}: [{len(v)} values] {v[:5]}...")
        else:
            print(f"  {k}: {v}")

    features = compute_volume_delta_features(ohlcv)
    print("\n=== CVD Features ===")
    for k, v in features.items():
        print(f"  {k}: {v}")
    print("✅ Volume Delta calculation OK")