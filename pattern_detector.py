"""
Pattern Detector — analyzes OHLCV data across multiple timeframes
Returns rich, formatted technical context for the AI brain to use.
"""
import numpy as np


def _rsi(series, period=14):
    """Compute RSI manually from a price series."""
    deltas = series.diff()
    gain = deltas.clip(lower=0).rolling(window=period).mean()
    loss = (-deltas.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _ema(series, span):
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _atr(df, period=14):
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ),
    )
    return tr.rolling(window=period).mean()


def _detect_support_resistance(df, num_levels=3):
    """
    Find key support / resistance levels by clustering recent lows and highs.
    Returns (support_levels, resistance_levels) as lists of floats.
    """
    if len(df) < 20:
        return [], []

    recent = df.tail(30)
    lows = recent["low"].values
    highs = recent["high"].values

    # Simple binning approach for supports (lows)
    low_sorted = np.sort(lows)
    low_clusters = []
    cluster = [low_sorted[0]]
    for v in low_sorted[1:]:
        if abs(v - cluster[-1]) / cluster[-1] < 0.015:  # within 1.5%
            cluster.append(v)
        else:
            low_clusters.append(np.median(cluster))
            cluster = [v]
    low_clusters.append(np.median(cluster))

    # Highs for resistance
    high_sorted = np.sort(highs)[::-1]
    high_clusters = []
    cluster = [high_sorted[0]]
    for v in high_sorted[1:]:
        if abs(v - cluster[-1]) / max(cluster[-1], 0.0001) < 0.015:
            cluster.append(v)
        else:
            high_clusters.append(np.median(cluster))
            cluster = [v]
    high_clusters.append(np.median(cluster))

    supports = sorted(set(round(x, 4) for x in low_clusters))[:num_levels]
    resistances = sorted(set(round(x, 4) for x in high_clusters), reverse=True)[:num_levels]
    return supports, resistances


def _trend_direction(df):
    """
    Determine trend direction by comparing short vs long EMAs.
    Returns "bullish", "bearish", or "neutral".
    """
    if len(df) < 26:
        return "neutral"
    try:
        ema_fast = _ema(df["close"], 9).iloc[-1]
        ema_mid = _ema(df["close"], 21).iloc[-1]
        ema_slow = _ema(df["close"], 50).iloc[-1] if len(df) >= 50 else None
    except (IndexError, KeyError):
        return "neutral"

    if ema_fast > ema_mid:
        if ema_slow is not None and ema_mid > ema_slow:
            return "bullish"
        return "mildly-bullish"
    elif ema_fast < ema_mid:
        if ema_slow is not None and ema_mid < ema_slow:
            return "bearish"
        return "mildly-bearish"
    return "neutral"


def _volatility_regime(df):
    """
    Classify volatility as "low", "medium", or "high" based on ATR vs close price.
    """
    if len(df) < 14:
        return "unknown"
    try:
        atr_val = _atr(df).iloc[-1]
        close_val = df["close"].iloc[-1]
        atr_pct = atr_val / close_val * 100
        if atr_pct < 0.8:
            return "low"
        elif atr_pct < 2.0:
            return "medium"
        return "high"
    except (IndexError, KeyError):
        return "unknown"


def _volume_anomaly(df, threshold=1.5):
    """
    Check if current volume is significantly above the 20-period average.
    Returns ("spike_up" | "spike_down" | "normal", ratio).
    """
    if len(df) < 20:
        return "normal", 1.0
    try:
        avg_vol = df["volume"].tail(20).mean()
        cur_vol = df["volume"].iloc[-1]
        ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        if ratio > threshold:
            # Check if close moved up or down
            if len(df) >= 2:
                close_change = df["close"].iloc[-1] - df["close"].iloc[-2]
                return ("spike_up" if close_change > 0 else "spike_down", round(ratio, 2))
            return ("spike_up", round(ratio, 2))
        return ("normal", round(ratio, 2))
    except (IndexError, KeyError):
        return "normal", 1.0


def _macd_crossover(df):
    """
    Check MACD line (EMA12 - EMA26) vs signal line (EMA9 of MACD).
    Returns ("bullish_cross" | "bearish_cross" | "no_cross").
    """
    if len(df) < 26:
        return "no_cross"
    try:
        ema12 = _ema(df["close"], 12)
        ema26 = _ema(df["close"], 26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        prev_macd = macd_line.iloc[-2] if len(macd_line) >= 2 else macd_line.iloc[-1]
        prev_sig = signal_line.iloc[-2] if len(signal_line) >= 2 else signal_line.iloc[-1]
        cur_macd = macd_line.iloc[-1]
        cur_sig = signal_line.iloc[-1]

        # Bullish cross: MACD crosses above signal
        if prev_macd <= prev_sig and cur_macd > cur_sig:
            return "bullish_cross"
        # Bearish cross: MACD crosses below signal
        if prev_macd >= prev_sig and cur_macd < cur_sig:
            return "bearish_cross"
        # MACD above signal = bullish momentum
        if cur_macd > cur_sig:
            return "bullish_momentum"
        return "bearish_momentum"
    except (IndexError, KeyError):
        return "no_cross"


def _rsi_signal(df):
    """
    Evaluate RSI extremes.
    Returns ("oversold" | "overbought" | "neutral", rsi_value).
    """
    if len(df) < 14:
        return "neutral", 50
    try:
        rsi_val = _rsi(df["close"]).iloc[-1]
        if np.isnan(rsi_val):
            return "neutral", 50
        rsi_rounded = round(rsi_val, 1)
        if rsi_val < 30:
            return "oversold", rsi_rounded
        elif rsi_val > 70:
            return "overbought", rsi_rounded
        return "neutral", rsi_rounded
    except (IndexError, KeyError):
        return "neutral", 50


def detect_patterns(symbol, df_15m=None, df_1h=None, df_4h=None):
    """
    Main entry point. Analyze OHLCV data across three timeframes.
    Accepts pandas DataFrames with columns: open, high, low, close, volume.
    Any timeframe can be None (skip that analysis).
    Returns a formatted string ready for injection into the AI prompt.
    """
    parts = []
    emojis = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪",
              "mildly-bullish": "🟡", "mildly-bearish": "🟠"}

    # ── Trend analysis per timeframe ──
    trends = {}
    for label, df in [("15m", df_15m), ("1h", df_1h), ("4h", df_4h)]:
        if df is not None:
            trends[label] = _trend_direction(df)
        else:
            trends[label] = "N/A"

    trend_line = "  ".join(
        f"{tf}:{trends[tf]}"
        for tf in ["15m", "1h", "4h"]
    )
    parts.append(f"📈 TECHNICAL: {symbol} {trend_line}")

    # ── RSI on the 1h timeframe (fallback to 15m, then 4h) ──
    rsi_df = df_1h if df_1h is not None else (df_15m if df_15m is not None else df_4h)
    if rsi_df is not None:
        rsi_status, rsi_val = _rsi_signal(rsi_df)
        rsi_emoji = "🔥" if rsi_status == "overbought" else "🧊" if rsi_status == "oversold" else ""
        parts.append(f"  RSI:{rsi_val} {rsi_emoji}{rsi_status}" if rsi_status != "neutral" else f"  RSI:{rsi_val}")
    else:
        parts.append("  RSI:N/A")

    # ── Volatility on 1h ──
    vol_df = df_1h if df_1h is not None else (df_4h if df_4h is not None else df_15m)
    if vol_df is not None:
        regime = _volatility_regime(vol_df)
        vol_emoji = {"low": "😴", "medium": "📊", "high": "⚡"}.get(regime, "")
        parts.append(f"  Vol:{regime}{vol_emoji}")
    else:
        parts.append("  Vol:N/A")

    # ── Volume anomaly on 15m (most recent action) ──
    if df_15m is not None:
        anomaly, ratio = _volume_anomaly(df_15m)
        if anomaly != "normal":
            vol_emoji = "📈" if "up" in anomaly else "📉"
            parts.append(f"  VolSpike:{anomaly.replace('_', ' ')} x{ratio}{vol_emoji}")
        else:
            parts.append(f"  Vol:{ratio:.1f}x avg")
    else:
        parts.append("  Vol:no 15m data")

    # ── MACD on 1h ──
    macd_df = df_1h if df_1h is not None else df_4h
    if macd_df is not None:
        macd = _macd_crossover(macd_df)
        macd_emoji = {"bullish_cross": "✅", "bullish_momentum": "📗",
                      "bearish_cross": "❌", "bearish_momentum": "📕"}.get(macd, "")
        parts.append(f"  MACD:{macd.replace('_', ' ')} {macd_emoji}" if macd_emoji else f"  MACD:{macd.replace('_', ' ')}")
    else:
        parts.append("  MACD:N/A")

    # ── Support / Resistance on 4h (strongest levels) ──
    sr_df = df_4h if df_4h is not None else (df_1h if df_1h is not None else df_15m)
    if sr_df is not None:
        supports, resistances = _detect_support_resistance(sr_df)
        current = sr_df["close"].iloc[-1] if len(sr_df) else 0
        sup_str = ", ".join(f"€{s:.4f}" for s in supports[:2])
        res_str = ", ".join(f"€{r:.4f}" for r in resistances[:2])
        parts.append(f"  Price: €{current:.4f}")
        if sup_str:
            parts.append(f"  Support @ {sup_str}")
        if res_str:
            parts.append(f"  Resistance @ {res_str}")

    # ── Multi-timeframe alignment ──
    tf_signals = [v for v in trends.values() if v not in ("N/A", "neutral", "mildly-bullish", "mildly-bearish")]
    if all(t == "bullish" for t in tf_signals):
        parts.append("  🎯 ALL TIMEFRAMES BULLISH — strong trend confirmation")
    elif all(t == "bearish" for t in tf_signals):
        parts.append("  ⚠️ ALL TIMEFRAMES BEARISH — strong selling pressure")
    elif "bullish" in tf_signals and "bearish" in tf_signals:
        parts.append("  ⚡ Mixed signals across timeframes — trade cautiously")

    return "\n".join(parts)


# ── Standalone test ──
if __name__ == "__main__":
    import pandas as pd

    # Generate synthetic OHLCV data for testing
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="15min")
    close = 1.0 + np.cumsum(np.random.randn(n) * 0.005)
    df_test = pd.DataFrame({
        "open": close - 0.001,
        "high": close + 0.005,
        "low": close - 0.005,
        "close": close,
        "volume": np.random.randint(10000, 50000, size=n),
    }, index=dates)

    print("=" * 60)
    print("TEST: detect_patterns with synthetic data")
    print("=" * 60)
    result = detect_patterns("XRP", df_15m=df_test, df_1h=df_test, df_4h=df_test)
    print(result)