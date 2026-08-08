#!/usr/bin/env python3
"""
Continuous Prediction Engine — "Perfect System" Core

Unlike the old threshold-based stack that only fires at extremes,
this engine outputs CONTINUOUS probability scores for up/down/flat
at 5 horizons (1m, 5m, 15m, 1h, 4h) on every evaluation.

Architecture:
  Layer 1: Order Book Microstructure (bid/ask imbalance, wall detection, absorption)
  Layer 2: Volume Profile Dynamics (POC migration, VA breaks, node transitions)
  Layer 3: CVD / Delta Divergence (cumulative volume delta vs price)
  Layer 4: Multi-TF Technical Alignment (RSI, MACD, BB across timeframes)
  Layer 5: Funding / OI Regime (mean reversion signal)
  Layer 6: ML Ensemble (XGBoost + LSTM) — if models loaded
  Layer 7: Adaptive Weighting (layer weights shift based on recent accuracy)

Output: {"1m": {"up": 45, "down": 30, "flat": 25}, "5m": {...}, ...}
"""
import json, os, time, math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_PATH = DATA_DIR / "continuous_weights.json"
ACCURACY_WINDOW = 20  # Number of recent predictions to track per layer

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class HorizonPrediction:
    """Probability distribution at one time horizon."""
    up: float = 0.0      # 0-100 probability of price going up
    down: float = 0.0    # 0-100 probability of price going down
    flat: float = 0.0    # 0-100 probability of staying flat (±0.3%)
    target_up: float = 0.0    # Predicted price if up
    target_down: float = 0.0  # Predicted price if down
    confidence: float = 0.0   # Overall confidence in this horizon (0-100)

@dataclass(slots=True)
class ContinuousPrediction:
    """Full prediction output at all horizons."""
    coin: str
    current_price: float
    timestamp: float
    horizons: dict[str, HorizonPrediction]  # "1m", "5m", "15m", "1h", "4h"
    layer_contributions: dict[str, float]   # Which layer contributed how much
    overall_bias: str  # "bullish", "bearish", "neutral"
    overall_confidence: float  # 0-100
    layer_confidences: dict[str, float] = field(default_factory=dict)
    horizon_drivers: dict = field(default_factory=dict)  # Per-horizon: [(layer, score), ...]

# ── Layer Weights (adaptive) ─────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "order_book": 0.16,
    "volume_profile": 0.11,
    "cvd_delta": 0.12,
    "multi_tf_technical": 0.14,
    "funding_oi": 0.07,
    "ml_ensemble": 0.13,
    "hurst_regime": 0.05,        # Hurst exponent regime signal (RAVEN paper)
    "higher_tf_align": 0.07,     # 4h/daily trend alignment — counter-trend protection
    "oi_delta": 0.03,            # Open Interest delta — starts low, learns
    "cross_exchange": 0.02,      # Binance divergence — starts low, learns
    "taker_ratio": 0.06,         # Real-time taker buy/sell aggression — strong signal
    "exhaustion": 0.04,          # Peak/bottom exhaustion from RSI divergence + vol climax
    "imbalance_trend": 0.04,     # Bid/ask imbalance building vs fading — trend of pressure
    "bollinger": 0.05,           # Bollinger Bands — overbought/oversold mean reversion
    "sr_levels": 0.05,           # Support/resistance swing levels — bounce/rejection zones
}

def _load_weights() -> dict[str, float]:
    if WEIGHTS_PATH.exists():
        try:
            return json.loads(WEIGHTS_PATH.read_text())
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)

def _save_weights(weights: dict[str, float]):
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))

# ── MTF Alignment Layer (4h + daily trend confirmation) ─────────────────────

def _layer_higher_tf_alignment(coin: str, mid: float,
                                candles_4h: list[dict],
                                candles_1d: list[dict]) -> dict[str, float]:
    """
    Check if higher timeframes confirm or contradict shorter-TF signals.

    This is the single most important protection against counter-trend entries.
    If 15m says BUY but 4h and daily are in a downtrend → reduce confidence heavily.
    If all TFs align → boost confidence.

    Returns confidence-scaled directional bias.
    """
    if not candles_4h and not candles_1d:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    def _close(c):
        return float(c.get("close", c.get("c", 0)))

    def _sma(closes, n):
        if len(closes) < n: return None
        return sum(closes[-n:]) / n

    # ── 4h trend: compare SMA20 vs SMA50 ──
    tf_4h_bullish = tf_4h_bearish = False
    if candles_4h and len(candles_4h) >= 20:
        closes_4h = [_close(c) for c in candles_4h if _close(c) > 0]
        sma20_4h = _sma(closes_4h, 20) if len(closes_4h) >= 20 else None
        sma50_4h = _sma(closes_4h, min(50, len(closes_4h))) if len(closes_4h) >= 30 else None
        if sma20_4h and sma50_4h:
            tf_4h_bullish = sma20_4h > sma50_4h and mid > sma20_4h
            tf_4h_bearish = sma20_4h < sma50_4h and mid < sma20_4h
        elif sma20_4h:
            tf_4h_bullish = mid > sma20_4h
            tf_4h_bearish = mid < sma20_4h

    # ── Daily trend: compare SMA20 vs SMA50 ──
    tf_1d_bullish = tf_1d_bearish = False
    if candles_1d and len(candles_1d) >= 20:
        closes_1d = [_close(c) for c in candles_1d if _close(c) > 0]
        sma20_1d = _sma(closes_1d, 20) if len(closes_1d) >= 20 else None
        sma50_1d = _sma(closes_1d, min(50, len(closes_1d))) if len(closes_1d) >= 30 else None
        if sma20_1d and sma50_1d:
            tf_1d_bullish = sma20_1d > sma50_1d and mid > sma20_1d
            tf_1d_bearish = sma20_1d < sma50_1d and mid < sma20_1d
        elif sma20_1d:
            tf_1d_bullish = mid > sma20_1d
            tf_1d_bearish = mid < sma20_1d

    # ── Score: how aligned are higher TFs? ──
    bull_signals = sum([tf_4h_bullish, tf_1d_bullish])
    bear_signals = sum([tf_4h_bearish, tf_1d_bearish])

    no_data = not (tf_4h_bullish or tf_4h_bearish or tf_1d_bullish or tf_1d_bearish)
    if no_data:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    # Strong alignment: both 4h and daily agree
    if bull_signals == 2 and bear_signals == 0:
        return {"up": 60, "down": 20, "flat": 20, "confidence": 70}
    if bear_signals == 2 and bull_signals == 0:
        return {"up": 20, "down": 60, "flat": 20, "confidence": 70}

    # Partial alignment: one TF agrees
    if bull_signals >= 1 and bear_signals == 0:
        return {"up": 50, "down": 28, "flat": 22, "confidence": 45}
    if bear_signals >= 1 and bull_signals == 0:
        return {"up": 28, "down": 50, "flat": 22, "confidence": 45}

    # Conflicting: 4h and daily disagree → neutral
    return {"up": 33, "down": 33, "flat": 34, "confidence": 20}

# ── Bollinger Bands Layer (price relative to volatility bands) ──────────────

def _layer_bollinger(coin: str, mid: float, candles: list[dict]) -> dict[str, float]:
    """
    Bollinger Bands (20-period, 2 std dev).
    - Price near upper band → overextended, expect reversion down
    - Price near lower band → oversold, expect reversion up
    - Bands squeezing → breakout imminent (low confidence directional)
    - Bands wide → strong trend, follow it
    """
    if not candles or len(candles) < 20:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    def _c(candle):
        return float(candle.get("close", candle.get("c", 0)))

    closes = [_c(c) for c in candles[-20:] if _c(c) > 0]
    if len(closes) < 20:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    mean = sum(closes) / len(closes)
    variance = sum((x - mean) ** 2 for x in closes) / len(closes)
    std = variance ** 0.5
    if std <= 0:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    upper = mean + 2 * std
    lower = mean - 2 * std
    bandwidth = (upper - lower) / mean  # Bandwidth as % of price

    # Position within bands (0 = lower, 1 = upper)
    if upper > lower:
        position = (mid - lower) / (upper - lower)
    else:
        position = 0.5

    # Band squeeze: width < 2% = tight, breakout coming
    is_squeeze = bandwidth < 0.02

    if is_squeeze:
        # Squeeze → neutral directional, moderate confidence for breakout
        return {"up": 40, "down": 40, "flat": 20, "confidence": 25}

    # Near upper band (>90% of band)
    if position > 0.90:
        confidence = min(65, 30 + (position - 0.9) * 300)
        return {"up": 25, "down": 55, "flat": 20, "confidence": confidence}

    # Near lower band (<10% of band)
    if position < 0.10:
        confidence = min(65, 30 + (0.1 - position) * 300)
        return {"up": 55, "down": 25, "flat": 20, "confidence": confidence}

    # Above middle → slight bullish
    if position > 0.60:
        conf = 10 + (position - 0.5) * 40
        return {"up": 40, "down": 30, "flat": 30, "confidence": conf}
    # Below middle → slight bearish
    if position < 0.40:
        conf = 10 + (0.5 - position) * 40
        return {"up": 30, "down": 40, "flat": 30, "confidence": conf}

    return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

# ── Support/Resistance Swing Levels Layer ───────────────────────────────────

def _layer_sr_levels(coin: str, mid: float, candles: list[dict]) -> dict[str, float]:
    """
    Detect swing highs and lows to identify support/resistance.
    - Price near resistance → bearish (expect rejection)
    - Price near support → bullish (expect bounce)
    - Multiple touches = stronger level
    """
    if not candles or len(candles) < 20:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    def _h(c): return float(c.get("high", c.get("h", 0)))
    def _l(c): return float(c.get("low", c.get("l", 0)))
    def _c(c): return float(c.get("close", c.get("c", 0)))

    # Find swing highs (local maxima) and swing lows (local minima)
    # with lookback of 3 candles each side
    lookback = 3
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(candles) - lookback):
        window_highs_before = [_h(candles[j]) for j in range(i - lookback, i) if _h(candles[j]) > 0]
        window_highs_after = [_h(candles[j]) for j in range(i + 1, i + lookback + 1) if _h(candles[j]) > 0]
        window_lows_before = [_l(candles[j]) for j in range(i - lookback, i) if _l(candles[j]) > 0]
        window_lows_after = [_l(candles[j]) for j in range(i + 1, i + lookback + 1) if _l(candles[j]) > 0]
        current_high = _h(candles[i])
        current_low = _l(candles[i])

        if current_high > 0 and window_highs_before and window_highs_after:
            if current_high > max(window_highs_before) and current_high > max(window_highs_after):
                swing_highs.append(current_high)
        if current_low > 0 and window_lows_before and window_lows_after:
            if current_low < min(window_lows_before) and current_low < min(window_lows_after):
                swing_lows.append(current_low)

    # Cluster nearby levels (±0.5% of price)
    def cluster_levels(levels, threshold_pct=0.005):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - current_cluster[-1]) / current_cluster[-1] < threshold_pct:
                current_cluster.append(lvl)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [lvl]
        clusters.append(sum(current_cluster) / len(current_cluster))
        return clusters

    resistances = cluster_levels(swing_highs)
    supports = cluster_levels(swing_lows)

    # Find nearest resistance and support
    nearest_res = None
    nearest_sup = None
    for r in resistances:
        if r > mid and (nearest_res is None or r < nearest_res):
            nearest_res = r
    for s in supports:
        if s < mid and (nearest_sup is None or s > nearest_sup):
            nearest_sup = s

    # Distance to nearest S/R as % of price
    res_dist = ((nearest_res - mid) / mid * 100) if nearest_res else 999
    sup_dist = ((mid - nearest_sup) / mid * 100) if nearest_sup else 999

    # ── Score based on proximity ──
    # Very close to resistance → bearish (rejection likely)
    if res_dist < 1.0:
        confidence = min(75, 60 - res_dist * 20)
        return {"up": 25, "down": 55, "flat": 20, "confidence": confidence}
    if res_dist < 3.0:
        confidence = 30 - res_dist * 5
        return {"up": 30, "down": 45, "flat": 25, "confidence": max(15, confidence)}

    # Very close to support → bullish (bounce likely)
    if sup_dist < 1.0:
        confidence = min(75, 60 - sup_dist * 20)
        return {"up": 55, "down": 25, "flat": 20, "confidence": confidence}
    if sup_dist < 3.0:
        confidence = 30 - sup_dist * 5
        return {"up": 45, "down": 30, "flat": 25, "confidence": max(15, confidence)}

    # Mid-range — neutral
    return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

# ── Layer 1: Order Book Microstructure ──────────────────────────────────────

def _layer_order_book(coin: str, mid: float, order_book: Optional[dict] = None,
                       funding_rate: float = 0.0) -> dict[str, float]:
    """
    Analyze order book for directional bias.
    Returns {"up": 0-100, "down": 0-100, "flat": 0-100, "confidence": 0-100}
    """
    if not order_book or not order_book.get("bids") or not order_book.get("asks"):
        # Fallback: use funding rate as proxy for positioning
        # Negative funding → shorts paying → contrarian long bias
        # Positive funding → longs paying → contrarian short bias
        if abs(funding_rate) > 0.0001:  # >0.01%/hr
            if funding_rate > 0.005:  # >0.5%/hr — extreme long positioning
                return {"up": 25, "down": 50, "flat": 25, "confidence": 45}
            elif funding_rate < -0.003:  # extreme short positioning
                return {"up": 50, "down": 25, "flat": 25, "confidence": 40}
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    # ── Normalize accessors: Hyperliquid WS returns [{px, sz, n}, ...] dicts ──
    def _px(level):
        if isinstance(level, dict): return float(level["px"])
        return float(level[0])
    def _sz(level):
        if isinstance(level, dict): return float(level["sz"])
        return float(level[1])

    # Calculate depth at 0.5%, 1%, 2% from mid
    bid_depth_05 = sum(_sz(b) for b in bids if _px(b) >= mid * 0.995)
    bid_depth_1 = sum(_sz(b) for b in bids if _px(b) >= mid * 0.99)
    bid_depth_2 = sum(_sz(b) for b in bids if _px(b) >= mid * 0.98)
    ask_depth_05 = sum(_sz(a) for a in asks if _px(a) <= mid * 1.005)
    ask_depth_1 = sum(_sz(a) for a in asks if _px(a) <= mid * 1.01)
    ask_depth_2 = sum(_sz(a) for a in asks if _px(a) <= mid * 1.02)

    total_bid = max(bid_depth_2, 0.001)
    total_ask = max(ask_depth_2, 0.001)
    imbalance = (total_bid - total_ask) / (total_bid + total_ask)  # -1 to +1

    # Wall detection: sudden large orders
    top_bid_size = _sz(bids[0]) if bids else 0
    top_ask_size = _sz(asks[0]) if asks else 0
    avg_bid_size = sum(_sz(b) for b in bids[:10]) / max(len(bids[:10]), 1)
    avg_ask_size = sum(_sz(a) for a in asks[:10]) / max(len(asks[:10]), 1)
    bid_wall = top_bid_size > avg_bid_size * 3
    ask_wall = top_ask_size > avg_ask_size * 3

    # Spread
    best_bid = _px(bids[0]) if bids else mid * 0.999
    best_ask = _px(asks[0]) if asks else mid * 1.001
    spread_pct = (best_ask - best_bid) / mid * 100

    score = 50  # Start neutral

    # Imbalance contribution: ±25 max
    score += imbalance * 25

    # Wall contribution: ±15
    if bid_wall and not ask_wall:
        score += 15  # Support wall → bullish
    elif ask_wall and not bid_wall:
        score -= 15  # Resistance wall → bearish

    # Spread: tight spread = indecision/coiling → neutral
    # Wide spread = low liquidity → reduce confidence
    confidence = 40
    if spread_pct > 0.5:
        confidence -= 15  # Wide spread = unreliable order book

    # ── Queue Imbalance (L1 OFI) — strongest microstructure predictor ──
    # bid_qty / (bid_qty + ask_qty). >0.65 = bullish, <0.35 = bearish.
    # Research: deep-ofi (github.com/jaefit/deep-ofi), alpha-engine
    bid_qty = sum(_sz(b) for b in bids[:3])
    ask_qty = sum(_sz(a) for a in asks[:3])
    total_qty = bid_qty + ask_qty
    if total_qty > 0:
        queue_imb = bid_qty / total_qty  # 0-1, 0.5 = balanced
        if queue_imb > 0.65:
            score += 12  # Heavy bid queue → bullish pressure
            confidence += 10
        elif queue_imb < 0.35:
            score -= 12  # Heavy ask queue → bearish pressure
            confidence += 10
        elif queue_imb > 0.55:
            score += 5   # Slight bid bias
        elif queue_imb < 0.45:
            score -= 5   # Slight ask bias

    # Depth ratio at 1%: bid depth vs ask depth
    depth_ratio = bid_depth_1 / max(ask_depth_1, 0.001)
    if depth_ratio > 2:
        score += 10  # Heavy bid support
    elif depth_ratio < 0.5:
        score -= 10  # Heavy ask resistance

    score = max(0, min(100, score))
    up = score
    down = 100 - score
    flat = max(0, 100 - abs(up - 50) * 2)  # More flat when near 50
    flat = min(flat, 50)

    # Normalize
    total = up + down + flat
    if total > 0:
        up = up / total * 100
        down = down / total * 100
        flat = flat / total * 100

    return {"up": round(up, 1), "down": round(down, 1), "flat": round(flat, 1),
            "confidence": min(100, confidence)}

# ── Layer 2: Volume Profile Dynamics ────────────────────────────────────────

def _layer_volume_profile(coin: str, mid: float, candles: list[dict]) -> dict[str, float]:
    """
    Track POC (Point of Control) migration and value area shifts.
    A rising POC with price = accumulation (bullish).
    A falling POC with price = distribution (bearish).
    Price breaking above VA high = breakout (bullish).
    Price breaking below VA low = breakdown (bearish).
    """
    if len(candles) < 20:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    # Build volume profile from last 100 candles
    recent = candles[-100:]
    prices = []
    volumes = []
    for c in recent:
        try:
            p = (float(c["h"]) + float(c["l"]) + float(c["c"])) / 3  # Typical price
            v = float(c["v"])
            prices.append(p)
            volumes.append(v)
        except (KeyError, ValueError):
            continue

    if len(prices) < 10:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    # Simple POC: highest volume price zone
    price_range = max(prices) - min(prices) if prices else 1
    if price_range <= 0:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    num_zones = min(20, len(prices) // 2)
    zone_width = price_range / num_zones
    zones = {}
    for p, v in zip(prices, volumes):
        zone = int((p - min(prices)) / zone_width)
        zone = min(zone, num_zones - 1)
        zones[zone] = zones.get(zone, 0) + v

    if not zones:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    poc_zone = max(zones, key=zones.get)
    poc_price = min(prices) + (poc_zone + 0.5) * zone_width

    # Value area: 70% of volume
    total_vol = sum(zones.values())
    sorted_zones = sorted(zones.items(), key=lambda x: x[1], reverse=True)
    cum_vol = 0
    va_zones = set()
    for z, v in sorted_zones:
        cum_vol += v
        va_zones.add(z)
        if cum_vol >= total_vol * 0.7:
            break

    va_high = min(prices) + (max(va_zones) + 1) * zone_width if va_zones else poc_price
    va_low = min(prices) + min(va_zones) * zone_width if va_zones else poc_price

    # Score based on price position relative to POC and VA
    score = 50

    # POC relative to price
    if mid > poc_price:
        score += 10  # Price above POC → bullish structure
    else:
        score -= 10  # Price below POC → bearish structure

    # VA position
    if mid > va_high:
        score += 15  # Above value area → strong bullish
        confidence = 60
    elif mid < va_low:
        score -= 15  # Below value area → strong bearish
        confidence = 60
    elif mid > poc_price:
        score += 5   # In value area, above POC → mild bullish
        confidence = 35
    else:
        score -= 5   # In value area, below POC → mild bearish
        confidence = 35

    # POC migration speed (last 3 POCs vs first 3)
    if len(recent) >= 50:
        first_slice = recent[:50]
        last_slice = recent[-50:]
        first_prices = [(float(c["h"]) + float(c["l"]) + float(c["c"])) / 3 for c in first_slice]
        last_prices = [(float(c["h"]) + float(c["l"]) + float(c["c"])) / 3 for c in last_slice]

        first_avg = sum(first_prices) / len(first_prices) if first_prices else mid
        last_avg = sum(last_prices) / len(last_prices) if last_prices else mid
        poc_delta_pct = (last_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0

        if poc_delta_pct > 2:
            score += 10
            confidence += 10
        elif poc_delta_pct < -2:
            score -= 10
            confidence += 10

    score = max(0, min(100, score))
    up = score
    down = 100 - score
    flat = max(0, 100 - abs(up - 50) * 2)
    flat = min(flat, 40)

    total = up + down + flat
    if total > 0:
        up = up / total * 100
        down = down / total * 100
        flat = flat / total * 100

    return {"up": round(up, 1), "down": round(down, 1), "flat": round(flat, 1),
            "confidence": min(100, confidence)}

# ── Layer 3: CVD / Delta Divergence ──────────────────────────────────────────

def _layer_cvd_delta(coin: str, mid: float, candles: list[dict]) -> dict[str, float]:
    """
    Cumulative Volume Delta divergence detection.
    If price makes higher high but CVD makes lower high → bearish divergence.
    If price makes lower low but CVD makes higher low → bullish divergence.
    """
    if len(candles) < 30:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    try:
        # Estimate delta from candle position (close relative to range)
        # close in upper third = buying pressure; lower third = selling
        deltas = []
        for c in candles[-50:]:
            try:
                h = float(c["h"]); l = float(c["l"]); o = float(c["o"])
                close = float(c["c"]); vol = float(c["v"])
                if h == l:
                    deltas.append(0)
                    continue
                # Close position in range: 0 (at low) to 1 (at high)
                pos = (close - l) / (h - l)
                # Convert to delta: -vol to +vol
                delta = (pos - 0.5) * 2 * vol
                deltas.append(delta)
            except (KeyError, ValueError):
                continue

        if len(deltas) < 10:
            return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

        # CVD
        cvd = []
        running = 0
        for d in deltas:
            running += d
            cvd.append(running)

        # Price highs/lows from last 20 candles
        closes = []
        for c in candles[-20:]:
            try:
                closes.append(float(c["c"]))
            except (KeyError, ValueError):
                continue

        if len(closes) < 10 or len(cvd) < 20:
            return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

        # Find recent price high and CVD at that point
        recent_closes = closes[-10:]
        recent_cvd = cvd[-10:] if len(cvd) >= 10 else cvd

        max_close = max(recent_closes)
        max_close_idx = recent_closes.index(max_close)
        cvd_at_max = recent_cvd[max_close_idx] if max_close_idx < len(recent_cvd) else 0

        # Compare to earlier period
        earlier_closes = closes[:10] if len(closes) > 10 else []
        earlier_cvd = cvd[5:15] if len(cvd) >= 15 else []

        if earlier_closes and earlier_cvd:
            prev_max = max(earlier_closes)
            prev_max_idx = earlier_closes.index(prev_max)
            prev_cvd = earlier_cvd[prev_max_idx] if prev_max_idx < len(earlier_cvd) else 0

            # Bearish divergence: price higher high, CVD lower high
            if max_close > prev_max * 1.005 and cvd_at_max < prev_cvd:
                return {"up": 15, "down": 65, "flat": 20, "confidence": 55}

            # Bullish divergence: price lower low, CVD higher low
            min_close = min(recent_closes)
            prev_min = min(earlier_closes)
            min_idx = recent_closes.index(min_close)
            prev_min_idx = earlier_closes.index(prev_min)
            cvd_at_min = recent_cvd[min_idx] if min_idx < len(recent_cvd) else 0
            prev_cvd_min = earlier_cvd[prev_min_idx] if prev_min_idx < len(earlier_cvd) else 0

            if min_close < prev_min * 0.995 and cvd_at_min > prev_cvd_min:
                return {"up": 65, "down": 15, "flat": 20, "confidence": 55}

        # ── Absorption & Stacked Imbalance (CryptoFlowEngine research) ──
        try:
            from cvd_divergence import detect_absorption, detect_stacked_imbalance
            absorption = detect_absorption(candles)
            stacked = detect_stacked_imbalance(candles)
            
            if absorption["type"] != "none" and absorption["confidence"] > 40:
                if absorption["type"] == "bullish":
                    return {"up": 60, "down": 20, "flat": 20, "confidence": absorption["confidence"]}
                elif absorption["type"] == "bearish":
                    return {"up": 20, "down": 60, "flat": 20, "confidence": absorption["confidence"]}
            
            if stacked["direction"] != "none" and stacked["confidence"] > 45:
                if stacked["direction"] == "bullish":
                    return {"up": 55 + stacked["count"] * 5, "down": 30 - stacked["count"] * 3,
                            "flat": 15, "confidence": stacked["confidence"]}
                else:
                    return {"up": 30 - stacked["count"] * 3, "down": 55 + stacked["count"] * 5,
                            "flat": 15, "confidence": stacked["confidence"]}
        except Exception:
            pass

        # CVD trend direction
        cvd_trend = recent_cvd[-1] - recent_cvd[0] if len(recent_cvd) > 1 else 0
        if cvd_trend > 0:
            return {"up": 50, "down": 25, "flat": 25, "confidence": 30}
        elif cvd_trend < 0:
            return {"up": 25, "down": 50, "flat": 25, "confidence": 30}

        return {"up": 33, "down": 33, "flat": 34, "confidence": 15}

    except Exception:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

# ── Layer 4: Multi-TF Technical Alignment ────────────────────────────────────

def _layer_multi_tf_technical(coin: str, mid: float, candles_1m: list[dict],
                                candles_5m: list[dict], candles_15m: list[dict],
                                candles_1h: list[dict]) -> dict[str, float]:
    """
    Check RSI and MACD alignment across timeframes.
    More timeframes agreeing = higher confidence.
    """
    def calc_rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50
        gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
        losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calc_macd(closes):
        if len(closes) < 26:
            return 0, 0
        ema12 = sum(closes[-12:]) / 12
        ema26 = sum(closes[-26:]) / 26
        for i in range(5):  # Smooth
            if len(closes) > 12 + i:
                ema12 = closes[-(12+i)] * (2/13) + ema12 * (11/13)
            if len(closes) > 26 + i:
                ema26 = closes[-(26+i)] * (2/27) + ema26 * (25/27)
        macd_line = ema12 - ema26
        return macd_line, macd_line * 0.5  # Signal approx

    def get_closes(candles):
        return [float(c["c"]) for c in candles if "c" in c]

    scores = []
    confidences = []

    for tf_name, candles in [("1m", candles_1m), ("5m", candles_5m),
                               ("15m", candles_15m), ("1h", candles_1h)]:
        closes = get_closes(candles)
        if len(closes) < 20:
            continue

        rsi = calc_rsi(closes)
        macd, signal = calc_macd(closes)

        tf_score = 50

        # RSI contribution
        if rsi < 30:
            tf_score += 15  # Oversold → bullish
        elif rsi > 70:
            tf_score -= 15  # Overbought → bearish
        elif rsi > 60:
            tf_score += 5   # Momentum up
        elif rsi < 40:
            tf_score -= 5   # Momentum down

        # MACD contribution
        if macd > signal:
            tf_score += 10
        else:
            tf_score -= 10

        scores.append(tf_score)
        confidences.append(40)  # Base confidence per timeframe

    if not scores:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}

    avg_score = sum(scores) / len(scores)
    # More agreement = higher confidence
    agreement = 1.0 - (max(scores) - min(scores)) / 100 if len(scores) > 1 else 0.5
    confidence = min(80, sum(confidences) / len(confidences) * agreement)

    avg_score = max(0, min(100, avg_score))
    up = avg_score
    down = 100 - avg_score
    flat = max(0, 100 - abs(up - 50) * 2)
    flat = min(flat, 40)

    total = up + down + flat
    if total > 0:
        up = up / total * 100
        down = down / total * 100
        flat = flat / total * 100

    return {"up": round(up, 1), "down": round(down, 1), "flat": round(flat, 1),
            "confidence": round(confidence, 1)}

# ── Layer 5: Funding / OI Regime ────────────────────────────────────────────

def _layer_funding_oi(coin: str, mid: float, funding_rate: float,
                       open_interest: float = 0) -> dict[str, float]:
    """
    Funding rate mean reversion signal.
    Extreme positive funding → too many longs → contrarian short bias.
    Extreme negative funding → too many shorts → contrarian long bias.
    """
    abs_funding = abs(funding_rate)
    hourly_pct = abs_funding * 100  # Convert to % per hour

    if hourly_pct < 0.001:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # Funding severity (0-100)
    # 0.01%/hr = mild, 0.1%/hr = extreme
    severity = min(100, hourly_pct * 1000)

    confidence = min(70, severity * 1.2)

    if funding_rate > 0:
        # Longs paying → contrarian short
        return {"up": round(30 - severity * 0.2, 1),
                "down": round(50 + severity * 0.2, 1),
                "flat": max(10, round(20 - severity * 0.1, 1)),
                "confidence": round(confidence, 1)}
    else:
        # Shorts paying → contrarian long
        return {"up": round(50 + severity * 0.2, 1),
                "down": round(30 - severity * 0.2, 1),
                "flat": max(10, round(20 - severity * 0.1, 1)),
                "confidence": round(confidence, 1)}

# ── Main Engine ──────────────────────────────────────────────────────────────

def predict_continuous(
    coin: str,
    current_price: float,
    candles_1m: list[dict] = None,
    candles_5m: list[dict] = None,
    candles_15m: list[dict] = None,
    candles_1h: list[dict] = None,
    candles_4h: list[dict] = None,
    candles_1d: list[dict] = None,
    order_book: dict = None,
    funding_rate: float = 0.0,
    open_interest: float = 0.0,
    oi_delta: dict = None,     # OI change signal {"direction": "up"/"down"/"flat", "confidence": 0-100}
    ml_signal: dict = None,    # From unified_predictor
    perfect_signal = None,     # From perfect_predictor
    exhaustion_signal = None,  # From peak_exhaustion_detector
    hurst_H: float = 0.5,      # Hurst exponent (0-1) — regime filter
    cvd_override: dict = None, # Real CVD from WebSocket trades (replaces OHLCV proxy)
    cross_exchange_divergence: dict = None,  # {"divergence_pct": X, "direction": "hl_premium"/"hl_discount"}
    taker_ratio: dict = None,  # {"direction": "bullish"/"bearish", "confidence": 0-100}
    imbalance_trend: dict = None,  # {"signal": "bullish"/"bearish", "confidence": 0-100}
) -> ContinuousPrediction:
    """
    THE CORE. Run all layers and produce continuous probabilities at every horizon.

    Returns probability distributions for up/down/flat at 1m, 5m, 15m, 1h, 4h, 1d.
    No thresholds. No binary signals. Pure probability.
    """
    candles_1m = candles_1m or []
    candles_5m = candles_5m or []
    candles_15m = candles_15m or []
    candles_1h = candles_1h or []
    candles_4h = candles_4h or []
    candles_1d = candles_1d or []

    weights = _load_weights()
    layer_results = {}

    # ── Run all layers ──
    layer_results["order_book"] = _layer_order_book(coin, current_price, order_book, funding_rate)
    layer_results["volume_profile"] = _layer_volume_profile(coin, current_price, candles_15m)
    layer_results["cvd_delta"] = _layer_cvd_delta(coin, current_price, candles_15m)
    layer_results["bollinger"] = _layer_bollinger(coin, current_price, candles_15m)
    layer_results["sr_levels"] = _layer_sr_levels(coin, current_price, candles_15m)
    
    # ── Override CVD with real trade-level data when available ──
    if cvd_override and cvd_override.get("confidence", 0) > 10:
        layer_results["cvd_delta"] = cvd_override  # Real delta beats OHLCV proxy
    layer_results["multi_tf_technical"] = _layer_multi_tf_technical(
        coin, current_price, candles_1m, candles_5m, candles_15m, candles_1h)
    layer_results["funding_oi"] = _layer_funding_oi(coin, current_price, funding_rate, open_interest)

    # ML ensemble integration
    if ml_signal and hasattr(ml_signal, 'direction') and hasattr(ml_signal, 'confidence'):
        ml_conf = ml_signal.confidence
        ml_dir = ml_signal.direction
        if ml_dir == "up":
            layer_results["ml_ensemble"] = {"up": 40 + ml_conf * 0.5, "down": 30 - ml_conf * 0.3,
                                             "flat": 30 - ml_conf * 0.2, "confidence": ml_conf}
        elif ml_dir == "down":
            layer_results["ml_ensemble"] = {"up": 30 - ml_conf * 0.3, "down": 40 + ml_conf * 0.5,
                                             "flat": 30 - ml_conf * 0.2, "confidence": ml_conf}
        else:
            layer_results["ml_ensemble"] = {"up": 30, "down": 30, "flat": 40, "confidence": ml_conf * 0.5}
    else:
        layer_results["ml_ensemble"] = {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # ── Hurst regime layer (trending vs mean-reverting detection) ──
    # H > 0.55: trending → momentum bias (slight directional tilt)
    # H < 0.45: mean-reverting → contrarian bias
    # H ≈ 0.50: random → neutral, low confidence
    from hurst_detector import hurst_regime as hr, hurst_confidence as hc
    regime = hr(hurst_H)
    conf = hc(hurst_H)
    if regime == "trending":
        # Trending: slight momentum bias — don't fight the trend
        layer_results["hurst_regime"] = {"up": 45, "down": 35, "flat": 20, "confidence": conf * 0.5}
    elif regime == "mean_reverting":
        # Mean-reverting: contrarian bias — fade the move
        layer_results["hurst_regime"] = {"up": 35, "down": 35, "flat": 30, "confidence": conf * 0.5}
    else:
        # Random: no bias, very low confidence
        layer_results["hurst_regime"] = {"up": 33, "down": 33, "flat": 34, "confidence": conf * 0.3}

    # ── Higher TF alignment layer (4h + daily trend confirmation) ──
    # If short TFs say one thing but 4h/daily say the opposite, reduce confidence.
    # If all TFs align, boost confidence. This prevents counter-trend entries.
    layer_results["higher_tf_align"] = _layer_higher_tf_alignment(
        coin, current_price, candles_4h, candles_1d)

    # ── OI Delta layer (Open Interest change direction) ──
    # Rising OI + price = trend confirmation. Divergence = reversal signal.
    if oi_delta and oi_delta.get("confidence", 0) > 0:
        oi_dir = oi_delta.get("direction", "flat")
        oi_conf = oi_delta.get("confidence", 10)
        if oi_dir == "up":
            layer_results["oi_delta"] = {"up": 40 + oi_conf * 0.3, "down": 30 - oi_conf * 0.2,
                                          "flat": 30 - oi_conf * 0.1, "confidence": oi_conf}
        elif oi_dir == "down":
            layer_results["oi_delta"] = {"up": 30 - oi_conf * 0.2, "down": 40 + oi_conf * 0.3,
                                          "flat": 30 - oi_conf * 0.1, "confidence": oi_conf}
        else:
            layer_results["oi_delta"] = {"up": 33, "down": 33, "flat": 34, "confidence": oi_conf * 0.3}
    else:
        layer_results["oi_delta"] = {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # ── Cross-exchange divergence filter ──
    # If HL price is significantly different from Binance, flag potential manipulation
    if cross_exchange_divergence and cross_exchange_divergence.get("divergence_pct", 0) > 1.0:
        div_pct = cross_exchange_divergence.get("divergence_pct", 0)
        div_dir = cross_exchange_divergence.get("direction", "")
        # High divergence = lower confidence in HL signal
        div_penalty = min(30, div_pct * 8)
        layer_results["cross_exchange"] = {
            "up": 33, "down": 33, "flat": 34,
            "confidence": max(5, 50 - div_penalty),
            "divergence_warning": f"HL vs Binance: {div_pct:.1f}% {div_dir}"
        }
    else:
        layer_results["cross_exchange"] = {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # ── Taker buy/sell ratio layer (real-time order flow aggression) ──
    # Tells you who's in control RIGHT NOW from actual trade direction.
    if taker_ratio and taker_ratio.get("confidence", 0) > 5:
        tk_dir = taker_ratio.get("direction", "neutral")
        tk_conf = taker_ratio.get("confidence", 10)
        if tk_dir == "bullish":
            layer_results["taker_ratio"] = {"up": 40 + tk_conf * 0.35, "down": 30 - tk_conf * 0.2,
                                             "flat": 30 - tk_conf * 0.15, "confidence": tk_conf}
        elif tk_dir == "bearish":
            layer_results["taker_ratio"] = {"up": 30 - tk_conf * 0.2, "down": 40 + tk_conf * 0.35,
                                             "flat": 30 - tk_conf * 0.15, "confidence": tk_conf}
        else:
            layer_results["taker_ratio"] = {"up": 33, "down": 33, "flat": 34, "confidence": tk_conf * 0.2}
    else:
        layer_results["taker_ratio"] = {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # ── Exhaustion layer (peak/bottom detection from RSI divergence + volume climax) ──
    # Was previously dead (exh=2) because exhaustion_signal was always None.
    # Now properly wired from peak_exhaustion_detector.detect_peak_exhaustion().
    if exhaustion_signal and getattr(exhaustion_signal, 'score', 0) > 0:
        exh_score = exhaustion_signal.score
        exh_dir = exhaustion_signal.direction  # "up" = bottom (bullish rev), "down" = peak (bearish rev)
        if exh_dir == "down":  # Peak detected → bearish exhaustion → expect reversal down
            layer_results["exhaustion"] = {"up": 30 - exh_score * 0.3, "down": 40 + exh_score * 0.35,
                                            "flat": 30 - exh_score * 0.05, "confidence": min(80, exh_score * 0.8)}
        elif exh_dir == "up":  # Bottom detected → bullish exhaustion → expect reversal up
            layer_results["exhaustion"] = {"up": 40 + exh_score * 0.35, "down": 30 - exh_score * 0.3,
                                            "flat": 30 - exh_score * 0.05, "confidence": min(80, exh_score * 0.8)}
        else:
            layer_results["exhaustion"] = {"up": 33, "down": 33, "flat": 34, "confidence": exh_score * 0.3}
    else:
        layer_results["exhaustion"] = {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # ── Bid/Ask imbalance trend (building vs fading pressure) ──
    # Is buying pressure accelerating or decelerating? The TREND matters more than the snapshot.
    if imbalance_trend and imbalance_trend.get("confidence", 0) > 5:
        imb_dir = imbalance_trend.get("signal", "neutral")
        imb_conf = imbalance_trend.get("confidence", 10)
        if imb_dir == "bullish":
            layer_results["imbalance_trend"] = {"up": 38 + imb_conf * 0.3, "down": 32 - imb_conf * 0.15,
                                                 "flat": 30 - imb_conf * 0.15, "confidence": imb_conf}
        elif imb_dir == "bearish":
            layer_results["imbalance_trend"] = {"up": 32 - imb_conf * 0.15, "down": 38 + imb_conf * 0.3,
                                                 "flat": 30 - imb_conf * 0.15, "confidence": imb_conf}
        else:
            layer_results["imbalance_trend"] = {"up": 33, "down": 33, "flat": 34, "confidence": imb_conf * 0.2}
    else:
        layer_results["imbalance_trend"] = {"up": 33, "down": 33, "flat": 34, "confidence": 5}

    # ── Weighted ensemble across layers ──
    total_weight = sum(weights.get(k, 0) for k in layer_results)
    if total_weight <= 0:
        total_weight = 1.0

    weighted_up = 0; weighted_down = 0; weighted_flat = 0; weighted_conf = 0
    layer_contributions = {}
    layer_confidences = {}

    for layer_name, result in layer_results.items():
        w = weights.get(layer_name, 0) / total_weight
        layer_conf = result.get("confidence", 10)
        weighted_up += result["up"] * w * (layer_conf / 50)  # Confidence-adjusted
        weighted_down += result["down"] * w * (layer_conf / 50)
        weighted_flat += result["flat"] * w * (layer_conf / 50)
        weighted_conf += layer_conf * w
        layer_contributions[layer_name] = round(w * 100, 1)
        layer_confidences[layer_name] = round(layer_conf, 1)

    # Normalize
    total = weighted_up + weighted_down + weighted_flat
    if total > 0:
        weighted_up = weighted_up / total * 100
        weighted_down = weighted_down / total * 100
        weighted_flat = weighted_flat / total * 100

    # ── Distribute to horizons with time-based decay ──
    # Short horizons: weight order_book + technical more
    # Long horizons: weight volume_profile + funding + higher_tf more
    horizon_configs = {
        "1m":  {"ob": 1.5, "tech": 1.3, "cvd": 1.2, "vp": 0.7, "funding": 0.5, "ml": 0.8, "hurst": 0.3, "htf": 0.2, "oi": 0.5, "xex": 0.2, "tkr": 1.3, "exh": 0.8, "imb": 1.2, "bb": 0.9, "sr": 0.8},
        "5m":  {"ob": 1.2, "tech": 1.2, "cvd": 1.2, "vp": 0.9, "funding": 0.7, "ml": 0.9, "hurst": 0.4, "htf": 0.3, "oi": 0.7, "xex": 0.3, "tkr": 1.1, "exh": 0.7, "imb": 1.0, "bb": 1.0, "sr": 0.9},
        "15m": {"ob": 0.8, "tech": 1.1, "cvd": 1.1, "vp": 1.2, "funding": 0.9, "ml": 1.0, "hurst": 0.6, "htf": 0.6, "oi": 0.9, "xex": 0.5, "tkr": 0.9, "exh": 0.8, "imb": 0.8, "bb": 1.1, "sr": 1.0},
        "1h":  {"ob": 0.4, "tech": 0.9, "cvd": 0.9, "vp": 1.4, "funding": 1.2, "ml": 1.2, "hurst": 0.8, "htf": 0.9, "oi": 1.1, "xex": 0.7, "tkr": 0.6, "exh": 1.0, "imb": 0.6, "bb": 1.0, "sr": 1.1},
        "4h":  {"ob": 0.2, "tech": 0.6, "cvd": 0.7, "vp": 1.5, "funding": 1.4, "ml": 1.3, "hurst": 1.0, "htf": 1.2, "oi": 1.3, "xex": 0.8, "tkr": 0.3, "exh": 1.2, "imb": 0.4, "bb": 0.8, "sr": 1.2},
        "1d":  {"ob": 0.1, "tech": 0.3, "cvd": 0.4, "vp": 1.8, "funding": 1.6, "ml": 1.5, "hurst": 1.2, "htf": 1.5, "oi": 1.4, "xex": 1.0, "tkr": 0.2, "exh": 1.3, "imb": 0.3, "bb": 0.6, "sr": 1.3},
    }

    layer_keys = {"ob": "order_book", "tech": "multi_tf_technical", "cvd": "cvd_delta",
                   "vp": "volume_profile", "funding": "funding_oi", "ml": "ml_ensemble",
                   "hurst": "hurst_regime", "htf": "higher_tf_align", "oi": "oi_delta",
                   "xex": "cross_exchange", "tkr": "taker_ratio", "exh": "exhaustion",
                   "imb": "imbalance_trend", "bb": "bollinger", "sr": "sr_levels"}

    horizons = {}
    horizon_drivers = {}  # Per-horizon: which layers drove the prediction
    atr = current_price * 0.015  # Default ATR ~1.5%

    for h_name, h_mult in horizon_configs.items():
        h_up = 0; h_down = 0; h_flat = 0; h_conf = 0; h_weight_total = 0
        layer_scores = []  # (layer_name, contribution)

        for short_key, layer_key in layer_keys.items():
            mult = h_mult.get(short_key, 1.0)
            result = layer_results.get(layer_key, {})
            h_weight_total += mult
            h_up += result.get("up", 33) * mult
            h_down += result.get("down", 33) * mult
            h_flat += result.get("flat", 34) * mult
            h_conf += result.get("confidence", 10) * mult
            contrib = (result.get("up",33) - result.get("down",33)) * mult  # directional contribution
            layer_scores.append((short_key, contrib))

        # Top 2 driving layers per horizon (compact)
        layer_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top2 = layer_scores[:2]
        horizon_drivers[h_name] = [(name, round(score, 0)) for name, score in top2]

        if h_weight_total > 0:
            h_up /= h_weight_total
            h_down /= h_weight_total
            h_flat /= h_weight_total
            h_conf /= h_weight_total

        # Price targets based on ATR scaled by horizon
        time_mult = {"1m": 0.3, "5m": 0.6, "15m": 1.0, "1h": 2.0, "4h": 4.0, "1d": 8.0}
        move = atr * time_mult.get(h_name, 1.0)

        horizons[h_name] = HorizonPrediction(
            up=round(h_up, 1),
            down=round(h_down, 1),
            flat=round(h_flat, 1),
            target_up=round(current_price * (1 + move * 0.01), 4),
            target_down=round(current_price * (1 - move * 0.01), 4),
            confidence=round(h_conf, 1),
        )

    # Overall bias
    if weighted_up > weighted_down + 15:
        bias = "bullish"
    elif weighted_down > weighted_up + 15:
        bias = "bearish"
    else:
        bias = "neutral"

    return ContinuousPrediction(
        coin=coin,
        current_price=current_price,
        timestamp=time.time(),
        horizons=horizons,
        layer_contributions=layer_contributions,
        overall_bias=bias,
        overall_confidence=round(weighted_conf, 1),
        layer_confidences=layer_confidences,
        horizon_drivers=horizon_drivers,  # NEW: per-horizon layer drivers
    )


def update_layer_accuracy(layer_name: str, was_correct: bool):
    """Called after a trade closes to adjust layer weights."""
    weights = _load_weights()
    if layer_name not in weights:
        return

    # Reward correct layers, penalize wrong ones
    adjustment = 0.02 if was_correct else -0.01
    weights[layer_name] = max(0.05, min(0.40, weights.get(layer_name, 0.15) + adjustment))

    # Re-normalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    _save_weights(weights)


def format_prediction_compact(pred: ContinuousPrediction) -> str:
    """Single-line format for daemon log."""
    parts = [f"{pred.coin} bias={pred.overall_bias} conf={pred.overall_confidence:.0f}%"]
    for h in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        hp = pred.horizons.get(h)
        if hp:
            parts.append(f"{h}:↑{hp.up:.0f}↓{hp.down:.0f}→{hp.flat:.0f}")
    return " | ".join(parts)


def format_prediction_detailed(pred: ContinuousPrediction) -> str:
    """Multi-line format for detailed logging."""
    lines = [
        f"╔══ {pred.coin} @ ${pred.current_price:.4f} — {pred.overall_bias.upper()} ({pred.overall_confidence:.0f}% conf)",
    ]
    for h in ["1m", "5m", "15m", "1h", "4h"]:
        hp = pred.horizons.get(h)
        if hp:
            bar_up = "█" * int(hp.up / 5)
            bar_down = "█" * int(hp.down / 5)
            bar_flat = "░" * int(hp.flat / 5)
            lines.append(f"║ {h:>3s} ↑{bar_up} {hp.up:.0f}% ↓{bar_down} {hp.down:.0f}% →{bar_flat} {hp.flat:.0f}%  "
                         f"tgt:${hp.target_up:.4f}/${hp.target_down:.4f}")
    lines.append(f"╚══ layers: " + " ".join(f"{k}:{v:.0f}%" for k, v in pred.layer_contributions.items()))
    return "\n".join(lines)


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with synthetic data
    import random
    random.seed(42)

    # Generate fake candles
    base = 100.0
    candles_1m = [{"h": base + random.uniform(-0.5, 0.5), "l": base + random.uniform(-1, 0),
                    "o": base + random.uniform(-0.3, 0.3), "c": base + random.uniform(-0.5, 0.5),
                    "v": random.uniform(100, 1000)}
                   for _ in range(100)]
    for i, c in enumerate(candles_1m):
        base += random.uniform(-0.1, 0.15)

    candles_5m = candles_1m[::5]
    candles_15m = candles_1m[::15]
    candles_1h = candles_1m[::60]

    order_book = {
        "bids": [[99.8, 500], [99.7, 1000], [99.5, 2000], [99.0, 5000]],
        "asks": [[100.2, 400], [100.3, 800], [100.5, 1500], [101.0, 3000]],
    }

    pred = predict_continuous(
        "TEST", 100.0,
        candles_1m=candles_1m, candles_5m=candles_5m,
        candles_15m=candles_15m, candles_1h=candles_1h,
        order_book=order_book, funding_rate=0.005,
    )

    print(format_prediction_detailed(pred))
    print(f"\nCompact: {format_prediction_compact(pred)}")
    print("\n✓ Continuous prediction engine ready")
