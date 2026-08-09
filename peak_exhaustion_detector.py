"""
Peak & Exhaustion Detector — identifies local tops/bottoms for predictive exits.
Detects exhaustion signals so we can sell at peaks and rebuy at bottoms.
Wired into Hyperliquid daemon's AI context every cycle.

Signals detected:
  1. RSI divergence (bearish/bullish) — price makes higher high, RSI makes lower high → top
  2. Volume climax — massive volume spike at a high → distribution/exhaustion
  3. Momentum exhaustion — decreasing rate of change over last N candles
  4. BB squeeze direction — tight bands about to expand, which way?
  5. VWAP deviation — how far above/below VWAP are we?
  6. Composite exhaustion score 0-100

Design philosophy: No hardcoded thresholds — continuous scoring, AI makes final call.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExhaustionSignal:
    """Detected exhaustion — potential peak (sell) or bottom (buy)."""
    symbol: str
    direction: str  # "peak" (overbought, sell) or "bottom" (oversold, buy)
    score: float  # 0-100 composite exhaustion score
    rsi_divergence: str  # "bearish", "bullish", or "none"
    volume_climax: bool
    momentum_decay: float  # 0-1 how much momentum has decayed
    bb_position: str  # "upper", "lower", "middle"
    vwap_deviation_pct: float  # % away from VWAP
    reasoning: str  # 1-line summary for AI context
    # ── Price targets (NEW) ──
    target_price: float = 0.0  # predicted price in 1-4h
    target_confidence: float = 0.0  # 0-100 confidence in target
    action: str = ""  # "sell_now" | "tighten_stop" | "hold_through"


def _ema(values, period):
    """Exponential moving average."""
    if len(values) < period:
        return float(sum(values) / len(values)) if values else 0.0
    alpha = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _rsi(closes, period=14):
    """Compute RSI series."""
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    if len(gains) < period:
        return []
    rsi_values = []
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return rsi_values


def _bollinger(closes, period=20, std_mult=2.0):
    """Compute Bollinger Bands (middle, upper, lower, width)."""
    if len(closes) < period:
        return None
    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std = math.sqrt(variance)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma if sma > 0 else 0
    return {"middle": sma, "upper": upper, "lower": lower, "width": width, "std": std}


def _vwap(highs, lows, closes, volumes):
    """Compute rolling VWAP."""
    if not closes or not volumes or len(closes) != len(volumes):
        return 0.0
    typicals = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    total_pv = sum(t * v for t, v in zip(typicals, volumes))
    total_v = sum(volumes)
    return total_pv / total_v if total_v > 0 else 0.0


def _detect_rsi_divergence(closes, highs, period=14, lookback=20):
    """
    Detect RSI divergence.
    Bearish: price makes higher high, RSI makes lower high → exhaustion top
    Bullish: price makes lower low, RSI makes higher low → exhaustion bottom
    Returns: "bearish", "bullish", or "none"
    """
    if len(closes) < period + lookback:
        return "none"
    rsi_vals = _rsi(closes, period)
    if len(rsi_vals) < lookback:
        return "none"

    # Take last `lookback` candles
    recent_closes = closes[-lookback:]
    recent_rsi = rsi_vals[-lookback:]
    recent_highs = highs[-lookback:] if len(highs) >= lookback else closes[-lookback:]

    # Find two peaks in price (local highs)
    mid = lookback // 2
    first_half_closes = recent_closes[:mid]
    second_half_closes = recent_closes[mid:]
    first_half_rsi = recent_rsi[:mid]
    second_half_rsi = recent_rsi[mid:]

    max1_close = max(first_half_closes)
    max2_close = max(second_half_closes)
    max1_rsi = max(first_half_rsi)
    max2_rsi = max(second_half_rsi)

    min1_close = min(first_half_closes)
    min2_close = min(second_half_closes)
    min1_rsi = min(first_half_rsi)
    min2_rsi = min(second_half_rsi)

    # Bearish divergence: higher price high, lower RSI high
    if max2_close > max1_close * 1.01 and max2_rsi < max1_rsi:
        return "bearish"

    # Bullish divergence: lower price low, higher RSI low
    if min2_close < min1_close * 0.99 and min2_rsi > min1_rsi:
        return "bullish"

    return "none"


def _detect_volume_climax(volumes, closes, period=10):
    """
    Detect volume climax at a price extreme.
    Massive volume spike at a high → distribution (selling into strength).
    Returns: bool
    """
    if len(volumes) < period * 2 or len(closes) < period * 2:
        return False
    recent_vol = volumes[-period:]
    prev_vol = volumes[-period * 2:-period]
    avg_prev = sum(prev_vol) / period if prev_vol else 1
    avg_recent = sum(recent_vol) / period
    vol_ratio = avg_recent / avg_prev if avg_prev > 0 else 1

    # Price at high end of range
    recent_closes = closes[-period:]
    all_closes = closes[-period * 2:]
    range_high = max(all_closes)
    range_low = min(all_closes)
    if range_high <= range_low:
        return False
    price_position = (sum(recent_closes) / period - range_low) / (range_high - range_low)

    # Volume climax: 2.5x+ normal volume AND price near top of range
    return vol_ratio > 2.5 and price_position > 0.75


def _momentum_decay(closes, period=6):
    """
    Measure momentum decay: is price acceleration decreasing?
    Returns 0-1 where 1 = momentum fully exhausted (about to reverse).
    """
    if len(closes) < period + 2:
        return 0.0
    # Rate of change over last `period`
    roc = [(closes[i] - closes[i - period]) / closes[i - period] for i in range(period, len(closes))]
    if len(roc) < 3:
        return 0.0
    # Is ROC decreasing?
    recent_roc = roc[-3:]
    if recent_roc[0] <= 0:
        return 0.0  # Already declining
    decay = 1.0 - (recent_roc[-1] / recent_roc[0]) if recent_roc[0] > 0 else 0.0
    return max(0.0, min(1.0, decay))


def _bb_position_score(closes, period=20):
    """Where is price relative to BB? 1.0 = at upper band (peak), 0.0 = at lower (bottom)."""
    bb = _bollinger(closes, period)
    if not bb or bb["upper"] <= bb["lower"]:
        return 0.5
    current = closes[-1]
    position = (current - bb["lower"]) / (bb["upper"] - bb["lower"])
    return max(0.0, min(1.0, position))


def detect_peak_exhaustion(
    symbol: str,
    candles: list,
    current_price: float,
) -> Optional[ExhaustionSignal]:
    """
    Main entry point: detect if a coin is showing peak or bottom exhaustion.

    Args:
        symbol: Coin symbol (e.g. "SOL")
        candles: List of OHLCV candle dicts (requires: close, high, low, volume)
        current_price: Current mid price

    Returns: ExhaustionSignal or None if no clear signal.
    """
    if not candles or len(candles) < 30:
        return None

    closes = [float(c.get("close", c.get("c", 0))) for c in candles]
    highs = [float(c.get("high", c.get("h", 0))) for c in candles]
    lows = [float(c.get("low", c.get("l", 0))) for c in candles]
    volumes = [float(c.get("volume", c.get("v", 0))) for c in candles]

    if not closes or closes[-1] <= 0:
        return None

    # 1. RSI divergence
    div = _detect_rsi_divergence(closes, highs)

    # 2. Volume climax
    vol_climax = _detect_volume_climax(volumes, closes)

    # 3. Momentum decay
    mom_decay = _momentum_decay(closes)

    # 4. BB position
    bb_pos = _bb_position_score(closes)

    # 5. VWAP deviation
    vwap_val = _vwap(highs[-50:], lows[-50:], closes[-50:], volumes[-50:])
    vwap_dev = ((current_price - vwap_val) / vwap_val * 100) if vwap_val > 0 else 0.0

    # ── Composite scoring ──
    peak_score = 0.0
    bottom_score = 0.0
    reasons = []

    # RSI divergence
    if div == "bearish":
        peak_score += 30
        reasons.append("RSI bearish divergence (price ↑, RSI ↓)")
    elif div == "bullish":
        bottom_score += 30
        reasons.append("RSI bullish divergence (price ↓, RSI ↑)")

    # Volume climax
    if vol_climax:
        peak_score += 25
        reasons.append("volume climax at range high")

    # Momentum decay
    if mom_decay > 0.5:
        peak_score += mom_decay * 20
        if mom_decay > 0.7:
            reasons.append(f"momentum exhaustion ({mom_decay:.0%})")

    # BB position
    if bb_pos > 0.85:
        peak_score += 15
        reasons.append("at upper BB band")
    elif bb_pos < 0.15:
        bottom_score += 15
        reasons.append("at lower BB band")

    # VWAP deviation
    if vwap_dev > 3.0:
        peak_score += min(15, vwap_dev * 2)
        reasons.append(f"{vwap_dev:.1f}% above VWAP")
    elif vwap_dev < -3.0:
        bottom_score += min(15, abs(vwap_dev) * 2)
        reasons.append(f"{abs(vwap_dev):.1f}% below VWAP")

    # Determine direction
    if peak_score >= 40:
        return ExhaustionSignal(
            symbol=symbol,
            direction="peak",
            score=min(100, peak_score),
            rsi_divergence=div,
            volume_climax=vol_climax,
            momentum_decay=mom_decay,
            bb_position="upper" if bb_pos > 0.7 else "middle",
            vwap_deviation_pct=vwap_dev,
            reasoning="; ".join(reasons) if reasons else "borderline peak signal",
        )
    elif bottom_score >= 40:
        return ExhaustionSignal(
            symbol=symbol,
            direction="bottom",
            score=min(100, bottom_score),
            rsi_divergence=div,
            volume_climax=False,  # volume climax at bottom is accumulation, not distribution
            momentum_decay=mom_decay,
            bb_position="lower" if bb_pos < 0.3 else "middle",
            vwap_deviation_pct=vwap_dev,
            reasoning="; ".join(reasons) if reasons else "borderline bottom signal",
        )

    return None


def predict_price_target(signal: ExhaustionSignal, candles: list, current_price: float) -> ExhaustionSignal:
    """
    Enrich an exhaustion signal with price targets and confidence-gated action.

    Confidence-gating rules:
      - HIGH confidence (score >= 70, divergence confirmed): hold through, predict exact target
      - MEDIUM confidence (score 50-69): tighten stops, cautious target
      - LOW confidence (score 40-49): SELL NOW, don't wait — we could be wrong
      - Momentum decay > 0.7 + volume climax: SELL IMMEDIATELY (exhaustion confirmed)
    """
    if not candles or len(candles) < 20:
        signal.action = "tighten_stop"
        signal.target_confidence = 30
        return signal

    closes = [float(c.get("close", c.get("c", 0))) for c in candles]
    highs = [float(c.get("high", c.get("h", 0))) for c in candles]
    lows = [float(c.get("low", c.get("l", 0))) for c in candles]

    # ── ATR for target range ──
    tr_values = []
    for i in range(1, min(15, len(highs))):
        h, l, pc = highs[-i], lows[-i], closes[-i - 1] if i + 1 < len(closes) else closes[-i]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_values.append(tr)
    atr = sum(tr_values) / len(tr_values) if tr_values else current_price * 0.01
    atr_pct = atr / current_price * 100 if current_price > 0 else 2.0

    # ── Support/resistance from recent candles ──
    recent_lows = sorted(lows[-20:])
    recent_highs = sorted(highs[-20:], reverse=True)
    support = sum(recent_lows[:3]) / 3 if len(recent_lows) >= 3 else current_price * 0.95
    resistance = sum(recent_highs[:3]) / 3 if len(recent_highs) >= 3 else current_price * 1.05

    # ── Confidence calculation ──
    confidence = signal.score  # base from exhaustion score

    # Divergence confirmed = much higher confidence
    if signal.rsi_divergence != "none":
        confidence = min(100, confidence + 15)
    else:
        confidence = max(20, confidence - 10)  # penalty for no divergence

    # Volume climax at peak = very reliable
    if signal.volume_climax:
        confidence = min(100, confidence + 10)

    # Momentum fully decayed = reversal is happening NOW
    if signal.momentum_decay > 0.7:
        confidence = min(100, confidence + 10)
    elif signal.momentum_decay < 0.3:
        confidence = max(20, confidence - 10)  # still has momentum — might not reverse yet

    signal.target_confidence = confidence

    # ── Target price prediction ──
    if signal.direction == "peak":
        # Predicting DROP — target is where we expect to rebuy
        # Conservative target: support level
        # Aggressive target: 1.5-2x ATR below current
        drop_pct = atr_pct * (1.5 if confidence >= 70 else 1.0)
        signal.target_price = current_price * (1 - drop_pct / 100)
        # Clamp to nearest support
        if support < signal.target_price:
            signal.target_price = support
    else:
        # Predicting RISE — target is where we expect to sell
        rise_pct = atr_pct * (1.5 if confidence >= 70 else 1.0)
        signal.target_price = current_price * (1 + rise_pct / 100)
        if resistance > signal.target_price:
            signal.target_price = resistance

    # ── Action: confidence-gated ──
    if signal.momentum_decay > 0.7 and signal.volume_climax:
        signal.action = "sell_now"  # Exhaustion is REAL — get out immediately
    elif confidence >= 70:
        signal.action = "tighten_stop"  # Confident in direction but not panicking
    elif confidence >= 50:
        signal.action = "sell_early"  # Not sure enough to hold — sell before it drops further
    else:
        signal.action = "sell_now"  # Low confidence — better to sell wrong than hold through a drop

    return signal


def format_exhaustion_for_ai(signal: Optional[ExhaustionSignal]) -> str:
    """Format exhaustion signal as compact text for AI context."""
    if not signal:
        return "⚡ EXHAUSTION: No clear peak/bottom signal detected."
    emoji = "🔴 PEAK EXHAUSTION" if signal.direction == "peak" else "🟢 BOTTOM EXHAUSTION"
    lines = [
        f"⚡ {emoji} ({signal.symbol}): score={signal.score:.0f}/100",
        f"  RSI div: {signal.rsi_divergence} | Vol climax: {signal.volume_climax}",
        f"  BB pos: {signal.bb_position} | VWAP dev: {signal.vwap_deviation_pct:+.1f}%",
        f"  Momentum decay: {signal.momentum_decay:.0%}",
        f"  → {signal.reasoning}",
    ]
    if signal.target_price > 0:
        lines.append(f"  🎯 Target: ${signal.target_price:.4f} (conf={signal.target_confidence:.0f}%)")
    if signal.action:
        action_emoji = {"sell_now": "🚨", "sell_early": "⚠️", "tighten_stop": "📐"}.get(signal.action, "→")
        lines.append(f"  {action_emoji} Action: {signal.action}")
    return "\n".join(lines)


# ── Self-test ──

# ── Tick-Level Peak Detection (merged from tick_peak_detector.py) ──
# Aug 9: merged — complementary to candle-level exhaustion detection

@dataclass
class TickPeakSignal:
    """Real-time peak/bottom detection — updated every 10 seconds."""
    symbol: str
    timestamp: float
    current_price: float
    is_peak: bool  # True = at/near top, False = at/near bottom
    confidence: float  # 0-100
    predicted_extreme: float  # exact predicted top or bottom price
    predicted_reversal_pct: float  # how far price will go before reversing
    seconds_to_extreme: int  # estimated seconds until we hit the extreme
    # Component scores
    ob_thinning: float = 0.0
    tick_flow: float = 0.0
    cvd_divergence: float = 0.0
    vwap_extension: float = 0.0
    liq_magnet: float = 0.0
    multi_tf: float = 0.0
    # Action
    action: str = ""  # "sell_now", "sell_soon", "hold", "buy_soon", "buy_now"
    reasoning: str = ""


def _get_ws_data(coin: str) -> dict:
    """Get live WebSocket data: order book, trades."""
    try:
        from hyperliquid_ws import get_latest
        return get_latest()
    except Exception:
        return {}


def _compute_order_book_signal(coin: str, current_price: float) -> tuple[float, float, str]:
    """
    Check if order book shows thinning — a classic peak/bottom signal.
    At a peak: bid depth shrinks (buyers exhausted), ask depth grows (sellers stepping in).
    At a bottom: ask depth shrinks (sellers exhausted), bid depth grows.

    Returns: (peak_score 0-100, bottom_score 0-100, reasoning)
    """
    try:
        ws = _get_ws_data(coin)
        ob = ws.get("orderbooks", {}).get(coin, {})
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        if not bids or not asks or len(bids) < 3 or len(asks) < 3:
            return 0.0, 0.0, "no_book"

        # Top 5 levels in USD
        bid_depth = sum(float(b.get("px", 0)) * float(b.get("sz", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("px", 0)) * float(a.get("sz", 0)) for a in asks[:5])

        # Count levels within 1% of mid
        mid = current_price
        near_bids = sum(float(b.get("sz", 0)) for b in bids if float(b.get("px", 0)) > mid * 0.99)
        near_asks = sum(float(a.get("sz", 0)) for a in asks if float(a.get("px", 0)) < mid * 1.01)

        total_near = near_bids + near_asks
        if total_near <= 0:
            return 0.0, 0.0, "no_near"

        bid_ratio = near_bids / total_near
        ask_ratio = near_asks / total_near

        # Peak signal: ask wall dominating (>65% of near-book volume is asks)
        peak_score = 0.0
        bottom_score = 0.0
        reason = "balanced_book"

        if ask_ratio > 0.65:
            peak_score = (ask_ratio - 0.5) * 200  # 0.65 → 30, 0.80 → 60
            reason = f"ask_wall:{ask_ratio:.0%}"
        elif ask_ratio > 0.55:
            peak_score = (ask_ratio - 0.5) * 100
            reason = "slight_sell_pressure"

        if bid_ratio > 0.65:
            bottom_score = (bid_ratio - 0.5) * 200
            reason = f"bid_wall:{bid_ratio:.0%}"
        elif bid_ratio > 0.55:
            bottom_score = (bid_ratio - 0.5) * 100
            reason = "slight_buy_pressure"

        # Depth imbalance (total bid depth vs ask depth)
        total = bid_depth + ask_depth
        if total > 0:
            depth_imbalance = (ask_depth - bid_depth) / total
            if depth_imbalance > 0.3:
                peak_score = max(peak_score, depth_imbalance * 80)
                reason += f"+depth_imb:{depth_imbalance:.1%}"
            elif depth_imbalance < -0.3:
                bottom_score = max(bottom_score, abs(depth_imbalance) * 80)
                reason += f"+depth_imb:{depth_imbalance:.1%}"

        return min(100, peak_score), min(100, bottom_score), reason

    except Exception:
        return 0.0, 0.0, "error"


def _compute_tick_flow_signal(coin: str) -> tuple[float, float, str]:
    """
    Analyze recent trades: big sells at bid = distribution (peak), big buys at ask = accumulation (bottom).
    """
    try:
        ws = _get_ws_data(coin)
        trade = ws.get("trades", {}).get(coin, {})
        if not trade:
            return 0.0, 0.0, "no_trades"

        side = trade.get("side", "")
        size = float(trade.get("size", 0))
        price = float(trade.get("price", 0))
        ts = int(trade.get("ts", 0))

        if size <= 0 or price <= 0:
            return 0.0, 0.0, "no_size"

        # Large sell = bearish flow (peak)
        # Large buy = bullish flow (bottom)
        age_ms = int(time.time() * 1000) - ts
        freshness = max(0, 1.0 - age_ms / 5000)  # decay over 5 seconds

        peak_score = 0.0
        bottom_score = 0.0

        if side == "sell":
            peak_score = min(60, size * 10) * freshness  # size 5+ = strong signal
        elif side == "buy":
            bottom_score = min(60, size * 10) * freshness

        return peak_score, bottom_score, f"{side}:{size:.2f}@{price:.4f}"

    except Exception:
        return 0.0, 0.0, "error"


def _compute_cvd_divergence(candles: list, current_price: float) -> float:
    """
    CVD divergence: price making higher high but CVD making lower high → peak.
    Approximate CVD from OHLCV: sum of (close - open) * volume when close > open → buy vol.
    """
    if not candles or len(candles) < 20:
        return 0.0

    closes = [float(c.get("close", c.get("c", 0))) for c in candles[-20:]]
    opens = [float(c.get("open", c.get("o", 0))) for c in candles[-20:]]
    highs = [float(c.get("high", c.get("h", 0))) for c in candles[-20:]]
    volumes = [float(c.get("volume", c.get("v", 0))) for c in candles[-20:]]

    # Approximate CVD
    cvd = []
    running = 0.0
    for i in range(len(closes)):
        delta = closes[i] - opens[i]
        running += delta * volumes[i] / max(closes[i], 0.0001)
        cvd.append(running)

    half = max(1, len(closes) // 2)
    first_closes = closes[:half]
    second_closes = closes[half:]
    first_cvd = cvd[:half]
    second_cvd = cvd[half:]

    # Peak divergence: price higher, CVD lower
    if max(second_closes) > max(first_closes) and max(second_cvd) < max(first_cvd):
        return min(80, (max(first_cvd) - max(second_cvd)) / max(abs(max(first_cvd)), 1) * 100)

    # Bottom divergence: price lower, CVD higher
    if min(second_closes) < min(first_closes) and min(second_cvd) > min(first_cvd):
        return -min(80, (min(second_cvd) - min(first_cvd)) / max(abs(min(first_cvd)), 1) * 100)

    return 0.0


def _compute_vwap_extension(coin: str, candles: list, current_price: float) -> float:
    """VWAP extension: how many std devs from VWAP."""
    if not candles or len(candles) < 20:
        return 0.0

    recent = candles[-20:]
    highs = [float(c.get("high", c.get("h", 0))) for c in recent]
    lows = [float(c.get("low", c.get("l", 0))) for c in recent]
    closes = [float(c.get("close", c.get("c", 0))) for c in recent]
    volumes = [float(c.get("volume", c.get("v", 0))) for c in recent]

    typicals = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    total_pv = sum(t * v for t, v in zip(typicals, volumes))
    total_v = sum(volumes)
    if total_v <= 0:
        return 0.0
    vwap = total_pv / total_v

    # VWAP std dev (price-weighted)
    variance = sum(((t - vwap) ** 2) * v for t, v in zip(typicals, volumes)) / total_v
    std = math.sqrt(variance)

    if std <= 0:
        return 0.0

    return (current_price - vwap) / std  # positive = above VWAP, negative = below


def detect_tick_peak(
    coin: str,
    current_price: float,
    candles_15m: list = None,
    candles_1m: list = None,
    side: str = "LONG",
    pnl_pct: float = 0.0,
) -> Optional[TickPeakSignal]:
    """
    High-frequency peak/bottom detection using ALL available real-time signals.

    Called every 10 seconds from the position monitor.
    Uses 1m candles for CVD (research: 1-min best for reversal detection).
    Uses 15m candles for VWAP and multi-TF alignment.
    Returns exact predicted extreme price with confidence.

    pnl_pct: current unrealized PnL % — lowers exit thresholds when in profit
             (e.g. +3% profit makes a 25% conf signal actionable instead of ignored)
    """
    candles = candles_15m or []
    reasons = []
    peak_score = 0.0
    bottom_score = 0.0

    # 1. Order book thinning (35% weight — was 25%)
    # Increased: OB is the most reliable real-time signal. A 98% ask wall
    # means sellers are in full control regardless of what other signals say.
    ob_peak, ob_bottom, ob_reason = _compute_order_book_signal(coin, current_price)
    if ob_peak > 0:
        peak_score += ob_peak * 0.35
        reasons.append(f"OB:{ob_reason}")
    if ob_bottom > 0:
        bottom_score += ob_bottom * 0.35
        reasons.append(f"OB:{ob_reason}")

    # 2. Tick flow (10% — was 20%)
    # Reduced: single-trade flow is noisy and often lags the order book.
    tf_peak, tf_bottom, tf_reason = _compute_tick_flow_signal(coin)
    if tf_peak > 0:
        peak_score += tf_peak * 0.10
        reasons.append(f"Flow:{tf_reason}")
    if tf_bottom > 0:
        bottom_score += tf_bottom * 0.10

    # 3. CVD divergence (20%) — uses 1m candles for accuracy (research: 1-min best for reversals)
    cvd_candles = candles_1m if candles_1m and len(candles_1m) >= 30 else candles
    if cvd_candles:
        cvd = _compute_cvd_divergence(cvd_candles, current_price)
        if cvd > 0:
            peak_score += cvd * 0.20
            reasons.append(f"CVD_div:+{cvd:.0f}")
        elif cvd < 0:
            bottom_score += abs(cvd) * 0.20
            reasons.append(f"CVD_div:{cvd:.0f}")

    # 4. VWAP extension (15%)
    if candles:
        vwap_ext = _compute_vwap_extension(coin, candles, current_price)
        if vwap_ext > 2.0:
            peak_score += min(100, vwap_ext * 7.5) * 0.15
            reasons.append(f"VWAP:+{vwap_ext:.1f}σ")
        elif vwap_ext < -2.0:
            bottom_score += min(100, abs(vwap_ext) * 7.5) * 0.15
            reasons.append(f"VWAP:{vwap_ext:.1f}σ")

    # 5. Multi-timeframe RSI alignment (10%)
    if candles and len(candles) >= 14:
        from peak_exhaustion_detector import _rsi
        closes = [float(c.get("close", c.get("c", 0))) for c in candles]
        rsi14 = _rsi(closes, 14)
        if rsi14:
            last_rsi = rsi14[-1]
            if last_rsi > 70:
                peak_score += (last_rsi - 70) * 0.10
                reasons.append(f"RSI:{last_rsi:.0f}")
            elif last_rsi < 30:
                bottom_score += (30 - last_rsi) * 0.10

    # ── Consensus ──
    is_peak = peak_score > bottom_score
    score = max(peak_score, bottom_score)

    if score < 20:
        return None  # No clear signal

    # ── Multi-confirmation bonus: when both OB + CVD are strong, boost score ──
    # A 91% ask wall + 80 CVD divergence together is much stronger than either alone
    ob_strong = (ob_peak if is_peak else ob_bottom) > 60
    cvd_strong = abs(cvd) > 50
    ob_extreme = (ob_peak if is_peak else ob_bottom) > 80  # >80 = >90% ask/bid wall dominance

    if ob_strong and cvd_strong:
        score += 10  # Compound signal bonus — both order book AND flow agree
    elif ob_extreme:
        score += 8   # Extreme wall bonus — 95%+ ask/bid wall is decisive on its own
    elif ob_strong or cvd_strong:
        score += 5   # Single strong component bonus (was 4)

    # ── Predicted extreme price ──
    atr = current_price * 0.01  # default 1%
    if candles and len(candles) >= 15:
        tr_vals = []
        for i in range(1, 15):
            h = float(candles[-i].get("high", candles[-i].get("h", 0)))
            l = float(candles[-i].get("low", candles[-i].get("l", 0)))
            pc = float(candles[-i-1].get("close", candles[-i-1].get("c", current_price)))
            tr_vals.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(tr_vals) / len(tr_vals)

    # Peak: predict how high before reversal
    # Bottom: predict how low before bounce
    if is_peak:
        predicted_extreme = current_price + atr * (0.5 + score / 200)  # 0.5-1.0 ATR above
        predicted_reversal_pct = (predicted_extreme / current_price - 1) * 100
    else:
        predicted_extreme = current_price - atr * (0.5 + score / 200)
        predicted_reversal_pct = (predicted_extreme / current_price - 1) * 100

    # ── Action — profit-aware thresholds ──
    # Base thresholds: sell_now ≥ 50, sell_soon ≥ 35, hold < 35
    # Lowered from 60/40 because strong multi-signal peaks (91% ask wall + CVD div)
    # were being ignored at 26-37% conf — the composite score underweights real signal strength.
    # When in profit, thresholds drop proportionally — a 3% profit position
    # treats a 30% conf signal like a 45% conf signal from a flat position.
    # This prevents winning positions from reversing without securing gains.

    # Profit multiplier: 0% profit → 1.0x (base), 5%+ profit → 0.5x (half threshold)
    profit_pct = abs(pnl_pct)
    if profit_pct >= 5.0:
        threshold_mult = 0.50
    elif profit_pct >= 3.0:
        threshold_mult = 0.60
    elif profit_pct >= 2.0:
        threshold_mult = 0.70
    elif profit_pct >= 1.0:
        threshold_mult = 0.80
    elif profit_pct >= 0.3:
        threshold_mult = 0.90
    else:
        threshold_mult = 1.0  # No profit → base thresholds

    sell_now_threshold = 50 * threshold_mult  # Was 60
    sell_soon_threshold = 35 * threshold_mult  # Was 40
    # Minimum threshold floor: never go below 18% for sell_soon or 30% for sell_now
    sell_now_threshold = max(30, sell_now_threshold)
    sell_soon_threshold = max(18, sell_soon_threshold)

    if is_peak and side == "LONG":
        # LONG position seeing a peak → sell to exit
        if score >= sell_now_threshold:
            action = "sell_now"
        elif score >= sell_soon_threshold:
            action = "sell_soon"
        else:
            action = "hold"
            if pnl_pct > 0.5 and score >= 20:
                action = "hold_profit"
    elif not is_peak and side == "SHORT":
        # SHORT position seeing a bottom → buy to cover
        if score >= sell_now_threshold:
            action = "buy_now"
        elif score >= sell_soon_threshold:
            action = "buy_soon"
        else:
            action = "hold"
            if pnl_pct > 0.5 and score >= 20:
                action = "hold_profit"
    elif is_peak and side == "SHORT":
        # SHORT position seeing a peak → add to short (sell more)!
        # This was the MISSED OPPORTUNITY: peaks were detected but ignored for shorts
        if score >= sell_now_threshold:
            action = "sell_more_now"
        elif score >= sell_soon_threshold:
            action = "sell_more_soon"
        else:
            action = "hold"
            if pnl_pct > 0.5 and score >= 20:
                action = "hold_profit"
    elif not is_peak and side == "LONG":
        # LONG position seeing a bottom → add to long (buy more)!
        if score >= sell_now_threshold:
            action = "buy_more_now"
        elif score >= sell_soon_threshold:
            action = "buy_more_soon"
        else:
            action = "hold"
            if pnl_pct > 0.5 and score >= 20:
                action = "hold_profit"
    else:
        action = "hold"

    return TickPeakSignal(
        symbol=coin,
        timestamp=time.time(),
        current_price=current_price,
        is_peak=is_peak,
        confidence=score,
        predicted_extreme=predicted_extreme,
        predicted_reversal_pct=predicted_reversal_pct,
        seconds_to_extreme=max(30, int(600 * (1 - score / 100))),  # 30-600 seconds
        ob_thinning=ob_peak if is_peak else ob_bottom,
        tick_flow=tf_peak if is_peak else tf_bottom,
        cvd_divergence=cvd if is_peak else abs(cvd),
        vwap_extension=abs(vwap_ext),
        multi_tf=0.0,
        action=action,
        reasoning="; ".join(reasons) if reasons else "weak_signal",
    )


def format_tick_peak(signal: Optional[TickPeakSignal]) -> str:
    """Compact log format for tick-level peak."""
    if not signal:
        return ""
    direction = "🔴 PEAK" if signal.is_peak else "🟢 BOTTOM"
    action_map = {"sell_now": "🚨SELL", "sell_soon": "⚠️SELL_SOON", "hold": "✋HOLD",
                  "buy_now": "🚨BUY", "buy_soon": "⚠️BUY_SOON",
                  "hold_profit": "💰HOLD_PROFIT",
                  "sell_more_now": "🔥ADD_SHORT", "sell_more_soon": "📉ADD_SHORT_SOON",
                  "buy_more_now": "🔥ADD_LONG", "buy_more_soon": "📈ADD_LONG_SOON"}
    return (
        f"{direction} {signal.symbol} @ ${signal.current_price:.4f} "
        f"conf={signal.confidence:.0f}% "
        f"extreme=${signal.predicted_extreme:.4f} ({signal.predicted_reversal_pct:+.2f}%) "
        f"in ~{signal.seconds_to_extreme}s "
        f"[{action_map.get(signal.action, '?')}] "
        f"| {signal.reasoning}"
    )

if __name__ == "__main__":
    # Generate synthetic candle data for testing
    import random
    random.seed(42)

    # Simulate a peak pattern: steady climb then exhaustion
    base = 100.0
    candles = []
    for i in range(60):
        if i < 40:
            # Uptrend
            change = random.uniform(0.1, 1.0)
        elif i < 50:
            # Exhaustion — slowing rise
            change = random.uniform(-0.5, 0.5)
        else:
            # Reversal — dropping
            change = random.uniform(-1.5, -0.3)
        base += change
        high = base + random.uniform(0, 2)
        low = base - random.uniform(0, 2)
        vol = random.uniform(100, 300) * (3.0 if i >= 45 and i <= 48 else 1.0)  # volume spike at top
        candles.append({"close": base, "high": high, "low": low, "volume": vol})

    current = candles[-1]["close"]
    sig = detect_peak_exhaustion("TEST", candles, current)
    print(format_exhaustion_for_ai(sig))
    print(f"\nPeak score expected: high (RSI divergence + volume climax + momentum decay)")
    if sig:
        print(f"Actual score: {sig.score:.0f}/100 — {sig.direction}")
        print("TEST PASSED" if sig.score >= 50 and sig.direction == "peak" else "TEST WEAK — check scoring")
    else:
        print("NO SIGNAL — test data may need adjustment")
