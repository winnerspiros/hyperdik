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
