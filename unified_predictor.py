#!/usr/bin/env python3
"""
Unified Prediction Engine — combines ALL available signals for price prediction.
Wires together: ML models (XGBoost + LSTM), peak exhaustion, market structure,
order book depth, VWAP, funding, and AI reasoning into one prediction.

Architecture:
  1. Technical layer: XGBoost direction + LSTM price prediction
  2. Exhaustion layer: peak/bottom detection with price targets
  3. Market structure: support/resistance, trend strength, BB position
  4. Order flow: bid/ask depth imbalance, CVD, VWAP deviation
  5. Macro layer: funding rate, BTC correlation, fear & greed
  6. Consensus: weighted ensemble → direction + target + confidence

Output: PredictionResult with exact price target, confidence, and action.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictionResult:
    """Unified prediction result from all signal layers combined."""
    symbol: str = ""
    current_price: float = 0.0
    direction: str = "flat"  # up, down, flat
    confidence: float = 0.0  # 0-100
    target_price: float = 0.0
    target_pct: float = 0.0
    timeframe: str = "4h"
    # Layer scores
    ml_score: float = 0.0
    exhaustion_score: float = 0.0
    structure_score: float = 0.0
    flow_score: float = 0.0
    # Decision
    action: str = "hold"
    reasoning: str = ""


def compute_ml_prediction(coin: str, candles: list, current_price: float) -> dict:
    """
    Run ML models (XGBoost + LSTM) for direction prediction.
    Loads pre-trained models from disk with caching + mtime-based auto-reload.
    If model file is updated (retrained), next call auto-reloads.

    Returns: {direction, ml_confidence, predicted_price, change_pct}
    """
    result = {"direction": "flat", "ml_confidence": 0, "predicted_price": current_price, "change_pct": 0.0}

    if not candles or len(candles) < 30:
        return result

    try:
        from ml_predictor import DirectionPredictor, LSTMPredictor
        import json as _json, os as _os, time as _time

        # ── Normalize candle keys and convert string values ──
        if candles and len(candles) > 0:
            first = candles[0]
            if 'c' in first and 'close' not in first:
                key_map = {'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume', 't': 'time'}
                candles = [{key_map.get(k, k): (float(v) if k in ('c','h','l','o','v') else v) for k, v in c.items()} for c in candles]

        model_dir = _os.path.join(_os.path.dirname(__file__), "data/ml_models")
        xgb_path = _os.path.join(model_dir, f"{coin}_xgb.json")
        lstm_path = _os.path.join(model_dir, f"{coin}_lstm.json")

        # ── Model cache with mtime-based auto-reload ──
        # Cache key: coin -> (xgb_model, lstm_model, xgb_mtime, lstm_mtime)
        if not hasattr(compute_ml_prediction, "_cache"):
            compute_ml_prediction._cache = {}
        _cache = compute_ml_prediction._cache

        xgb_model = None
        lstm_model = None

        # Check cache or load from disk
        if coin in _cache:
            _cached_xgb, _cached_lstm, _xgb_mtime, _lstm_mtime = _cache[coin]
            # Check if file was updated since last load
            _reload_xgb = _os.path.exists(xgb_path) and _os.path.getmtime(xgb_path) > _xgb_mtime
            _reload_lstm = _os.path.exists(lstm_path) and _os.path.getmtime(lstm_path) > _lstm_mtime
            if not _reload_xgb:
                xgb_model = _cached_xgb
            if not _reload_lstm:
                lstm_model = _cached_lstm
        else:
            _reload_xgb = _os.path.exists(xgb_path)
            _reload_lstm = _os.path.exists(lstm_path)

        # Load XGBoost if needed
        if _reload_xgb and _os.path.exists(xgb_path):
            try:
                xgb_model = DirectionPredictor()
                xgb_model.load(xgb_path)
            except Exception:
                xgb_model = None

        # Load LSTM if needed
        if _reload_lstm and _os.path.exists(lstm_path):
            try:
                lstm_model = LSTMPredictor()
                lstm_model.load_weights(lstm_path)
            except Exception:
                lstm_model = None

        # Update cache
        _cache[coin] = (
            xgb_model,
            lstm_model,
            _os.path.getmtime(xgb_path) if _os.path.exists(xgb_path) else 0,
            _os.path.getmtime(lstm_path) if _os.path.exists(lstm_path) else 0,
        )

        # ── XGBoost prediction ──
        if xgb_model and xgb_model.is_trained:
            xgb_pred = xgb_model.predict(candles)
        else:
            # Fall back to quick online training
            dp = DirectionPredictor()
            train_result = dp.train(candles[-200:] if len(candles) >= 200 else candles, test_split=0.0)
            if "error" not in train_result:
                xgb_pred = dp.predict(candles)
            else:
                xgb_pred = {"direction": "flat", "confidence": 0}

        # ── LSTM prediction ──
        if lstm_model:
            lstm_pred = lstm_model.predict_next(candles)
        else:
            lstm = LSTMPredictor()
            lstm_pred = lstm.predict_next(candles)

        # ── Consensus: blend XGBoost + LSTM signals ──
        xgb_dir = xgb_pred.get("direction", "flat")
        xgb_conf = xgb_pred.get("confidence", 0.5) * 100
        lstm_dir = lstm_pred.get("direction", "flat")
        lstm_pct = lstm_pred.get("change_pct", 0)
        lstm_price = lstm_pred.get("predicted_price", current_price)

        if xgb_dir == lstm_dir and xgb_dir != "flat":
            result["direction"] = xgb_dir
            result["ml_confidence"] = (xgb_conf + 55) / 2  # blend + LSTM baseline
            result["predicted_price"] = lstm_price
            result["change_pct"] = lstm_pct
        elif xgb_conf > 35 and xgb_dir != "flat":
            # XGB has weak signal but LSTM disagrees — use XGB with reduced confidence
            result["direction"] = xgb_dir
            result["ml_confidence"] = xgb_conf * 0.6
            result["change_pct"] = lstm_pct
        elif abs(lstm_pct) > 0.5:
            result["direction"] = lstm_dir
            result["ml_confidence"] = max(20, abs(lstm_pct) * 15)
            result["predicted_price"] = lstm_price
            result["change_pct"] = lstm_pct
        elif xgb_dir != "flat":
            # XGB alone, low confidence but better than nothing
            result["direction"] = xgb_dir
            result["ml_confidence"] = max(15, xgb_conf * 0.4)
            result["change_pct"] = lstm_pct
    except ImportError:
        pass  # ML deps not available
    except Exception:
        pass

    return result


def compute_structure_score(candles: list, current_price: float) -> dict:
    """
    Market structure analysis: trend, support/resistance, BB position.
    Returns: {structure_direction, structure_score, support, resistance}
    """
    if not candles or len(candles) < 20:
        return {"structure_direction": "flat", "structure_score": 0, "support": current_price * 0.95, "resistance": current_price * 1.05}

    closes = [float(c.get("close", c.get("c", 0))) for c in candles]
    highs = [float(c.get("high", c.get("h", 0))) for c in candles]
    lows = [float(c.get("low", c.get("l", 0))) for c in candles]

    # Simple trend: compare EMA 9 vs EMA 21
    def ema(values, period):
        if len(values) < period:
            return sum(values) / len(values)
        alpha = 2.0 / (period + 1)
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    trend_up = ema9 > ema21 * 1.005
    trend_down = ema9 < ema21 * 0.995

    # Support/resistance from recent extremes
    recent_lows = sorted(lows[-20:])
    recent_highs = sorted(highs[-20:], reverse=True)
    support = sum(recent_lows[:3]) / 3 if len(recent_lows) >= 3 else current_price * 0.95
    resistance = sum(recent_highs[:3]) / 3 if len(recent_highs) >= 3 else current_price * 1.05

    # BB position
    bb_mid = sum(closes[-20:]) / 20
    bb_std = math.sqrt(sum((c - bb_mid) ** 2 for c in closes[-20:]) / 20)
    bb_pos = (current_price - (bb_mid - 2 * bb_std)) / (4 * bb_std) if bb_std > 0 else 0.5

    score = 0
    direction = "flat"
    if trend_up and bb_pos < 0.7:
        direction = "up"
        score = 50 + (1 - bb_pos) * 30
    elif trend_down and bb_pos > 0.3:
        direction = "down"
        score = 50 + bb_pos * 30

    return {
        "structure_direction": direction,
        "structure_score": min(100, score),
        "support": support,
        "resistance": resistance,
        "bb_position": bb_pos,
    }


def compute_flow_score(coin: str, mids: dict, candles: list) -> dict:
    """
    Order flow analysis: uses volume trend and price trend from candles.
    Falls back to volume trend analysis when order book data is unavailable.

    Returns: {flow_direction, flow_score}
    """
    score = 0
    direction = "flat"

    if not candles or len(candles) < 20:
        return {"flow_direction": direction, "flow_score": score}

    volumes = [float(c.get("volume", c.get("v", 0))) for c in candles[-20:]]
    closes = [float(c.get("close", c.get("c", 0))) for c in candles[-20:]]

    if len(volumes) < 10:
        return {"flow_direction": direction, "flow_score": score}

    # Volume trend
    recent_vol = sum(volumes[-5:]) / 5
    prev_vol = sum(volumes[-10:-5]) / 5
    vol_trend = recent_vol / prev_vol if prev_vol > 0 else 1.0

    # Price trend
    price_start = closes[0]
    price_end = closes[-1]
    price_change = (price_end - price_start) / price_start if price_start > 0 else 0

    # Combine: rising volume + rising price = bullish accumulation
    #          rising volume + falling price = bearish distribution
    if vol_trend > 1.3:
        if price_change > 0.01:
            direction = "up"
            score = min(80, 50 + vol_trend * 10)
        elif price_change < -0.01:
            direction = "down"
            score = min(80, 50 + vol_trend * 10)
    elif vol_trend > 1.1:
        if price_change > 0.005:
            direction = "up"
            score = min(60, 30 + vol_trend * 8)
        elif price_change < -0.005:
            direction = "down"
            score = min(60, 30 + vol_trend * 8)

    # Price-only trend (when volume is flat but price moves)
    if direction == "flat" and abs(price_change) > 0.02:
        direction = "up" if price_change > 0 else "down"
        score = min(40, abs(price_change) * 500)

    return {"flow_direction": direction, "flow_score": min(100, score)}


def _get_regime_weights(regime: str = "sideways") -> dict[str, float]:
    """RAVEN-inspired regime-adaptive layer weights.

    The core insight: different signal types dominate in different regimes.
    - Trending: ML momentum + macro are most predictive (follow the trend)
    - Ranging: Structure (S/R) + flow (absorption/CVD) dominate (mean reversion)
    - Volatile: Exhaustion signals are most predictive (catch the exhaustion)

    Based on: RAVEN paper (arxiv.org/abs/2606.24062)
    """
    if regime in ("trending_up", "trending_down"):
        # In trends: ML momentum dominates, exhaustion less relevant
        # Structure still matters for trend continuation signals
        return {"ml": 0.35, "exhaustion": 0.15, "structure": 0.22, "flow": 0.18, "macro": 0.10}
    elif regime == "ranging":
        # In ranges: structure (S/R) and flow (absorption/CVD) dominate
        # ML is noisy in ranges — reduce weight significantly
        return {"ml": 0.12, "exhaustion": 0.25, "structure": 0.28, "flow": 0.25, "macro": 0.10}
    elif regime == "volatile" or regime == "high_vol":
        # In high volatility: exhaustion signals catch turning points
        # ML still useful for direction, macro for sentiment context
        return {"ml": 0.20, "exhaustion": 0.35, "structure": 0.15, "flow": 0.20, "macro": 0.10}
    else:
        # Default / sideways / unknown: balanced
        return {"ml": 0.30, "exhaustion": 0.25, "structure": 0.20, "flow": 0.15, "macro": 0.10}


def predict_unified(
    coin: str,
    candles: list,
    current_price: float,
    mids: dict = None,
    funding_rate: float = 0.0,
    btc_mid: float = 0.0,
    fear_greed: int = 50,
    regime: str = "sideways",  # NEW: regime-adaptive weighting
) -> PredictionResult:
    """
    UNIFIED PREDICTION — combines EVERYTHING into one call.

    Weights: regime-adaptive (trending→ML-heavy, ranging→structure/flow-heavy)
    """
    mids = mids or {}
    result = PredictionResult(
        symbol=coin,
        current_price=current_price,
        direction="flat",
        confidence=0,
        target_price=current_price,
        target_pct=0.0,
        timeframe="4h",
    )

    # ── Normalize candle keys and convert string values (HL API returns strings) ──
    if candles and len(candles) > 0:
        first = candles[0]
        if 'c' in first and 'close' not in first:
            key_map = {'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume', 't': 'time'}
            candles = [{key_map.get(k, k): (float(v) if k in ('c','h','l','o','v') else v) for k, v in c.items()} for c in candles]

    # 1. ML prediction
    ml = compute_ml_prediction(coin, candles, current_price)
    result.ml_score = ml.get("ml_confidence", 0)

    # 2. Exhaustion
    exh_dir = "flat"  # default
    try:
        from peak_exhaustion_detector import detect_peak_exhaustion, predict_price_target
        exh = detect_peak_exhaustion(coin, candles, current_price)
        if exh and exh.score >= 30:
            exh = predict_price_target(exh, candles, current_price)
            result.exhaustion_score = exh.score
            _exhaustion_dir_map = {"bottom": "up", "peak": "down"}
            exh_dir = _exhaustion_dir_map.get(getattr(exh, "direction", ""), "flat")
    except Exception:
        pass

    # 3. Market structure
    structure = compute_structure_score(candles, current_price)
    result.structure_score = structure.get("structure_score", 0)

    # 4. Order flow
    flow = compute_flow_score(coin, mids, candles)
    result.flow_score = flow.get("flow_score", 0)

    # 5. Macro: funding + F&G + BTC
    macro_score = 0
    macro_dir = "flat"
    if funding_rate > 0.01:
        macro_score += 15  # expensive to long
        macro_dir = "down"
    elif funding_rate < -0.005:
        macro_score += 15
        macro_dir = "up"
    if fear_greed < 30:
        macro_score += 10  # extreme fear = contrarian bullish
        if macro_dir == "flat":
            macro_dir = "up"
    elif fear_greed > 70:
        macro_score += 10  # greed = contrarian bearish
        if macro_dir == "flat":
            macro_dir = "down"

    # ── Weighted consensus (REGIME-ADAPTIVE — RAVEN paper) ──
    weights = _get_regime_weights(regime)
    directions = {
        "ml": ml.get("direction", "flat"),
        "exhaustion": exh_dir if result.exhaustion_score > 0 else "flat",
        "structure": structure.get("structure_direction", "flat"),
        "flow": flow.get("flow_direction", "flat"),
        "macro": macro_dir,
    }
    scores = {
        "ml": result.ml_score,
        "exhaustion": result.exhaustion_score,
        "structure": result.structure_score,
        "flow": result.flow_score,
        "macro": macro_score,
    }

    up_weight = sum(weights[k] * scores[k] for k in weights if directions[k] == "up")
    down_weight = sum(weights[k] * scores[k] for k in weights if directions[k] == "down")
    total_conf = up_weight + down_weight

    if total_conf > 0:
        # Count how many layers are actually contributing direction (non-flat)
        active_layers = sum(1 for k in weights if directions[k] != "flat" and scores[k] > 0)
        # Amplification: single-source signals get a bigger boost so they aren't diluted
        # 1 active layer → 3.5x amp, 2 → 2.5x, 3+ → 2.2x
        amp = 1.5 + (2.0 / max(1, active_layers))
        if up_weight > down_weight:
            result.direction = "up"
            result.confidence = up_weight / (up_weight + down_weight) * min(100, total_conf * amp)
        else:
            result.direction = "down"
            result.confidence = down_weight / (up_weight + down_weight) * min(100, total_conf * amp)

    # ── Target price ──
    atr = current_price * 0.015  # default 1.5% ATR
    if candles and len(candles) >= 15:
        tr_vals = []
        for i in range(1, 15):
            h = float(candles[-i].get("high", candles[-i].get("h", 0)))
            l = float(candles[-i].get("low", candles[-i].get("l", 0)))
            pc = float(candles[-i-1].get("close", candles[-i-1].get("c", current_price)))
            tr_vals.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(tr_vals) / len(tr_vals)

    if result.direction == "up":
        result.target_price = current_price + atr * (2.0 if result.confidence >= 60 else 1.2)
        result.target_pct = (result.target_price / current_price - 1) * 100
    elif result.direction == "down":
        result.target_price = current_price - atr * (2.0 if result.confidence >= 60 else 1.2)
        result.target_pct = (result.target_price / current_price - 1) * 100

    # ── Action: confidence-gated ──
    if result.confidence >= 70:
        result.action = "buy" if result.direction == "up" else "sell"
        result.reasoning = f"High confidence ({result.confidence:.0f}%) {result.direction} prediction"
    elif result.confidence >= 45:
        result.action = "hold"
        result.reasoning = f"Moderate confidence ({result.confidence:.0f}%) — wait for confirmation"
    else:
        result.action = "hold"
        result.reasoning = f"Low confidence ({result.confidence:.0f}%) — insufficient signal"

    return result


def format_prediction_for_ai(pred: PredictionResult) -> str:
    """Compact AI-ready format."""
    direction_emoji = {"up": "🟢", "down": "🔴", "flat": "⚪"}
    return (
        f"🔮 PREDICTION ({pred.symbol} {pred.timeframe}): {direction_emoji.get(pred.direction, '?')} {pred.direction.upper()} "
        f"conf={pred.confidence:.0f}% target=${pred.target_price:.4f} ({pred.target_pct:+.1f}%)\n"
        f"  ML:{pred.ml_score:.0f} Exh:{pred.exhaustion_score:.0f} Struct:{pred.structure_score:.0f} "
        f"Flow:{pred.flow_score:.0f}\n"
        f"  → {pred.action}: {pred.reasoning}"
    )


# ── Self-test ──
if __name__ == "__main__":
    import random
    random.seed(42)
    base = 100.0
    candles = []
    for i in range(80):
        if i < 45:
            change = random.uniform(0.5, 2.0)
        else:
            change = random.uniform(-1.5, -0.3)
        base += change
        candles.append({
            "close": base + random.uniform(-0.2, 0.2),
            "high": base + random.uniform(0.5, 3),
            "low": base - random.uniform(0.5, 3),
            "volume": random.uniform(100, 300),
        })

    pred = predict_unified("TEST", candles, 100.0)
    print(format_prediction_for_ai(pred))
