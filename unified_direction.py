#!/usr/bin/env python3
"""
Unified Direction Predictor — continuous directional read per coin.

Combines ALL available data sources into a single directional probability.
Each source votes: up, down, or neutral with a confidence weight.

ARCHITECTURE:
  Layer 1: Microstructure (fastest — tick data, order book, CVD, taker ratio)
  Layer 2: Technical (ML models, VWAP, Hurst, patterns)
  Layer 3: Sentiment (funding, OI delta, whale activity, liquidations)
  Layer 4: Regime classifier (extreme vs normal vs trending)

Output: DirectionPredicton(direction, confidence, regime, expected_move_pct, horizon_s)
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DirectionPrediction:
    """Unified directional prediction for a coin."""
    coin: str
    direction: str          # "up", "down", "flat"
    confidence: float       # 0-100
    regime: str             # "extreme_up", "extreme_down", "normal", "trending_up", "trending_down"
    expected_move_pct: float  # expected % move in horizon
    horizon_seconds: int      # prediction time horizon
    # Breakdown
    layer1_score: float = 0.0   # microstructure score (-1 to +1)
    layer2_score: float = 0.0   # technical score
    layer3_score: float = 0.0   # sentiment score
    votes_up: int = 0
    votes_down: int = 0
    votes_neutral: int = 0
    # For regime
    is_extreme: bool = False
    vwap_sigma: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════
# LAYER 1: Microstructure (fast — real-time tick/order book data)
# ═══════════════════════════════════════════════════════════════

def _layer1_microstructure(coin: str, ws_data: dict, mids: dict,
                           cvd_signal: float = 0.0,
                           taker_ratio: float = 0.5) -> tuple[float, int, int]:
    """
    Fastest signals: order book imbalance, last trade direction, CVD, taker ratio.
    Returns (score -1..+1, votes_up, votes_down).
    """
    score = 0.0
    votes_up = 0
    votes_down = 0

    trades = ws_data.get("trades", {})
    best_bid = ws_data.get("best_bid", 0)
    best_ask = ws_data.get("best_ask", 0)
    bid_depth = ws_data.get("bid_depth_usd", 0)
    ask_depth = ws_data.get("ask_depth_usd", 0)

    # 1. Order book depth imbalance
    total_depth = bid_depth + ask_depth
    if total_depth > 0:
        depth_imbalance = (bid_depth - ask_depth) / total_depth  # + = bid heavy, - = ask heavy
        score += depth_imbalance * 0.25
        if depth_imbalance > 0.05:
            votes_up += 1
        elif depth_imbalance < -0.05:
            votes_down += 1

    # 2. Last trade direction
    if coin in trades:
        t = trades[coin]
        trade_side = t.get("side", "")
        trade_sz = float(t.get("size", 0))
        if trade_side == "B" and trade_sz > 0:
            score += 0.10
            votes_up += 1
        elif trade_side == "A" and trade_sz > 0:
            score -= 0.10
            votes_down += 1

    # 3. CVD (cumulative volume delta — sustained buying/selling pressure)
    if abs(cvd_signal) > 0.01:
        score += cvd_signal * 0.30
        if cvd_signal > 0.02:
            votes_up += 1
        elif cvd_signal < -0.02:
            votes_down += 1

    # 4. Taker ratio (buy vs sell volume)
    if taker_ratio > 0.55:
        score += 0.10
        votes_up += 1
    elif taker_ratio < 0.45:
        score -= 0.10
        votes_down += 1

    return max(-1.0, min(1.0, score)), votes_up, votes_down


# ═══════════════════════════════════════════════════════════════
# LAYER 2: Technical (ML models, VWAP, Hurst, patterns)
# ═══════════════════════════════════════════════════════════════

def _layer2_technical(coin: str, current_price: float, mids: dict,
                      ml_direction: str = "flat", ml_confidence: float = 0,
                      vwap_sigma: float = 0.0,
                      hurst: float = 0.5,
                      exhaustion_score: float = 0,
                      exhaustion_direction: str = "") -> tuple[float, int, int, float]:
    """
    Medium-speed signals: ML models, VWAP deviation, Hurst exponent, peak exhaustion.
    Returns (score -1..+1, votes_up, votes_down, vwap_sigma_out).
    """
    score = 0.0
    votes_up = 0
    votes_down = 0

    # 1. ML prediction (highest weight in this layer)
    if ml_direction == "up" and ml_confidence >= 30:
        weight = ml_confidence / 100 * 0.40
        score += weight
        votes_up += 1
    elif ml_direction == "down" and ml_confidence >= 30:
        weight = ml_confidence / 100 * 0.40
        score -= weight
        votes_down += 1

    # 2. VWAP signal — mean reversion OR trend continuation
    # If ML agrees with VWAP direction, it's trend continuation (don't fade it)
    # If ML disagrees or is flat, VWAP deviation = mean reversion signal
    is_trending = hurst > 0.55
    vwap_weight = 0.10 if is_trending else 0.25
    
    ml_says_up = ml_direction == "up" and ml_confidence >= 50
    ml_says_down = ml_direction == "down" and ml_confidence >= 50
    
    if vwap_sigma > 1.0:
        if not ml_says_up:
            score -= vwap_weight * min(vwap_sigma / 2.0, 1.0)
            votes_down += 1
    elif vwap_sigma < -1.0:
        if not ml_says_down:
            score += vwap_weight * min(abs(vwap_sigma) / 2.0, 1.0)
            votes_up += 1

    # 3. Hurst exponent — trending vs mean-reverting bias
    if hurst > 0.60:
        # Strongly trending — follow the trend
        # Trend direction from recent price momentum
        pass  # Trend direction handled by Layer 3
    elif hurst < 0.45:
        # Mean-reverting — fade extremes
        if vwap_sigma > 0.8:
            score -= 0.05
        elif vwap_sigma < -0.8:
            score += 0.05

    # 4. Peak exhaustion — reversal signal
    # direction from detector: "peak" = bearish reversal, "bottom" = bullish reversal
    if exhaustion_score >= 40:
        if exhaustion_direction == "peak" and ml_direction != "down":
            score -= 0.15
            votes_down += 1
        elif exhaustion_direction == "bottom" and ml_direction != "up":
            score += 0.15
            votes_up += 1

    return max(-1.0, min(1.0, score)), votes_up, votes_down, vwap_sigma


# ═══════════════════════════════════════════════════════════════
# LAYER 3: Sentiment & Macro (funding, OI, whales, liquidations)
# ═══════════════════════════════════════════════════════════════

def _layer3_sentiment(coin: str, funding_rate: float = 0.0,
                      oi_delta_pct: float = 0.0,
                      whale_buying: bool = False,
                      whale_selling: bool = False,
                      liq_clusters_above: float = 0,
                      liq_clusters_below: float = 0,
                      current_price: float = 0.0) -> tuple[float, int, int]:
    """
    Slower signals: funding rates, open interest, whale activity, liquidation levels.
    Returns (score -1..+1, votes_up, votes_down).
    """
    score = 0.0
    votes_up = 0
    votes_down = 0

    # 1. Funding rate — continuous score scaled to typical HL ranges
    # HL funding is per-hour fraction. Typical range: -0.0001 to +0.0001 for majors
    # -0.00005 = funding is negative → shorts paying → bullish (contrarian)
    # +0.00005 = funding positive → longs paying → bearish (contrarian)
    funding_score = -funding_rate * 1500  # Scale: -0.00005 → +0.075, +0.00005 → -0.075
    score += max(-0.15, min(0.15, funding_score))
    if funding_score > 0.03:
        votes_up += 1
    elif funding_score < -0.03:
        votes_down += 1

    # 2. OI delta — continuous score
    # Even 0.1% OI change in 5min is meaningful
    oi_score = oi_delta_pct * 0.05  # 1% delta → 0.05 score
    score += max(-0.10, min(0.10, oi_score))
    if oi_delta_pct > 0.3:
        votes_up += 1
    elif oi_delta_pct < -0.3:
        votes_down += 1

    # 3. Whale activity
    if whale_buying:
        score += 0.12
        votes_up += 1
    if whale_selling:
        score -= 0.12
        votes_down += 1

    # 4. Liquidation magnets — price gravitates toward large liquidation clusters
    if current_price > 0:
        dist_above = (liq_clusters_above - current_price) / current_price if liq_clusters_above > 0 else 999
        dist_below = (current_price - liq_clusters_below) / current_price if liq_clusters_below > 0 else 999
        # Closer cluster pulls harder
        if dist_above < 0.02 and dist_above < dist_below:
            score += 0.10  # Price likely to rise toward liquidation cluster
            votes_up += 1
        elif dist_below < 0.02 and dist_below < dist_above:
            score -= 0.10  # Price likely to fall toward liquidation cluster
            votes_down += 1

    return max(-1.0, min(1.0, score)), votes_up, votes_down


# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def classify_regime(vwap_sigma: float, recent_volatility: float,
                    hurst: float, ml_confidence: float,
                    is_extreme_move: bool,
                    price_change_5m_pct: float) -> str:
    """
    Classify market regime for a coin.

    extreme_up: Unrealistic spike up — hold position for max profit
    extreme_down: Unrealistic drop — dangerous, could be whale manipulation
    trending_up/down: Clear directional trend
    normal: Oscillating within range
    """
    if is_extreme_move:
        if price_change_5m_pct > 5.0:
            return "extreme_up"
        elif price_change_5m_pct < -5.0:
            return "extreme_down"

    if abs(vwap_sigma) > 3.0:
        if vwap_sigma > 0:
            return "extreme_up"
        else:
            return "extreme_down"

    if hurst > 0.55:
        if ml_confidence >= 40:
            if "up" in str(ml_confidence):  # will be fixed by caller
                return "trending_up"
            return "trending_down"
        # Use VWAP as trend proxy
        if vwap_sigma > 0.3:
            return "trending_up"
        elif vwap_sigma < -0.3:
            return "trending_down"

    return "normal"


# ═══════════════════════════════════════════════════════════════
# UNIFIED PREDICTOR
# ═══════════════════════════════════════════════════════════════

def predict_direction(coin: str, current_price: float, mids: dict,
                      ws_data: dict | None = None,
                      # Layer 1 inputs
                      cvd_signal: float = 0.0,
                      taker_ratio: float = 0.5,
                      # Layer 2 inputs
                      ml_direction: str = "flat",
                      ml_confidence: float = 0.0,
                      vwap_sigma: float = 0.0,
                      hurst: float = 0.5,
                      exhaustion_score: float = 0,
                      exhaustion_direction: str = "",
                      # Layer 3 inputs
                      funding_rate: float = 0.0,
                      oi_delta_pct: float = 0.0,
                      whale_buying: bool = False,
                      whale_selling: bool = False,
                      liq_clusters_above: float = 0,
                      liq_clusters_below: float = 0,
                      # Regime inputs
                      price_change_5m_pct: float = 0.0,
                      recent_volatility: float = 0.0,
                      ) -> DirectionPrediction:
    """
    Unified directional prediction combining all data layers.

    Returns DirectionPrediction with direction, confidence, regime, and
    expected move size + time horizon.
    """
    ws = ws_data or {}

    # ── Layer 1: Microstructure ──
    l1_score, l1_up, l1_down = _layer1_microstructure(
        coin, ws, mids, cvd_signal, taker_ratio)

    # ── Layer 2: Technical ──
    l2_score, l2_up, l2_down, _vwap = _layer2_technical(
        coin, current_price, mids, ml_direction, ml_confidence,
        vwap_sigma, hurst, exhaustion_score, exhaustion_direction)

    # ── Layer 3: Sentiment ──
    l3_score, l3_up, l3_down = _layer3_sentiment(
        coin, funding_rate, oi_delta_pct,
        whale_buying, whale_selling,
        liq_clusters_above, liq_clusters_below, current_price)

    # ── Combine layers (weighted: microstructure 35%, technical 40%, sentiment 25%) ──
    combined = l1_score * 0.35 + l2_score * 0.40 + l3_score * 0.25

    total_up = l1_up + l2_up + l3_up
    total_down = l1_down + l2_down + l3_down
    total_neutral = (3 + 4 + 4) - total_up - total_down  # max possible votes

    # ── Determine direction ──
    if combined > 0.08:
        direction = "up"
    elif combined < -0.08:
        direction = "down"
    else:
        direction = "flat"

    # ── Confidence: scaled from combined score and vote agreement ──
    base_confidence = abs(combined) * 100
    # Count active layers (score ≠ 0 means it had input data)
    active_layers = sum(1 for s in (l1_score, l2_score, l3_score) if abs(s) > 0.001)
    # Single-source signals are real but underweighted — amplify them
    # 1 layer → 2.5x, 2 layers → 1.5x, 3 layers → 1.0x
    amp = 1.0 + (1.5 / max(1, active_layers))
    base_confidence *= amp
    agreement_bonus = min((total_up + total_down) / max(total_up + total_down + total_neutral, 1), 1.0) * 20
    if direction == "flat":
        confidence = min(base_confidence * 0.5, 40)
    else:
        confidence = min(base_confidence + agreement_bonus, 95)

    # ── Regime classification ──
    is_extreme = abs(vwap_sigma) > 3.0 or abs(price_change_5m_pct) > 5.0
    regime = classify_regime(vwap_sigma, recent_volatility, hurst,
                             ml_confidence, is_extreme, price_change_5m_pct)

    # ── Expected move size and horizon ──
    if regime in ("extreme_up", "extreme_down"):
        expected_move_pct = abs(price_change_5m_pct) * 0.5  # continuation move
        horizon_s = 300  # 5 min — extreme moves are fast
    elif regime in ("trending_up", "trending_down"):
        expected_move_pct = abs(vwap_sigma) * 0.2
        horizon_s = 600  # 10 min
    else:
        # Normal oscillation — small moves, fast reversals
        expected_move_pct = max(abs(vwap_sigma) * 0.15, 0.1)
        horizon_s = 180  # 3 min

    return DirectionPrediction(
        coin=coin,
        direction=direction,
        confidence=confidence,
        regime=regime,
        expected_move_pct=expected_move_pct,
        horizon_seconds=horizon_s,
        layer1_score=l1_score,
        layer2_score=l2_score,
        layer3_score=l3_score,
        votes_up=total_up,
        votes_down=total_down,
        votes_neutral=total_neutral,
        is_extreme=is_extreme,
        vwap_sigma=vwap_sigma,
    )


# ═══════════════════════════════════════════════════════════════
# TRADE DECISION ENGINE
# ═══════════════════════════════════════════════════════════════

def should_enter(pred: DirectionPrediction, has_position: bool = False,
                 min_confidence: float = 55.0) -> tuple[bool, str, str]:
    """
    Decide whether to enter a position based on unified prediction.

    Returns (should_enter, action, reason).
    Action is "BUY", "SELL", or "HOLD".
    """
    if pred.direction == "flat":
        return False, "HOLD", f"no direction (conf={pred.confidence:.0f})"

    if pred.confidence < min_confidence:
        return False, "HOLD", f"low confidence ({pred.confidence:.0f}<{min_confidence:.0f})"

    # Extreme moves: wait for confirmation, then enter WITH the move
    if pred.regime == "extreme_up":
        if pred.direction == "up":
            return True, "BUY", f"extreme_up continuation (conf={pred.confidence:.0f})"
        else:
            return False, "HOLD", f"extreme_up but direction is {pred.direction}"

    if pred.regime == "extreme_down":
        # Extreme down is DANGEROUS — only enter if very high confidence
        if pred.direction == "down" and pred.confidence >= 75:
            return True, "SELL", f"extreme_down with high conf ({pred.confidence:.0f})"
        else:
            return False, "HOLD", f"extreme_down too risky (conf={pred.confidence:.0f})"

    # Trending: follow the trend
    if pred.regime in ("trending_up",) and pred.direction == "up":
        return True, "BUY", f"trending_up ({pred.confidence:.0f}%)"
    if pred.regime in ("trending_down",) and pred.direction == "down":
        return True, "SELL", f"trending_down ({pred.confidence:.0f}%)"

    # Normal oscillation: trade the rhythm — buy low, sell high
    if pred.regime == "normal":
        if pred.direction == "up" and pred.vwap_sigma < -0.3:
            # Price below VWAP + prediction up = good buy
            return True, "BUY", f"normal bounce (VWAP:{pred.vwap_sigma:+.1f}σ, conf={pred.confidence:.0f})"
        elif pred.direction == "down" and pred.vwap_sigma > 0.3:
            return True, "SELL", f"normal fade (VWAP:{pred.vwap_sigma:+.1f}σ, conf={pred.confidence:.0f})"
        elif pred.direction == "up":
            return True, "BUY", f"normal up (conf={pred.confidence:.0f})"
        elif pred.direction == "down":
            return True, "SELL", f"normal down (conf={pred.confidence:.0f})"

    return False, "HOLD", f"no clear signal (regime={pred.regime}, dir={pred.direction})"


def should_exit(pred: DirectionPrediction, entry_direction: str,
                entry_price: float, current_price: float,
                in_profit: bool, hold_seconds: float = 0) -> tuple[bool, str]:
    """
    Decide whether to exit a position.

    Exit conditions:
    1. Direction flips against us with confidence >= 50
    2. In extreme regime: trail stop, don't exit early
    3. In normal regime: cut fast if direction changes
    4. Time-based: if prediction horizon passed without profit
    """
    pnl_pct = ((current_price - entry_price) / entry_price * 100 *
               (1 if entry_direction == "BUY" else -1))

    # ── Direction flip: prediction now goes against our position ──
    direction_flipped = (
        (entry_direction == "BUY" and pred.direction == "down" and pred.confidence >= 50) or
        (entry_direction == "SELL" and pred.direction == "up" and pred.confidence >= 50)
    )

    # ── Extreme regime: don't exit on small flips, trail the move ──
    if pred.regime in ("extreme_up", "extreme_down"):
        if direction_flipped and pred.confidence >= 70:
            return True, f"extreme regime direction flip (conf={pred.confidence:.0f})"
        return False, "holding extreme move"

    # ── Trending regime: exit only on strong reversal signal ──
    if pred.regime in ("trending_up", "trending_down"):
        if direction_flipped and pred.confidence >= 60:
            return True, f"trend reversal (conf={pred.confidence:.0f})"
        if pnl_pct < -1.5:
            return True, f"stop loss in trend ({pnl_pct:+.2f}%)"
        return False, "trend intact"

    # ── Normal regime: cut fast on direction change or small loss ──
    if pred.regime == "normal":
        if direction_flipped:
            return True, f"direction flipped (conf={pred.confidence:.0f})"
        if pnl_pct < -0.5:
            return True, f"tight stop in normal regime ({pnl_pct:+.2f}%)"
        # Time-based: if we've held past prediction horizon and no profit, exit
        if hold_seconds > pred.horizon_seconds and pnl_pct < 0.1:
            return True, f"prediction expired ({hold_seconds:.0f}s, no profit)"
        return False, "position OK"

    return False, "hold"


def get_position_size_pct(pred: DirectionPrediction, equity: float,
                          max_position_pct: float = 0.10) -> float:
    """
    Kelly-inspired position sizing based on prediction confidence and regime.

    Extreme regime: smaller size (unpredictable)
    Trending regime: larger size (higher probability)
    Normal regime: base size
    """
    base_pct = pred.confidence / 100 * max_position_pct

    if pred.regime in ("extreme_up", "extreme_down"):
        base_pct *= 0.5  # Half size for extreme moves
    elif pred.regime in ("trending_up", "trending_down"):
        base_pct *= 1.2  # 20% more for trending

    # Cap at max_position_pct
    return min(base_pct, max_position_pct, 0.20)  # Hard cap 20%
