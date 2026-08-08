"""
Perfect Prediction Engine — combines the most reliable signals in crypto into
multi-horizon probability distributions.

THREE IRREDUCIBLE SIGNALS (backed by market microstructure research):
  1. FUNDING RATE EXTREMES → mean reversion (most reliable signal in crypto)
     - When funding > 0.01%/hr → too many longs → price corrects down
     - When funding < -0.005%/hr → too many shorts → price squeezes up
     - Predictive horizon: 1-4 hours

  2. ORDER BOOK IMBALANCE → short-term direction (milliseconds to minutes)
     - Bid depth > ask depth → buying pressure → up in next 1-5m
     - Ask wall building → selling pressure → down in next 1-5m
     - Predictive horizon: seconds to 5 minutes

  3. MULTI-TIMEFRAME CONFLUENCE → conviction multiplier
     - When 1m, 5m, 15m, 1h all point same direction → strong signal
     - When timeframes disagree → low conviction, stay flat
     - Predictive horizon: 15m to 4h

OUTPUT: Probability distribution at each timeframe with exact price targets.
"""

import math, time, json, os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HorizonPrediction:
    """Prediction at one timeframe."""
    timeframe: str  # "1m", "5m", "15m", "1h", "4h"
    direction: str  # "up", "down", "flat"
    confidence: float  # 0-100
    target_price: float
    target_pct: float  # expected % move
    prob_up: float = 0.33
    prob_down: float = 0.33
    prob_flat: float = 0.34


@dataclass  
class PerfectPrediction:
    """Multi-horizon prediction with probability distributions."""
    symbol: str
    current_price: float
    timestamp: float
    # Per-horizon predictions
    horizons: dict = field(default_factory=dict)  # "1m" → HorizonPrediction, etc.
    # Composite
    consensus_direction: str = "flat"
    consensus_confidence: float = 0
    consensus_target: float = 0
    # Signal sources
    funding_signal: str = ""  # "extreme_long", "extreme_short", "neutral"
    ob_signal: str = ""  # "buy_pressure", "sell_pressure", "balanced"
    multi_tf_alignment: int = 0  # 0-5 how many timeframes agree
    action: str = ""  # "buy", "sell", "hold"


# ── Caching to avoid excessive API calls ──
_funding_cache: dict = {}  # coin → {signal, strength, funding, cached_at}
_ob_cache: dict = {}  # coin → {signal, strength, imbalance, cached_at}
FUNDING_CACHE_TTL = 300  # 5 min (funding changes hourly)
OB_CACHE_TTL = 30  # 30 seconds (order book changes rapidly)


def _get_funding_signal(coin: str) -> dict:
    """Get funding rate and determine if it's at an extreme. Cached 5 min."""
    now = time.time()
    if coin in _funding_cache and now - _funding_cache[coin].get("cached_at", 0) < FUNDING_CACHE_TTL:
        return _funding_cache[coin]
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        result = info.meta_and_asset_ctxs()
        
        universe = result[0].get("universe", []) if isinstance(result, tuple) else result.get("universe", [])
        ctxs = result[1] if isinstance(result, tuple) else result.get("asset_ctxs", result.get("context", []))
        
        funding = 0.0
        open_interest = 0.0
        
        for i, asset in enumerate(universe):
            if asset.get("name", "").upper() == coin.upper():
                if i < len(ctxs):
                    ctx = ctxs[i]
                    funding = float(ctx.get("funding", 0))
                    open_interest = float(ctx.get("openInterest", 0))
                break
        
        # Determine signal
        if funding > 0.0001:  # > 0.01%/hr → extreme longs
            signal = "extreme_long"
            strength = min(100, funding * 500)  # 0.02% → 10, 0.1% → 50
        elif funding < -0.00005:  # < -0.005%/hr → extreme shorts
            signal = "extreme_short"
            strength = min(100, abs(funding) * 1000)
        else:
            signal = "neutral"
            strength = 0
        
        result = {
            "signal": signal,
            "strength": strength,
            "funding": funding,
            "open_interest": open_interest,
            "cached_at": time.time(),
        }
        _funding_cache[coin] = result
        return result
    except Exception:
        return {"signal": "neutral", "strength": 0, "funding": 0, "open_interest": 0, "cached_at": 0}


def _get_order_book_signal(coin: str, current_price: float) -> dict:
    """Analyze order book for immediate direction. Cached 30s."""
    now = time.time()
    if coin in _ob_cache and now - _ob_cache[coin].get("cached_at", 0) < OB_CACHE_TTL:
        return _ob_cache[coin]
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        book = info.l2_snapshot(coin)
        
        levels = book.get("levels", [])
        if not levels or len(levels) < 2:
            return {"signal": "balanced", "strength": 0}
        
        # levels[0] = asks (sells), levels[1] = bids (buys)
        asks = levels[0] if len(levels) > 0 else []
        bids = levels[1] if len(levels) > 1 else []
        
        if not asks or not bids:
            return {"signal": "balanced", "strength": 0}
        
        # Sum top 5 levels in native units (size)
        ask_depth = sum(float(a.get("sz", 0)) for a in asks[:5])
        bid_depth = sum(float(b.get("sz", 0)) for b in bids[:5])
        total = ask_depth + bid_depth
        
        if total <= 0:
            result = {"signal": "balanced", "strength": 0, "imbalance": 0, "cached_at": now}
            _ob_cache[coin] = result
            return result
        
        imbalance = (bid_depth - ask_depth) / total  # + = buy pressure, - = sell pressure
        
        if imbalance > 0.15:
            signal = "buy_pressure"
            strength = min(100, imbalance * 200)
        elif imbalance < -0.15:
            signal = "sell_pressure"
            strength = min(100, abs(imbalance) * 200)
        else:
            signal = "balanced"
            strength = abs(imbalance) * 100
        
        result = {
            "signal": signal,
            "strength": strength,
            "imbalance": imbalance,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "cached_at": now,
        }
        _ob_cache[coin] = result
        return result
    except Exception:
        result = {"signal": "balanced", "strength": 0, "imbalance": 0, "cached_at": 0}
        return result


def _compute_multi_tf_direction(candles_1m: list, candles_5m: list,
                                 candles_15m: list, candles_1h: list) -> dict:
    """Check if multiple timeframes agree on direction."""
    def tf_direction(candles):
        if not candles or len(candles) < 5:
            return "flat"
        closes = [float(c.get("close", c.get("c", 0))) for c in candles]
        if len(closes) < 5:
            return "flat"
        short_ema = sum(closes[-3:]) / 3
        long_ema = sum(closes[-8:]) / min(8, len(closes))
        if short_ema > long_ema * 1.002:
            return "up"
        elif short_ema < long_ema * 0.998:
            return "down"
        return "flat"
    
    dirs = {
        "1m": tf_direction(candles_1m) if candles_1m else "flat",
        "5m": tf_direction(candles_5m) if candles_5m else "flat",
        "15m": tf_direction(candles_15m) if candles_15m else "flat",
        "1h": tf_direction(candles_1h) if candles_1h else "flat",
    }
    
    up_count = sum(1 for d in dirs.values() if d == "up")
    down_count = sum(1 for d in dirs.values() if d == "down")
    alignment = max(up_count, down_count)
    
    consensus = "flat"
    if up_count >= 3:
        consensus = "up"
    elif down_count >= 3:
        consensus = "down"
    
    return {"directions": dirs, "alignment": alignment, "consensus": consensus}


def predict_perfect(
    coin: str,
    current_price: float,
    candles_1m: list = None,
    candles_5m: list = None,
    candles_15m: list = None,
    candles_1h: list = None,
    atr_15m: float = 0,
) -> PerfectPrediction:
    """
    THE PERFECT PREDICTION — combines funding extremes + order book + multi-TF confluence
    into probability distributions at every timeframe.
    
    This is the highest-confidence prediction possible with available data.
    """
    pred = PerfectPrediction(
        symbol=coin,
        current_price=current_price,
        timestamp=time.time(),
    )
    
    # ── 1. Funding extreme signal (most reliable in crypto) ──
    funding = _get_funding_signal(coin)
    pred.funding_signal = funding["signal"]
    
    # ── 2. Order book imbalance (short-term) ──
    ob = _get_order_book_signal(coin, current_price)
    pred.ob_signal = ob["signal"]
    
    # ── 3. Multi-timeframe confluence ──
    mtf = _compute_multi_tf_direction(candles_1m, candles_5m, candles_15m, candles_1h)
    pred.multi_tf_alignment = mtf["alignment"]
    
    # ── ATR for target sizing ──
    atr = atr_15m if atr_15m > 0 else current_price * 0.005
    atr_pct_1m = atr / current_price * 100 * 0.3
    atr_pct_5m = atr / current_price * 100 * 0.5
    atr_pct_15m = atr / current_price * 100 * 0.8
    atr_pct_1h = atr / current_price * 100 * 1.5
    atr_pct_4h = atr / current_price * 100 * 3.0
    
    # ── Build per-horizon predictions ──
    
    # 1-minute: purely order book driven
    if ob["signal"] == "buy_pressure":
        h1 = HorizonPrediction("1m", "up", ob["strength"] * 0.8,
                               current_price * (1 + atr_pct_1m/100),
                               atr_pct_1m, prob_up=0.45, prob_down=0.25, prob_flat=0.30)
    elif ob["signal"] == "sell_pressure":
        h1 = HorizonPrediction("1m", "down", ob["strength"] * 0.8,
                               current_price * (1 - atr_pct_1m/100),
                               -atr_pct_1m, prob_up=0.25, prob_down=0.45, prob_flat=0.30)
    else:
        h1 = HorizonPrediction("1m", "flat", 10,
                               current_price, 0, prob_up=0.33, prob_down=0.33, prob_flat=0.34)
    pred.horizons["1m"] = h1
    
    # 5-minute: OB + short-term trend
    ob_strength = ob["strength"] * 0.6
    mtf_5m_dir = mtf["directions"].get("5m", "flat")
    
    if ob["signal"] == "buy_pressure" and mtf_5m_dir == "up":
        conf = ob_strength + 20
        h5 = HorizonPrediction("5m", "up", min(85, conf),
                               current_price * (1 + atr_pct_5m/100),
                               atr_pct_5m, prob_up=0.50, prob_down=0.20, prob_flat=0.30)
    elif ob["signal"] == "sell_pressure" and mtf_5m_dir == "down":
        conf = ob_strength + 20
        h5 = HorizonPrediction("5m", "down", min(85, conf),
                               current_price * (1 - atr_pct_5m/100),
                               -atr_pct_5m, prob_up=0.20, prob_down=0.50, prob_flat=0.30)
    elif mtf_5m_dir != "flat":
        h5 = HorizonPrediction("5m", mtf_5m_dir, 40,
                               current_price * (1 + atr_pct_5m/100 * (1 if mtf_5m_dir=="up" else -1)),
                               atr_pct_5m * (1 if mtf_5m_dir=="up" else -1))
    else:
        h5 = HorizonPrediction("5m", "flat", 15, current_price, 0)
    pred.horizons["5m"] = h5
    
    # 15-minute: multi-TF + momentum
    mtf_15m_dir = mtf["directions"].get("15m", "flat")
    mtf_1h_dir = mtf["directions"].get("1h", "flat")
    
    if mtf_15m_dir == mtf_1h_dir and mtf_15m_dir != "flat":
        conf = 60 + mtf["alignment"] * 8
        h15 = HorizonPrediction("15m", mtf_15m_dir, min(90, conf),
                                current_price * (1 + atr_pct_15m/100 * (1 if mtf_15m_dir=="up" else -1)),
                                atr_pct_15m * (1 if mtf_15m_dir=="up" else -1),
                                prob_up=0.50 if mtf_15m_dir=="up" else 0.20,
                                prob_down=0.50 if mtf_15m_dir=="down" else 0.20)
    elif mtf_15m_dir != "flat":
        h15 = HorizonPrediction("15m", mtf_15m_dir, 45,
                                current_price * (1 + atr_pct_15m/100 * (1 if mtf_15m_dir=="up" else -1)),
                                atr_pct_15m * (1 if mtf_15m_dir=="up" else -1))
    else:
        h15 = HorizonPrediction("15m", "flat", 20, current_price, 0)
    pred.horizons["15m"] = h15
    
    # 1-hour: trend + funding
    if mtf_1h_dir == "up" and funding["signal"] != "extreme_long":
        conf = 50 if mtf["alignment"] >= 3 else 35
        h1h = HorizonPrediction("1h", "up", conf,
                                current_price * (1 + atr_pct_1h/100),
                                atr_pct_1h)
    elif mtf_1h_dir == "down" and funding["signal"] != "extreme_short":
        conf = 50 if mtf["alignment"] >= 3 else 35
        h1h = HorizonPrediction("1h", "down", conf,
                                current_price * (1 - atr_pct_1h/100),
                                -atr_pct_1h)
    elif funding["signal"] == "extreme_long":
        # Funding extreme → mean reversion down in 1-4h
        h1h = HorizonPrediction("1h", "down", funding["strength"],
                                current_price * (1 - atr_pct_1h/100 * 1.5),
                                -atr_pct_1h * 1.5,
                                prob_up=0.20, prob_down=0.55, prob_flat=0.25)
    elif funding["signal"] == "extreme_short":
        h1h = HorizonPrediction("1h", "up", funding["strength"],
                                current_price * (1 + atr_pct_1h/100 * 1.5),
                                atr_pct_1h * 1.5,
                                prob_up=0.55, prob_down=0.20, prob_flat=0.25)
    else:
        h1h = HorizonPrediction("1h", "flat", 25, current_price, 0)
    pred.horizons["1h"] = h1h
    
    # 4-hour: regime + funding dominant
    if funding["signal"] == "extreme_long":
        h4h = HorizonPrediction("4h", "down", funding["strength"] * 0.9,
                                current_price * (1 - atr_pct_4h/100 * 2.0),
                                -atr_pct_4h * 2.0,
                                prob_up=0.15, prob_down=0.60, prob_flat=0.25)
    elif funding["signal"] == "extreme_short":
        h4h = HorizonPrediction("4h", "up", funding["strength"] * 0.9,
                                current_price * (1 + atr_pct_4h/100 * 2.0),
                                atr_pct_4h * 2.0,
                                prob_up=0.60, prob_down=0.15, prob_flat=0.25)
    elif mtf["alignment"] >= 4:
        dir_4h = mtf["consensus"]
        h4h = HorizonPrediction("4h", dir_4h, 45,
                                current_price * (1 + atr_pct_4h/100 * (1 if dir_4h=="up" else -1)),
                                atr_pct_4h * (1 if dir_4h=="up" else -1))
    else:
        h4h = HorizonPrediction("4h", "flat", 30, current_price, 0)
    pred.horizons["4h"] = h4h
    
    # ── Consensus: weighted by timeframe reliability ──
    dirs_weighted = {"up": 0, "down": 0, "flat": 0}
    weights_tf = {"1m": 1, "5m": 2, "15m": 3, "1h": 4, "4h": 2}  # 1h most reliable
    
    for tf, h in pred.horizons.items():
        w = weights_tf.get(tf, 1) * (h.confidence / 100)
        dirs_weighted[h.direction] += w
    
    total = sum(dirs_weighted.values())
    if total > 0:
        best_dir = max(dirs_weighted, key=dirs_weighted.get)
        pred.consensus_direction = best_dir
        pred.consensus_confidence = dirs_weighted[best_dir] / total * 100
    
    # ── Target from 1h horizon (most actionable) ──
    h1h = pred.horizons.get("1h")
    if h1h and h1h.direction != "flat":
        pred.consensus_target = h1h.target_price
    
    # ── Action: only act when consensus is clear ──
    if pred.consensus_direction != "flat" and pred.consensus_confidence >= 55:
        pred.action = "buy" if pred.consensus_direction == "up" else "sell"
    elif pred.multi_tf_alignment >= 3:
        pred.action = "buy" if mtf["consensus"] == "up" else "sell"
    else:
        pred.action = "hold"
    
    return pred


def format_perfect_prediction(pred: PerfectPrediction) -> str:
    """Human-readable multi-horizon prediction."""
    lines = [f"🎯 PERFECT PREDICTION ({pred.symbol} @ ${pred.current_price:.4f})"]
    lines.append(f"  Funding: {pred.funding_signal} | OB: {pred.ob_signal} | TF align: {pred.multi_tf_alignment}/5")
    lines.append(f"  Consensus: {pred.consensus_direction.upper()} conf={pred.consensus_confidence:.0f}% → ${pred.consensus_target:.4f}")
    
    for tf in ["1m", "5m", "15m", "1h", "4h"]:
        h = pred.horizons.get(tf)
        if h:
            dir_arrow = {"up": "↑", "down": "↓", "flat": "→"}
            lines.append(
                f"  {tf:4s} {dir_arrow.get(h.direction, '?'):1s} conf={h.confidence:.0f}% "
                f"→ ${h.target_price:.4f} ({h.target_pct:+.2f}%) "
                f"[up:{h.prob_up:.0%} down:{h.prob_down:.0%} flat:{h.prob_flat:.0%}]"
            )
    
    lines.append(f"  Action: {pred.action.upper()}")
    return "\n".join(lines)


# ── Self-test ──
if __name__ == "__main__":
    # Quick test with SOL
    from hyperliquid_daemon import _fetch_candles
    from hyperliquid_client import get_all_mids
    
    mids = get_all_mids()
    current = float(mids.get("SOL", 77))
    candles_1m = _fetch_candles("SOL", "1m", 60)
    candles_5m = _fetch_candles("SOL", "5m", 60)
    candles_15m = _fetch_candles("SOL", "15m", 60)
    candles_1h = _fetch_candles("SOL", "1h", 24)
    
    pred = predict_perfect("SOL", current, candles_1m, candles_5m, candles_15m, candles_1h)
    print(format_perfect_prediction(pred))
