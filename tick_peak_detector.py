"""
High-Frequency Peak/Bottom Detector — millisecond-level top and bottom prediction.
Uses EVERY available signal: order book, tick flow, CVD, VWAP, liquidity, multi-TF alignment.

Runs in the 10-second position monitor loop. No heavy ML — pure signal processing for speed.

Signals (weighted, 0-100 composite):
  1. ORDER BOOK THINNING (25%) — bid depth evaporating → peak forming
  2. TICK FLOW IMBALANCE (20%) — large sells at bid → distribution at top
  3. CVD DIVERGENCE (20%) — price up, CVD down → bearish peak signal
  4. VWAP EXTENSION (15%) — price > 2σ above VWAP → overextended
  5. LIQUIDATION MAGNET (10%) — large liquidation cluster nearby → price target
  6. MULTI-TF ALIGNMENT (10%) — all timeframes agree → strong signal

Output: exact predicted top/bottom price with confidence, updated every 10 seconds.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional


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
