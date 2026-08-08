#!/usr/bin/env python3
"""
Context Enricher — feeds ALL available intelligence modules into the AI context.

Each module produces a compact text summary. The enricher combines them into a
single "intel" block that gets appended to the AI prompt. <100 tokens per module.

Modules wired:
- volume_delta: CVD direction, buying/selling pressure
- cvd_divergence: CVD vs price divergence (leading reversal signal)
- vwap_signal: VWAP bands mean reversion / trend confirmation
- liquidation_cascade: mass liquidation bounce probability
- oi_delta: Open Interest change (trend confirmation / exhaustion)
- market_structure: support/resistance levels, market phase
- liquidation_data: where stops cluster (avoid hunts)
- chart_patterns: double top/bottom, H&S, flags
- pump_detector: rapid price moves (avoid FOMO, catch early)
- economic_calendar: high-impact news events
- correlation_tracker: BTC correlation, pair relationships
- hyperliquid_whale: large-wallet activity and convergence signals
- hyperliquid_delta_neutral: active funding arb positions and PnL
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("enricher")

TOKEN_BUDGET = 150  # Max tokens for all enrichment combined


def _safe_get(func, default=""):
    """Call func, return default on any error."""
    try:
        return func()
    except Exception:
        return default


# ── Individual module enrichments ──

def _enrich_volume_delta(candles_15m: list) -> str:
    """CVD direction and buying/selling pressure from 15m candles."""
    if len(candles_15m) < 20:
        return ""
    try:
        from volume_delta import compute_volume_delta_features
        ohlcv = []
        for c in candles_15m[-50:]:
            ohlcv.append({
                "o": float(c.get("o", c.get("open", 0))),
                "h": float(c.get("h", c.get("high", 0))),
                "l": float(c.get("l", c.get("low", 0))),
                "c": float(c.get("c", c.get("close", 0))),
                "v": float(c.get("v", c.get("volume", 0))),
            })
        features = compute_volume_delta_features(ohlcv)
        cvd_trend = features.get("cvd_trend", "flat")
        pressure = features.get("pressure", "neutral")
        delta_ratio = features.get("delta_ratio", 0.5)
        parts = [f"CVD:{cvd_trend}"]
        if pressure != "neutral":
            parts.append(f"pressure:{pressure}")
        if delta_ratio > 0.6:
            parts.append("buyers_dominate")
        elif delta_ratio < 0.4:
            parts.append("sellers_dominate")
        return " ".join(parts)
    except Exception:
        return ""


def _enrich_market_structure(coin: str, candles_15m: list) -> str:
    """Support/resistance levels and market phase from swing points."""
    if len(candles_15m) < 30:
        return ""
    try:
        closes = np.array([float(c.get("c", c.get("close", 0))) for c in candles_15m[-50:]])
        highs = np.array([float(c.get("h", c.get("high", 0))) for c in candles_15m[-50:]])
        lows = np.array([float(c.get("l", c.get("low", 0))) for c in candles_15m[-50:]])

        from market_structure import classify_market_structure, find_swing_points

        swings = find_swing_points(closes, highs, lows, left_bars=3, right_bars=3, min_strength=2)
        structure = classify_market_structure(closes, highs, lows,
                                              left_bars=3, right_bars=3,
                                              swing_strength=2)

        phase = structure.get("phase", "unknown")
        nearest_support = structure.get("nearest_support", 0)
        nearest_resistance = structure.get("nearest_resistance", 0)
        price = closes[-1]

        parts = [f"Structure:{phase}"]
        if nearest_support > 0:
            dist = (price - nearest_support) / price * 100
            parts.append(f"S:${nearest_support:.2f}({dist:.1f}%away)")
        if nearest_resistance > 0 and nearest_resistance != nearest_support:
            dist = (nearest_resistance - price) / price * 100
            parts.append(f"R:${nearest_resistance:.2f}({dist:.1f}%away)")
        return " ".join(parts)
    except Exception:
        return ""


def _enrich_liquidations(coin: str) -> str:
    """Where are stops clustered? Avoid getting hunted.
    Uses liquidation_zones module which analyzes order book depth
    clusters from Hyperliquid l2_snapshot to estimate liquidation zones."""
    try:
        from liquidation_zones import get_liquidation_context
        result = get_liquidation_context(coin)
        if result and len(result) > 5:
            return result[:180]
        return ""
    except Exception:
        return ""


def _enrich_chart_patterns(candles_15m: list) -> str:
    """Detect double tops/bottoms, H&S, flags on 15m candles."""
    if len(candles_15m) < 25:
        return ""
    try:
        closes = np.array([float(c.get("c", c.get("close", 0))) for c in candles_15m[-50:]])
        highs = np.array([float(c.get("h", c.get("high", 0))) for c in candles_15m[-50:]])
        lows = np.array([float(c.get("l", c.get("low", 0))) for c in candles_15m[-50:]])

        from chart_patterns import detect_double_bottom, detect_double_top, detect_head_and_shoulders

        dt = detect_double_top(closes, highs, lows, lookback=25)
        db = detect_double_bottom(closes, highs, lows, lookback=25)
        hs = detect_head_and_shoulders(closes, highs, lows, lookback=30)

        if db.get("found"):
            return f"Pattern:double_bottom(conf={db.get('confidence',0):.0f}%)"
        if dt.get("found"):
            return f"Pattern:double_top(conf={dt.get('confidence',0):.0f}%)"
        if hs.get("found"):
            return f"Pattern:{hs.get('type','h&s')}(conf={hs.get('confidence',0):.0f}%)"
        return ""
    except Exception:
        return ""


def _enrich_pump_alert(coin: str) -> str:
    """Rapid price move detection — avoid FOMO entries."""
    try:
        from pump_detector import is_pumping
        pumping, pct, reason = is_pumping(coin)
        if pumping:
            return f"⚠️PUMP:{coin}+{pct:.1f}%({reason})"
        return ""
    except Exception:
        return ""


def _enrich_economic_calendar() -> str:
    """High-impact news events in the next 4 hours."""
    try:
        from economic_calendar import get_upcoming_events
        events = get_upcoming_events(hours_ahead=4, min_impact="high")
        if events:
            return f"News:{events[0][:60]}"
        return ""
    except Exception:
        return ""


def _enrich_correlation(coin: str) -> str:
    """BTC correlation — divergence warning."""
    try:
        from correlation_tracker import get_btc_correlation
        corr = get_btc_correlation(coin)
        if abs(corr) > 0.8:
            return f"BTC_corr:{corr:+.2f}"
        return ""
    except Exception:
        return ""


def _enrich_cvd_divergence(candles_15m: list) -> str:
    """CVD vs price divergence — leading reversal signal."""
    if len(candles_15m) < 40:
        return ""
    try:
        from cvd_divergence import detect_cvd_divergence
        ohlcv = []
        for c in candles_15m[-50:]:
            ohlcv.append({
                "o": float(c.get("o", c.get("open", 0))),
                "h": float(c.get("h", c.get("high", 0))),
                "l": float(c.get("l", c.get("low", 0))),
                "c": float(c.get("c", c.get("close", 0))),
                "v": float(c.get("v", c.get("volume", 0))),
            })
        div = detect_cvd_divergence(ohlcv)
        if div["divergence"] != "none" and div["confidence"] > 40:
            arrow = "📉" if "bearish" in div["divergence"] else "📈"
            return f"CVDdiv:{arrow}{div['divergence']}({div['confidence']:.0f}%)"
        return ""
    except Exception:
        return ""


def _enrich_vwap(candles_15m: list) -> str:
    """VWAP bands — mean reversion potential."""
    if len(candles_15m) < 20:
        return ""
    try:
        from vwap_signal import compute_vwap_bands
        vwap = compute_vwap_bands(candles_15m)
        if abs(vwap.deviation_sigma) >= 1.0 and vwap.confidence > 30:
            sig_char = {"vwap_mean_revert_buy": "MR_BUY", "vwap_mean_revert_sell": "MR_SELL",
                        "vwap_trend_buy": "T_BUY", "vwap_trend_sell": "T_SELL"}.get(vwap.signal.value, "?")
            return f"VWAP:{sig_char}@{vwap.deviation_sigma:+.1f}σ({vwap.confidence:.0f}%)"
        return ""
    except Exception:
        return ""


def _enrich_cascade(candles_15m: list, symbol: str) -> str:
    """Liquidation cascade — bounce probability after mass liq."""
    if len(candles_15m) < 12:
        return ""
    try:
        from liquidation_cascade import detect_cascade_from_candles
        cascade = detect_cascade_from_candles(candles_15m, symbol)
        if cascade.cascade_detected and cascade.bounce_probability > 0.3:
            return f"LiqCascade:{cascade.cascade_side}(bounce={cascade.bounce_probability:.0%})"
        return ""
    except Exception:
        return ""


def _enrich_liquidation_activity(coin: str) -> str:
    """Real liquidation events from userFills WebSocket (not estimates).

    Reads actual liquidation fills with the liquidation flag from the
    hyperliquid_ws userFills stream. Complements _enrich_liquidations
    (estimated zones from order book) and _enrich_cascade (OI-proxy).
    """
    try:
        from liquidation_monitor import get_liquidation_context, get_self_liquidation_summary
        ctx = get_liquidation_context(coin)
        if not ctx or ctx.startswith("Liq:none"):
            return ""
        # Also check for self-liquidation alert
        self_liq = get_self_liquidation_summary()
        if self_liq:
            ctx += " | " + self_liq
        return ctx
    except Exception:
        return ""


# ── Main API ──

def _enrich_whale_activity(whale_tracker=None) -> str:
    if whale_tracker is None:
        return ""
    try:
        signals = whale_tracker.get_recent_signals(max_age=300)
        ctx = whale_tracker.get_whale_context()
        if not signals:
            return ""
        # Compact: list top 2 whale signals with confidence + count
        top = sorted(signals, key=lambda s: s.confidence, reverse=True)[:2]
        parts = []
        for s in top:
            parts.append(f"{'🐳' if s.side == 'BUY' else '🐻'}{s.symbol}:{s.confidence:.0f}%({s.whale_count}w)")
        return "Whales: " + " ".join(parts)
    except Exception:
        return ""


def _enrich_delta_neutral(arb_state=None) -> str:
    """Delta-neutral funding arb — active positions and returns."""
    if arb_state is None:
        return ""
    try:
        from hyperliquid_delta_neutral import get_arb_context
        ctx = get_arb_context(arb_state)
        # Only include if there are active positions
        if "No active" in ctx:
            return ""
        # Compact: summarize active arb positions
        active_count = len([v for v in arb_state.positions.values() if getattr(v, 'active', False)])
        total = arb_state.total_funding_collected
        if active_count == 0:
            return ""
        return f"DN-Arb:{active_count}pos collected${total:+.1f}"
    except Exception:
        return ""


# ── Main API ──

def enrich_context(
    coin: str,
    candles_15m: list | None = None,
    whale_tracker=None,
    arb_state=None,
) -> str:
    """Build enriched context string for AI prompt.

    Returns a compact string (under TOKEN_BUDGET tokens) with signals from
    all available intelligence modules. Gracefully degrades if modules fail.
    """
    candles = candles_15m or []
    lines = []

    # Volume delta (CVD) — who's in control?
    vd = _safe_get(lambda: _enrich_volume_delta(candles))
    if vd:
        lines.append(vd)

    # Market structure — where are S/R levels?
    ms = _safe_get(lambda: _enrich_market_structure(coin, candles))
    if ms:
        lines.append(ms)

    # Liquidations — where are stops clustered?
    liq = _safe_get(lambda: _enrich_liquidations(coin))
    if liq:
        lines.append(liq)

    # Chart patterns — confirm TA signals
    pat = _safe_get(lambda: _enrich_chart_patterns(candles))
    if pat:
        lines.append(pat)

    # Pump detection — avoid FOMO
    pump = _safe_get(lambda: _enrich_pump_alert(coin))
    if pump:
        lines.append(pump)

    # Economic calendar — news events
    econ = _safe_get(lambda: _enrich_economic_calendar())
    if econ:
        lines.append(econ)

    # BTC correlation — divergence risk
    corr = _safe_get(lambda: _enrich_correlation(coin))
    if corr:
        lines.append(corr)

    # CVD divergence — leading reversal signal (NEW)
    cvd_div = _safe_get(lambda: _enrich_cvd_divergence(candles))
    if cvd_div:
        lines.append(cvd_div)

    # VWAP bands — mean reversion / trend (NEW)
    vwap = _safe_get(lambda: _enrich_vwap(candles))
    if vwap:
        lines.append(vwap)

    # Liquidation cascade — bounce probability (NEW)
    cascade = _safe_get(lambda: _enrich_cascade(candles, coin))
    if cascade:
        lines.append(cascade)

    # Real liquidation events from WS — actual fills (NEW)
    liq_activity = _safe_get(lambda: _enrich_liquidation_activity(coin))
    if liq_activity:
        lines.append(liq_activity)

    # Whale activity — recent large-wallet trades (NEW)
    whale = _safe_get(lambda: _enrich_whale_activity(whale_tracker))
    if whale:
        lines.append(whale)

    # Delta-neutral arb status (NEW)
    arb = _safe_get(lambda: _enrich_delta_neutral(arb_state))
    if arb:
        lines.append(arb)

    result = " | ".join(lines)
    # Truncate to token budget
    if len(result) > TOKEN_BUDGET * 4:
        result = result[:TOKEN_BUDGET * 4]
    return result
