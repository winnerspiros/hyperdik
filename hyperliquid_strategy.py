#!/usr/bin/env python3
"""
Hyperliquid Strategy Engine v3 — 5-pillar weighted composite signal
with ensemble architecture and 5-regime detection.

PILLARS:
  1. Trend (0.30) — EMA crossover + ADX direction + Supertrend
  2. Momentum (0.25) — RSI + MACD histogram + volume confirmation
  3. Mean-Reversion (0.15) — Bollinger Band position + StochRSI
  4. Volume (0.15) — OBV trend + price-volume correlation
  5. Volatility (0.15) — ATR expansion/contraction + BB squeeze

REGIMES:
  TRENDING_UP, TRENDING_DOWN, SIDEWAYS, HIGH_VOL, CRISIS

ENSEMBLE: Sub-strategies weighted per regime, confluence scoring,
         counter-trend guards, BTC beta filter.

Built from research across:
  - master-confluence (Enki1444) — ensemble architecture, regime detection
  - moss-trade-bot-skills — 5-pillar weighted composite
  - passivbot (enarjord) — dynamic distance scaling, exposure gating
  - ref-perp-bot (CShear) — whale/funding/liquidation strategies
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

# ============================================================
# Enums & Types
# ============================================================

class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


class StrategyFamily(str, Enum):
    TREND_FOLLOW = "trend_follow"
    MEAN_REVERT = "mean_revert"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    DEFENSIVE = "defensive"


SignalSide = Literal["BUY", "SELL", "EXIT", "HOLD"]


@dataclass(frozen=True, slots=True)
class SubSignal:
    """Signal from a single sub-strategy."""
    family: StrategyFamily
    side: SignalSide
    confidence: float
    reason: str
    stop_distance_atr: float = 1.8
    tp_distance_atr: float = 0.6


@dataclass(frozen=True, slots=True)
class PillarWeights:
    """5-pillar composite weights."""
    trend: float = 0.30
    momentum: float = 0.25
    mean_reversion: float = 0.15
    volume: float = 0.15
    volatility: float = 0.15

    def __post_init__(self):
        total = self.trend + self.momentum + self.mean_reversion + self.volume + self.volatility
        if abs(total - 1.0) > 0.001:
            scale = 1.0 / total
            object.__setattr__(self, 'trend', self.trend * scale)
            object.__setattr__(self, 'momentum', self.momentum * scale)
            object.__setattr__(self, 'mean_reversion', self.mean_reversion * scale)
            object.__setattr__(self, 'volume', self.volume * scale)
            object.__setattr__(self, 'volatility', self.volatility * scale)


@dataclass(frozen=True, slots=True)
class MasterSignal:
    """Final aggregated signal from the strategy engine."""
    symbol: str
    side: SignalSide
    reason: str
    entry_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    atr: float | None = None
    regime: MarketRegime = MarketRegime.SIDEWAYS
    rsi: float | None = None
    confidence: float = 0.0
    strategy_family: str = "ensemble"
    sub_signals: int = 0
    kelly_fraction: float = 0.0
    composite_score: float = 0.0  # [-1, +1] from weighted composite


# ============================================================
# Regime weights: how much each sub-strategy contributes per regime
# ============================================================

REGIME_WEIGHTS: dict[MarketRegime, dict[StrategyFamily, float]] = {
    MarketRegime.TRENDING_UP: {
        StrategyFamily.TREND_FOLLOW: 1.0,
        StrategyFamily.MOMENTUM: 0.0,
        StrategyFamily.BREAKOUT: 0.7,
        StrategyFamily.MEAN_REVERT: 0.2,
        StrategyFamily.DEFENSIVE: 1.5,
    },
    MarketRegime.TRENDING_DOWN: {
        StrategyFamily.TREND_FOLLOW: 1.0,
        StrategyFamily.MOMENTUM: 0.0,
        StrategyFamily.BREAKOUT: 0.7,
        StrategyFamily.MEAN_REVERT: 0.2,
        StrategyFamily.DEFENSIVE: 1.5,
    },
    MarketRegime.SIDEWAYS: {
        StrategyFamily.TREND_FOLLOW: 0.1,
        StrategyFamily.MOMENTUM: 0.0,
        StrategyFamily.BREAKOUT: 0.3,
        StrategyFamily.MEAN_REVERT: 2.0,
        StrategyFamily.DEFENSIVE: 1.5,
    },
    MarketRegime.HIGH_VOL: {
        StrategyFamily.TREND_FOLLOW: 0.8,
        StrategyFamily.MOMENTUM: 0.0,
        StrategyFamily.BREAKOUT: 1.0,
        StrategyFamily.MEAN_REVERT: 0.3,
        StrategyFamily.DEFENSIVE: 2.0,
    },
    MarketRegime.CRISIS: {
        StrategyFamily.TREND_FOLLOW: 0.1,
        StrategyFamily.MOMENTUM: 0.0,
        StrategyFamily.BREAKOUT: 0.2,
        StrategyFamily.MEAN_REVERT: 0.1,
        StrategyFamily.DEFENSIVE: 3.0,
    },
}

# Correlation groups for BTC beta filter and portfolio constraints
CRYPTO_CORR_GROUPS: dict[str, list[str]] = {
    "btc_ecosystem": ["BTC"],
    "eth_ecosystem": ["ETH", "LINK", "AAVE", "UNI"],
    "sol_ecosystem": ["SOL"],
    "defi_midcap": ["AVAX", "LINK", "AAVE", "NEAR"],
    "meme_narrative": ["HYPE", "DOGE", "WIF", "PEPE"],
    "ai_narrative": ["RNDR", "TAO"],
    "exchange_tokens": ["BNB"],
}

# ============================================================
# 1. INDICATORS
# ============================================================

def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """Convert Hyperliquid candle snapshot to DataFrame."""
    if not candles:
        return pd.DataFrame()
    frame = pd.DataFrame(candles).copy()
    rename = {"t": "open_time", "T": "close_time", "o": "open", "h": "high",
              "l": "low", "c": "close", "v": "volume"}
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "open_time" in frame.columns:
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.dropna(subset=[c for c in ["open", "high", "low", "close"] if c in frame.columns])
    frame = frame.sort_values("open_time" if "open_time" in frame.columns else frame.columns[0])
    return frame.reset_index(drop=True)


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMAs, RSI, ATR, Bollinger, MACD, OBV, StochRSI."""
    if len(df) < 20:
        return df
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    vol = out["volume"].astype(float)

    # EMAs
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    out["rsi14"] = 100 - (100 / (1 + rs))
    out["rsi14"] = out["rsi14"].fillna(50)

    # RSI2 (Connors-style fast RSI)
    delta2 = close.diff()
    gain2 = delta2.clip(lower=0)
    loss2 = (-delta2).clip(lower=0)
    avg_gain2 = gain2.ewm(span=2, adjust=False).mean()
    avg_loss2 = loss2.ewm(span=2, adjust=False).mean()
    rs2 = avg_gain2 / avg_loss2.replace(0, float('nan'))
    out["rsi2"] = 100 - (100 / (1 + rs2))
    out["rsi2"] = out["rsi2"].fillna(50)

    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(span=14, adjust=False).mean()

    # ATR percentile (volatility ranking)
    atr_pct_series = (out["atr14"] / close * 100)
    out["atr_pctl"] = atr_pct_series.rolling(50, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_high"] = bb_mid + 2 * bb_std
    out["bb_low"] = bb_mid - 2 * bb_std
    out["bb_mid"] = bb_mid
    out["bb_width"] = (out["bb_high"] - out["bb_low"]) / out["bb_mid"].replace(0, float('nan'))

    # MACD
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    out["macd"] = ema8 - ema17
    out["macd_signal"] = out["macd"].ewm(span=5, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_hist_prev"] = out["macd_hist"].shift(1)

    # ADX
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14_adx = true_range.ewm(span=14, adjust=False).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr14_adx.replace(0, float('nan')))
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr14_adx.replace(0, float('nan')))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan'))
    out["adx14"] = dx.ewm(span=14, adjust=False).mean()
    out["adx14"] = out["adx14"].fillna(15)
    out["plus_di"] = plus_di.fillna(0)
    out["minus_di"] = minus_di.fillna(0)

    # OBV
    obv = (vol * ((close > close.shift()).astype(int) * 2 - 1)).cumsum()
    out["obv"] = obv
    out["obv_ema20"] = obv.ewm(span=20, adjust=False).mean()

    # Volatility z-score
    vol_ma = vol.rolling(48).mean()
    vol_std = vol.rolling(48).std()
    out["vol_z"] = (vol - vol_ma) / vol_std.replace(0, float('nan'))
    out["vol_z"] = out["vol_z"].fillna(0)

    # EMA slope
    out["ema20_slope"] = out["ema20"].pct_change(3).fillna(0)

    # StochRSI (approximation)
    rsi_14 = out["rsi14"]
    rsi_low = rsi_14.rolling(14).min()
    rsi_high = rsi_14.rolling(14).max()
    out["stochrsi"] = (rsi_14 - rsi_low) / (rsi_high - rsi_low).replace(0, float('nan'))
    out["stochrsi"] = out["stochrsi"].fillna(0.5).clip(0, 1)

    # Body ratio
    out["body_ratio"] = ((close - out["open"]).abs() / (high - low).replace(0, float('nan'))).fillna(0.5)

    # ATR expansion
    out["atr_expansion"] = out["atr14"] / out["atr14"].rolling(20).mean().replace(0, float('nan'))
    out["atr_expansion"] = out["atr_expansion"].fillna(1.0)

    return out


# ============================================================
# 2. REGIME DETECTION
# ============================================================

_high_vol_streaks: dict[str, int] = {}


def detect_regime(df: pd.DataFrame, symbol: str = "") -> tuple[MarketRegime, dict]:
    """Classify market into 5 regimes using ADX + EMA alignment + ATR percentile + vol spikes."""
    global _high_vol_streaks
    if len(df) < 50:
        return MarketRegime.SIDEWAYS, {"reason": "insufficient_data"}

    last = df.iloc[-1]
    close = float(last["close"])
    adx = _safe_float(last, "adx14", 0)
    rsi = _safe_float(last, "rsi14", 50)
    vol_z = _safe_float(last, "vol_z", 0)
    ema20 = _safe_float(last, "ema20", close)
    ema50 = _safe_float(last, "ema50", close)
    ema200 = _safe_float(last, "ema200", close)

    # ATR percentile
    atr_pctl = _safe_float(last, "atr_pctl", 0.5)
    bb_pctl = _safe_float(last, "bb_width", 0)

    # BB width percentile
    bb_series = df["bb_width"].dropna()
    if len(bb_series) > 20:
        bb_pctl = float(bb_series.rank(pct=True).iloc[-1])

    # EMA alignment
    ema_aligned_bull = close > ema20 > ema50 > ema200 if ema200 > 0 else False
    ema_aligned_bear = close < ema20 < ema50 < ema200 if ema200 > 0 else False
    ema_mixed = not ema_aligned_bull and not ema_aligned_bear

    vol_extreme = abs(vol_z) > 2.5

    diagnostics = {"adx": adx, "atr_pctl": atr_pctl, "bb_pctl": bb_pctl,
                   "vol_z": vol_z, "rsi": rsi}

    # Crisis: extreme vol + extreme volume spike
    if atr_pctl > 0.95 and vol_extreme:
        return MarketRegime.CRISIS, diagnostics

    # High vol: extreme ATR or BB + strong ADX, requires persistence
    if atr_pctl > 0.92 or (bb_pctl > 0.95 and adx > 35):
        _high_vol_streaks[symbol] = _high_vol_streaks.get(symbol, 0) + 1
        if _high_vol_streaks[symbol] >= 1:
            return MarketRegime.HIGH_VOL, diagnostics
    else:
        _high_vol_streaks[symbol] = 0

    # Trending
    if adx > 20:
        if ema_aligned_bull and rsi > 48:
            return MarketRegime.TRENDING_UP, {**diagnostics, "trend": "bull"}
        if ema_aligned_bear and rsi > 32:
            return MarketRegime.TRENDING_DOWN, {**diagnostics, "trend": "bear"}
        # Simpler trend detection: don't require perfect EMA stacking
        if close > ema20 and close > ema50 and rsi > 48:
            return MarketRegime.TRENDING_UP, {**diagnostics, "trend": "bull_simple"}
        if close < ema20 and close < ema50 and rsi < 55:
            return MarketRegime.TRENDING_DOWN, {**diagnostics, "trend": "bear_simple"}

    # Sideways
    if adx < 20 and ema_mixed:
        return MarketRegime.SIDEWAYS, diagnostics

    # Default
    if ema_aligned_bull and rsi > 50:
        return MarketRegime.TRENDING_UP, {**diagnostics, "trend": "bull_weak"}
    if ema_aligned_bear and rsi > 38:
        return MarketRegime.TRENDING_DOWN, {**diagnostics, "trend": "bear_weak"}

    return MarketRegime.SIDEWAYS, diagnostics


# ============================================================
# 3. 5-PILLAR COMPOSITE SIGNAL
# ============================================================

def _safe_float(series: pd.Series, key: str, default: float = 0.0) -> float:
    """Get float from Series, returning default for missing/NaN."""
    val = series.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _pillar_trend(last: pd.Series, prev: pd.Series) -> float:
    """Pillar 1: Trend (EMA crossover + ADX direction).
    Returns normalized score [-1, +1]."""
    close = float(last["close"])
    ema20 = _safe_float(last, "ema20", close)
    ema50 = _safe_float(last, "ema50", close)
    adx = _safe_float(last, "adx14", 15)
    plus_di = _safe_float(last, "plus_di", 0)
    minus_di = _safe_float(last, "minus_di", 0)

    prev_ema20 = _safe_float(prev, "ema20", close)
    prev_ema50 = _safe_float(prev, "ema50", close)

    # EMA crossover signal
    bullish_cross = prev_ema20 <= prev_ema50 and ema20 > ema50
    bearish_cross = prev_ema20 >= prev_ema50 and ema20 < ema50

    ema_sig = 0.0
    if bullish_cross:
        ema_sig = 0.6
    elif bearish_cross:
        ema_sig = -0.6
    elif close > ema20 > ema50:
        ema_sig = 0.3
    elif close < ema20 < ema50:
        ema_sig = -0.3

    # ADX as confidence amplifier (not attenuator)
    # ADX 25+ = full weight, ADX 15-25 = 50% weight, ADX < 15 = 25% weight
    adx_factor = min(1.0, max(0.25, adx / 25.0))

    # DI direction bonus
    di_sig = 0.0
    if plus_di > minus_di * 1.2:
        di_sig = 0.3
    elif minus_di > plus_di * 1.2:
        di_sig = -0.3

    raw = (ema_sig * 0.7 + di_sig * 0.3) * adx_factor
    return max(-1.0, min(1.0, raw))


def _pillar_momentum(last: pd.Series, prev: pd.Series) -> float:
    """Pillar 2: Momentum (RSI + MACD histogram).
    Returns normalized score [-1, +1]."""
    rsi = _safe_float(last, "rsi14", 50)
    macd_hist = _safe_float(last, "macd_hist", 0)
    macd_hist_prev = _safe_float(last, "macd_hist_prev", 0)
    close = float(last["close"])

    # RSI: centered at 50, normalized to [-1, +1]
    rsi_sig = (rsi - 50) / 50.0
    rsi_sig = max(-1.0, min(1.0, rsi_sig))

    # MACD: histogram vs price
    macd_sig = macd_hist / (close * 0.002) if close > 0 else 0
    macd_sig = max(-1.0, min(1.0, macd_sig))

    # MACD divergence bonus
    if macd_hist > 0 and macd_hist > macd_hist_prev:
        macd_sig = max(macd_sig, macd_sig * 1.2)
    elif macd_hist < 0 and macd_hist < macd_hist_prev:
        macd_sig = min(macd_sig, macd_sig * 1.2)

    return rsi_sig * 0.5 + macd_sig * 0.5


def _pillar_mean_revert(last: pd.Series) -> float:
    """Pillar 3: Mean-Reversion (Bollinger position + StochRSI).
    Returns normalized score [-1, +1] (positive = oversold → bounce up)."""
    close = float(last["close"])
    bb_high = _safe_float(last, "bb_high", 0)
    bb_low = _safe_float(last, "bb_low", 0)
    stochrsi = _safe_float(last, "stochrsi", 0.5)

    if bb_high <= bb_low:
        return 0.0

    # Position in BB band: 0 = at top, 1 = at bottom
    bb_position = (close - bb_low) / (bb_high - bb_low)

    # Signal: -position (revert toward midline)
    # BB position 0.2 (near bottom) → +0.6 (buy signal)
    # BB position 0.5 (midline) → 0.0 (no signal)
    # BB position 0.8 (near top) → -0.6 (sell signal)
    mr_sig = (0.5 - bb_position) * 0.8 * 2  # scaled

    # StochRSI amplification
    if stochrsi < 0.2 and mr_sig > 0:
        mr_sig *= 1.3
    elif stochrsi > 0.8 and mr_sig < 0:
        mr_sig *= 1.3

    return max(-1.0, min(1.0, mr_sig))


def _pillar_volume(last: pd.Series) -> float:
    """Pillar 4: Volume (OBV trend + price-volume correlation).
    Returns normalized score [-1, +1]."""
    obv = _safe_float(last, "obv", 0)
    obv_ema = _safe_float(last, "obv_ema20", 0)
    vol_z = _safe_float(last, "vol_z", 0)
    close = float(last["close"])
    ema20 = _safe_float(last, "ema20", close)

    # OBV trend
    if obv_ema > 0 and obv > obv_ema:
        obv_sig = 0.5 + min(0.5, (obv - obv_ema) / obv_ema * 10)
    elif obv_ema > 0 and obv < obv_ema:
        obv_sig = -0.5 - min(0.5, (obv_ema - obv) / obv_ema * 10)
    else:
        obv_sig = 0.0

    # Volume confirmation
    vol_sig = 0.0
    if abs(vol_z) > 1.5:
        vol_sig = 0.3 * (1 if close > ema20 else -1) if ema20 > 0 else 0

    return max(-1.0, min(1.0, obv_sig * 0.7 + vol_sig * 0.3))


def _pillar_volatility(last: pd.Series) -> float:
    """Pillar 5: Volatility (ATR expansion + BB squeeze).
    Returns normalized score [-1, +1] (positive = expanding, good for trend).
    Actually returns: expansion signal — positive = vol expanding (trend-friendly),
    negative = vol contracting (mean-reversion-friendly)."""
    atr_expansion = _safe_float(last, "atr_expansion", 1.0)
    bb_width = _safe_float(last, "bb_width", 0)

    # ATR expansion: >1 = expanding, <1 = contracting
    atr_sig = (atr_expansion - 1.0) * 2.0
    atr_sig = max(-1.0, min(1.0, atr_sig))

    # BB squeeze: tight bands → -1, wide bands → +1
    # Typical bb_width of 0.02-0.15 for crypto
    bb_sig = (bb_width - 0.05) / 0.1
    bb_sig = max(-1.0, min(1.0, bb_sig))

    return atr_sig * 0.5 + bb_sig * 0.5


def compute_composite(df: pd.DataFrame, weights: PillarWeights | None = None,
                      long_bias: float = 0.5) -> float:
    """Compute the 5-pillar weighted composite signal [-1, +1].
    > 0 = bullish, < 0 = bearish, 0 = neutral."""
    if len(df) < 50:
        return 0.0

    weights = weights or PillarWeights()
    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_sig = _pillar_trend(last, prev)
    momentum_sig = _pillar_momentum(last, prev)
    mr_sig = _pillar_mean_revert(last)
    vol_sig = _pillar_volume(last)
    vola_sig = _pillar_volatility(last)

    composite = (
        weights.trend * trend_sig +
        weights.momentum * momentum_sig +
        weights.mean_reversion * mr_sig +
        weights.volume * vol_sig +
        weights.volatility * vola_sig
    )

    # Direction bias: attenuate opposite-direction signals
    if composite > 0 and long_bias < 0.3:
        composite *= (1.0 - (0.3 - long_bias))  # dampen longs when bearish bias
    elif composite < 0 and long_bias > 0.7:
        composite *= (1.0 - (long_bias - 0.7))  # dampen shorts when bullish bias

    return max(-1.0, min(1.0, composite))


# ============================================================
# 4. SUB-STRATEGIES (Ensemble)
# ============================================================

def _trend_follow_signal(df: pd.DataFrame, last: pd.Series, prev: pd.Series) -> SubSignal | None:
    """Trend following: EMA crossover + momentum + RSI filter.
    Research: 68.7% WR with 0.6x ATR TP."""
    close = float(last["close"])
    ema20 = _safe_float(last, "ema20", close)
    ema50 = _safe_float(last, "ema50", close)
    rsi = _safe_float(last, "rsi14", 50)
    macd_hist = _safe_float(last, "macd_hist", 0)
    macd_hist_prev = _safe_float(last, "macd_hist_prev", 0)

    prev_ema20 = _safe_float(prev, "ema20", close)
    prev_ema50 = _safe_float(prev, "ema50", close)

    bullish_cross = prev_ema20 <= prev_ema50 and ema20 > ema50
    bearish_cross = prev_ema20 >= prev_ema50 and ema20 < ema50
    bull_momentum = close > ema20 and close > ema50 and 45 <= rsi <= 68
    bear_momentum = close < ema20 and close < ema50 and 28 <= rsi <= 55
    macd_bull = macd_hist > 0 or (macd_hist > macd_hist_prev and macd_hist_prev < 0)
    macd_bear = macd_hist < 0 or (macd_hist < macd_hist_prev and macd_hist_prev > 0)

    if (bullish_cross or bull_momentum) and macd_bull and rsi < 72:
        conf = 0.4 if bullish_cross else 0.25
        conf += 0.15 if 48 <= rsi <= 62 else 0
        conf += 0.10 if macd_bull else 0
        return SubSignal(StrategyFamily.TREND_FOLLOW, "BUY", conf,
                         f"trend_bull_ema_rsi{rsi:.0f}", tp_distance_atr=1.5, stop_distance_atr=1.5)

    if (bearish_cross or bear_momentum) and macd_bear and rsi > 38:
        conf = 0.4 if bearish_cross else 0.25
        conf += 0.15 if 38 <= rsi <= 52 else 0
        conf += 0.10 if macd_bear else 0
        return SubSignal(StrategyFamily.TREND_FOLLOW, "SELL", conf,
                         f"trend_bear_ema_rsi{rsi:.0f}", tp_distance_atr=1.5, stop_distance_atr=1.5)

    return None


def _mean_revert_signal(df: pd.DataFrame, last: pd.Series) -> SubSignal | None:
    """Mean reversion: Bollinger Band bounce + RSI oversold/overbought.
    Research: 76.9% WR in sideways, PF 1.27."""
    close = float(last["close"])
    bb_high = _safe_float(last, "bb_high", 0)
    bb_low = _safe_float(last, "bb_low", 0)
    rsi = _safe_float(last, "rsi14", 50)
    stochrsi = _safe_float(last, "stochrsi", 0.5)
    adx = _safe_float(last, "adx14", 25)

    if bb_high <= bb_low:
        return None

    bb_pct = (close - bb_low) / (bb_high - bb_low)

    # Mean reversion: enhanced — works WITH trend when ADX is high
    # When ADX > 30: trend is strong, sell rallies in downtrend, buy dips in uptrend
    # When ADX < 30: sideways mean reversion as before
    if adx > 30:
        # Trend mean reversion: fade extremes IN trend direction
        ema20 = _safe_float(last, "ema20", close)
        ema50 = _safe_float(last, "ema50", close)
        is_uptrend = close > ema20 > ema50
        is_downtrend = close < ema20 < ema50
        
        if is_downtrend and bb_pct > 0.50:
            # Sell bounces in a downtrend
            conf = 0.20
            conf += 0.10 if adx > 35 else 0
            conf += 0.10 if bb_pct > 0.70 else 0
            return SubSignal(StrategyFamily.MEAN_REVERT, "SELL", conf,
                             f"mr_trend_sell_adx{adx:.0f}_bb{bb_pct:.0%}", stop_distance_atr=1.0, tp_distance_atr=1.0)
        
        if is_uptrend and bb_pct < 0.50:
            # Buy dips in an uptrend
            conf = 0.20
            conf += 0.10 if adx > 35 else 0
            conf += 0.10 if bb_pct < 0.30 else 0
            return SubSignal(StrategyFamily.MEAN_REVERT, "BUY", conf,
                             f"mr_trend_buy_adx{adx:.0f}_bb{bb_pct:.0%}", stop_distance_atr=1.0, tp_distance_atr=1.0)
        
        return None

    # Buy: near lower BB + oversold + StochRSI oversold
    if bb_pct < 0.35 and rsi < 45 and stochrsi < 0.40:
        conf = 0.30
        conf += 0.15 if rsi < 35 else 0
        conf += 0.10 if stochrsi < 0.20 else 0
        conf += 0.10 if bb_pct < 0.15 else 0
        return SubSignal(StrategyFamily.MEAN_REVERT, "BUY", conf,
                         f"mr_bb_low_rsi{rsi:.0f}", stop_distance_atr=1.5, tp_distance_atr=1.5)

    # Sell: near upper BB + overbought
    if bb_pct > 0.65 and rsi > 55 and stochrsi > 0.60:
        conf = 0.30
        conf += 0.15 if rsi > 65 else 0
        conf += 0.10 if stochrsi > 0.80 else 0
        conf += 0.10 if bb_pct > 0.85 else 0
        return SubSignal(StrategyFamily.MEAN_REVERT, "SELL", conf,
                         f"mr_bb_high_rsi{rsi:.0f}", stop_distance_atr=1.5, tp_distance_atr=1.5)

    return None


def _breakout_signal(df: pd.DataFrame, last: pd.Series) -> SubSignal | None:
    """Breakout: post-Bollinger squeeze with volume confirmation."""
    close = float(last["close"])
    vol_z = _safe_float(last, "vol_z", 0)
    bb_width = _safe_float(last, "bb_width", 0)
    atr_expansion = _safe_float(last, "atr_expansion", 1.0)
    ema20_slope = _safe_float(last, "ema20_slope", 0)

    # BB squeeze breakout: bands were tight, now expanding
    bb_prev = df["bb_width"].iloc[-6:-1].mean() if len(df) > 5 else 0
    squeeze_breakout = bb_width > bb_prev * 1.2 and bb_prev < 0.04

    if squeeze_breakout and abs(vol_z) > 1.0 and atr_expansion > 1.1:
        if ema20_slope > 0.001:
            return SubSignal(StrategyFamily.BREAKOUT, "BUY", 0.25,
                             "breakout_bull_squeeze", tp_distance_atr=2.0, stop_distance_atr=1.5)
        elif ema20_slope < -0.001:
            return SubSignal(StrategyFamily.BREAKOUT, "SELL", 0.25,
                             "breakout_bear_squeeze", tp_distance_atr=2.0, stop_distance_atr=1.5)

    return None


def _defensive_signal(df: pd.DataFrame, last: pd.Series) -> SubSignal | None:
    """Defensive: exit signals on extreme exhaustion or vol spikes."""
    rsi = _safe_float(last, "rsi14", 50)
    rsi2 = _safe_float(last, "rsi2", 50)
    atr_expansion = _safe_float(last, "atr_expansion", 1.0)

    if rsi2 > 95 and rsi > 72:
        return SubSignal(StrategyFamily.DEFENSIVE, "SELL", 0.50,
                         f"exhaustion_rsi2_{rsi2:.0f}", stop_distance_atr=1.0, tp_distance_atr=0.3)
    if rsi2 < 5 and rsi < 28:
        return SubSignal(StrategyFamily.DEFENSIVE, "BUY", 0.50,
                         f"panic_rsi2_{rsi2:.0f}", stop_distance_atr=1.0, tp_distance_atr=0.3)
    if atr_expansion > 2.0:
        return SubSignal(StrategyFamily.DEFENSIVE, "EXIT", 0.40,
                         "vol_spike", stop_distance_atr=1.0, tp_distance_atr=0.3)

    return None


# ============================================================
# 5. ENSEMBLE SIGNAL AGGREGATOR
# ============================================================

def aggregate_signals(sub_signals: list[SubSignal], regime: MarketRegime,
                      min_confidence: float = 0.25) -> tuple[SignalSide, float, str, int]:
    """Aggregate sub-strategy signals into ensemble decision.
    Returns (side, confidence, reason, num_fired)."""
    if not sub_signals:
        return "HOLD", 0.0, "no_sub_signals", 0

    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[MarketRegime.SIDEWAYS])
    buy_score = 0.0
    sell_score = 0.0
    exit_score = 0.0
    reasons = []
    fired = 0

    for sig in sub_signals:
        w = weights.get(sig.family, 0.5)
        weighted_conf = sig.confidence * w
        if sig.side == "BUY":
            buy_score += weighted_conf
        elif sig.side == "SELL":
            sell_score += weighted_conf
        elif sig.side == "EXIT":
            exit_score += weighted_conf
        if sig.side in ("BUY", "SELL", "EXIT"):
            fired += 1
            reasons.append(f"{sig.family.value}:{sig.confidence:.2f}")

    # Defensive EXIT always wins
    if exit_score > 0.3:
        return "EXIT", min(exit_score, 1.0), f"ensemble_exit:{','.join(reasons)}", fired

    best_score = max(buy_score, sell_score)
    if best_score < min_confidence:
        return "HOLD", best_score, f"low_conf:{best_score:.2f}", fired

    if buy_score > sell_score and buy_score >= min_confidence:
        return "BUY", min(buy_score, 1.0), f"ensemble_buy:{','.join(reasons)}", fired
    if sell_score > buy_score and sell_score >= min_confidence:
        return "SELL", min(sell_score, 1.0), f"ensemble_sell:{','.join(reasons)}", fired

    return "HOLD", best_score, f"neutral:b={buy_score:.2f}_s={sell_score:.2f}", fired


# ============================================================
# 6. KELLY CRITERION
# ============================================================

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   fraction: float = 0.25) -> float:
    """Fractional Kelly criterion for position sizing."""
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.01
    b = avg_win / avg_loss
    p = win_rate
    kelly = (b * p - (1 - p)) / b
    kelly = max(0.0, min(kelly, 0.25))
    return max(0.005, kelly * fraction)


def estimate_kelly_from_history(trades: list[dict] | None) -> float:
    """Estimate Kelly fraction from trade history."""
    if not trades or len(trades) < 20:
        return kelly_fraction(0.55, 1.2, 1.0, 0.25)
    wins = [t["ret"] for t in trades if t.get("ret", 0) > 0]
    losses = [abs(t["ret"]) for t in trades if t.get("ret", 0) < 0]
    if not wins or not losses:
        return kelly_fraction(0.55, 1.2, 1.0, 0.25)
    wr = len(wins) / len(trades)
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    return kelly_fraction(wr, avg_win, avg_loss, 0.25)


# ============================================================
# 7. BTC BETA FILTER + CORRELATION GROUPS
# ============================================================

def get_correlation_group(symbol: str) -> str | None:
    """Get correlation group for a symbol."""
    for group, members in CRYPTO_CORR_GROUPS.items():
        if symbol.upper() in members:
            return group
    return None


def get_correlation_members(symbol: str) -> list[str]:
    """Get all members of a symbol's correlation group."""
    group = get_correlation_group(symbol)
    if not group:
        return [symbol.upper()]
    return CRYPTO_CORR_GROUPS.get(group, [symbol.upper()])


# ============================================================
# 8. ENTRY LEVEL CALCULATION (Support/Resistance)
# ============================================================

def calculate_entry_levels(candles: list[dict], coin: str,
                           current_price: float) -> dict:
    """Calculate support/resistance levels from multi-timeframe analysis."""
    df = candles_to_frame(candles)
    if len(df) < 20:
        return {"support": current_price * 0.97, "resistance": current_price * 1.03,
                "support_distance_pct": 3.0, "resistance_distance_pct": 3.0,
                "rsi": 50, "signals": []}

    df = add_basic_indicators(df)
    last = df.iloc[-1]
    rsi = _safe_float(last, "rsi14", 50)

    # Swing low/high from last 20 candles for S/R
    recent = df.tail(20)
    swing_low = float(recent["low"].min())
    swing_high = float(recent["high"].max())
    bb_low = _safe_float(last, "bb_low", swing_low)
    bb_high = _safe_float(last, "bb_high", swing_high)

    # Support: nearest of swing low, BB low
    supports = [s for s in [swing_low, bb_low] if s < current_price]
    support = max(supports) if supports else current_price * 0.97

    # Resistance: nearest of swing high, BB high
    resistances = [r for r in [swing_high, bb_high] if r > current_price]
    resistance = min(resistances) if resistances else current_price * 1.03

    support_dist = (current_price - support) / current_price * 100
    res_dist = (resistance - current_price) / current_price * 100

    # Signals
    sig_list = []
    if rsi < 35:
        sig_list.append("RSI_OVERSOLD")
    elif rsi > 65:
        sig_list.append("RSI_OVERBOUGHT")
    if support_dist < 2:
        sig_list.append("NEAR_SUPPORT")
    if res_dist < 2:
        sig_list.append("NEAR_RESISTANCE")

    # Check EMA alignment
    ema20 = _safe_float(last, "ema20", current_price)
    ema50 = _safe_float(last, "ema50", current_price)
    if current_price > ema20 > ema50:
        sig_list.append("BULL_ALIGNED")
    elif current_price < ema20 < ema50:
        sig_list.append("BEAR_ALIGNED")

    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "support_distance_pct": round(support_dist, 2),
        "resistance_distance_pct": round(res_dist, 2),
        "rsi": round(rsi, 1),
        "signals": sig_list,
    }


# ============================================================
# 9. MASTER SIGNAL GENERATOR
# ============================================================

def generate_master_signal(
    symbol: str,
    candles: list[dict],
    trades: list[dict] | None = None,
    btc_candles: list[dict] | None = None,
    open_positions: dict[str, dict] | None = None,
    equity: float = 10000.0,
    min_confidence: float = 0.25,
) -> MasterSignal:
    """Generate the master trading signal — the brain of the bot.

    Flow:
      1. Build indicators
      2. Detect market regime
      3. Check BTC beta (for alts)
      4. Run all 4 sub-strategies
      5. Aggregate with regime weights
      6. Apply Kelly sizing
      7. Calculate entry/stop/target levels
    """
    df = candles_to_frame(candles)
    if len(df) < 55:
        return MasterSignal(symbol=symbol, side="HOLD", reason="insufficient_candles")

    df = add_basic_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])
    atr = _safe_float(last, "atr14", 0)
    rsi = _safe_float(last, "rsi14", 50)

    if atr <= 0 or math.isnan(atr):
        return MasterSignal(symbol=symbol, side="HOLD", reason="indicator_not_ready")

    # === STEP 1: Regime Detection ===
    regime, diagnostics = detect_regime(df, symbol)

    # === STEP 2: BTC Beta Filter ===
    btc_is_bearish = False
    btc_is_bullish = False
    if btc_candles and symbol.upper() not in ("BTC",):
        btc_df = add_basic_indicators(candles_to_frame(btc_candles))
        if len(btc_df) >= 55:
            btc_regime, _ = detect_regime(btc_df)
            btc_is_bearish = btc_regime in (MarketRegime.TRENDING_DOWN, MarketRegime.CRISIS)
            btc_is_bullish = btc_regime == MarketRegime.TRENDING_UP

    # === STEP 3: 5-Pillar Composite ===
    composite = compute_composite(df)

    # === STEP 4: Run Sub-Strategies ===
    sub_signals: list[SubSignal] = []

    tf = _trend_follow_signal(df, last, prev)
    if tf:
        sub_signals.append(tf)

    mr = _mean_revert_signal(df, last)
    if mr:
        sub_signals.append(mr)

    brk = _breakout_signal(df, last)
    if brk:
        sub_signals.append(brk)

    dfn = _defensive_signal(df, last)
    if dfn:
        sub_signals.append(dfn)

    # === STEP 5: Aggregate ===
    if regime in (MarketRegime.SIDEWAYS,):
        min_conf = max(min_confidence, 0.30)
    elif regime == MarketRegime.HIGH_VOL:
        min_conf = max(min_confidence, 0.35)
    else:
        min_conf = min_confidence

    # ── Regime-following signal: trade the trend even when sub-strategies are silent ──
    # If market is clearly trending but no strong sub-signal exists, add a regime-based signal.
    # This prevents the system from sitting out obvious moves (e.g. SOL -3% but no signal).
    regime_has_signal = any(
        (s.side == "SELL" and regime == MarketRegime.TRENDING_DOWN) or
        (s.side == "BUY" and regime == MarketRegime.TRENDING_UP)
        for s in sub_signals
    )
    if not regime_has_signal and regime in (MarketRegime.TRENDING_DOWN, MarketRegime.TRENDING_UP):
        ema20 = _safe_float(last, "ema20", close)
        trend_strength = max(0.05, abs(composite))
        if regime == MarketRegime.TRENDING_DOWN and close < ema20:
            conf = 0.25 + trend_strength * 0.25  # 0.30-0.50
            sub_signals.append(SubSignal(StrategyFamily.TREND_FOLLOW, "SELL", conf,
                f"regime_trend_down_comp{composite:.2f}", tp_distance_atr=2.0, stop_distance_atr=1.5))
        elif regime == MarketRegime.TRENDING_UP and close > ema20:
            conf = 0.25 + trend_strength * 0.25
            sub_signals.append(SubSignal(StrategyFamily.TREND_FOLLOW, "BUY", conf,
                f"regime_trend_up_comp{composite:.2f}", tp_distance_atr=2.0, stop_distance_atr=1.5))

    side, confidence, reason, num_fired = aggregate_signals(sub_signals, regime, min_conf)

    # === STEP 6: Counter-trend guard ===
    if side == "SELL" and regime == MarketRegime.TRENDING_UP:
        ema20 = _safe_float(last, "ema20", close)
        ema50 = _safe_float(last, "ema50", close)
        if close > ema20 > ema50 and confidence < 0.95:
            return MasterSignal(symbol=symbol, side="HOLD",
                                reason=f"countertrend_blocked_short_in_uptrend",
                                atr=atr, regime=regime, rsi=rsi, composite_score=composite)

    if side == "BUY" and regime == MarketRegime.TRENDING_DOWN:
        ema20 = _safe_float(last, "ema20", close)
        ema50 = _safe_float(last, "ema50", close)
        if close < ema20 < ema50 and confidence < 0.95:
            return MasterSignal(symbol=symbol, side="HOLD",
                                reason=f"countertrend_blocked_long_in_downtrend",
                                atr=atr, regime=regime, rsi=rsi, composite_score=composite)

    # === STEP 7: BTC Beta Filter ===
    # Allow counter-BTC trades when coin's own regime disagrees OR coin shows independent strength
    if btc_is_bearish and side == "BUY" and regime not in (MarketRegime.TRENDING_UP,):
        return MasterSignal(symbol=symbol, side="HOLD",
                            reason=f"btc_bear_filter_blocked_alt_long",
                            atr=atr, regime=regime, rsi=rsi, composite_score=composite)

    if btc_is_bullish and side == "SELL" and regime not in (MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOL, MarketRegime.SIDEWAYS):
        # Only block shorts when coin is clearly trending UP against BTC
        # Sideways/high_vol/trending_down coins can diverge from BTC
        return MasterSignal(symbol=symbol, side="HOLD",
                            reason=f"btc_bull_filter_blocked_alt_short",
                            atr=atr, regime=regime, rsi=rsi, composite_score=composite)

    # === STEP 8: Kelly Sizing ===
    kelly = estimate_kelly_from_history(trades)

    # === STEP 9: Build Final Signal ===
    if side in ("BUY", "SELL"):
        direction_subs = [s for s in sub_signals if s.side == side]
        if direction_subs:
            regime_w = REGIME_WEIGHTS.get(regime, {})
            best_sub = max(direction_subs,
                           key=lambda s: s.confidence * regime_w.get(s.family, 0.5))
        else:
            best_sub = sub_signals[0]

        stop_dist = best_sub.stop_distance_atr * atr
        tp_dist = best_sub.tp_distance_atr * atr

        if side == "BUY":
            stop = close - stop_dist
            tp = close + tp_dist
        else:
            stop = close + stop_dist
            tp = close - tp_dist

        return MasterSignal(
            symbol=symbol, side=side,
            reason=f"master_{regime.value}:{reason}",
            entry_price=close, stop_price=stop, take_profit_price=tp,
            atr=atr, regime=regime, rsi=rsi,
            confidence=confidence, strategy_family=best_sub.family.value if direction_subs else "ensemble",
            sub_signals=num_fired, kelly_fraction=kelly,
            composite_score=composite,
        )

    if side == "EXIT":
        return MasterSignal(
            symbol=symbol, side="EXIT",
            reason=f"master_{regime.value}:{reason}",
            atr=atr, regime=regime, rsi=rsi,
            confidence=confidence, sub_signals=num_fired, composite_score=composite,
        )

    return MasterSignal(
        symbol=symbol, side="HOLD",
        reason=f"master_{regime.value}:{reason}",
        atr=atr, regime=regime, rsi=rsi,
        confidence=confidence, sub_signals=num_fired, composite_score=composite,
    )


# ============================================================
# 10. MULTI-MODULE SIGNAL ENRICHMENT
#    Layers additional gold signals onto the master signal:
#    - CVD divergence (bearish/bullish divergence detection)
#    - VWAP bands (mean reversion / trend confirmation)
#    - Liquidation cascade (bounce probability after mass liq)
#    - OI delta (trend confirmation / exhaustion)
#    - Order book imbalance (short-term direction)
#    - Volume profile POC (magnetic price levels)
# ============================================================

@dataclass(slots=True)
class EnrichedSignal:
    """Master signal with all enrichment modules appended."""
    base_signal: MasterSignal
    
    # CVD
    cvd_divergence: str = "none"
    cvd_confidence: float = 0.0
    cvd_description: str = ""
    
    # VWAP
    vwap_signal: str = "neutral"
    vwap_deviation_pct: float = 0.0
    vwap_deviation_sigma: float = 0.0
    vwap_price: float = 0.0
    vwap_context: str = ""
    
    # Liquidation cascade
    cascade_detected: bool = False
    cascade_side: str = ""
    cascade_bounce_prob: float = 0.0
    cascade_action: str = "HOLD"
    cascade_context: str = ""
    
    # OI delta
    oi_signal: str = "neutral"
    oi_delta_pct: float = 0.0
    oi_confidence: float = 0.0
    oi_context: str = ""
    
    # Composite enrichment
    enriched_confidence: float = 0.0
    enrichment_boost: float = 0.0
    enrichment_reason: str = ""
    
    # ── Delegate common attributes to base_signal ──
    @property
    def side(self) -> str:
        return self.base_signal.side
    
    @property
    def symbol(self) -> str:
        return self.base_signal.symbol
    
    @property
    def confidence(self) -> float:
        """Use enriched confidence if available, otherwise base."""
        if self.enriched_confidence > 0:
            return max(self.base_signal.confidence, self.enriched_confidence) if self.enrichment_boost > 0 else self.base_signal.confidence
        return self.base_signal.confidence
    
    @property
    def regime(self):
        return self.base_signal.regime
    
    @property
    def composite_score(self) -> float:
        return self.base_signal.composite_score
    
    @property
    def reason(self) -> str:
        return self.base_signal.reason
    
    @property
    def atr(self) -> float | None:
        return self.base_signal.atr
    
    @property
    def entry_price(self) -> float | None:
        """Use VWAP-adjusted entry if enrichment suggests it."""
        if self.vwap_price > 0 and abs(self.vwap_deviation_sigma) > 1.5:
            raw = self.base_signal.entry_price or self.vwap_price
            w = min(0.3, abs(self.vwap_deviation_sigma) / 10)
            return raw * (1 - w) + self.vwap_price * w
        return self.base_signal.entry_price
    
    @property
    def stop_price(self) -> float | None:
        return self.base_signal.stop_price
    
    @property
    def take_profit_price(self) -> float | None:
        return self.base_signal.take_profit_price
    
    @property
    def strategy_family(self) -> str:
        return self.base_signal.strategy_family
    
    @property
    def sub_signals(self) -> int:
        return self.base_signal.sub_signals
    
    @property
    def kelly_fraction(self) -> float:
        return self.base_signal.kelly_fraction
    
    @property
    def rsi(self) -> float:
        return self.base_signal.rsi
    
    @property
    def enriched_entry_price(self) -> float | None:
        """VWAP-adjusted entry price if applicable."""
        return self.entry_price  # Already handled in entry_price property


def enrich_master_signal(
    base: MasterSignal,
    candles: list[dict],
    mids: dict[str, float] | None = None,
    hl_client=None,
    active_positions: list | None = None,
) -> EnrichedSignal:
    """
    Enrich a master signal with all available modules.
    
    Args:
        base: The base master signal from generate_master_signal()
        candles: OHLCV candles for this coin (15m)
        mids: Current mid prices dict (for OI context)
        hl_client: Hyperliquid client (for OI data)
        active_positions: Current open positions
    """
    enriched = EnrichedSignal(base_signal=base)
    boosts = []  # Collect enrichment boosts
    
    coin = base.symbol
    price = float(mids.get(coin, 0)) if mids else (base.entry_price or 0)
    
    # ── 1. CVD Divergence ──
    try:
        from cvd_divergence import detect_cvd_divergence
        cvd = detect_cvd_divergence(candles)
        enriched.cvd_divergence = cvd["divergence"]
        enriched.cvd_confidence = cvd["confidence"]
        enriched.cvd_description = cvd["description"]
        
        if cvd["divergence"] != "none" and cvd["confidence"] > 50:
            div_type = cvd["divergence"]
            # Bullish divergence aligns with BUY
            if "bullish" in div_type and base.side == "BUY":
                boost = min(15, cvd["confidence"] / 6)
                boosts.append(f"CVD+{boost:.0f}")
            # Bearish divergence aligns with SELL
            elif "bearish" in div_type and base.side == "SELL":
                boost = min(15, cvd["confidence"] / 6)
                boosts.append(f"CVD+{boost:.0f}")
            # Divergence contradicts signal — reduce confidence
            elif "bullish" in div_type and base.side == "SELL":
                boost = -min(10, cvd["confidence"] / 8)
                boosts.append(f"CVD{boost:.0f}")
            elif "bearish" in div_type and base.side == "BUY":
                boost = -min(10, cvd["confidence"] / 8)
                boosts.append(f"CVD{boost:.0f}")
    except Exception:
        pass
    
    # ── 2. VWAP Bands ──
    try:
        from vwap_signal import compute_vwap_bands, VWAPSignal
        vwap_result = compute_vwap_bands(candles)
        enriched.vwap_signal = vwap_result.signal.value
        enriched.vwap_deviation_pct = vwap_result.deviation_pct
        enriched.vwap_deviation_sigma = vwap_result.deviation_sigma
        enriched.vwap_price = vwap_result.vwap
        enriched.vwap_context = (
            f"VWAP={vwap_result.vwap:.4f} dev={vwap_result.deviation_pct:+.2f}% "
            f"({vwap_result.deviation_sigma:+.1f}σ) "
            f"bands=[{vwap_result.lower_band_2:.4f}..{vwap_result.upper_band_2:.4f}]"
        )
        
        vw_sig = vwap_result.signal
        if vw_sig == VWAPSignal.MEAN_REVERT_BUY and base.side == "BUY":
            boost = min(12, vwap_result.confidence / 7)
            boosts.append(f"VWAP+{boost:.0f}")
        elif vw_sig == VWAPSignal.MEAN_REVERT_SELL and base.side == "SELL":
            boost = min(12, vwap_result.confidence / 7)
            boosts.append(f"VWAP+{boost:.0f}")
    except Exception:
        pass
    
    # ── 3. Liquidation Cascade ──
    try:
        from liquidation_cascade import detect_cascade_from_candles
        cascade = detect_cascade_from_candles(candles, coin)
        enriched.cascade_detected = cascade.cascade_detected
        enriched.cascade_side = cascade.cascade_side
        enriched.cascade_bounce_prob = cascade.bounce_probability
        enriched.cascade_action = cascade.suggested_action
        enriched.cascade_context = (
            f"LiqCascade:{cascade.cascade_side} "
            f"severity={cascade.severity:.2f} "
            f"bounce={cascade.bounce_probability:.0%} "
            f"action={cascade.suggested_action} "
            f"conf={cascade.confidence:.0f}%"
        )
        
        if cascade.cascade_detected and cascade.suggested_action == base.side:
            boost = min(20, cascade.bounce_probability * 25)
            boosts.append(f"LC+{boost:.0f}")
    except Exception:
        pass
    
    # ── 4. OI Delta ──
    try:
        if hl_client and price > 0:
            from oi_delta import get_oi_delta
            oi_current, _ = None, None
            try:
                # Try to get OI from client
                meta = hl_client.meta()
                universe = meta.get("universe", [])
                for asset in universe:
                    if asset.get("name", "").upper() == coin.upper():
                        oi_current = float(asset.get("openInterest", 0))
                        break
            except Exception:
                pass
            
            if oi_current and oi_current > 0:
                oi_sig = get_oi_delta(coin, oi_current, price)
                enriched.oi_signal = oi_sig.signal
                enriched.oi_delta_pct = oi_sig.oi_delta_pct
                enriched.oi_confidence = oi_sig.confidence
                enriched.oi_context = (
                    f"OI:{oi_sig.signal} "
                    f"oi={oi_sig.oi_delta_pct:+.2f}% "
                    f"px={oi_sig.price_delta_pct:+.2f}% "
                    f"conf={oi_sig.confidence:.0f}%"
                )
                
                if oi_sig.signal == "trend_confirm_buy" and base.side == "BUY":
                    boost = min(12, oi_sig.confidence / 7)
                    boosts.append(f"OI+{boost:.0f}")
                elif oi_sig.signal == "trend_confirm_sell" and base.side == "SELL":
                    boost = min(12, oi_sig.confidence / 7)
                    boosts.append(f"OI+{boost:.0f}")
                elif oi_sig.signal == "exhaustion_bullish" and base.side == "BUY":
                    boost = min(8, oi_sig.confidence / 10)
                    boosts.append(f"OI_exh+{boost:.0f}")
                elif oi_sig.signal == "exhaustion_bearish" and base.side == "SELL":
                    boost = min(8, oi_sig.confidence / 10)
                    boosts.append(f"OI_exh+{boost:.0f}")
    except Exception:
        pass
    
    # ── Calculate total enrichment ──
    total_boost = sum(float(b.split("+")[-1].split("-")[-1]) * 
                      (1 if "+" in b else -1) 
                      for b in boosts if "+" in b or any(c.isdigit() for c in b.split("+")[-1]))
    
    # Cap total boost
    total_boost = max(-15, min(25, total_boost))
    enriched.enrichment_boost = round(total_boost, 1)
    enriched.enriched_confidence = min(99, max(0, base.confidence + total_boost))
    enriched.enrichment_reason = ",".join(boosts) if boosts else "no_enrichment"

    # ── ENRICHMENT OVERRIDE: When base signal is weak, enrichment can suggest direction ──
    if base.side == "HOLD" or base.confidence < 10:
        # Count directional signals from enrichment
        buy_signals = 0
        sell_signals = 0
        
        if enriched.cvd_divergence != "none" and enriched.cvd_confidence > 40:
            if "bullish" in enriched.cvd_divergence:
                buy_signals += 1
            elif "bearish" in enriched.cvd_divergence:
                sell_signals += 1
        
        if enriched.vwap_signal in ("mean_revert_buy", "oversold"):
            buy_signals += 1
        elif enriched.vwap_signal in ("mean_revert_sell", "overbought"):
            sell_signals += 1
        
        if enriched.cascade_detected and enriched.cascade_bounce_prob > 0.5:
            if enriched.cascade_action == "BUY":
                buy_signals += 1
            elif enriched.cascade_action == "SELL":
                sell_signals += 1
        
        if enriched.oi_signal in ("trend_confirm_buy", "exhaustion_bullish"):
            buy_signals += 1
        elif enriched.oi_signal in ("trend_confirm_sell", "exhaustion_bearish"):
            sell_signals += 1
        
        # 2+ enrichment signals pointing same way → suggests direction
        if buy_signals >= 2 and sell_signals == 0:
            enriched.base_signal.side = "BUY"
            enriched.base_signal.reason = f"enrichment_override:buy:{','.join(boosts) if boosts else 'cvd_vwap'}"
            enriched.enriched_confidence = max(enriched.enriched_confidence, 25)
        elif sell_signals >= 2 and buy_signals == 0:
            enriched.base_signal.side = "SELL"
            enriched.base_signal.reason = f"enrichment_override:sell:{','.join(boosts) if boosts else 'cvd_vwap'}"
            enriched.enriched_confidence = max(enriched.enriched_confidence, 25)

    return enriched


def get_enrichment_context(enriched: EnrichedSignal) -> str:
    """
    Get a compact context string of all enrichments for logging/AI context.
    Target: <100 chars.
    """
    parts = []
    
    if enriched.cvd_divergence != "none" and enriched.cvd_confidence > 40:
        arrow = "📉" if "bearish" in enriched.cvd_divergence else "📈"
        parts.append(f"CVD:{arrow}{enriched.cvd_divergence}({enriched.cvd_confidence:.0f}%)")
    
    if enriched.vwap_signal not in ("neutral", "") and enriched.vwap_deviation_sigma != 0:
        parts.append(f"VWAP:{enriched.vwap_deviation_sigma:+.1f}σ")
    
    if enriched.cascade_detected:
        parts.append(f"LiqCascade:{enriched.cascade_action}({enriched.cascade_bounce_prob:.0%})")
    
    if enriched.oi_signal != "neutral" and enriched.oi_confidence > 20:
        parts.append(f"OI:{enriched.oi_signal}({enriched.oi_confidence:.0f}%)")
    
    if enriched.enrichment_boost != 0:
        arrow = "↑" if enriched.enrichment_boost > 0 else "↓"
        parts.append(f"Enrich:{arrow}{enriched.enrichment_boost:+.0f}%")
    
    return " | ".join(parts) if parts else ""


# ============================================================
# 11. CONVENIENCE — compute all signals for a batch of coins
# ============================================================

def compute_all_signals(
    candles_by_coin: dict[str, list[dict]],
    btc_candles: list[dict] | None = None,
    open_positions: dict[str, dict] | None = None,
    equity: float = 10000.0,
) -> dict[str, MasterSignal]:
    """Compute signals for all coins, sorted by confidence."""
    signals = {}
    for coin, candles in candles_by_coin.items():
        if len(candles) < 50:
            continue
        sig = generate_master_signal(
            coin, candles,
            btc_candles=btc_candles,
            open_positions=open_positions,
            equity=equity,
        )
        signals[coin] = sig
    return signals


# ============================================================
# 11. CONVICTION SCORER — 5-factor quality gate (alphastrike)
# ============================================================

class ConvictionTier:
    """Position sizing tier based on conviction score."""
    NO_TRADE = "no_trade"   # score < 70
    SMALL = "small"          # 70-84: 15% position, 0.25% risk
    MEDIUM = "medium"        # 85-94: 30% position, 0.40% risk
    LARGE = "large"          # 95+:   50% position, 0.50% risk


# Tier sizing: (position_pct_of_equity, risk_per_trade_pct, stop_atr_mult)
CONVICTION_TIERS = {
    ConvictionTier.NO_TRADE: (0.0, 0.0, 0.0),
    ConvictionTier.SMALL:    (0.10, 0.0020, 1.5),   # 10% position, 0.20% risk, wide stop
    ConvictionTier.MEDIUM:   (0.20, 0.0030, 1.3),
    ConvictionTier.LARGE:    (0.30, 0.0040, 1.1),
}

MIN_CONVICTION_SCORE = 70  # Must score 70+ to trade at all


def score_conviction(
    signal: MasterSignal,
    composite: float,
    regime: MarketRegime,
    volume_ratio: float = 1.0,
    atr_ratio: float = 1.0,
    model_agreement_pct: float = 0.0,
    bb_width: float = 0.0,
) -> tuple[float, str, dict]:
    """Score trade conviction on 5 factors (0-100 scale).

    Returns (score, tier, breakdown).

    5 Factors:
      1. Signal Quality (0-30): confidence * 30 → how strong the signal is
      2. Regime Clarity (0-20): ADX > 25 = full, ADX 15-25 = half
      3. Composite Alignment (0-20): composite score aligned with signal direction
      4. Volume Confirmation (0-15): volume_ratio above 1.0 = confirmed
      5. Technical Setup (0-15): RSI in sweet spot + near support/resistance

    Score < 70 → NO TRADE (ultra-conservative gate)
    """
    breakdown = {}
    score = 0.0

    # 1. Signal Quality (0-30): raw confidence from ensemble
    sq = min(30.0, signal.confidence * 30.0)
    breakdown["signal_quality"] = sq
    score += sq

    # 2. Regime Clarity (0-20): how clear is the regime?
    if regime == MarketRegime.CRISIS:
        rc = 5.0  # Crisis = unclear, anything can happen
    elif regime == MarketRegime.HIGH_VOL:
        rc = 10.0  # High vol = somewhat unclear
    elif regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
        rc = 20.0  # Trending = clearest
    else:
        rc = 15.0  # Sideways = moderately clear
    breakdown["regime_clarity"] = rc
    score += rc

    # 3. Composite Alignment (0-20): does composite agree with signal?
    if signal.side == "BUY" and composite > 0:
        ca = 10.0 + abs(composite) * 10.0  # 10-20 range
    elif signal.side == "SELL" and composite < 0:
        ca = 10.0 + abs(composite) * 10.0
    elif composite == 0:
        ca = 5.0
    else:
        ca = 0.0  # Composite disagrees → penalty
    breakdown["composite_alignment"] = ca
    score += ca

    # 4. Volume Confirmation (0-15): above-average volume
    if volume_ratio > 2.0:
        vc = 15.0
    elif volume_ratio > 1.5:
        vc = 12.0
    elif volume_ratio > 1.0:
        vc = 8.0
    elif volume_ratio > 0.8:
        vc = 5.0
    else:
        vc = 2.0
    breakdown["volume_confirmation"] = vc
    score += vc

    # 5. Technical Setup (0-15): RSI + support/resistance + BB squeeze
    rsi = signal.rsi or 50
    ts = 0.0
    if signal.side == "BUY":
        if rsi < 35:
            ts += 10.0  # Oversold bounce setup
        elif rsi < 45:
            ts += 7.0
        elif rsi < 55:
            ts += 5.0
        else:
            ts += 2.0  # Buying into overbought = weak setup
        # Near support bonus
        if signal.entry_price and signal.stop_price:
            dist_pct = abs(signal.entry_price - signal.stop_price) / signal.entry_price * 100
            if dist_pct < 2:
                ts += 5.0  # Tight stop = good setup
    else:  # SELL
        if rsi > 65:
            ts += 10.0  # Overbought fade setup
        elif rsi > 55:
            ts += 7.0
        elif rsi > 45:
            ts += 5.0
        else:
            ts += 2.0  # Selling into oversold = weak setup
        if signal.entry_price and signal.stop_price:
            dist_pct = abs(signal.stop_price - signal.entry_price) / signal.entry_price * 100
            if dist_pct < 2:
                ts += 5.0

    # BB squeeze bonus: tight bands + about to expand = powerful setup
    if bb_width > 0 and bb_width < 0.04:
        ts += 3.0  # BB squeeze = about to move big

    breakdown["technical_setup"] = min(15.0, ts)
    score += min(15.0, ts)

    # Cap at 100
    score = max(0.0, min(100.0, score))

    # Determine tier
    if score >= 95:
        tier = ConvictionTier.LARGE
    elif score >= 85:
        tier = ConvictionTier.MEDIUM
    elif score >= MIN_CONVICTION_SCORE:
        tier = ConvictionTier.SMALL
    else:
        tier = ConvictionTier.NO_TRADE

    # Extra discrimination: counter-trend trade = HALVE the tier
    if signal.side == "BUY" and regime == MarketRegime.TRENDING_DOWN:
        if tier == ConvictionTier.LARGE:
            tier = ConvictionTier.MEDIUM
        elif tier == ConvictionTier.MEDIUM:
            tier = ConvictionTier.SMALL
        elif tier == ConvictionTier.SMALL:
            tier = ConvictionTier.NO_TRADE
    if signal.side == "SELL" and regime == MarketRegime.TRENDING_UP:
        if tier == ConvictionTier.LARGE:
            tier = ConvictionTier.MEDIUM
        elif tier == ConvictionTier.MEDIUM:
            tier = ConvictionTier.SMALL
        elif tier == ConvictionTier.SMALL:
            tier = ConvictionTier.NO_TRADE

    return score, tier, breakdown


def get_tier_sizing(tier: str) -> tuple[float, float, float]:
    """Get (position_pct, risk_pct, stop_atr_mult) for a tier."""
    return CONVICTION_TIERS.get(tier, CONVICTION_TIERS[ConvictionTier.NO_TRADE])
