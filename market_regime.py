"""
Market Regime Detection — classifies the market as Trending_Bull, Trending_Bear, Ranging, Volatile, or Low_Vol.
Day trader and scalper use this to coordinate their behavior.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class MarketRegime:
    BULL = "trending_bull"
    BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOL = "low_vol"


def classify_regime(df_4h: Optional[pd.DataFrame] = None,
                    df_1h: Optional[pd.DataFrame] = None,
                    btc_24h_change: Optional[float] = None,
                    fear_greed: Optional[int] = None) -> dict:
    """
    Classify current market regime using multiple signals.
    
    Returns dict with:
      - regime: one of MarketRegime.*
      - strength: 0-1 how strong the signal is
      - description: human-readable explanation
      - action_bias: "scalp", "swing", "both", or "wait"
    """
    result = {
        "regime": MarketRegime.LOW_VOL,
        "strength": 0.0,
        "description": "Insufficient data",
        "action_bias": "wait",
    }

    signals = {"trend": 0, "volatility": 0, "volume": 0, "fear_greed": 0}
    reasons = []

    # 1. TREND SIGNAL from BTC 24h change
    if btc_24h_change is not None:
        if btc_24h_change > 3:
            signals["trend"] = 1  # Strong bull
            reasons.append(f"BTC +{btc_24h_change:.1f}% (bullish)")
        elif btc_24h_change > 1:
            signals["trend"] = 0.5  # Mild bull
            reasons.append(f"BTC +{btc_24h_change:.1f}% (mild bull)")
        elif btc_24h_change < -3:
            signals["trend"] = -1  # Strong bear
            reasons.append(f"BTC {btc_24h_change:.1f}% (bearish)")
        elif btc_24h_change < -1:
            signals["trend"] = -0.5  # Mild bear
            reasons.append(f"BTC {btc_24h_change:.1f}% (mild bear)")
        else:
            signals["trend"] = 0  # Neutral / ranging
            reasons.append(f"BTC {btc_24h_change:+.1f}% (ranging)")

    # 2. VOLATILITY from 4h ATR
    if df_4h is not None and len(df_4h) >= 14:
        closes = df_4h["close"].values.astype(float)
        highs = df_4h["high"].values.astype(float)
        lows = df_4h["low"].values.astype(float)
        atr = np.mean([highs[i] - lows[i] for i in range(-14, 0)])
        mean_price = np.mean(closes[-14:])
        atr_pct = atr / mean_price * 100 if mean_price > 0 else 0
        
        if atr_pct > 5:
            signals["volatility"] = 1  # High vol
            reasons.append(f"ATR {atr_pct:.1f}% (high vol)")
        elif atr_pct > 2.5:
            signals["volatility"] = 0.5  # Medium vol
            reasons.append(f"ATR {atr_pct:.1f}% (med vol)")
        else:
            signals["volatility"] = 0  # Low vol
            reasons.append(f"ATR {atr_pct:.1f}% (low vol)")

    # 3. FEAR & GREED
    if fear_greed is not None:
        if fear_greed >= 75:
            signals["fear_greed"] = 1  # Extreme greed
            reasons.append(f"F&G {fear_greed} (extreme greed)")
        elif fear_greed >= 55:
            signals["fear_greed"] = 0.5  # Greed
            reasons.append(f"F&G {fear_greed} (greed)")
        elif fear_greed <= 25:
            signals["fear_greed"] = -1  # Extreme fear
            reasons.append(f"F&G {fear_greed} (extreme fear)")
        elif fear_greed <= 45:
            signals["fear_greed"] = -0.5  # Fear
            reasons.append(f"F&G {fear_greed} (fear)")
        else:
            signals["fear_greed"] = 0  # Neutral
            reasons.append(f"F&G {fear_greed} (neutral)")

    # 4. VOLUME ANOMALY from 1h
    if df_1h is not None and len(df_1h) >= 24:
        volumes = df_1h["volume"].values.astype(float)
        recent_vol = np.mean(volumes[-4:])  # last 4h
        avg_vol = np.mean(volumes[-24:])  # last 24h
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        if vol_ratio > 2:
            signals["volume"] = 1
            reasons.append(f"Vol {vol_ratio:.1f}x avg (anomaly)")
        elif vol_ratio > 1.5:
            signals["volume"] = 0.5
            reasons.append(f"Vol {vol_ratio:.1f}x avg (elevated)")
        else:
            signals["volume"] = 0

    # COMBINE SIGNALS
    trend_score = signals["trend"]
    vol_score = signals["volatility"]
    fg_score = signals["fear_greed"]
    vol_ratio_score = signals["volume"]

    high_vol = vol_score >= 0.5
    trending = abs(trend_score) >= 0.5
    fear = fg_score < 0
    extreme_sentiment = abs(fg_score) >= 0.5

    if high_vol and (trending or extreme_sentiment):
        result["regime"] = MarketRegime.VOLATILE
        result["description"] = "High volatility with directional bias — trade small, tight stops"
        result["action_bias"] = "both"
    elif trending and trend_score > 0:
        result["regime"] = MarketRegime.BULL
        result["strength"] = abs(trend_score)
        result["description"] = f"Trending up — favor swing/trend following"
        result["action_bias"] = "swing"
    elif trending and trend_score < 0:
        result["regime"] = MarketRegime.BEAR
        result["strength"] = abs(trend_score)
        result["description"] = f"Trending down — sit out or short, no long scalps"
        result["action_bias"] = "wait"
    elif not trending and not high_vol:
        result["regime"] = MarketRegime.RANGING
        result["strength"] = 0.3
        result["description"] = f"Ranging — favor support scalping, avoid trend chasing"
        result["action_bias"] = "scalp"
    else:
        result["regime"] = MarketRegime.LOW_VOL
        result["description"] = f"Low volatility — no clear edge, reduce size"
        result["action_bias"] = "wait"

    # Build summary
    result["signals"] = signals
    result["reasons"] = reasons
    return result


def get_regime_summary(df_4h, df_1h, btc_change, fear_greed) -> str:
    """Returns a one-line market regime summary for the AI"""
    regime = classify_regime(df_4h, df_1h, btc_change, fear_greed)
    direction_icons = {
        MarketRegime.BULL: "🟢",
        MarketRegime.BEAR: "🔴",
        MarketRegime.RANGING: "🟡",
        MarketRegime.VOLATILE: "🟠",
        MarketRegime.LOW_VOL: "⚪",
    }
    icon = direction_icons.get(regime["regime"], "⚪")
    return (
        f"{icon} REGIME: {regime['regime']} | "
        f"Bias: {regime['action_bias']} | "
        f"Strength: {regime['strength']:.0%} | "
        f"{' | '.join(regime.get('reasons', [])[:2])}"
    )