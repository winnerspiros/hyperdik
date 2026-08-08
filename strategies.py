"""
Trading Strategies — multiple strategies that the agent evaluates
Each strategy returns a signal: {"direction": "buy"/"sell"/"neutral", "confidence": 0-100, "reason": "..."}
"""
import time

class TechnicalStrategy:
    """Strategy based on technical indicators (RSI, MACD, EMAs, Bollinger Bands)"""
    
    def evaluate(self, snapshot):
        """Evaluate technical setup from market snapshot"""
        if not snapshot:
            return {"direction": "neutral", "confidence": 0, "reason": "no_data"}
        
        reasons = []
        score = 0  # Positive = buy, negative = sell
        tf_weights = {"15m": 0.15, "1h": 0.35, "4h": 0.50}
        
        for tf, weight in tf_weights.items():
            rsi = snapshot.get(f"{tf}_rsi", 50)
            macd = snapshot.get(f"{tf}_macd", 0)
            macd_sig = snapshot.get(f"{tf}_macd_signal", 0)
            ema9 = snapshot.get(f"{tf}_ema_9", 0)
            ema21 = snapshot.get(f"{tf}_ema_21", 0)
            ema50 = snapshot.get(f"{tf}_ema_50", 0)
            vol_ratio = snapshot.get(f"{tf}_volume_ratio", 1)
            bb_upper = snapshot.get(f"{tf}_bb_upper", 0)
            bb_lower = snapshot.get(f"{tf}_bb_lower", 0)
            price = snapshot.get(f"{tf}_close", 0)
            change = snapshot.get(f"{tf}_change", 0)
            
            tf_score = 0
            
            # RSI analysis
            if rsi < 25:
                tf_score += 30  # Strong oversold bounce potential
                reasons.append(f"{tf} RSI {rsi:.0f} (oversold)")
            elif rsi < 30:
                tf_score += 20
                reasons.append(f"{tf} RSI {rsi:.0f} near oversold")
            elif rsi < 40:
                tf_score += 10
            elif rsi > 75:
                tf_score -= 30  # Strong overbought
                reasons.append(f"{tf} RSI {rsi:.0f} (overbought)")
            elif rsi > 70:
                tf_score -= 20
                reasons.append(f"{tf} RSI {rsi:.0f} near overbought")
            elif rsi > 60:
                tf_score -= 10
            
            # MACD crossover
            if macd > macd_sig and macd > 0:
                tf_score += 15
                reasons.append(f"{tf} MACD bullish")
            elif macd < macd_sig and macd < 0:
                tf_score -= 15
                reasons.append(f"{tf} MACD bearish")
            elif macd > macd_sig:
                tf_score += 8
            elif macd < macd_sig:
                tf_score -= 8
            
            # EMA alignment (trend)
            if ema9 > ema21 > ema50 and ema9 > 0:
                tf_score += 12
                reasons.append(f"{tf} bullish trend")
            elif ema9 < ema21 < ema50 and ema9 > 0:
                tf_score -= 12
                reasons.append(f"{tf} bearish trend")
            
            # Bollinger Band position
            if price > 0 and bb_upper > 0 and bb_lower > 0:
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = (price - bb_lower) / bb_range
                    if bb_position < 0.2:
                        tf_score += 10  # Near lower band - bounce potential
                        reasons.append(f"{tf} near BB lower")
                    elif bb_position > 0.8:
                        tf_score -= 10  # Near upper band - reversal risk
                        reasons.append(f"{tf} near BB upper")
            
            # Volume confirmation  
            if vol_ratio > 2:
                tf_score += 10 * (1 if tf_score > 0 else -1)
            elif vol_ratio > 1.5:
                tf_score += 5 * (1 if tf_score > 0 else -1)
            
            # Momentum
            if abs(change) > 0.02:  # 2% move in timeframe
                tf_score += change * 500  # Momentum adds to signal
            
            score += tf_score * weight
        
        # Determine direction and confidence
        if score > 10:
            direction = "buy"
            confidence = min(100, abs(score))
        elif score < -10:
            direction = "sell"
            confidence = min(100, abs(score))
        else:
            direction = "neutral"
            confidence = 0
        
        return {
            "direction": direction,
            "confidence": int(confidence),
            "reason": "; ".join(reasons[:5]) if reasons else "neutral_tech",
            "score": round(score, 1),
        }


class SentimentStrategy:
    """Strategy based on news and social sentiment"""
    
    def evaluate(self, sentiment_result):
        if not sentiment_result:
            return {"direction": "neutral", "confidence": 0, "reason": "no_sentiment_data"}
        
        composite = sentiment_result.get("composite", 0)
        
        if composite > 3:
            return {"direction": "buy", "confidence": 70, "reason": "strong_bullish_sentiment", "score": composite}
        elif composite > 1:
            return {"direction": "buy", "confidence": 40, "reason": "mildly_bullish_sentiment", "score": composite}
        elif composite < -3:
            return {"direction": "sell", "confidence": 70, "reason": "strong_bearish_sentiment", "score": composite}
        elif composite < -1:
            return {"direction": "sell", "confidence": 40, "reason": "mildly_bearish_sentiment", "score": composite}
        else:
            return {"direction": "neutral", "confidence": 0, "reason": "neutral_sentiment", "score": composite}


class MeanReversionStrategy:
    """Strategy based on mean reversion - buy dips, sell pumps"""
    
    def evaluate(self, snapshot):
        if not snapshot:
            return {"direction": "neutral", "confidence": 0, "reason": "no_data"}
        
        reasons = []
        score = 0
        
        # Check hourly RSI for mean reversion signals
        rsi_1h = snapshot.get("1h_rsi", 50)
        rsi_4h = snapshot.get("4h_rsi", 50)
        close = snapshot.get("last_price", 0)
        bb_lower = snapshot.get("1h_bb_lower", 0)
        bb_upper = snapshot.get("1h_bb_upper", 0)
        
        # Deep oversold = buy mean reversion
        if rsi_1h < 25 and rsi_4h < 35:
            score += 35
            reasons.append(f"deep_oversold_1h_{rsi_1h:.0f}")
        
        # Deep overbought = sell mean reversion
        if rsi_1h > 75 and rsi_4h > 65:
            score -= 35
            reasons.append(f"deep_overbought_1h_{rsi_1h:.0f}")
        
        # Bollinger band touches
        if close > 0 and bb_lower > 0 and close <= bb_lower * 1.01:
            score += 20
            reasons.append("touch_bb_lower_1h")
        if close > 0 and bb_upper > 0 and close >= bb_upper * 0.99:
            score -= 20
            reasons.append("touch_bb_upper_1h")
        
        if score > 15:
            direction = "buy"
        elif score < -15:
            direction = "sell"
        else:
            direction = "neutral"
        
        return {
            "direction": direction,
            "confidence": min(100, abs(score) * 2),
            "reason": "; ".join(reasons) if reasons else "neutral_mr",
            "score": round(score, 1),
        }


class TrendFollowingStrategy:
    """Strategy based on trend following - buy in uptrends, sell in downtrends"""
    
    def evaluate(self, snapshot):
        if not snapshot:
            return {"direction": "neutral", "confidence": 0, "reason": "no_data"}
        
        score = 0
        reasons = []
        
        # Multi-timeframe trend alignment
        tfs = ["15m", "1h", "4h"]
        bullish_trends = sum(1 for tf in tfs if snapshot.get(f"{tf}_trend") == "bullish")
        bearish_trends = sum(1 for tf in tfs if snapshot.get(f"{tf}_trend") == "bearish")
        
        if bullish_trends >= 2:
            score += 25
            reasons.append(f"{bullish_trends}/3 bullish_trends")
        if bearish_trends >= 2:
            score -= 25
            reasons.append(f"{bearish_trends}/3 bearish_trends")
        
        # MACD alignment across timeframes
        for tf in tfs:
            macd = snapshot.get(f"{tf}_macd", 0)
            macd_sig = snapshot.get(f"{tf}_macd_signal", 0)
            if macd > macd_sig and macd > 0 and macd_sig > 0:
                score += 10
            elif macd < macd_sig and macd < 0 and macd_sig < 0:
                score -= 10
        
        if score > 20:
            direction = "buy"
        elif score < -20:
            direction = "sell"
        else:
            direction = "neutral"
        
        return {
            "direction": direction,
            "confidence": min(100, abs(score) * 2),
            "reason": "; ".join(reasons) if reasons else "neutral_trend",
            "score": round(score, 1),
        }


class StrategyOrchestrator:
    """Combines multiple strategies into a final decision"""
    
    def __init__(self):
        self.technical = TechnicalStrategy()
        self.sentiment = SentimentStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.trend_following = TrendFollowingStrategy()
        
        # Strategy weights (can be tuned)
        self.weights = {
            "technical": 0.40,
            "trend": 0.25,
            "mean_reversion": 0.20,
            "sentiment": 0.15,
        }
    
    def evaluate(self, snapshot=None, sentiment=None):
        """Run all strategies and produce consensus"""
        results = {}
        
        # Run each strategy
        results["technical"] = self.technical.evaluate(snapshot)
        results["trend"] = self.trend_following.evaluate(snapshot)
        results["mean_reversion"] = self.mean_reversion.evaluate(snapshot)
        results["sentiment"] = self.sentiment.evaluate(sentiment)
        
        # Weighted vote
        buy_weight = 0
        sell_weight = 0
        total_confidence = 0
        all_reasons = []
        
        for strat_name, result in results.items():
            w = self.weights.get(strat_name, 0.15)
            if result["direction"] == "buy":
                buy_weight += w * (result["confidence"] / 100)
                total_confidence += result["confidence"] * w
            elif result["direction"] == "sell":
                sell_weight += w * (result["confidence"] / 100)
                total_confidence += result["confidence"] * w
            
            if result["reason"] != "neutral" and result["reason"]:
                all_reasons.append(f"{strat_name}: {result['reason']}")
        
        # Decision
        if buy_weight > sell_weight and buy_weight > 0.15:
            direction = "buy"
            confidence = min(100, int(total_confidence * 1.5))
        elif sell_weight > buy_weight and sell_weight > 0.15:
            direction = "sell"
            confidence = min(100, int(total_confidence * 1.5))
        else:
            direction = "neutral"
            confidence = 0
        
        return {
            "direction": direction,
            "confidence": confidence,
            "reasons": all_reasons[:5],
            "breakdown": {k: v["direction"] for k, v in results.items()},
            "buy_weight": round(buy_weight, 2),
            "sell_weight": round(sell_weight, 2),
        }