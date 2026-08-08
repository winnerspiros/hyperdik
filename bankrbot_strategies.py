"""
BankrBot-inspired strategies — pump-risk filtering, narrative tracking, persistent trends
Plus Revolut X native TP/SL conditional orders
"""
import time

class PumpRiskFilter:
    """BankrBot pump-risk flags — filter out fake volume / low liquidity pairs"""
    
    PUMP_RISK_COINS = {"BONK", "MEW", "NEIRO", "FLOKI", "PEPE", "TOSHI", "MOBILE"}
    
    @staticmethod
    def assess(symbol, snapshot):
        """Return risk flags for a pair. Two or more flags = exclude."""
        flags = []
        coin = symbol.split("-")[0] if "-" in symbol else symbol
        
        # Fresh listing risk (new pairs with sudden volume)
        bb_upper = snapshot.get("1h_bb_upper", 0)
        bb_lower = snapshot.get("1h_bb_lower", 0)
        price = snapshot.get("last_price", 0)
        
        if bb_lower > 0 and price > 0:
            bb_width = (bb_upper - bb_lower) / price * 100
            if bb_width > 10:
                flags.append("extreme_volatility")  # BB >10% wide
        
        # Low liquidity
        spread = snapshot.get("spread_pct", 100)
        if spread > 1:
            flags.append("wide_spread")
        
        # Meme coin risk (known pump-n-dump territory)
        if coin in PumpRiskFilter.PUMP_RISK_COINS:
            flags.append("high_risk_coin")
        
        # Volume anomaly
        vol_ratio = snapshot.get("1h_volume_ratio", 1)
        if vol_ratio > 5:
            flags.append("volume_spike_anomaly")
        
        # No volume at all
        if vol_ratio < 0.1:
            flags.append("dead_pair")
        
        return flags


class PersistentTrendStrategy:
    """BankrBot: persistence > pump magnitude. Same trend for 3+ sessions = real"""
    
    def __init__(self, memory_file="/tmp/revolut_trend_memory.json"):
        self.memory_file = memory_file
        self.trend_memory = {}
        self._load()
    
    def _load(self):
        import json, os
        try:
            with open(self.memory_file) as f:
                self.trend_memory = json.load(f)
        except: pass
    
    def _save(self):
        import json
        with open(self.memory_file, "w") as f:
            json.dump(self.trend_memory, f)
    
    def evaluate(self, snapshot):
        if not snapshot:
            return {"direction": "neutral", "confidence": 0, "reason": "no_data"}
        
        symbol = snapshot.get("symbol", "unknown")
        now = int(time.time())
        
        # Track trend direction over time
        trend_1h = snapshot.get("1h_trend", "mixed")
        trend_4h = snapshot.get("4h_trend", "mixed")
        rsi_1h = snapshot.get("1h_rsi", 50)
        
        entry = self.trend_memory.setdefault(symbol, {
            "trends": [], "persistent_since": now, "direction": "unknown"
        })
        
        entry["trends"].append({
            "time": now,
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "rsi_1h": rsi_1h,
        })
        entry["trends"] = entry["trends"][-24:]  # Keep last 24 checks
        
        # Count persistent trend
        bullish_count = sum(1 for t in entry["trends"][-6:] if t["trend_1h"] == "bullish")
        bearish_count = sum(1 for t in entry["trends"][-6:] if t["trend_1h"] == "bearish")
        
        self._save()
        
        # Signal: persistent same-direction trend for 3+ checks
        score = 0
        reasons = []
        
        if bullish_count >= 4:
            score += 25
            reasons.append(f"persistent_bullish_{bullish_count}/6")
            # Check for pullback within uptrend (buy the dip)
            if rsi_1h < 45:
                score += 15
                reasons.append("dip_in_uptrend")
        elif bearish_count >= 4:
            score -= 25
            reasons.append(f"persistent_bearish_{bearish_count}/6")
        
        # RSI confirmation
        if score > 0 and 40 < rsi_1h < 60:
            score += 10  # Room to run
        if score < 0 and 40 < rsi_1h < 60:
            score += 10  # Room to fall
        
        if score > 20:
            return {"direction": "buy", "confidence": min(100, abs(score) * 2), 
                    "reason": "; ".join(reasons), "score": round(score, 1)}
        elif score < -20:
            return {"direction": "sell", "confidence": min(100, abs(score) * 2),
                    "reason": "; ".join(reasons), "score": round(score, 1)}
        
        return {"direction": "neutral", "confidence": 0, "reason": "no_persistent_trend", "score": 0}


class VolumeProfileStrategy:
    """Volume analysis: high relative volume confirms moves, anomalies signal exhaustion"""
    
    def evaluate(self, snapshot):
        if not snapshot:
            return {"direction": "neutral", "confidence": 0, "reason": "no_data"}
        
        score = 0
        reasons = []
        
        vol_ratio = snapshot.get("1h_volume_ratio", 1)
        change = snapshot.get("1h_change", 0)
        price = snapshot.get("last_price", 0)
        
        # Volume confirmation on breakout
        if vol_ratio > 2 and abs(change) > 0.01:
            score += 20 if change > 0 else -20
            reasons.append(f"volume_{vol_ratio:.1f}x_confirms_{'up' if change > 0 else 'down'}")
        
        # Volume climax (exhaustion) — 5x+ volume with stall = reversal
        if vol_ratio > 5 and abs(change) < 0.005:
            score -= 15  # Exhaustion, fade the move
            reasons.append("volume_climax_exhaustion")
        
        # Low volume drift (manipulation risk)
        if vol_ratio < 0.3 and abs(change) > 0.02:
            score -= 20  # Big move on thin volume = fake
            reasons.append("thin_volume_move")
        
        if score > 15:
            return {"direction": "buy", "confidence": min(70, abs(score) * 3),
                    "reason": "; ".join(reasons), "score": round(score, 1)}
        elif score < -15:
            return {"direction": "sell", "confidence": min(70, abs(score) * 3),
                    "reason": "; ".join(reasons), "score": round(score, 1)}
        
        return {"direction": "neutral", "confidence": 0, "reason": "normal_volume", "score": 0}