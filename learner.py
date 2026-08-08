"""
Learning System — tracks strategy performance and auto-adjusts weights
Learns from every trade: which strategies work, which pairs profit, market regimes
"""
import json
import time
import os

class LearningSystem:
    def __init__(self, state_path="/tmp/revolut_learner.json"):
        self.state_path = state_path
        self.state = {
            "strategy_weights": {
                "technical": 0.40,
                "trend": 0.25,
                "mean_reversion": 0.20,
                "sentiment": 0.15,
            },
            "strategy_performance": {},  # strategy_name -> {wins, losses, total_pnl, trades}
            "pair_performance": {},       # symbol -> {wins, losses, total_pnl, trades}
            "market_regime": "unknown",   # trending, ranging, volatile
            "regime_history": [],
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "total_pnl": 0.0,
            "best_pair": None,
            "worst_day": None,
            "daily_pnl": 0.0,
            "last_trade_time": 0,
            "adaptation_history": [],
        }
        self.load()
    
    def load(self):
        try:
            with open(self.state_path) as f:
                saved = json.load(f)
                # Merge saved state, preserving defaults for new keys
                for k, v in saved.items():
                    if k in self.state:
                        self.state[k] = v
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def record_trade_outcome(self, symbol, strategy, side, entry_price, exit_price, size, pnl_pct):
        """Record a completed trade outcome"""
        is_win = pnl_pct > 0
        self.state["total_trades"] += 1
        if is_win:
            self.state["total_wins"] += 1
        else:
            self.state["total_losses"] += 1
        
        # Track per strategy
        if strategy not in self.state["strategy_performance"]:
            self.state["strategy_performance"][strategy] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        perf = self.state["strategy_performance"][strategy]
        perf["trades"] += 1
        if is_win:
            perf["wins"] += 1
        else:
            perf["losses"] += 1
        perf["pnl"] += pnl_pct
        
        # Track per pair
        if symbol not in self.state["pair_performance"]:
            self.state["pair_performance"][symbol] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        pp = self.state["pair_performance"][symbol]
        pp["trades"] += 1
        if is_win:
            pp["wins"] += 1
        else:
            pp["losses"] += 1
        pp["pnl"] += pnl_pct
        
        # Update best pair
        if self.state["best_pair"] is None:
            self.state["best_pair"] = symbol
        else:
            best = self.state["pair_performance"].get(self.state["best_pair"], {})
            if pp["pnl"] > best.get("pnl", 0):
                self.state["best_pair"] = symbol
        
        self.state["last_trade_time"] = int(time.time())
        self.state["daily_pnl"] += pnl_pct
        self.state["total_pnl"] += pnl_pct
        self.save()
    
    def adapt_strategy_weights(self):
        """Auto-adjust strategy weights based on performance"""
        if self.state["total_trades"] < 5:
            return  # Not enough data
        
        total_pnl = sum(p["pnl"] for p in self.state["strategy_performance"].values())
        if total_pnl == 0:
            return
        
        weights = self.state["strategy_weights"]
        new_weights = {}
        
        for strategy, perf in self.state["strategy_performance"].items():
            if perf["trades"] < 2:
                new_weights[strategy] = weights.get(strategy, 0.15)
                continue
            
            win_rate = perf["wins"] / max(perf["trades"], 1)
            avg_pnl = perf["pnl"] / max(perf["trades"], 1)
            
            # Performance score: win rate * 50 + avg pnl * 50
            perf_score = (win_rate * 50) + (max(-1, min(1, avg_pnl)) * 50)
            new_weights[strategy] = max(0.05, min(0.50, perf_score / 100))
        
        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            for s in new_weights:
                new_weights[s] = round(new_weights[s] / total, 2)
        
        # Ensure the sum is exactly 1.0
        new_total = sum(new_weights.values())
        if new_total != 1.0 and new_weights:
            # Add rounding difference to the largest weight
            largest = max(new_weights, key=new_weights.get)
            new_weights[largest] = round(new_weights[largest] + (1.0 - new_total), 2)
        
        self.state["strategy_weights"] = new_weights
        self.state["adaptation_history"].append({
            "time": int(time.time()),
            "weights": dict(new_weights),
            "total_trades": self.state["total_trades"],
        })
        self.save()
    
    def detect_market_regime(self, market_snapshot):
        """Detect if market is trending, ranging, or volatile"""
        # Simple regime detection based on ATR and BB width
        regime = "unknown"
        
        if market_snapshot:
            atr_1h = market_snapshot.get("1h_atr", 0)
            price = market_snapshot.get("last_price", 1)
            bb_upper = market_snapshot.get("1h_bb_upper", 0)
            bb_lower = market_snapshot.get("1h_bb_lower", 0)
            
            if price > 0 and atr_1h > 0:
                volatility = atr_1h / price * 100  # ATR as % of price
                if volatility > 3:
                    regime = "volatile"
                elif volatility > 1:
                    regime = "ranging"
                else:
                    regime = "trending"
            
            # BB width confirms
            if bb_upper > 0 and bb_lower > 0 and price > 0:
                bb_width = (bb_upper - bb_lower) / price * 100
                if bb_width > 8:
                    regime = "volatile"
        
        self.state["market_regime"] = regime
        self.state["regime_history"].append({
            "time": int(time.time()),
            "regime": regime,
        })
        self.save()
        return regime
    
    def get_optimal_strategy(self, regime):
        """Return which strategy to favor based on market regime"""
        # In trending markets: trend following wins
        # In ranging markets: mean reversion wins
        # In volatile markets: reduce size, use technical
        if regime == "trending":
            return {"favor": "trend", "reduce": "mean_reversion", "note": "trend_following_preferred"}
        elif regime == "volatile":
            return {"favor": "technical", "reduce": "sentiment", "note": "reduce_size_volatile"}
        else:  # ranging
            return {"favor": "mean_reversion", "reduce": "trend", "note": "mean_reversion_preferred"}
    
    def get_summary(self):
        """Get a readable summary of learning state"""
        s = self.state
        lines = []
        lines.append(f"Total trades: {s['total_trades']} (W:{s['total_wins']} L:{s['total_losses']})")
        if s['total_trades'] > 0:
            win_rate = s['total_wins'] / s['total_trades'] * 100
            lines.append(f"Win rate: {win_rate:.1f}% | Total PnL: {s['total_pnl']:+.2f}%")
        lines.append(f"Market regime: {s['market_regime']}")
        lines.append(f"Weights: {json.dumps(s['strategy_weights'])}")
        if s['best_pair']:
            bp = s['pair_performance'].get(s['best_pair'], {})
            lines.append(f"Best pair: {s['best_pair']} ({bp.get('wins',0)}W/{bp.get('losses',0)}L, {bp.get('pnl',0):+.2f}%)")
        return "\n".join(lines)