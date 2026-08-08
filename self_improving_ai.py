"""
Self-Improving AI — performance feedback loop that analyzes past decisions
and adjusts the AI prompt/strategy based on what works.
"""
import json
import os
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger("self_improving_ai")


class SelfImprovingAI:
    """
    AI self-improvement loop.
    
    How it works:
    1. Reads past decisions from SQLite database
    2. Analyzes win/loss patterns by: market condition, coin, time of day, strategy
    3. Generates prompt modifications to improve future decisions
    4. Tracks which modifications helped and which didn't
    
    Usage:
        improve = SelfImprovingAI("/home/ubuntu/revolut-x-trader/data/trader.db")
        prompt_additions = improve.get_prompt_additions(strategy="day_trader")
        # Add prompt_additions to the AI's system prompt
    """
    
    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "trader.db"
            )
        self.db_path = db_path
        self.prompt_memory_file = "/tmp/ai_prompt_memory.json"
        self.prompt_memory = self._load_prompt_memory()
    
    def _load_prompt_memory(self) -> Dict:
        default = {
            "additions": [],  # List of prompt additions tried
            "results": {},    # addition_id -> win_rate_improvement
            "last_update": None,
        }
        if os.path.exists(self.prompt_memory_file):
            try:
                with open(self.prompt_memory_file) as f:
                    return json.load(f)
            except:
                pass
        return default
    
    def _save_prompt_memory(self):
        try:
            with open(self.prompt_memory_file, "w") as f:
                json.dump(self.prompt_memory, f)
        except:
            pass
    
    def _get_db(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except:
            return None
    
    def analyze_patterns(self, strategy: str = "day_trader") -> Dict:
        """
        Analyze past decisions and find patterns in wins/losses.
        
        Returns:
            Dict with patterns found and suggested prompt changes
        """
        conn = self._get_db()
        if not conn:
            return {"has_data": False, "suggestions": []}
        
        # Get recent decisions - check trades table for PnL
        rows = conn.execute(
            """SELECT d.direction, d.symbol, t.pnl as pnl_result, d.market_conditions,
                      DATE(d.timestamp) as trade_date,
                      strftime('%H', d.timestamp) as trade_hour
               FROM decisions d
               LEFT JOIN trades t ON t.symbol = d.symbol AND t.strategy = d.strategy
               WHERE d.strategy=? AND d.direction IN ('buy','sell')
               ORDER BY d.id DESC LIMIT 50""",
            (strategy,)
        ).fetchall()
        
        conn.close()
        
        if len(rows) < 5:
            return {"has_data": False, "suggestions": ["Need at least 5 evaluated trades for self-analysis"]}
        
        # Analyze patterns
        total = len(rows)
        wins = [r for r in rows if r["pnl_result"] and float(r["pnl_result"]) > 0]
        losses = [r for r in rows if r["pnl_result"] and float(r["pnl_result"]) <= 0]
        win_rate = len(wins) / total if total > 0 else 0
        
        suggestions = []
        
        # Pattern 1: Which coins win/lose most?
        coin_results = {}
        for r in rows:
            coin = r["symbol"] or "unknown"
            if coin not in coin_results:
                coin_results[coin] = {"wins": 0, "losses": 0, "total": 0}
            coin_results[coin]["total"] += 1
            if r["pnl_result"] and float(r["pnl_result"]) > 0:
                coin_results[coin]["wins"] += 1
            elif r["pnl_result"]:
                coin_results[coin]["losses"] += 1
        
        for coin, stats in coin_results.items():
            if stats["total"] >= 3:
                cr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
                if cr < 0.3:
                    suggestions.append(f"⚠️ {coin}: {cr*100:.0f}% win rate ({stats['total']} trades) — AVOID")
                elif cr > 0.7:
                    suggestions.append(f"✅ {coin}: {cr*100:.0f}% win rate ({stats['total']} trades) — PREFER")
        
        # Pattern 2: Best/worst hours
        hour_results = {}
        for r in rows:
            h = r["trade_hour"] or "00"
            if h not in hour_results:
                hour_results[h] = {"wins": 0, "total": 0}
            hour_results[h]["total"] += 1
            if r["pnl_result"] and float(r["pnl_result"]) > 0:
                hour_results[h]["wins"] += 1
        
        best_hour = max(hour_results, key=lambda h: hour_results[h]["wins"] / max(hour_results[h]["total"], 1))
        worst_hour = min(hour_results, key=lambda h: hour_results[h]["wins"] / max(hour_results[h]["total"], 1))
        
        if hour_results.get(best_hour, {}).get("total", 0) >= 3:
            suggestions.append(f"⏰ Best trading hour: {best_hour}:00 UTC")
        if worst_hour != best_hour and hour_results.get(worst_hour, {}).get("total", 0) >= 3:
            suggestions.append(f"⏰ Worst trading hour: {worst_hour}:00 UTC — AVOID")
        
        # Pattern 3: Overall performance
        if win_rate < 0.3:
            suggestions.append("🔧 Overall win rate below 30% — consider more conservative strategy")
        elif win_rate > 0.7:
            suggestions.append("🔧 Overall win rate above 70% — current strategy working well")
        
        return {
            "has_data": True,
            "total_decisions": total,
            "win_rate": round(win_rate * 100, 1),
            "suggestions": suggestions,
        }
    
    def get_prompt_additions(self, strategy: str = "day_trader") -> str:
        """
        Generate prompt additions based on self-analysis.
        
        Returns:
            String to append to the AI system prompt
        """
        analysis = self.analyze_patterns(strategy)
        
        if not analysis.get("has_data"):
            return ""
        
        lines = ["📚 AI SELF-LEARNING NOTES:"]
        for s in analysis.get("suggestions", []):
            lines.append(f"  {s}")
        
        return "\n".join(lines)


# Global convenience function
_global_self_improve = None

def get_self_improvement_context(strategy: str = "day_trader") -> str:
    """Get self-improvement context for the AI prompt"""
    global _global_self_improve
    if _global_self_improve is None:
        _global_self_improve = SelfImprovingAI()
    return _global_self_improve.get_prompt_additions(strategy)


if __name__ == "__main__":
    # Test
    improve = SelfImprovingAI()
    print(improve.get_prompt_additions())
    print()
    print(get_self_improvement_context())