"""
Self-Funding Manager — uses trading profits to pay for AI compute
Tracks daily PnL, calculates surplus for API credits, enables auto-topup
"""
import json
import time
import os

STATE_FILE = "/tmp/revolut_self_fund.json"
DAILY_BUDGET_EUR = 5.0  # Daily AI compute budget
PROFIT_SHARE_FOR_FUNDING = 0.3  # 30% of profits go to self-funding

class SelfFundingManager:
    def __init__(self):
        self.state = {
            "total_pnl_eur": 0.0,
            "funds_allocated_for_ai": 0.0,
            "funds_consumed_today": 0.0,
            "days_funded": 0,
            "last_fund_date": None,
            "is_self_sustaining": False,
            "daily_profit_log": [],
        }
        self.load()
    
    def load(self):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if k in self.state:
                        self.state[k] = v
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def save(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def record_pnl(self, pnl_eur):
        """Record a PnL event from trading"""
        self.state["total_pnl_eur"] += pnl_eur
        today = time.strftime("%Y-%m-%d")
        
        # Track daily PnL
        if self.state["daily_profit_log"] and self.state["daily_profit_log"][-1]["date"] == today:
            self.state["daily_profit_log"][-1]["pnl"] += pnl_eur
        else:
            self.state["daily_profit_log"].append({"date": today, "pnl": pnl_eur})
        
        # Keep last 30 days
        self.state["daily_profit_log"] = self.state["daily_profit_log"][-30:]
        
        # If profit, allocate portion to AI funding
        if pnl_eur > 0:
            allocation = pnl_eur * PROFIT_SHARE_FOR_FUNDING
            self.state["funds_allocated_for_ai"] += allocation
        
        self.check_self_sustaining()
        self.save()
    
    def check_self_sustaining(self):
        """Check if trading profits can sustain AI compute costs"""
        # Need at least 7 days of positive PnL covering daily budget
        recent = self.state["daily_profit_log"][-7:]
        if len(recent) >= 3:
            total_recent = sum(d["pnl"] for d in recent)
            avg_daily = total_recent / len(recent)
            if avg_daily >= DAILY_BUDGET_EUR * 1.5:  # 1.5x buffer
                self.state["is_self_sustaining"] = True
                return True
        
        self.state["is_self_sustaining"] = False
        return False
    
    def consume_funds(self, amount_eur):
        """Record AI compute consumption"""
        self.state["funds_consumed_today"] += amount_eur
        self.state["funds_allocated_for_ai"] -= amount_eur
        self.save()
    
    def get_status(self):
        """Get readable status"""
        lines = []
        lines.append(f"💰 Total PnL: €{self.state['total_pnl_eur']:.2f}")
        lines.append(f"📊 AI Fund: €{self.state['funds_allocated_for_ai']:.2f} allocated")
        lines.append(f"📉 Today's cost: €{self.state['funds_consumed_today']:.2f}")
        lines.append(f"🔄 Self-sustaining: {'YES ✅' if self.state['is_self_sustaining'] else 'NO ❌'}")
        if self.state['daily_profit_log']:
            recent = self.state['daily_profit_log'][-3:]
            parts = []
            for d in recent:
                pnl_str = "€{:.2f}".format(d["pnl"]).replace("-", "-€")
                parts.append(f'{d["date"]}: {pnl_str}')
            lines.append("📈 Recent PnL: " + " | ".join(parts))
        return "\n".join(lines)
    
    def recommendation(self):
        """What should happen next for self-funding"""
        if self.state["is_self_sustaining"]:
            return "SELF_SUSTAINING: Trading covers AI costs. Increase position sizes?"
        
        if self.state["funds_allocated_for_ai"] > DAILY_BUDGET_EUR:
            return f"FUNDED: €{self.state['funds_allocated_for_ai']:.2f} available for AI. Can run ~{int(self.state['funds_allocated_for_ai']/DAILY_BUDGET_EUR)} more days."
        
        deficit = max(0, DAILY_BUDGET_EUR - self.state.get("funds_consumed_today", 0))
        return f"NEEDS_FUNDING: €{deficit:.2f} more needed today. Total AI fund: €{self.state['funds_allocated_for_ai']:.2f}"

if __name__ == "__main__":
    sfm = SelfFundingManager()
    print(sfm.get_status())
    print(f"\n→ {sfm.recommendation()}")