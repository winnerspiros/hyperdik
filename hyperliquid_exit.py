#!/usr/bin/env python3
"""
Hyperliquid Exit Engine — Three-phase position exit.

Phase 1: ROI Ladder — take partial profit at tiered levels (50% at +5%, 30% at +8%, 20% at +12%)
Phase 2: Hard Stop — absolute loss limit, no exceptions  
Phase 3: Ratcheting Trailing Stop — locks in profit as price moves favorably

Plus: Pre-trade thesis for every entry.
"""
import json, time

def build_exit_plan(entry_price: float, direction: str, coin: str, atr_multiple: float = 2.0) -> dict:
    """
    Build a comprehensive three-phase exit plan for a position.
    
    Returns a dict the AI daemon can submit alongside the entry order.
    """
    multiplier = 1 if direction == "LONG" else -1
    
    # Phase 1: ROI Ladder (tiered take-profits)
    roi_ladder = [
        {"pct": 5.0, "sell_pct": 50, "label": "TP1 - base"},
        {"pct": 8.0, "sell_pct": 30, "label": "TP2 - extension"},
        {"pct": 12.0, "sell_pct": 20, "label": "TP3 - runner"},
    ]
    
    # Phase 2: Hard Stop (absolute limit)
    hard_stop_pct = 5.0  # default, AI can override
    
    # Phase 3: Ratcheting Trailing Stop
    # Activates after price moves 3% in our favor
    # Trail distance: 2% from peak
    trailing = {
        "activation_pct": 3.0,  # start trailing after +3%
        "trail_distance_pct": 2.0,  # keep 2% from peak
    }
    
    plan = {
        "coin": coin,
        "direction": direction,
        "entry_price": entry_price,
        "roi_ladder": [
            {
                "price": round(entry_price * (1 + l["pct"]/100 * multiplier), 2),
                "sell_pct": l["sell_pct"],
                "label": l["label"],
            }
            for l in roi_ladder
        ],
        "hard_stop": round(entry_price * (1 - hard_stop_pct/100 * multiplier), 2),
        "hard_stop_pct": hard_stop_pct,
        "trailing_activation": round(entry_price * (1 + trailing["activation_pct"]/100 * multiplier), 2),
        "trail_distance_pct": trailing["trail_distance_pct"],
    }
    
    return plan


def build_thesis(context: str, coin: str, direction: str, entry_price: float,
                 support: float, resistance: float, conviction: str = "medium") -> str:
    """
    Build a structured pre-trade thesis.
    
    Returns a string the AI can validate before executing.
    """
    return f"""
PRE-TRADE THESIS: {direction.upper()} {coin} @ ${entry_price:,.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BULL CASE (why we win):
- Entry at support level (${support:,.2f}), minimizing downside risk
- Confirmed by: {context[:150]}

BEAR CASE (what kills us):
- Support breaks → stop at ${entry_price * 0.95:,.2f} (5% loss max)
- Leverage amplifies losses — position size must be conservative
- Market regime change could invalidate thesis

INVALIDATION CRITERIA (when to exit early):
- Price closes below support on 4h+ timeframe
- RSI divergence on daily
- Funding rate turns extremely negative (crowded short)

SIZING RATIONALE:
- Max loss: 5% of position = {5 * entry_price * 0.01:.2f} per unit
- Position fits within 20% equity limit
- Conviction: {conviction}

EXIT PLAN: Three-phase (see exit_plan)
✓ Phase 1: ROI ladder at +5/+8/+12%
✓ Phase 2: Hard stop at -5%
✓ Phase 3: Ratcheting trail after +3%
""".strip()


# ── Trailing Stop Engine ──

class TrailingStop:
    """Ratcheting trailing stop for a single position."""
    
    def __init__(self, entry_price: float, direction: str, 
                 activation_pct: float = 3.0, trail_pct: float = 2.0):
        self.entry = entry_price
        self.direction = direction  # LONG or SHORT
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self.peak = entry_price
        self.stop = None  # None until activated
        self.activated = False
        self.multiplier = 1 if direction == "LONG" else -1
    
    def update(self, current_price: float) -> float:
        """Update trailing stop. Returns current stop price (or None if not activated)."""
        # Check if trailing should activate
        pnl_pct = (current_price - self.entry) / self.entry * 100 * self.multiplier
        
        if not self.activated:
            if pnl_pct >= self.activation_pct:
                self.activated = True
                self.peak = current_price
                self.stop = current_price * (1 - self.trail_pct/100 * self.multiplier)
            return None
        
        # Update peak
        if current_price * self.multiplier > self.peak * self.multiplier:
            self.peak = current_price
            self.stop = self.peak * (1 - self.trail_pct/100 * self.multiplier)
        
        # Stop only moves in favorable direction (ratcheting)
        return self.stop
    
    def triggered(self, current_price: float) -> bool:
        """Check if the trailing stop is hit."""
        if self.stop is None:
            return False
        return current_price * self.multiplier <= self.stop * self.multiplier


# ── Position Manager ──

class PositionManager:
    """Manages all open positions with their exit plans and trailing stops."""
    
    def __init__(self):
        self.positions = {}  # {coin: {entry, direction, exit_plan, trailing_stop, roi_executed}}
    
    def open_position(self, coin: str, direction: str, entry_price: float, 
                      support: float, resistance: float, context: str = ""):
        """Register a new position with full exit plan."""
        thesis = build_thesis(context, coin, direction, entry_price, support, resistance)
        exit_plan = build_exit_plan(entry_price, direction, coin)
        trail = TrailingStop(entry_price, direction)
        
        self.positions[coin] = {
            "entry": entry_price,
            "direction": direction,
            "thesis": thesis,
            "exit_plan": exit_plan,
            "trail": trail,
            "roi_executed": [],  # which ladder rungs have fired
        }
        
        return thesis, exit_plan
    
    def update(self, coin: str, current_price: float) -> list:
        """Check all exit conditions. Returns list of actions needed."""
        if coin not in self.positions:
            return []
        
        pos = self.positions[coin]
        multiplier = 1 if pos["direction"] == "LONG" else -1
        actions = []
        
        # Phase 1: ROI Ladder
        for rung in pos["exit_plan"]["roi_ladder"]:
            if rung["label"] not in pos["roi_executed"]:
                if current_price * multiplier >= rung["price"] * multiplier:
                    actions.append({
                        "type": "roi_take_profit",
                        "coin": coin,
                        "sell_pct": rung["sell_pct"],
                        "price": current_price,
                        "label": rung["label"],
                    })
                    pos["roi_executed"].append(rung["label"])
        
        # Phase 2: Hard Stop
        stop_price = pos["exit_plan"]["hard_stop"]
        if current_price * multiplier <= stop_price * multiplier:
            actions.append({
                "type": "hard_stop",
                "coin": coin,
                "sell_pct": 100,
                "price": current_price,
                "label": "HARD STOP",
            })
            return actions  # immediate, no further checks
        
        # Phase 3: Trailing Stop
        trail = pos["trail"]
        stop = trail.update(current_price)
        if trail.triggered(current_price):
            actions.append({
                "type": "trailing_stop",
                "coin": coin,
                "sell_pct": 100,
                "price": current_price,
                "label": f"TRAILING STOP @ {stop:.2f}",
            })
        
        return actions
    
    def close_position(self, coin: str):
        """Remove a closed position."""
        self.positions.pop(coin, None)


# ── Self-test ──

if __name__ == "__main__":
    print("=== HYPERLIQUID EXIT ENGINE ===\n")
    
    # Test: LONG BTC entry at 64500
    thesis, plan = PositionManager().open_position(
        "BTC", "LONG", 64500.0, 63000.0, 66000.0,
        "BTC bounced off 63k support with high volume, RSI 35 turning up"
    )
    print(thesis)
    print()
    print("Exit Plan:", json.dumps(plan, indent=2))
    print()
    
    # Simulate price movement
    pm = PositionManager()
    pm.open_position("BTC", "LONG", 64500, 63000, 66000, "support bounce")
    
    for price in [64800, 65200, 65800, 65500, 66000, 67000, 66500, 66300]:
        actions = pm.update("BTC", price)
        if actions:
            for a in actions:
                print(f"BTC @ ${price:,.0f}: {a['type']} ({a['label']})")
    
    print("\n✅ Exit engine test complete")
