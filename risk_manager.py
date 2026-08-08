"""
Risk Manager — position sizing, portfolio limits, stop-loss management
"""
import json
import time

class RiskManager:
    def __init__(self, config=None):
        self.config = config or {
            "max_position_size_pct": 30,   # Max % of portfolio per position
            "max_total_positions": 3,       # Max concurrent positions
            "default_stop_loss_pct": 5,     # Default stop loss %
            "default_take_profit_pct": 15,  # Default take profit %
            "trailing_stop_activation_pct": 5,  # Profit % to activate trailing stop
            "trailing_stop_distance_pct": 3,    # Trailing stop distance
            "min_balance_to_trade": 3,      # Min EUR to trade
            "max_daily_loss_pct": 10,       # Max daily loss before pause
            "max_slippage_pct": 0.5,        # Max acceptable slippage
            "min_spread_to_trade": 0.01,    # Min spread % (avoid thin markets)
            "order_book_imbalance_min": -0.3,  # Min imbalance score
            "order_book_imbalance_max": 0.3,   # Max imbalance score
        }
        self.daily_pnl = 0
        self.trade_log = []
        self.consecutive_losses = 0
        self.is_paused = False
    
    def load_state(self, path="/tmp/revolut_trader_state.json"):
        """Load saved state"""
        try:
            with open(path) as f:
                state = json.load(f)
            self.daily_pnl = state.get("daily_pnl", 0)
            self.consecutive_losses = state.get("consecutive_losses", 0)
            self.is_paused = state.get("is_paused", False)
            self.trade_log = state.get("trade_log", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def save_state(self, path="/tmp/revolut_trader_state.json"):
        """Save state"""
        with open(path, "w") as f:
            json.dump({
                "daily_pnl": self.daily_pnl,
                "consecutive_losses": self.consecutive_losses,
                "is_paused": self.is_paused,
                "trade_log": self.trade_log[-50:],  # Keep last 50 trades
            }, f)
    
    def get_portfolio_value_usd(self, client):
        """Calculate total portfolio value in EUR"""
        balances = client.get_balances()
        total = 0
        for b in balances:
            currency = b["currency"]
            available = float(b["available"])
            if currency in ("EUR", "USDC", "USD", "EURC"):
                total += available
            elif available > 0 and currency in ("BTC", "ETH"):
                # Small approximation - skip for now, count only stable
                pass
        return max(total, 0.01)
    
    def get_available_usdc(self, client):
        """Get available balance in order of preference: EUR > USDC > USD > EURC"""
        balances = client.get_balances()
        for priority in ["EUR", "USDC", "USD", "EURC"]:
            for b in balances:
                if b["currency"] == priority:
                    avail = float(b["available"])
                    if avail > 0:
                        return avail
        return 0
    
    def calculate_position_size(self, client, symbol, confidence_score, atr=None):
        """Calculate how much to trade based on risk parameters"""
        usdc = self.get_available_usdc(client)
        total_value = usdc
        
        if total_value < self.config["min_balance_to_trade"]:
            return 0, "balance_too_low"
        
        # Base size from portfolio allocation
        max_per_pos = total_value * (self.config["max_position_size_pct"] / 100)
        print(f"  Portfolio: ${total_value:.2f}, Max/pos: ${max_per_pos:.2f}")
        
        # Adjust by confidence (score 0-100, maps to 0-1.5x multiplier)
        confidence_mult = min(1.5, max(0.3, confidence_score / 50))
        size = max_per_pos * confidence_mult
        
        # ATR-based adjustment (volatility)
        if atr and atr > 0:
            # If ATR is high relative to price, reduce size
            vol_ratio = atr / 100  # If atr is 3% of price, reduce by 0.7x
            vol_adjust = max(0.5, min(1.0, 1.0 - vol_ratio * 0.1))
            size *= vol_adjust
        
        # Daily loss limit check
        max_daily_loss = total_value * (self.config["max_daily_loss_pct"] / 100)
        if self.daily_pnl < -max_daily_loss * 0.5:
            print(f"  Daily loss near limit: ${self_daily_pnl:.2f}")
            size *= 0.5  # Half size when nearing daily loss limit
        
        # Consecutive losses reduction
        if self.consecutive_losses >= 3:
            size *= 0.3
            print(f"  {self.consecutive_losses} consecutive losses, reducing to 30%")
        elif self.consecutive_losses >= 2:
            size *= 0.6
            print(f"  {self.consecutive_losses} consecutive losses, reducing to 60%")
        
        # Enforce minimum order size
        min_order = 1  # Min 1 EUR/R order
        size = max(min_order, min(size, max_per_pos * 1.5))
        
        # Round to 2 decimal places
        size = round(size, 2)
        
        # Also check we don't exceed available balance
        size = min(size, usdc * 0.95)  # Keep 5% buffer
        
        return size, "ok" if size >= min_order else "too_small"
    
    def should_pause_trading(self):
        """Check if trading should be paused"""
        if self.is_paused:
            return True, "trading_paused"
        return False, "ok"
    
    def record_trade_result(self, trade, pnl=None):
        """Record trade outcome for risk management"""
        self.trade_log.append({
            "time": int(time.time()),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "size": trade.get("size"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "pnl": pnl,
            "reason": trade.get("reason", ""),
        })
        
        if pnl is not None:
            self.daily_pnl += pnl
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Auto-pause on large loss
            if self.daily_pnl < -100:
                self.is_paused = True
        
        self.save_state()
    
    def get_stop_loss_price(self, entry_price, side):
        """Calculate stop loss price"""
        pct = self.config["default_stop_loss_pct"] / 100
        if side == "buy":
            return round(entry_price * (1 - pct), 2)
        else:
            return round(entry_price * (1 + pct), 2)
    
    def get_take_profit_price(self, entry_price, side):
        """Calculate take profit price"""
        pct = self.config["default_take_profit_pct"] / 100
        if side == "buy":
            return round(entry_price * (1 + pct), 2)
        else:
            return round(entry_price * (1 - pct), 2)
    
    def validate_market_conditions(self, snapshot):
        """Validate if market conditions are safe to trade"""
        spread = snapshot.get("spread_pct", 0)
        if spread > self.config["max_slippage_pct"] * 2:
            return False, f"spread too high: {spread:.3f}%"
        
        ob_imb = snapshot.get("order_book_imbalance", 0)
        if ob_imb < self.config["order_book_imbalance_min"] or ob_imb > self.config["order_book_imbalance_max"]:
            return False, f"order book too imbalanced: {ob_imb:.2f}"
        
        return True, "ok"