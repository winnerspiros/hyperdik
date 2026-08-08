"""
Pump Chaser — detects sudden price spikes on low-cap/meme coins and triggers fast trades.
Runs alongside the main day_trader as a sub-minute monitor.

Strategy:
1. Watch a list of volatile meme coins every 15 seconds
2. Detect >20% price spike in <2 minutes
3. Buy market order with tight stop-loss (-5% trailing)
4. Take profit at +30% or time-exit after 60 seconds
5. Only risk small amounts (€5-10 per pump)
"""
import time
import logging
import json
import os
import threading
from datetime import datetime

log = logging.getLogger("PumpChaser")

# Low-cap / meme coins on Revolut X that are prone to pumps
PUMP_WATCHLIST = [
    "MEW/USD", "MEW/USDC", "MEW/EUR",
    "BONK/USD", "BONK/EUR", "BONK/USDC",
    "PEPE/USD", "PEPE/EUR", "PEPE/USDC",
    "MAGIC/USD", "MAGIC/EUR",
    "TRUMP/USD", "TRUMP/EUR", "TRUMP/USDC",
    "PENGU/USD", "PENGU/EUR",
    "WIF/USD", "WIF/EUR",
    "SHIB/USD", "SHIB/EUR", "SHIB/USDC",
]

# Speed settings
CHECK_INTERVAL = 15        # seconds between price checks
PUMP_THRESHOLD_PCT = 20    # % gain to trigger pump detection
TRAILING_STOP_PCT = 5      # % below peak to sell
TIME_EXIT_SECONDS = 60     # max hold time
MAX_PUMP_RISK = 10         # max EUR per pump trade
COOLDOWN_SECONDS = 300     # don't re-buy same coin for 5 min after a trade

class PumpChaser:
    def __init__(self, client, logger=None):
        self.client = client
        self.running = False
        self.thread = None
        self.price_cache = {}  # symbol -> [(timestamp, price), ...]
        self.active_position = None  # current pump trade if any
        self.cooldowns = {}  # symbol -> timestamp of last trade
        self.trade_history = []
        self.last_alert = None  # current pump alert for AI context
        # Use provided logger or set up our own
        global log
        if logger:
            log = logger

    def start(self):
        """Start the pump monitor thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        log.info("🚀 Pump Chaser started — monitoring 21 meme coins every 15s")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _get_price(self, symbol):
        """Get current price for a symbol"""
        try:
            tickers = self.client.get_tickers()
            data = tickers if isinstance(tickers, list) else tickers.get("data", [])
            for t in data:
                if t.get("symbol") == symbol:
                    bid = float(t.get("bid", 0))
                    ask = float(t.get("ask", 0))
                    last = float(t.get("last_price", 0))
                    return last or (bid + ask) / 2 if bid and ask else 0
        except:
            pass
        return 0

    def _detect_pump(self, symbol, current_price):
        """Check if current price indicates a pump in progress"""
        if symbol not in self.price_cache:
            return False
        history = self.price_cache[symbol]
        # Need at least 2 data points over the last 2 minutes
        recent = [(t, p) for t, p in history if time.time() - t < 120]
        if len(recent) < 2:
            return False
        oldest_price = recent[0][1]
        if oldest_price <= 0:
            return False
        gain_pct = (current_price - oldest_price) / oldest_price * 100
        return gain_pct >= PUMP_THRESHOLD_PCT

    def _buy_pump(self, symbol, price):
        """Execute a fast pump buy — market order with tight plan"""
        base = symbol.split("/")[0]
        pair = symbol
        size = min(MAX_PUMP_RISK, 10)  # Always small risk

        try:
            result = self.client.place_order(pair, "buy", order_type="market", quote_size=str(round(size, 2)))
            state = result.get("data", {}).get("state", "unknown")
            fill_price = float(result.get("data", {}).get("price", price))
            log.info(f"⚡ PUMP BUY: {symbol} @ €{fill_price:.6f} (€{size:.2f}) state={state}")

            self.active_position = {
                "symbol": symbol,
                "entry_price": fill_price,
                "peak_price": fill_price,
                "size": size,
                "base": base,
                "entry_time": time.time(),
                "stop_price": fill_price * (1 - TRAILING_STOP_PCT / 100),
            }
            self.trade_history.append({
                "time": datetime.now().isoformat(),
                "symbol": symbol,
                "side": "buy",
                "size": size,
                "price": fill_price,
                "type": "pump_chase",
            })
            self.last_alert = {
                "symbol": symbol,
                "entry_price": fill_price,
                "entry_time": time.time(),
                "status": "active",
            }
            return True
        except Exception as e:
            log.error(f"   Pump buy failed: {e}")
            return False

    def _check_active_position(self):
        """Check trailing stop, take profit, or time exit for active pump positions"""
        if not self.active_position:
            return

        pos = self.active_position
        current = self._get_price(pos["symbol"])
        if current <= 0:
            return

        elapsed = time.time() - pos["entry_time"]
        gain_pct = (current - pos["entry_price"]) / pos["entry_price"] * 100

        # Update peak
        if current > pos["peak_price"]:
            pos["peak_price"] = current
            pos["stop_price"] = current * (1 - TRAILING_STOP_PCT / 100)

        # Check trailing stop
        if current <= pos["stop_price"]:
            log.info(f"   Trailing stop hit: {pos['symbol']} at €{current:.6f} (peak was €{pos['peak_price']:.6f})")
            self._sell_position("trailing_stop")
            return

        # Time exit — sell regardless after TIME_EXIT_SECONDS
        if elapsed >= TIME_EXIT_SECONDS:
            pnl = gain_pct
            log.info(f"   Time exit: {pos['symbol']} held {elapsed:.0f}s, PnL: {pnl:+.1f}%")
            self._sell_position("time_exit")
            return

        # Take profit — if up 30%+, sell
        if gain_pct >= 30:
            log.info(f"   Pump target hit: {pos['symbol']} up {gain_pct:+.1f}%, selling")
            self._sell_position("take_profit")

    def _sell_position(self, reason):
        """Sell the active pump position"""
        pos = self.active_position
        if not pos:
            return
        base = pos["base"]
        pair = pos["symbol"]
        try:
            balance = 0
            balances = self.client.get_balances()
            bal_data = balances if isinstance(balances, list) else balances.get("data", [])
            for b in bal_data:
                if b.get("currency") == pos["base"]:
                    balance = float(b.get("available", 0))
                    break
            if balance > 0:
                result = self.client.place_order(pair, "sell", order_type="market", base_size=str(round(balance, 6)))
                state = result.get("data", {}).get("state", "unknown")
                current_price = self._get_price(pos["symbol"])
                pnl = (current_price - pos["entry_price"]) / pos["entry_price"] * 100 if current_price > 0 else 0
                log.info(f"   Sold {pos['symbol']}: PnL {pnl:+.1f}% reason={reason}")
                self.trade_history.append({
                    "time": datetime.now().isoformat(),
                    "symbol": pos["symbol"],
                    "side": "sell",
                    "size": balance,
                    "price": current_price,
                    "type": "pump_chase",
                    "reason": reason,
                    "pnl_pct": round(pnl, 2),
                })
        except Exception as e:
            log.error(f"   Pump sell error: {e}")

        # Set cooldown
        self.cooldowns[pos["base"]] = time.time()
        self.active_position = None
        if self.last_alert:
            self.last_alert["status"] = f"sold_{reason}"

    def _monitor_loop(self):
        """Main monitor loop — runs every CHECK_INTERVAL seconds"""
        cycle = 0
        while self.running:
            cycle += 1
            try:
                # Check active position first
                self._check_active_position()

                # Check all watched coins for pumps
                for symbol in PUMP_WATCHLIST:
                    base = symbol.split("/")[0]
                    # Skip if in cooldown or already has active position
                    if base in self.cooldowns and time.time() - self.cooldowns[base] < COOLDOWN_SECONDS:
                        continue
                    if self.active_position and self.active_position["base"] == base:
                        continue

                    price = self._get_price(symbol)
                    if price <= 0:
                        continue

                    # Store price history
                    if symbol not in self.price_cache:
                        self.price_cache[symbol] = []
                    self.price_cache[symbol].append((time.time(), price))
                    # Keep last 2 minutes
                    self.price_cache[symbol] = [x for x in self.price_cache[symbol] if time.time() - x[0] < 120]

                    # Check for pump
                    if self._detect_pump(symbol, price):
                        log.info(f"🔥 PUMP DETECTED: {symbol} at €{price:.6f}")
                        self.last_alert = {
                            "symbol": symbol,
                            "price": price,
                            "time": time.time(),
                            "status": "detected",
                        }
                        # Buy immediately — no AI needed for pump chases (speed is critical)
                        self._buy_pump(symbol, price)

            except Exception as e:
                log.debug(f"   Pump monitor cycle error: {e}")

            # Sleep for interval
            for _ in range(CHECK_INTERVAL):
                if not self.running:
                    break
                time.sleep(1)

    def get_pump_context(self):
        """Get pump alert context for the main AI brain"""
        if not self.last_alert:
            return ""
        alert = self.last_alert
        age = time.time() - alert.get("time", 0)
        if age > 60:  # Only show alerts < 1 minute old
            return ""
        sym = alert.get("symbol", "?")
        price = alert.get("price", 0)
        status = alert.get("status", "unknown")
        return f"🔥 PUMP ALERT: {sym} spiked! Price: €{price:.6f} | Status: {status}"

    def get_pump_history(self):
        """Get pump trade history for analysis"""
        recent = [t for t in self.trade_history if t.get("type") == "pump_chase"][-10:]
        if not recent:
            return ""
        lines = ["🔥 PUMP CHASE HISTORY:"]
        for t in recent:
            side = t.get("side", "?")
            sym = t.get("symbol", "?")
            price = t.get("price", 0)
            pnl = t.get("pnl_pct")
            reason = t.get("reason", "")
            if pnl is not None:
                lines.append(f"  {sym} {side} @ €{price:.6f} PnL: {pnl:+.1f}% ({reason})")
            else:
                lines.append(f"  {sym} {side} @ €{price:.6f}")
        return "\n".join(lines)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from revolut_client import RevolutXClient
    c = RevolutXClient()
    chaser = PumpChaser(c)
    print("Pump Chaser module loaded successfully")
    print(f"Watchlist: {len(PUMP_WATCHLIST)} coins")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"Pump threshold: {PUMP_THRESHOLD_PCT}%")
    print(f"Max risk: €{MAX_PUMP_RISK}")
    print(f"Time exit: {TIME_EXIT_SECONDS}s")
    print(f"Trailing stop: {TRAILING_STOP_PCT}%")
    print(f"Cooldown: {COOLDOWN_SECONDS}s")