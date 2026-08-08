"""
Revolut X AI-Powered Trading Daemon — 24/7 with real AI reasoning
The daemon gathers market data, I (the AI) analyze and decide, it executes.
This is where we actually make money — not dumb RSI scripts.
"""
import sys, os, json, time, signal, logging, uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revolut_client import RevolutXClient, RevolutXError
from market_analyzer import MarketAnalyzer
from sentiment import SentimentAnalyzer
from risk_manager import RiskManager
from learner import LearningSystem
from self_fund import SelfFundingManager
from ai_brain import AITradingBrain

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler('/tmp/revolut_trader.log'), logging.StreamHandler()])
log = logging.getLogger('revolut-ai')

HEARTBEAT = "/tmp/revolut_trader_heartbeat.txt"

class AITrader:
    def __init__(self):
        self.client = RevolutXClient()
        self.analyzer = MarketAnalyzer(self.client)
        self.sentiment = SentimentAnalyzer()
        self.risk = RiskManager()
        self.learner = LearningSystem()
        self.funding = SelfFundingManager()
        self.brain = AITradingBrain()
        self.running = True
        self.cycle_count = 0
        self.start_time = time.time()
        self.trade_history = []
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'running', False))
        log.info(f"🤖 AI-Powered Revolut X Trader starting")
        log.info(f"   Brain API key: {'✅ LOADED' if self.brain.api_key else '❌ MISSING'}")
    
    def log_hb(self):
        with open(HEARTBEAT, "w") as f:
            json.dump({"ts": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                       "cycle": self.cycle_count, "uptime": int(time.time()-self.start_time)}, f)
    
    def get_balance(self):
        b = self.client.get_balances()
        eur = 0
        holdings = {}
        for entry in b:
            cur, avail = entry["currency"], float(entry["available"])
            if avail > 0:
                if cur == "EUR": eur = avail
                elif cur not in ("USDC", "USD", "EURC", "GBP"): holdings[cur] = avail
        return eur, holdings
    
    def run_cycle(self):
        self.cycle_count += 1
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info(f"\n{'='*60}")
        log.info(f"🤖 AI CYCLE #{self.cycle_count} — {ts}")
        log.info(f"{'='*60}")
        self.log_hb()
        
        # 1. Get portfolio state
        eur_balance, holdings = self.get_balance()
        log.info(f"💰 EUR: €{eur_balance:.2f} | Holdings: {len(holdings)} positions")
        for sym, qty in sorted(holdings.items()):
            log.info(f"   {sym}: {qty:.6f}")
        
        if eur_balance < 1 and not holdings:
            log.warning("No balance to trade")
            return {"status": "no_funds"}
        
        # 2. Check active orders
        try:
            active = self.client.get_active_orders(limit=10).get("data", [])
            if active:
                log.info(f"📋 Active orders: {len(active)}")
                return {"status": "orders_active", "count": len(active)}
        except: pass
        
        # 3. Get comprehensive market data
        log.info("📊 Gathering market intelligence...")
        market_data = {}
        
        # Get top pairs tickers
        try:
            tickers = self.client.get_tickers()
            top_tickers = [t for t in tickers.get("data", []) if t["symbol"].endswith("/EUR")][:10]
            market_data["tickers"] = top_tickers
        except Exception as e:
            market_data["tickers"] = f"error: {e}"
        
        # Get Fear & Greed
        try:
            fng = self.sentiment.get_market_fear_greed()
            market_data["fear_greed"] = fng
        except: pass
        
        # Get BTC analysis (market proxy)
        try:
            btc_snap = self.analyzer.analyze_market_snapshot("BTC-EUR")
            market_data["btc"] = {
                "price": btc_snap.get("last_price"),
                "rsi_1h": btc_snap.get("1h_rsi"),
                "rsi_4h": btc_snap.get("4h_rsi"),
                "trend_1h": btc_snap.get("1h_trend"),
                "trend_4h": btc_snap.get("4h_trend"),
            }
        except: pass
        
        # Get sentiment for top coins
        for coin in ["BTC", "ETH", "SOL"]:
            try:
                s = self.sentiment.get_composite_sentiment(coin)
                market_data[f"{coin}_sentiment"] = s.get("sentiment")
            except: pass
        
        # Get ETH analysis
        try:
            eth_snap = self.analyzer.analyze_market_snapshot("ETH-EUR")
            market_data["eth"] = {"price": eth_snap.get("last_price"), "rsi_1h": eth_snap.get("1h_rsi")}
        except: pass
        
        # Get news headlines
        try:
            news = self.sentiment.get_market_overview()
            market_data["news"] = [s.get("title","")[:80] for s in news.get("top_stories", [])[:3]]
        except: pass
        
        # 4. Check if we should sell anything (check our holdings against current prices)
        if holdings:
            for coin, qty in holdings.items():
                if coin in ("XRP", "ZKJ", "BONK"):  # Skip dust
                    continue
                try:
                    sym = f"{coin}-EUR"
                    snap = self.analyzer.analyze_market_snapshot(sym)
                    rsi = snap.get("1h_rsi", 50)
                    price = snap.get("last_price", 0)
                    value = qty * price if price else 0
                    
                    # AI reasoning for exit: if RSI > 80 (overbought) and we're up, sell
                    if rsi > 80 and value > 2:
                        log.info(f"⚡ AI TRIGGER: {sym} RSI={rsi:.0f} overbought, selling €{value:.2f}")
                        result = self.client.place_order(
                            symbol=sym, side="sell", order_type="market", base_size=str(qty))
                        log.info(f"   Sold: {result.get('data',{}).get('state','?')}")
                        self.learner.record_trade_outcome(sym, "ai_exit", "sell", price, price, qty, 0)
                        self.funding.record_pnl(0)
                        
                except: pass
        
        # 5. Only buy if we have meaningful EUR balance
        if eur_balance < 2:
            log.info(f"💤 Only €{eur_balance:.2f} EUR — waiting for more (or profits to clear)")
            return {"status": "low_balance"}
        
        # 6. AI DECISION TIME — this is where actual intelligence happens
        log.info("🧠 Calling AI brain for trading decision...")
        
        portfolio = {
            "eur_balance": eur_balance,
            "holdings": {k: round(v, 6) for k, v in holdings.items()},
        }
        
        decision = self.brain.analyze_market_decision(market_data, portfolio, self.trade_history)
        
        log.info(f"📝 AI Decision: {decision.get('direction', 'unknown')}")
        reason = decision.get('reason', '')
        if reason:
            for line in reason.split('. '):
                log.info(f"   • {line.strip()}")
        
        # Log the raw analysis for debugging
        if decision.get("raw_analysis"):
            with open("/tmp/ai_brain_debug.txt", "w") as f:
                f.write(decision["raw_analysis"])
        
        # 7. Execute the AI's decision
        if "direction" not in decision:
            log.error(f"❌ AI returned unparseable response, raw content: {decision.get('raw_analysis','')[:300]}")
            return {"status": "parse_failed"}
        
        if decision["direction"] == "buy":
            symbol = decision.get("symbol", "BTC-EUR").replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
            
            # Parse size
            size_str = decision.get("size", "1")
            try:
                size = float(size_str.replace("€", "").replace("$", ""))
            except:
                size = min(2, eur_balance * 0.3)
            
            size = min(size, eur_balance * 0.3)  # Never risk more than 30%
            size = max(1, min(round(size, 2), eur_balance - 0.5))  # Keep buffer
            
            if size >= 1:
                log.info(f"⚡ EXECUTING AI BUY: {symbol} @ €{size}")
                try:
                    result = self.client.place_order(
                        symbol=symbol, side="buy", order_type="market", quote_size=str(size))
                    log.info(f"   Result: {result.get('data',{}).get('state','?')}")
                    
                    self.trade_history.append({
                        "time": int(time.time()), "symbol": symbol, "side": "buy", 
                        "size": size, "reason": decision.get("reason","")[:100]
                    })
                    self.funding.record_pnl(0)
                except Exception as e:
                    log.error(f"   Buy failed: {e}")
            else:
                log.info(f"   Size €{size:.2f} too small, skipping")
        
        elif decision["direction"] == "sell":
            symbol = decision.get("symbol", "").replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
            base = symbol.split("-")[0] if symbol else ""
            if base in holdings and holdings[base] > 0:
                qty = holdings[base]
                log.info(f"⚡ EXECUTING AI SELL: {symbol} ({qty:.6f})")
                try:
                    result = self.client.place_order(
                        symbol=symbol, side="sell", order_type="market", base_size=str(qty))
                    log.info(f"   Result: {result.get('data',{}).get('state','?')}")
                    self.trade_history.append({
                        "time": int(time.time()), "symbol": symbol, "side": "sell",
                        "size": qty, "reason": decision.get("reason","")[:100]
                    })
                except Exception as e:
                    log.error(f"   Sell failed: {e}")
            else:
                log.info(f"   Don't hold {base}, skipping sell")
        
        else:  # WAIT
            log.info(f"💤 AI says WAIT: {decision.get('reason','')[:150]}")
            if decision.get("condition"):
                log.info(f"   Will trade when: {decision['condition']}")
        
        # 8. Summary
        log.info(f"🤖 AI Brain stats: {self.brain.get_cost_summary()}")
        log.info(f"Cycle complete in {time.time() - self.start_time:.0f}s")
        return {"status": decision["direction"]}
    
    def run_forever(self, interval=60):
        """Run 24/7 — checks market every `interval` minutes with AI reasoning"""
        log.info(f"🚀 Entering 24/7 AI-powered trading loop ({interval}min interval)")
        
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                log.error(f"💥 Cycle crashed: {e}", exc_info=True)
                time.sleep(10)
            
            # Sleep with abort check
            for _ in range(interval * 60):
                if not self.running:
                    break
                time.sleep(1)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=60, help="Minutes between AI cycles")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    
    t = AITrader()
    if args.once:
        t.run_cycle()
    else:
        t.run_forever(args.interval)