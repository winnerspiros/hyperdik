"""
Revolut X Autonomous Trading Agent
Main entry point — scans markets, analyzes, decides, and executes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revolut_client import RevolutXClient, RevolutXError
from market_analyzer import MarketAnalyzer
from sentiment import SentimentAnalyzer
from strategies import StrategyOrchestrator
from risk_manager import RiskManager
import time
import json

class TradingAgent:
    def __init__(self, paper_mode=True):
        self.client = RevolutXClient()
        self.analyzer = MarketAnalyzer(self.client)
        self.sentiment = SentimentAnalyzer()
        self.strategies = StrategyOrchestrator()
        self.risk = RiskManager()
        self.paper_mode = paper_mode
        self.risk.load_state()
    
    def scan_opportunities(self, top_pairs=15):
        """Scan the market for the best trading opportunities"""
        print("=== MARKET SCAN ===")
        
        # Get top performing pairs
        print(f"Scanning top {top_pairs} USD pairs...")
        opportunities = self.analyzer.get_market_opportunities(top_pairs)
        
        if not opportunities:
            print("No strong opportunities found")
            return []
        
        print(f"Found {len(opportunities)} potential opportunities")
        return opportunities
    
    def analyze_opportunity(self, opp):
        """Deep analysis of a single opportunity"""
        symbol = opp["symbol"]
        print(f"\n  Analyzing {symbol} @ ${opp['price']}")
        print(f"  Tech score: {opp['total']} | Direction: {opp['direction']}")
        print(f"  RSI 1h: {opp['rsi_1h']:.0f} | RSI 4h: {opp['rsi_4h']:.0f}")
        print(f"  Trend 1h: {opp['trend_1h']} | Trend 4h: {opp['trend_4h']}")
        
        # Get detailed snapshot for strategy evaluation
        try:
            snapshot = self.analyzer.analyze_market_snapshot(symbol)
        except Exception as e:
            print(f"  Failed to get snapshot: {e}")
            return None
        
        # Get sentiment (only for major coins)
        coin = symbol.split("-")[0]
        sentiment = None
        if coin in ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"]:
            try:
                sentiment = self.sentiment.get_composite_sentiment(coin)
                print(f"  Sentiment ({coin}): {sentiment['sentiment']} ({sentiment['composite']:+.1f})")
            except Exception as e:
                print(f"  Sentiment unavailable: {e}")
        
        # Run strategy orchestrator
        decision = self.strategies.evaluate(snapshot, sentiment)
        print(f"  Strategy decision: {decision['direction']} (conf: {decision['confidence']}%)")
        if decision["reasons"]:
            for r in decision["reasons"]:
                print(f"    - {r}")
        
        return {
            "opportunity": opp,
            "snapshot": snapshot,
            "sentiment": sentiment,
            "decision": decision,
        }
    
    def make_trade(self, symbol, side, confidence, price, atr=None):
        """Execute a trade with proper risk management"""
        # Validate market conditions
        paused, reason = self.risk.should_pause_trading()
        if paused:
            print(f"  Trading paused: {reason}")
            return None
        
        # Calculate position size
        size, status = self.risk.calculate_position_size(
            self.client, symbol, confidence, atr
        )
        
        if status != "ok":
            print(f"  Position size check failed: {status}")
            return None
        
        if size < 1:
            print(f"  Position too small: ${size:.2f}")
            return None
        
        print(f"  Position size: ${size:.2f}")
        
        # Calculate stop loss and take profit
        sl_price = self.risk.get_stop_loss_price(price, side)
        tp_price = self.risk.get_take_profit_price(price, side)
        
        print(f"  Entry: ${price:.2f} | SL: ${sl_price:.2f} | TP: ${tp_price:.2f}")
        
        if self.paper_mode:
            print(f"  [PAPER] Would place {side} order: ${size} of {symbol}")
            trade = {
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "reason": "paper_mode",
            }
            return trade
        
        # REAL EXECUTION
        try:
            # Place the order
            coin = symbol.split("-")[0]
            result = self.client.place_order(
                symbol=symbol,
                side=side,
                order_type="market",
                quote_size=str(size),
            )
            
            print(f"  Order placed: {json.dumps(result, indent=2)[:200]}")
            
            if result.get("data", {}).get("state") in ("new", "partially_filled"):
                venue_id = result["data"]["venue_order_id"]
                
                # Could place SL/TP as conditional orders here
                # Note: Revolut X supports TP/SL but requires the original order to be filled first
                
                trade = {
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "entry_price": price,
                    "venue_order_id": venue_id,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "timestamp": int(time.time()),
                }
                return trade
            
        except RevolutXError as e:
            print(f"  Order failed: {e}")
        except Exception as e:
            print(f"  Unexpected error: {e}")
        
        return None
    
    def run_cycle(self, top_pairs=15):
        """Run one complete trading cycle"""
        print(f"\n{'='*60}")
        print(f"TRADING CYCLE - {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        print(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"{'='*60}")
        
        # 1. Check if we should be trading
        paused, reason = self.risk.should_pause_trading()
        if paused:
            print(f"Trading is paused: {reason}")
            self.risk.save_state()
            return {"status": "paused", "reason": reason}
        
        # 2. Check for existing positions
        try:
            active = self.client.get_active_orders(limit=10)
            active_orders = active.get("data", [])
            if active_orders:
                print(f"Active orders: {len(active_orders)}")
                # Don't enter new positions if we have active orders
                # (we only want to check if we need to exit)
                for order in active_orders:
                    print(f"  {order['side']} {order['symbol']} @ {order.get('price', 'market')} ({order['status']})")
                    print(f"  Filled: {order['filled_quantity']}/{order['quantity']}")
        except Exception as e:
            print(f"Could not check active orders: {e}")
        
        # 3. Check portfolio
        try:
            usdc = self.risk.get_available_usdc(self.client)
            print(f"\nAvailable USDC: ${usdc:.2f}")
        except Exception as e:
            print(f"Could not check portfolio: {e}")
            return {"status": "error", "reason": str(e)}
        
        # 4. Scan for opportunities
        opportunities = self.scan_opportunities(top_pairs)
        if not opportunities:
            print("\nNo opportunities found. Getting market overview...")
            try:
                overview = self.sentiment.get_market_overview()
                fng = overview.get("fear_greed", {})
                print(f"Fear & Greed: {fng.get('value')} ({fng.get('classification')})")
                print(f"Overall sentiment: {overview.get('overall')}")
            except Exception as e:
                print(f"Could not get overview: {e}")
            return {"status": "no_opportunities"}
        
        # 5. Deep analyze top opportunities
        print("\n=== DEEP ANALYSIS ===")
        best_trade = None
        
        for opp in opportunities[:5]:
            try:
                analysis = self.analyze_opportunity(opp)
                if analysis and analysis["decision"]["direction"] != "neutral":
                    dec = analysis["decision"]
                    if dec["confidence"] >= 50 and self._is_better_than(dec, best_trade):
                        best_trade = {
                            "analysis": analysis,
                            "decision": dec,
                        }
                        print(f"  --> CANDIDATE: {opp['symbol']} {dec['direction']} ({dec['confidence']}%)")
            except Exception as e:
                print(f"  Error analyzing {opp.get('symbol', '?')}: {e}")
                continue
        
        # 6. Execute best trade
        if best_trade:
            print("\n=== EXECUTING BEST TRADE ===")
            dec = best_trade["decision"]
            opp = best_trade["analysis"]["opportunity"]
            
            result = self.make_trade(
                symbol=opp["symbol"],
                side=dec["direction"],
                confidence=dec["confidence"],
                price=float(opp["price"]),
                atr=opp.get("atr_1h"),
            )
            
            if result:
                self.risk.record_trade_result(result)
                print(f"\nTrade placed: {result}")
                return {"status": "trade_placed", "trade": result}
            else:
                print("Could not place trade")
                return {"status": "trade_failed"}
        else:
            print("\nNo strong enough signals for any pair")
            return {"status": "no_strong_signals"}
    
    def _is_better_than(self, dec, best):
        """Compare two decisions"""
        if best is None:
            return True
        # Higher confidence is better
        return dec["confidence"] > best["decision"]["confidence"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Revolut X Trading Agent")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (default: paper)")
    parser.add_argument("--pairs", type=int, default=15, help="Number of pairs to scan")
    parser.add_argument("--cycle", action="store_true", help="Run one cycle")
    parser.add_argument("--scan", action="store_true", help="Just scan, no trades")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    parser.add_argument("--loop-minutes", type=int, default=0, help="Run continuously with N minute intervals")
    
    args = parser.parse_args()
    
    agent = TradingAgent(paper_mode=not args.live)
    
    if args.status:
        try:
            b = agent.client.get_balances()
            print(json.dumps(b, indent=2))
        except Exception as e:
            print(f"Error: {e}")
        return
    
    if args.scan:
        opps = agent.scan_opportunities(args.pairs)
        if opps:
            print("\nOpportunities:", json.dumps(opps[:5], indent=2, default=str))
        return
    
    if args.cycle:
        result = agent.run_cycle(args.pairs)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
        return
    
    if args.loop_minutes > 0:
        print(f"Running every {args.loop_minutes} minutes. Ctrl+C to stop.")
        while True:
            try:
                agent.run_cycle(args.pairs)
                print(f"\nSleeping {args.loop_minutes} minutes...")
                time.sleep(args.loop_minutes * 60)
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                print(f"Cycle error: {e}")
                time.sleep(60)
        return
    
    # Default: run one cycle
    result = agent.run_cycle(args.pairs)
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
