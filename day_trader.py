"""
Revolut X Day Trading Daemon — AI-powered, all decisions through AI
- 5-minute cycles
- Limit orders at support levels
- Multiple concurrent positions
- Self-healing, fully detached
"""
import logging, os, json, time, signal, sys, urllib.request, ssl, uuid, math
from datetime import datetime, timezone
from collections import defaultdict

TRADABLE_COINS = {"ADA","APT","ARB","ATOM","AVAX","BCH","BNB","BONK","BTC","CRV","DOGE","DOT","ENA","ETH","FIL","HYPE","ICP","INJ","LINK","NEAR","OP","PEPE","SEI","SHIB","SOL","SUI","TON","TRX","UNI","WIF","XRP","LTC"}

TRADER_DIR = "/home/ubuntu/revolut-x-trader"
os.chdir(TRADER_DIR)
sys.path.insert(0, TRADER_DIR)

from revolut_client import RevolutXClient, RevolutXError
from ai_brain import AITradingBrain
from market_regime import classify_regime
from pump_chaser import PumpChaser
import numpy as np

def setup_logging():
    os.makedirs(os.path.join(TRADER_DIR, "logs"), exist_ok=True)
    log = logging.getLogger("DayTrader")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(TRADER_DIR, "logs", "day_trader.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)
    return log

log = setup_logging()

class DayTrader:
    def __init__(self, interval_minutes=5):
        self.interval = interval_minutes
        self.running = True
        self.client = RevolutXClient()
        self.brain = AITradingBrain()
        self.trade_history = []
        self.pending_limit_orders = []
        self.last_sold = []
        self.trailing_stops = {}
        self.pump_chaser = PumpChaser(self.client, logger=log)
        try:
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'running', False))
            signal.signal(signal.SIGINT, lambda *_: setattr(self, 'running', False))
        except (ValueError, OSError):
            pass  # Running in background thread — signals handled by parent

    def get_analyzer(self):
        from market_analyzer import MarketAnalyzer
        return MarketAnalyzer(self.client)

    def get_candles(self, symbol, resolution="1h", limit=24):
        try:
            return self.client.get_candles(symbol, resolution, limit=limit)
        except:
            return []

    def get_candles_df(self, symbol, resolution="1h", hours_back=48):
        try:
            interval_map = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
            from market_analyzer import MarketAnalyzer
            ma = MarketAnalyzer(self.client)
            return ma.get_candles_df(symbol, resolution, hours_back)
        except:
            return None

    def calculate_support_levels(self, symbol):
        """Calculate support/resistance from EMA + volume POC — data-driven, no hardcoded percentages."""
        try:
            df = self.get_candles_df(symbol, "1h", hours_back=48)
            if df is None or len(df) < 10:
                return {"support": None, "resistance": None, "support_distance_pct": None, "buy_zone": None}
            closes = df["close"].values.astype(float)
            highs = df["high"].values.astype(float)
            lows = df["low"].values.astype(float)
            volumes = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(closes))
            current = closes[-1]
            # EMA(12) as dynamic support
            alpha = 2 / (12 + 1)
            ema = closes[0]
            for c in closes:
                ema = alpha * c + (1 - alpha) * ema
            # Volume-weighted support: prices where volume was high relative to recent
            vol_mean = np.mean(volumes[-10:])
            vol_support = np.mean([lows[i] for i in range(-int(len(lows)*0.3), 0) if volumes[i] > vol_mean * 1.2]) if any(volumes[-int(len(lows)*0.3):] > vol_mean * 1.2) else np.min(lows[-10:])
            # Use the closer of EMA and volume support
            ema_dist = abs(current - ema) / current * 100 if current > 0 else 999
            vol_dist = abs(current - vol_support) / current * 100 if current > 0 else 999
            support = ema if ema_dist < vol_dist else vol_support
            support_dist = (current - support) / current * 100 if current > 0 else 999
            buy_zone = support_dist > 0 and support_dist < 3
            resistance = np.mean(highs[-5:])
            return {"support": support, "resistance": resistance, "support_distance_pct": support_dist, "buy_zone": buy_zone}
        except Exception as e:
            log.debug(f"Support calc error: {e}")
            return {"support": None, "resistance": None, "support_distance_pct": None, "buy_zone": None}

    def _get_holdings_and_eur(self):
        holdings = {}
        eur = 0.0
        MIN_TRADE_AMOUNT = {
            "BTC": 0.0001, "ETH": 0.001, "XRP": 0.1, "SOL": 0.01,
            "LINK": 0.01, "ICP": 0.01, "ADA": 1, "DOGE": 1,
            "DOT": 0.01, "AVAX": 0.01, "ARB": 0.1, "APT": 0.01,
            "PEPE": 1000, "BONK": 1000, "SUI": 0.1,
        }
        try:
            balances = self.client.get_balances()
            bal_data = balances if isinstance(balances, list) else balances.get("data", [])
            for b in bal_data:
                curr = b.get("currency", "")
                avail = float(b.get("available", 0))
                if curr == "EUR":
                    eur = avail if avail > 0 else 0.0
                elif curr in TRADABLE_COINS and avail > 0:
                    # Filter out dust that can't be traded
                    min_amount = MIN_TRADE_AMOUNT.get(curr, 0.0001)
                    if avail >= min_amount:
                        holdings[curr] = avail
                    else:
                        log.info(f"   Filtering out dust: {curr} {avail:.8f} (min {min_amount})")
        except Exception as e:
            log.error(f"   Balance fetch error: {e}")
        return holdings, eur

    def _submit_pending_action(self, action_type, symbol, action_data, reasoning):
        """Write a pending action for validator->executor pipeline."""
        import json, uuid
        PENDING_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
        os.makedirs(PENDING_DIR, exist_ok=True)
        batch_id = f"batch_{int(time.time())}"
        record = {
            "type": action_type,
            "symbol": symbol,
            "side": action_type,
            "sell_all": action_data.get("sell_all", True),
            "sell_qty": action_data.get("sell_qty"),
            "size": action_data.get("size"),
            "price": action_data.get("price"),
            "order_type": action_data.get("order_type", "market"),
            "target": action_data.get("target"),
            "stop": action_data.get("stop"),
            "amount": action_data.get("amount"),
            "keep_on_exchange": False,
            "reasoning": reasoning,
            "source": "day_trader",
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        fname = f"{action_type}_{symbol.split('-')[0]}_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
        with open(os.path.join(PENDING_DIR, fname), "w") as f:
            json.dump(record, f, indent=2)
        log.info(f"  Submitted [{batch_id}] {action_type} {symbol}: {reasoning[:120]}")
        return record

    def _execute_ai_sell(self, decision, holdings):
        """Submit an AI sell decision to pending_actions for validation & execution."""
        symbol = decision.get("symbol", "")
        if not symbol:
            log.info("   No symbol for sell, skipping")
            return
        symbol = symbol.replace("/", "-").replace("-EUR-EUR", "-EUR")
        base = symbol.split("-")[0]

        # Scalp cooldown: don't sell if scalper just bought this coin
        try:
            now = time.time()
            with open("/tmp/revolut_scalp_cooldown") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2 and parts[0] == base:
                        bought_at = float(parts[1])
                        if now - bought_at < 300:  # 5 min cooldown
                            log.info(f"   ⏭️  skipping sell {base} — scalp cooldown ({now-bought_at:.0f}s ago)")
                            return
        except FileNotFoundError:
            pass
        qty = holdings.get(base, 0)
        if qty <= 0:
            log.info(f"   No {base} to sell")
            return
        # AI can specify sell_all=True (sell entire position) or sell_fraction (e.g. 0.5 = half)
        sell_all = decision.get("sell_all", True)
        sell_fraction = decision.get("sell_fraction", None)
        if not sell_all and sell_fraction is not None:
            sell_qty = qty * sell_fraction
            log.info(f"   AI partial sell: {sell_fraction*100:.0f}% of {base} ({sell_qty:.4f} of {qty:.4f})")
        else:
            sell_qty = qty
            sell_all = True
        self._submit_pending_action(
            "sell", f"{base}-EUR",
            {"sell_all": sell_all, "size": qty, "sell_qty": sell_qty},
            decision.get("reason", f"Sell {base} per AI decision")
        )
        # Record cooldown locally
        try:
            with open("/tmp/revolut_sold_cooldown", "a") as f:
                f.write(f"{base},{time.time()}\n")
        except: pass
        self.last_sold.append(base)
        self.trade_history.append({"symbol": f"{base}-EUR", "side": "sell", "size": qty, "type": "pending", "time": datetime.now().isoformat()})

    def _execute_ai_buy(self, decision, eur, open_bases=None):
        """Submit an AI buy decision — uses AI confidence to scale size."""
        symbol = decision.get("sym", decision.get("symbol", "BTC-EUR")).replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
        base = symbol.split("-")[0]
        if open_bases and base in open_bases:
            log.info(f"   Already an active order for {base} on exchange — skipping duplicate")
            return
        # Accept compact (sz, ep, tp, sl, conf) and verbose keys
        size_str = str(decision.get("sz", decision.get("size", "10")))
        try:
            size = float(size_str.replace("€", "").replace("EUR", "").strip())
        except:
            size = 10
        # ── AI confidence scales size: low conf = smaller, high conf = full size ──
        ai_conf = float(decision.get("conf", decision.get("confidence", 50)))
        conf_mult = max(0.3, min(1.0, ai_conf / 100.0))  # 30% min, 100% max
        size = size * conf_mult
        if size < 3:
            log.info(f"   AI conf={ai_conf:.0f}% → size €{size:.2f} below min — skipping")
            return
        max_size = max(3, round(eur * conf_mult, 2))
        if size > max_size:
            size = max_size
        # Accept compact entry price key
        entry = decision.get("ep", decision.get("entry", "market"))
        order_type = "limit" if entry and entry != "market" else "market"
        price = entry if entry != "market" else None
        # ── AI-suggested TP/SL: use if provided ──
        tp = decision.get("tp", decision.get("target"))
        sl = decision.get("sl", decision.get("stop"))
        log.info(f"⚡ AI BUY: {symbol} €{size:.2f} conf={ai_conf:.0f}% " + (f"TP={tp} SL={sl}" if tp or sl else ""))
        self._submit_pending_action(
            "buy", symbol,
            {"size": size, "order_type": order_type, "price": price, "target": tp, "stop": sl},
            f"Buy {symbol} €{size:.2f} AI_conf={ai_conf:.0f}%"
        )
        self.trade_history.append({"symbol": symbol, "side": "buy", "size": size, "price": price, "type": order_type, "time": datetime.now().isoformat()})

    def _execute_ai_limit_rebuy(self, decision, eur, targets, open_bases=None):
        """Submit an AI limit order signal to pending_actions."""
        symbol = decision.get("symbol", "").replace("/", "-").split("-")[0] + "-EUR"
        base = symbol.split("-")[0]
        if open_bases and base in open_bases:
            log.info(f"   Already an active order for {base} — skipping duplicate limit rebuy")
            return
        size_str = decision.get("size", "10")
        try:
            size = float(size_str.replace("€", "").replace("EUR", "").strip())
        except:
            size = 10
        max_size = max(3, round(eur * 0.9, 2))
        if size > max_size:
            log.info(f"   AI LIMIT requested €{size:.2f} — capping to €{max_size:.2f} (bal: €{eur:.2f})")
            size = max_size
        price_str = decision.get("limit_price", decision.get("entry", ""))
        if not price_str or size < 3:
            return
        try:
            price = float(price_str)
        except:
            return
        existing = any(symbol.split("-")[0] in o.get("symbol", "") for o in self.pending_limit_orders)
        if existing:
            log.info(f"   Already a pending order for {symbol}, skipping duplicate")
            return
        log.info(f"   AI LIMIT SIGNAL: {symbol} @ €{price:.4f} (€{size:.2f})")
        self._submit_pending_action(
            "buy", symbol,
            {"size": round(size, 2), "order_type": "limit", "price": price, "target": decision.get("target"), "stop": decision.get("stop")},
            decision.get("reason", f"Limit buy {symbol} @ €{price:.4f} per AI decision")
        )

    def _process_self_review(self, holdings):
        """Build enriched context from ALL modules for the AI."""
        enriched_context = ""
        try:
            from data_enricher import build_rich_context
            ec = build_rich_context(holdings, [])
            if ec:
                enriched_context += "\n\n" + str(ec)
        except: pass
        try:
            regime = classify_regime()
            if regime:
                enriched_context += "\n\n" + str(regime)
        except: pass
        try:
            from pattern_detector import detect_patterns
            pattern_coins = list(holdings.keys())
            try:
                eur_tickers = self._get_eur_tickers()[:5]
                for t in eur_tickers:
                    coin = t["symbol"].replace("/","-").split("-")[0]
                    if coin not in pattern_coins:
                        pattern_coins.append(coin)
            except: pass
            for coin in pattern_coins[:6]:
                sym = f"{coin}-EUR"
                df_15m = self.get_candles_df(sym, "15m", hours_back=24)
                df_1h = self.get_candles_df(sym, "1h", hours_back=72)
                df_4h = self.get_candles_df(sym, "4h", hours_back=168)
                if df_15m is not None:
                    tech = detect_patterns(sym, df_15m, df_1h, df_4h)
                    if tech:
                        enriched_context += "\n\n" + str(tech)
        except: pass
        try:
            from self_review import get_ai_review_context
            review = get_ai_review_context(strategy="day_trader")
            if review: enriched_context += "\n\n" + review
        except: pass
        try:
            from correlation_tracker import get_diversification_advice
            corr = get_diversification_advice(list(holdings.keys()), holdings)
            if corr: enriched_context += "\n\n" + corr
        except: pass
        try:
            from volume_profile import get_volume_profile_summary
            for coin in list(holdings.keys())[:3]:
                vp = get_volume_profile_summary(f"{coin}-EUR", self)
                if vp: enriched_context += "\n\n" + vp
        except: pass
        try:
            from market_structure import get_market_structure_summary
            for coin in list(holdings.keys())[:3]:
                ms = get_market_structure_summary(f"{coin}-EUR", self)
                if ms: enriched_context += "\n\n" + ms
        except: pass
        try:
            from kelly_sizing import get_kelly_summary
            kelly = get_kelly_summary("day_trader")
            if kelly: enriched_context += "\n\n" + kelly
        except: pass
        try:
            if self.pending_limit_orders:
                ol = ["📋 PENDING LIMIT ORDERS:"]
                for o in self.pending_limit_orders[:5]:
                    ol.append(f"  BUY {o.get('symbol','?')} @ €{o.get('price',0):.4f} (€{o.get('size',0):.2f})")
                enriched_context += "\n\n" + "\n".join(ol)
            else:
                enriched_context += "\n\n📋 No pending limit orders"
        except: pass
        try:
            from liquidation_data import format_liquidation_for_ai
            liq = format_liquidation_for_ai()
            if liq: enriched_context += "\n\n" + liq
        except: pass
        # Multi-day lows
        try:
            import numpy as np
            lows_lines = ["📉 MULTI-DAY LOWS & LIQUIDATION ZONES:"]
            tracked = ["BTC", "ETH", "SOL", "XRP"] + list(holdings.keys())[:3]
            for coin in set(tracked):
                df = self.get_candles_df(f"{coin}-EUR", "1h", hours_back=170)
                if df is None or len(df) < 24: continue
                closes = df["close"].values.astype(float)
                lows = df["low"].values.astype(float)
                current = closes[-1]
                low_1d = np.min(lows[-24:])
                low_3d = np.min(lows[-72:]) if len(lows) >= 72 else low_1d
                low_7d = np.min(lows) if len(lows) >= 168 else low_3d
                dist_1d = (current - low_1d) / current * 100 if current > 0 else 999
                dist_3d = (current - low_3d) / current * 100 if current > 0 else 999
                dist_7d = (current - low_7d) / current * 100 if current > 0 else 999
                cascade_risk = "none"
                if len(closes) >= 12:
                    recent = closes[-12:]
                    drops = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
                    if drops >= 8:
                        cascade_risk = "high" if dist_7d < 2 else "moderate"
                liq_zone = ""
                if dist_7d < 2: liq_zone = " 🟡 near multi-day low"
                lows_lines.append(f"  {coin}: 1d-low {dist_1d:.1f}% | 3d-low {dist_3d:.1f}% | 7d-low {dist_7d:.1f}% | cascade: {cascade_risk}{liq_zone}")
            if len(lows_lines) > 1: enriched_context += "\n\n" + "\n".join(lows_lines)
        except: pass
        try:
            from mempool_monitor import format_mempool_for_ai
            mp = format_mempool_for_ai()
            if mp: enriched_context += "\n\n" + mp
        except: pass
        try:
            from chart_patterns import detect_all_patterns, format_patterns_for_ai
            for coin in list(holdings.keys())[:3]:
                candles = self.get_candles_df(f"{coin}-EUR", "1h", hours_back=120)
                if candles is not None and len(candles) > 50:
                    cp = np.array(candles["close"].values, dtype=float)
                    hp = np.array(candles["high"].values, dtype=float)
                    lp = np.array(candles["low"].values, dtype=float)
                    pats = detect_all_patterns(cp, hp, lp)
                    if pats: enriched_context += "\n\n" + format_patterns_for_ai(pats)
        except: pass
        try:
            from risk_manager_v2 import CryptoRiskManager
            risk = CryptoRiskManager()
            enriched_context += "\n\n" + risk.get_risk_summary()
        except: pass
        try:
            from pump_detector import format_pump_context
            pump = format_pump_context()
            if pump: enriched_context += "\n\n" + pump
        except: pass
        try:
            from defi_monitor import format_defi_for_ai
            df = format_defi_for_ai()
            if df: enriched_context += "\n\n" + df
        except: pass
        try:
            from onchain_metrics import get_combined_onchain_summary
            oc = get_combined_onchain_summary()
            if oc: enriched_context += "\n\n" + oc
        except: pass
        try:
            from portfolio_rebalancer import format_rebalance_advice
            prices = {k: v.get("current_price", 0) for k, v in holdings.items()}
            pr = format_rebalance_advice(list(holdings.keys()), prices)
            if pr: enriched_context += "\n\n" + pr
        except: pass
        try:
            from self_improving_ai import get_self_improvement_context
            si = get_self_improvement_context("day_trader")
            if si: enriched_context += "\n\n" + si
        except: pass
        try:
            pc = self.pump_chaser.get_pump_context()
            if pc: enriched_context += "\n\n" + pc
        except: pass
        try:
            from funding_signals import get_funding_data, format_multi_price_for_ai
            fd = get_funding_data()
            if fd: enriched_context += "\n\n💸 FUNDING / OI:\n" + "\n".join(f"  • {f}" for f in fd[:5])
            mp = format_multi_price_for_ai(["BTC", "ETH", "XRP", "SOL", "ADA"])
            if mp: enriched_context += "\n\n" + mp
        except: pass
        # Rapid market alerts (fast moves detected by rapid_monitor)
        try:
            from rapid_monitor import get_recent_alerts
            ra = get_recent_alerts()
            if ra: enriched_context += "\n\n" + ra
        except: pass
        # Daily P&L
        try:
            pos_lines = ["📊 DAILY P&L:"]
            total_value = 0
            for coin, qty in list(holdings.items())[:8]:
                snap = self.get_analyzer().analyze_market_snapshot(f"{coin}-EUR")
                price = 0
                if snap: price = float(snap.get("last_price", 0) or snap.get("mid", 0) or 0)
                if price > 0:
                    val = qty * price
                    total_value += val
                    pos_lines.append(f"  {coin}: {qty:.4f} @ €{price:.4f} = €{val:.2f}")
            pos_lines.append(f"  Total portfolio: ~€{total_value:.2f}")
            enriched_context += "\n\n" + "\n".join(pos_lines)
        except: pass
        if enriched_context:
            for s in enriched_context.split("\n"):
                line = s.strip()
                if line and (line.startswith("🟢") or line.startswith("🟡") or line.startswith("🔴")
                    or line.startswith("🌐") or line.startswith("💰") or line.startswith("📊")
                    or line.startswith("📈") or line.startswith("🔬") or line.startswith("🏗")
                    or line.startswith("💧") or line.startswith("⛏") or line.startswith("🛡")
                    or line.startswith("📐") or line.startswith("📋") or line.startswith("🔄")
                    or line.startswith("🧠") or line.startswith("🔥") or line.startswith("📰")
                    or line.startswith("🐋") or line.startswith("📅") or line.startswith("🔗")
                    or line.startswith("🎯") or line.startswith("💸") or line.startswith("📉")
                    or s.startswith("  ") or line.startswith("•")):
                    if s.startswith("  "):
                        log.info(f"  AI_DATA|  {line}")
                    else:
                        log.info(f"  AI_CONTEXT| {line[:150]}")
        return enriched_context

    def _process_ai_decision(self, decision, eur, holdings, targets, open_bases=None):
        """Process the AI's trading decision."""
        direction = decision.get("direction", "wait").lower()
        ai_conf = decision.get("conf", decision.get("confidence", 0))
        ai_hold = decision.get("hold", 0)
        log.info(f"📝 AI Decision: {direction} conf={ai_conf} hold={ai_hold}m")
        raw_reason = decision.get("reason")
        if raw_reason:
            reason_str = ",".join(raw_reason) if isinstance(raw_reason, list) else str(raw_reason)[:80]
            log.info(f"   • {reason_str}")
        # DEBUG: log raw AI output to verify compact format
        raw = decision.get("raw_analysis", "")
        if raw:
            log.info(f"  RAW: {raw[:500]}")
        if direction == "sell":
            self._execute_ai_sell(decision, holdings)
        elif direction == "buy":
            # Confidence gate: skip low-confidence buys
            if ai_conf > 0 and ai_conf < 25:
                log.info(f"   ⏭️  skip buy — AI confidence too low ({ai_conf})")
                return
            self._execute_ai_buy(decision, eur, open_bases=open_bases)
        elif direction == "limit":
            self._execute_ai_limit_rebuy(decision, eur, targets, open_bases=open_bases)
        else:
            pass  # wait

    def _get_eur_tickers(self):
        """Get EUR trading pairs for analysis."""
        try:
            tickers = self.client.get_tickers()
            data = tickers if isinstance(tickers, list) else tickers.get("data", [])
            return [t for t in data if "/EUR" in t.get("symbol", "")]
        except:
            return []

    def run(self):
        log.info("🚀 Day Trader starting — AI-controlled, no hardcoded rules")
        try:
            with open("/tmp/revolut_trader_heartbeat.txt", "w") as f:
                f.write(f"started={time.time()}\n")
        except: pass
        cycle = 0
        consecutive_low = 0
        while self.running:
            cycle += 1
            log.info(f"\n{'='*50}")
            log.info(f"🔄 Cycle #{cycle} — {datetime.now().isoformat()}")
            try:
                self._trading_cycle()
            except Exception as e:
                log.error(f"   Cycle error: {e}", exc_info=True)
            log.info(f"✅ Cycle #{cycle} complete")
            try:
                with open("/tmp/revolut_trader_heartbeat.txt", "w") as f:
                    f.write(f"cycle={cycle} time={time.time()}\n")
            except: pass
            # Adaptive sleep: longer when nothing to do
            try:
                with open("/tmp/revolut_trader_state.json") as f:
                    state = json.load(f)
                has_eur = float(state.get("eur", 0)) >= 3
                has_holdings = bool(state.get("holdings", {}))
            except:
                has_eur = False
                has_holdings = False
            if not has_eur and not has_holdings:
                cycle_sleep = 60 * 60  # check once per hour max
            elif not has_eur and has_holdings:
                cycle_sleep = 15 * 60   # check every 15min — might want to sell
            elif consecutive_low >= 3:
                cycle_sleep = 30 * 60
            elif consecutive_low >= 2:
                cycle_sleep = 15 * 60
            else:
                cycle_sleep = self.interval * 60
            for _ in range(0, cycle_sleep, 5):
                if not self.running:
                    break
                time.sleep(5)
        log.info("🛑 Daemon stopped")

    def _trading_cycle(self):
        """Single trading cycle — get data, build context, ask AI, execute."""
        holdings, eur = self._get_holdings_and_eur()
        log.info(f"💰 EUR: €{eur:.2f} | Holdings: {len(holdings)} positions")
        log.info(f"   Positions: {dict(list(holdings.items())[:8])}" if holdings else "")
        with open("/tmp/revolut_trader_state.json", "w") as f:
            json.dump({"eur": eur, "holdings": holdings, "time": datetime.now().isoformat()}, f)
        if not holdings and eur < 3:
            log.info("   No capital to trade, skipping cycle")
            return
        
        # LOW CAPITAL MODE: EUR < 3 — still run AI for sell decisions
        low_capital = eur < 3
        if low_capital:
            log.info("   🟡 Low capital (€%.2f) — AI can still decide to SELL" % eur)
        
        # Get tickers and build targets (for sell PnL checks + stale order replacement)
        eur_tickers = self._get_eur_tickers()
        targets = []
        for t in eur_tickers[:10]:
            sym = t["symbol"].replace("/", "-")
            levels = self.calculate_support_levels(sym)
            if levels["support"] and levels["buy_zone"]:
                targets.append({"symbol": sym, "levels": levels})
                log.info(f"   {sym:12s} support @ {levels['support']:.6f} ({levels['support_distance_pct']:.1f}% away)")
        # Check existing open orders
        try:
            active_orders = self.client.get_active_orders()
            active_data = active_orders.get("data", active_orders) if isinstance(active_orders, dict) else active_orders
            exchange_orders = {}
            if isinstance(active_data, list):
                for o in active_data:
                    try:
                        sym = o.get("symbol", "").replace("/", "-")
                        base = sym.split("-")[0]
                        oid = o.get("id", "")
                        # Revolut X nests price/size inside order_configuration.limit
                        oconf = o.get("order_configuration", {})
                        limit_cfg = oconf.get("limit", {}) if isinstance(oconf, dict) else {}
                        price = limit_cfg.get("price") or o.get("price", "")
                        amount = limit_cfg.get("quote_size") or limit_cfg.get("base_size") or o.get("amount", "")
                        exchange_orders[base] = {
                            "id": oid,
                            "price": float(price) if price else 0,
                            "amount": float(amount) if amount else 0,
                        }
                    except (KeyError, ValueError, TypeError):
                        continue
                log.info(f"   Existing open orders: {len(active_data)} symbols: {sorted(exchange_orders.keys())}")
                # Auto-replace stale orders if support drifted
                calc_supports = {t["symbol"].split("-")[0]: t["levels"]["support"] for t in targets}
                for base, oinfo in list(exchange_orders.items()):
                    new_support = calc_supports.get(base)
                    if new_support and oinfo.get("price", 0) and abs(oinfo["price"] - new_support) / max(oinfo["price"], 1) > 0.01:
                        try:
                            self.client.cancel_order(oinfo["id"])
                            log.info(f"   Replaced {base}: old €{oinfo['price']:.4f} -> new €{new_support:.4f}")
                            del exchange_orders[base]
                        except:
                            pass
                self.pending_limit_orders = [{"symbol": f"{b}-EUR", "price": info["price"], "size": info["amount"]} for b, info in exchange_orders.items() if info.get("price", 0) > 0]
        except:
            pass
        # Super dip opportunities
        log.info("🔥 Checking for super dip opportunities...")
        super_dips = []
        for t in eur_tickers:
            sym = t["symbol"].replace("/", "-")
            try:
                chg_24h = float(t.get("change_24h", 0))
                if chg_24h <= -12:
                    price = (float(t.get("bid", 0)) + float(t.get("ask", 0))) / 2 if t.get("bid") and t.get("ask") else 0
                    if price > 0:
                        levels = self.calculate_support_levels(sym)
                        if levels["support"] and levels["support_distance_pct"] < 5:
                            super_dips.append({"symbol": sym, "price": price, "drop": chg_24h, "support": levels["support"]})
                            log.info(f"   Potential dip: {sym} down {chg_24h:.1f}%")
            except:
                pass
        # Get psychological levels
        current_prices = {}
        for coin in holdings:
            try:
                snap = self.get_analyzer().analyze_market_snapshot(f"{coin}-EUR")
                if snap:
                    current_prices[coin] = snap.get("last_price") or snap.get("mid") or snap.get("ask") or 0
            except:
                pass
        # Calculate PnL for each position
        pnl_summary = {}
        total_est = eur + sum(h * current_prices.get(c, 0) for c, h in holdings.items())
        for coin, qty in holdings.items():
            try:
                total_cost = 0
                total_qty = 0
                for th in self.trade_history:
                    if th.get("symbol", "").startswith(coin) and th.get("side") == "buy":
                        total_cost += float(th.get("price", 0)) * float(th.get("size", 0))
                        total_qty += float(th.get("size", 0))
                    elif th.get("symbol", "").startswith(coin) and th.get("side") == "sell":
                        total_qty -= float(th.get("size", 0))
                entry = total_cost / total_qty if total_qty > 0 else 0
                current = current_prices.get(coin, 0)
                if entry > 0 and current > 0:
                    pnl = (current - entry) / entry * 100
                    pnl_summary[coin] = {"entry": entry, "current": current, "pnl_pct": round(pnl, 2), "qty": qty, "value": current * qty}
            except:
                pass
        if pnl_summary:
            parts = []
            for c, p in pnl_summary.items():
                parts.append(f"{c}: {p['pnl_pct']:+.2f}%")
            log.info(f"   PnL: {' | '.join(parts)}")
        # Track recently sold
        recently_sold = set()
        try:
            if os.path.exists("/tmp/revolut_sold_cooldown"):
                with open("/tmp/revolut_sold_cooldown") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 2:
                            c, t = parts
                            recently_sold.add(c)
        except: pass
        # ── CIRCUIT BREAKER: check for rapid drops BEFORE AI call ──
        circuit_breakers = []
        for coin, qty in holdings.items():
            try:
                snap = self.get_analyzer().analyze_market_snapshot(f"{coin}-EUR")
                if not snap: continue
                current = float(snap.get("last_price", 0) or snap.get("mid", 0) or 0)
                if current <= 0: continue
                pnl_info = pnl_summary.get(coin, {})
                entry = pnl_info.get("entry", 0)
                if entry > 0:
                    drop_pct = (current - entry) / entry * 100
                    if drop_pct <= -5.0:
                        circuit_breakers.append({"coin": coin, "drop": drop_pct, "price": current})
                        log.warning(f" CIRCUIT BREAKER: {coin} dropped {drop_pct:.1f}% from entry! Submitting sell signal")
                        self._submit_pending_action(
                            "sell", f"{coin}-EUR",
                            {"sell_all": True, "size": qty},
                            f"Circuit breaker: {coin} dropped {drop_pct:.1f}% from entry (€{current:.4f})"
                        )
            except: pass

        # ── SMART AI SKIP: no EUR + no holdings = no AI call needed ──
        if not holdings and eur < 3:
            log.info("  No capital and no holdings — skipping AI call (saves tokens)")
            return

        # Build AI context from ALL modules
        is_weekend = datetime.now(timezone.utc).weekday() >= 5
        enriched_context = self._process_self_review(holdings)
        # Inject day-of-week context so AI knows actual market conditions
        if is_weekend:
            enriched_context = "📅 WEEKEND — lower volume, wider spreads. Be extra cautious.\n\n" + enriched_context
        else:
            enriched_context = "📅 WEEKDAY — normal market conditions and liquidity.\n\n" + enriched_context
        # Add strategy analysis (martingale, grid, partial exits, coin ranking)
        try:
            from strategy_analyzer import get_full_strategy_context
            strat_tickers = []
            for t in eur_tickers:
                d = t.get("data", t) if isinstance(t, dict) else {}
                strat_tickers.append({
                    "symbol": d.get("symbol", t.get("symbol", "")),
                    "bid": d.get("bid", t.get("bid", 0)),
                    "ask": d.get("ask", t.get("ask", 0)),
                    "volume_24h": d.get("volume_24h", t.get("volume_24h", 0)),
                    "change_24h": d.get("change_24h", t.get("change_24h", 0))
                })
            sa = get_full_strategy_context(
                holdings, current_prices, pnl_summary, eur,
                strat_tickers,
                {t["symbol"]: t["levels"] for t in targets}
            )
            if sa:
                enriched_context += "\n\n" + sa
        except Exception:
            pass
        # Low-capital hint for AI
        if low_capital:
            enriched_context += f"\n\n🟡 LOW CAPITAL MODE: €{eur:.2f} EUR available — focus on SELLING positions to free up cash"
        # Build market_data and portfolio for AI
        market_data = {
            "tickers": eur_tickers,
            "holdings": holdings,
            "eur_balance": eur,
            "support_levels": {t["symbol"]: t["levels"] for t in targets},
            "pnl_summary": pnl_summary,
            "super_dips": super_dips,
            "recently_sold": list(recently_sold),
        }
        portfolio = {"eur_balance": eur, "holdings": holdings, "total_est": total_est}
        decision = self.brain.analyze_market_decision(market_data, portfolio, self.trade_history, rich_context=enriched_context)
        # Get base symbols with active exchange orders for duplicate check
        try:
            open_bases = set(exchange_orders.keys())
        except:
            open_bases = None
        self._process_ai_decision(decision, eur, holdings, targets, open_bases=open_bases)
        # Log AI stats
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(TRADER_DIR, "data", "trader.db"))
            calls = conn.execute("SELECT COUNT(*) FROM decisions WHERE DATE(timestamp) = DATE('now')").fetchone()[0]
            cost = calls * 0.0005
            log.info(f"📊 AI Brain stats: AI calls: {calls}, estimated cost: €{cost:.4f}")
            conn.close()
        except:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=5, help="Minutes between cycles")
    args = parser.parse_args()
    trader = DayTrader(interval_minutes=args.interval)
    trader.run()