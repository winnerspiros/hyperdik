"""
Parallel scalper — checks for dip opportunities every 60 seconds
V4: AI-evaluated buy/sell decisions — every trade goes through the AI brain
"""
import sys, os, time, json, uuid, logging, json
from datetime import datetime, timedelta, timezone
sys.path.insert(0, "/home/ubuntu/hyperliquid-trader")
os.chdir("/home/ubuntu/hyperliquid-trader")
from revolut_client import RevolutXClient, RevolutXError
from market_analyzer import MarketAnalyzer
from ai_brain import AITradingBrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scalper")

client = RevolutXClient()
analyzer = MarketAnalyzer(client)
brain = AITradingBrain()

PAIRS = ["ADA-EUR", "DOGE-EUR", "XRP-EUR", "DOT-EUR", "APT-EUR", "LINK-EUR", "AVAX-EUR", "ARB-EUR"]

# Track recently sold coins so scalper doesn't rebuy immediately
RECENTLY_SOLD = {}  # coin -> timestamp

# Track positions bought by this scalper: coin -> {"qty": float, "entry": float, "time": float}
SCALP_POSITIONS = {}

log.info("🚀 Parallel Scalper V4 — AI evaluates every trade decision")
cycle_count = 0
while True:
    try:
        balances = client.get_balances()
        eur = 0
        held_coins = set()
        MIN_TRADE_AMOUNT = {
            "BTC": 0.0001, "ETH": 0.001, "XRP": 0.1, "SOL": 0.01,
            "LINK": 0.01, "ICP": 0.01, "ADA": 1, "DOGE": 1,
            "DOT": 0.01, "AVAX": 0.01, "ARB": 0.1, "APT": 0.01,
            "PEPE": 1000, "BONK": 1000, "SUI": 0.1,
        }
        for b in balances:
            avail = float(b["available"])
            if b["currency"] == "EUR":
                eur = avail
            elif avail > 0.001 and b["currency"] not in ("USDC", "USD", "EURC", "GBP", "USDT"):
                min_amount = MIN_TRADE_AMOUNT.get(b["currency"], 0.0001)
                if avail >= min_amount:
                    held_coins.add(b["currency"])

        if eur < 5:
            time.sleep(60)
            continue

        now = time.time()

        # --- SELL MANAGEMENT: ask AI before selling scalp positions ---
        for coin in list(SCALP_POSITIONS.keys()):
            pos = SCALP_POSITIONS[coin]
            try:
                snap = analyzer.analyze_market_snapshot(f"{coin}-EUR")
                if not snap:
                    continue
                current = snap.get("current_price", 0)
                if current <= 0:
                    continue
                entry = pos["entry"]
                pnl = (current - entry) / entry * 100 if entry > 0 else 0
                age_hours = (now - pos["time"]) / 3600

                # Ask AI if we should sell — compact JSON
                sell_prompt = (
                    f"SCALP:{coin} entry={entry:.4f} now={current:.4f} PnL={pnl:+.1f}% age={age_hours:.1f}h\n"
                    'JSON:{"d":"sell|hold","conf":0-100,"rf":["take_profit","cut_loss","free_capital","more_upside","bouncing","dying"]}'
                )
                ai_verdict = brain._call_llm(
                    "Scalp exit. Output ONLY valid JSON.",
                    sell_prompt,
                    temperature=0.2
                )
                ai_content = ai_verdict.get("content", '{"d":"hold"}')
                try:
                    import json as _j2
                    sd = _j2.loads(ai_content.strip().lstrip("```json").rstrip("```").strip())
                    should_sell = sd.get("d", "hold") == "sell"
                    sell_conf = int(sd.get("conf", 50))
                    # Use rf flags: dying → force sell, bouncing → hold
                    rf = sd.get("rf", [])
                    if isinstance(rf, str): rf = [rf]
                    if should_sell and sell_conf < 20 and "dying" not in [str(f).lower() for f in rf]:
                        should_sell = False
                        log.info(f"   ⏭️  {coin}: low sell conf={sell_conf} and not dying — holding")
                except:
                    should_sell = False  # JSON parse failed, don't guess

                if should_sell:
                    qty = pos["qty"]
                    PENDING_DIR = "/home/ubuntu/revolut-x-trader/data/pending_actions"
                    os.makedirs(PENDING_DIR, exist_ok=True)
                    record = {
                        "type": "sell", "symbol": f"{coin}-EUR", "side": "sell",
                        "sell_all": True, "size": qty,
                        "order_type": "market", "source": "scalper",
                        "reasoning": f"Sell {coin} scalp position (PnL {pnl:+.1f}%)",
                        "batch_id": f"scalp_{int(time.time())}",
                        "created_at": datetime.now().isoformat(),
                    }
                    fname = f"sell_{coin}_{int(time.time())}.json"
                    with open(os.path.join(PENDING_DIR, fname), "w") as f:
                        json.dump(record, f, indent=2)
                    log.info(f" Submitted SCALP SELL {coin} PnL {pnl:+.1f}% — waiting for validator")
                    RECENTLY_SOLD[coin] = now
                    with open("/tmp/revolut_sold_cooldown", "a") as f:
                        f.write(f"{coin},{now}\n")
                    del SCALP_POSITIONS[coin]
                else:
                    log.debug(f"   AI says HOLD {coin} ({pnl:+.1f}%)")
            except Exception as e:
                log.debug(f"   {coin} scalp check error: {e}")

        # Remove tracking for coins no longer held (sold by day trader)
        for coin in list(SCALP_POSITIONS.keys()):
            if coin not in held_coins:
                log.info(f"   📝 {coin} scalp position sold externally, removing tracking")
                del SCALP_POSITIONS[coin]
                RECENTLY_SOLD[coin] = now

        # Clean old cooldowns (30 min expiry)
        for coin in list(RECENTLY_SOLD.keys()):
            if now - RECENTLY_SOLD[coin] > 1800:
                del RECENTLY_SOLD[coin]

        # Also check shared cooldown file from day trader
        try:
            if os.path.exists("/tmp/revolut_sold_cooldown"):
                with open("/tmp/revolut_sold_cooldown", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and "," in line:
                            parts = line.split(",")
                            if len(parts) == 2:
                                coin, ts = parts[0], float(parts[1])
                                if now - ts < 1800:
                                    RECENTLY_SOLD[coin] = max(RECENTLY_SOLD.get(coin, 0), ts)
                with open("/tmp/revolut_sold_cooldown", "w") as f:
                    for coin, ts in RECENTLY_SOLD.items():
                        if now - ts < 1800:
                            f.write(f"{coin},{ts}\n")
        except: pass

        # Include rapid alerts in buy prompt context
        rapid_context = ""
        try:
            import rapid_monitor
            rapid_context = rapid_monitor.get_recent_alerts(max_alerts=3)
        except: pass

        bought = 0
        recently_sold_cooldown = set()
        try:
            if os.path.exists("/tmp/revolut_sold_cooldown"):
                with open("/tmp/revolut_sold_cooldown") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 2:
                            c, t = parts
                            if now - float(t) < 1800:
                                recently_sold_cooldown.add(c)
        except: pass

        for pair in PAIRS:
            if bought >= 2:
                break
            base = pair.split("-")[0]

            # Skip checks
            if base in held_coins or base in SCALP_POSITIONS or base in recently_sold_cooldown or base in RECENTLY_SOLD:
                continue

            try:
                candles = analyzer.get_candles_df(pair, "15m", hours_back=12)
                if candles is None or len(candles) < 10:
                    continue

                closes = candles["close"].values.astype(float)
                current = closes[-1]
                lows = sorted(candles["low"].astype(float).values[-15:])
                support = sum(lows[:3]) / 3 if len(lows) >= 3 else min(lows)
                dist = (current - support) / current * 100 if current > 0 else 999

                # Rapid check: is the pair near support? (no AI — just a pre-filter)
                if dist < 3.0:
                    # SKIP AI call if max trade size is below €3 minimum
                    max_trade = eur * 0.3
                    if max_trade < 3:
                        log.debug(f"   EUR too low for scalp (€{eur:.2f}, max trade €{max_trade:.2f} < €3 min) — skipping AI call")
                        continue

                    # Determine actual day-of-week for accurate AI context
                    is_weekend = datetime.now(timezone.utc).weekday() >= 5
                    weekday_str = "Weekend low-volume conditions" if is_weekend else "Weekday — normal volume conditions"

                    # Ask AI: scalp buy — compact JSON
                    buy_prompt = (
                        f"SCALP:{pair} price={current:.4f} support={support:.4f}({dist:.1f}%away) "
                        f"EUR={eur:.2f} pos={len(held_coins)} {weekday_str}\n"
                        + (f"RAPID:{rapid_context}\n" if rapid_context else "")
                        + 'JSON:{"d":"buy|skip","sz":EUR_amount,"conf":0-100,"rf":["real_dip","noise","bearish","bounce_confident","low_vol","risky"]}'
                    )

                    ai_verdict = brain._call_llm(
                        "Scalp entry at support. Output ONLY valid JSON.",
                        buy_prompt,
                        temperature=0.2
                    )
                    ai_content = ai_verdict.get("content", '{"d":"skip"}')

                    # Parse AI response — compact keys
                    try:
                        ai_json = json.loads(ai_content.strip().lstrip("```json").rstrip("```").strip())
                        should_buy = ai_json.get("d", "skip") == "buy"
                        ai_conf = int(ai_json.get("conf", 50))
                        ai_size = float(ai_json.get("sz", eur * 0.15))
                        # Use rf flags: noise→skip, risky→half size, low_vol→reduce
                        ai_rf = ai_json.get("rf", [])
                        if isinstance(ai_rf, str): ai_rf = [ai_rf]
                        ai_rf_lower = [str(f).lower() for f in ai_rf]
                        if should_buy and "noise" in ai_rf_lower:
                            should_buy = False
                            log.info(f"   ⏭️  {pair}: rf=noise — skipping")
                        if should_buy and "risky" in ai_rf_lower:
                            ai_size *= 0.5
                            log.info(f"   ⚠️  {pair}: rf=risky — halving size to €{ai_size:.2f}")
                        if should_buy and "low_vol" in ai_rf_lower:
                            ai_size *= 0.7
                        # Confidence gate: skip low-confidence scalp attempts
                        if should_buy and ai_conf < 25:
                            log.info(f"   ⏭️  {pair} scalp: AI says buy but conf={ai_conf} — skipping")
                            should_buy = False
                    except (json.JSONDecodeError, TypeError, ValueError):
                        should_buy = '"buy"' in ai_content.lower()
                        ai_size = eur * 0.15

                    # Execute buy if AI says yes (runs for both parse paths)
                    if should_buy:
                        # Rapid guard: max 30% per trade (safety cap, not AI override)
                        size = min(ai_size, eur * 0.3)
                        if size < 3:
                            log.info(f"   AI suggested {ai_size:.2f} but min trade is €3 — skipping")
                            continue
                        record = {
                            "type": "buy", "symbol": pair, "side": "buy",
                            "sell_all": False, "size": round(size, 2),
                            "order_type": "market", "source": "scalper",
                            "reasoning": f"Scalp buy {pair} at support (€{current:.4f})",
                            "batch_id": f"scalp_{int(time.time())}",
                            "created_at": datetime.now().isoformat(),
                        }
                        PENDING_DIR = "/home/ubuntu/revolut-x-trader/data/pending_actions"
                        fname = f"buy_{base}_{int(time.time())}.json"
                        with open(os.path.join(PENDING_DIR, fname), "w") as f:
                            json.dump(record, f, indent=2)
                        log.info(f" Submitted SCALP BUY {pair} €{size:.2f} — waiting for validator")
                        SCALP_POSITIONS[base] = {
                            "qty": size / current if current > 0 else 0,
                            "entry": current,
                            "time": now,
                        }
                        # Protect from day_trader selling immediately
                        with open("/tmp/revolut_scalp_cooldown", "a") as f:
                            f.write(f"{base},{now}\n")
                        bought += 1
                        eur -= size
                    else:
                        log.debug(f"🤖 AI says skip {pair} — {ai_content[:80]}")
            except Exception as e:
                log.debug(f"   {pair} scan error: {e}")

        if SCALP_POSITIONS:
            pos_summary = ", ".join(f"{c}: €{p['qty']*p['entry']:.1f}" for c, p in SCALP_POSITIONS.items())
            log.debug(f"📦 Scalp positions: {pos_summary}")

        # Heartbeat every 10 cycles (INFO) so we know scalper is alive
        if cycle_count % 10 == 0:
            log.info(f"💓 Scalper heartbeat — cycle #{cycle_count} | EUR €{eur:.2f} | {len(SCALP_POSITIONS)} scalp positions")
        cycle_count += 1
        time.sleep(60)
    except Exception as e:
        log.error(f"Scalper cycle error: {e}")
        time.sleep(60)