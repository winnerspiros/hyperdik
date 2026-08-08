#!/usr/bin/env python3
"""
Action Executor v2 — AI-powered execution of approved actions.
- Reads approved actions from data/approved_actions/
- Consults cheap AI (Gemini Flash Lite) before each execution
- AI decides: market vs limit order, price tweaks, timing, or cancel
- Executes via Revolut client when AI gives the green light
- Reports results back to calendar
- Runs every 1 minute via cron
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EXECUTOR] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "executor.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Executor")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
APPROVED_DIR = os.path.join(TRADER_DIR, "data", "approved_actions")
COMPLETED_DIR = os.path.join(TRADER_DIR, "data", "completed_actions")
os.makedirs(APPROVED_DIR, exist_ok=True)
os.makedirs(COMPLETED_DIR, exist_ok=True)

# ── Lazy imports ─────────────────────────────────────────────────────────────

def _import(name):
    try:
        return __import__(name, fromlist=[""])
    except Exception as e:
        return None

rev_client_mod = _import("revolut_client")
data_enricher_mod = _import("data_enricher")

# ── Order Tracking (TP/SL fill verification) ──────────────────────────────────

# Hyperliquid price decimals per coin — prevents float_to_wire rounding errors
_PX_DECIMALS_PRICE_MAP: dict[str, int] = {
    # Per-coin known tick sizes that don't follow the price-based heuristic
    "BTC": 1, "ETH": 2, "SOL": 2, "LINK": 3, "SUI": 4, "DOT": 3,
    "AVAX": 2, "DOGE": 5, "ADA": 4, "XRP": 4, "ARB": 4, "OP": 3,
    "PEPE": 8, "BONK": 8, "WIF": 4, "APT": 3, "ICP": 3,
    "LTC": 2, "TIA": 3, "INJ": 3,
}
# Default heuristic: determine tick size from price range
# Price > $1000 → tick $0.1 (1 dp), $100-$1000 → $0.1 (1 dp),
# $10-$100 → $0.01 (2 dp), $1-$10 → $0.001 (3 dp),
# $0.1-$1 → $0.0001 (4 dp), < $0.1 → 5+ dp

def _get_price_decimals(coin: str, price: float = 0.0) -> int:
    """Get correct price decimals for Hyperliquid wire protocol.
    
    Checks per-coin map first, then falls back to price-based heuristic.
    """
    coin_upper = coin.upper()
    if coin_upper in _PX_DECIMALS_PRICE_MAP:
        return _PX_DECIMALS_PRICE_MAP[coin_upper]
    p = abs(price) if price > 0 else 0
    if p <= 0:
        return 4  # Unknown coin, conservative default
    if p >= 1000:
        return 1
    elif p >= 100:
        return 1
    elif p >= 10:
        return 2
    elif p >= 1:
        return 3
    elif p >= 0.1:
        return 4
    else:
        return 5

_active_orders: dict[str, dict] = {}
def _extract_oid(result: dict, label: str = "") -> int:
    """Extract order ID from HL SDK response."""
    if isinstance(result, dict):
        oid = result.get("oid") or result.get("orderId") or result.get("data", {}).get("oid")
        if oid:
            return int(oid)
        # Some responses have order info nested in statuses
        for resp in result.get("response", {}).get("data", {}).get("statuses", []):
            if isinstance(resp, dict):
                # Check for exchange error first
                if "error" in resp:
                    return 0
                for st in ("filled", "resting", "triggered"):
                    inner = resp.get(st, {})
                    if isinstance(inner, dict) and "oid" in inner:
                        return int(inner["oid"])
                if "oid" in resp:
                    return int(resp["oid"])
    return 0

def _track_order(coin: str, info: dict):
    """Store order info for fill verification."""
    _active_orders[coin.upper()] = info

def get_active_orders() -> dict:
    """Return active TP/SL orders for daemon to check against fills."""
    return dict(_active_orders)

def check_order_fills(ws_data: dict) -> list[dict]:
    """Check WebSocket userFills against active orders. Returns list of filled order events."""
    fills = ws_data.get("fills", [])
    if not fills:
        return []
    events = []
    for fill in fills[-10:]:  # Check last 10 fills
        oid = fill.get("oid", 0)
        coin = fill.get("coin", "")
        if not oid or not coin:
            continue
        for tracked_coin, order in list(_active_orders.items()):
            if tracked_coin != coin:
                continue
            # Check SL
            if order.get("sl_oid") == oid:
                events.append({
                    "coin": coin, "type": "sl", "oid": oid,
                    "px": float(fill.get("px", 0)), "sz": float(fill.get("sz", 0)),
                    "time": fill.get("time", 0),
                })
                del _active_orders[tracked_coin]
                break
            # Check TPs
            tp_oids = order.get("tp_oids", [])
            if oid in tp_oids:
                events.append({
                    "coin": coin, "type": "tp", "oid": oid,
                    "px": float(fill.get("px", 0)), "sz": float(fill.get("sz", 0)),
                    "time": fill.get("time", 0),
                })
                # Remove this TP from tracking but keep SL
                order["tp_oids"] = [x for x in tp_oids if x != oid]
                break
    return events

# ── AI brain (cheap model) ───────────────────────────────────────────────────

class ExecutorBrain:
    """Cheap AI model (Gemini Flash Lite) to examine each approved action before execution."""

    def __init__(self):
        self.api_key = ""
        for source in [
            os.environ.get("OPENROUTER_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        ]:
            if source:
                self.api_key = source
                break
        if not self.api_key:
            for path in [
                os.path.expanduser("~/.hermes/.env"),
                "/home/ubuntu/.hermes/.env",
            ]:
                try:
                    with open(path) as f:
                        for line in f:
                            if "OPENROUTER_API_KEY" in line:
                                raw = line.split("=", 1)[1].strip()
                                key = raw.strip("'\" ").strip()
                                if key:
                                    self.api_key = key
                                    break
                except Exception:
                    pass
                if self.api_key:
                    break
        self.model = "google/gemini-2.5-flash-lite"

    def _call(self, system_prompt, user_prompt, temperature=0.2):
        import urllib.request, ssl
        if not self.api_key:
            return {"error": "no_api_key"}
        ctx = ssl.create_default_context()
        data = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 800,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=25) as resp:
                result = json.loads(resp.read())
                if "choices" in result and len(result["choices"]) > 0:
                    msg = result["choices"][0].get("message", {})
                    content = msg.get("content")
                    if content:
                        return {"content": content.strip()}
                    else:
                        return {"error": "empty_content"}
                else:
                    return {"error": "no_choices"}
        except Exception as e:
            return {"error": str(e)}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_approved_actions():
    actions = []
    for fname in os.listdir(APPROVED_DIR):
        if fname.startswith("approved_") and fname.endswith(".json"):
            try:
                with open(os.path.join(APPROVED_DIR, fname)) as f:
                    action = json.load(f)
                    action["_file"] = fname
                    actions.append(action)
            except Exception as e:
                log.warning(f"Bad approved file {fname}: {e}")
    # ── Prioritize real trades over alerts ──
    # Trades (buy/sell/close) go first, alerts last and limited
    trades = [a for a in actions if a.get("type") in ("buy", "sell", "close")]
    alerts = [a for a in actions if a.get("type") == "alert"]
    others = [a for a in actions if a.get("type") not in ("buy", "sell", "close", "alert")]
    # Only process max 5 alerts per run to prevent flooding
    return trades + others + alerts[:5]

def _load_calendar():
    try:
        with open(CALENDAR_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# ── AI evaluation of each action ─────────────────────────────────────────────

def _ai_evaluate_action(client, action_data, market_data):
    """Ask cheap AI: should we execute this action now? How?"""
    action = action_data.get("action", {})
    action_type = action.get("type", "unknown")
    symbol = action.get("symbol", "?")
    side = action.get("side", "?")
    orig_price = action_data.get("modified_price") or action.get("price")
    orig_size = action_data.get("modified_size") or action.get("size")
    sell_all = action.get("sell_all", False)
    use_limit = action_data.get("use_limit", False)
    limit_price = action_data.get("limit_price")
    validator_reasoning = action_data.get("validator_reasoning", "")
    validator_conf = action_data.get("validator_confidence", 50)

    # Get current price from tickers
    current_price = None
    try:
        if client:
            raw_tickers = client.get_tickers()
            tickers = raw_tickers if isinstance(raw_tickers, list) else raw_tickers.get("data", [])
            for t in tickers:
                if isinstance(t, dict) and t.get("symbol", "").replace("/USD", "").replace("-EUR", "") == symbol:
                    current_price = float(t.get("last_price", 0) or t.get("mid", 0) or 0)
                    break
    except Exception:
        pass

    # Use limit order if validator suggests it
    order_type = "limit" if (use_limit and limit_price) else "market"
    if order_type == "limit":
        log.info(f"  📊 {symbol}: validator suggests LIMIT order @ {limit_price} (market: {current_price})")

    brain = ExecutorBrain()
    if not brain.api_key:
        log.warning("  No AI key — executing as-is")
        return {"decision": "execute", "order_type": action.get("order_type", "market"), "price": orig_price, "size": orig_size}

    # Build prompt
    lines = [
        "You are an execution AI. You examine approved trading actions and decide HOW to execute them.",
        "Respond ONLY with JSON. No prose.",
        "",
        f"Action: {action_type} {side} {symbol}",
        f"Original price: {orig_price}",
        f"Original size: {orig_size}",
        f"Sell all: {sell_all}",
        f"Validator reasoning: {validator_reasoning[:200]}",
        f"Current market price: {current_price}",
        "",
    ]

    # Add market context
    fg = market_data.get("fear_greed", {})
    if fg:
        lines.append(f"Fear & Greed: {fg.get('value')} ({fg.get('classification')})")

    if market_data.get("regime"):
        lines.append(f"Market regime: {market_data['regime']}")

    lines.append("")
    lines.append("Decide:")
    lines.append('  "execute" — place the order now')
    lines.append('  "wait" — skip this run, try again later')
    lines.append('  "cancel" — this action is no longer valid')
    lines.append("")
    lines.append("If execute, choose order_type: 'market' (fastest, taker fee) or 'limit' (save fees, may not fill)")
    lines.append("If limit, set a price. If market, set price to null.")
    if sell_all:
        lines.append("This is a SELL-ALL order — you do NOT set size. Leave \"size\": null and the")
        lines.append("executor will look up and sell the exact held balance. Never guess a quantity.")
    else:
        lines.append("You can modify size within reason (e.g. reduce if market moved against).")
    lines.append("")
    lines.append('```json')
    lines.append('{')
    lines.append('  "decision": "execute | wait | cancel",')
    lines.append('  "order_type": "market | limit",')
    lines.append('  "price": null,')
    lines.append('  "size": null,')
    lines.append('  "reasoning": "why this decision"')
    lines.append('}')
    lines.append('```')

    prompt = "\n".join(lines)
    result = brain._call(
        "You are a cautious execution AI. Favor market orders for small amounts, limit orders for larger ones. Cancel if price moved >5% from original.",
        prompt,
    )

    if "error" in result:
        log.warning(f"  AI error: {result['error'][:80]} — executing as-is")
        return {"decision": "execute", "order_type": action.get("order_type", "market"), "price": orig_price, "size": orig_size}

    # Parse response
    content = result["content"]
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl > 0:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        ai = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try regex fallback
        m = re.search(r'({[\s\S]*?"decision"[\s\S]*?})', cleaned)
        if m:
            try:
                ai = json.loads(m.group(1))
            except json.JSONDecodeError:
                ai = {"decision": "execute", "order_type": "market", "price": None, "size": None, "reasoning": "parse_fallback"}
        else:
            ai = {"decision": "execute", "order_type": "market", "price": None, "size": None, "reasoning": "parse_fallback"}

    decision = ai.get("decision", "execute")
    log.info(f"  AI says: {decision} | order_type={ai.get('order_type','?')} price={ai.get('price','?')} size={ai.get('size','?')} | {ai.get('reasoning','')[:100]}")

    # Hard guard: never trust the AI's "size" for a sell_all order. The exact
    # held quantity must come from the exchange balance at execution time —
    # an AI hallucinated size (seen: 10000 units when only 11 were held)
    # would either fail outright or, worse, succeed with wrong math. Force
    # size=None here so _execute_sell always re-fetches the real balance.
    final_size = ai.get("size") or orig_size
    if sell_all:
        final_size = None

    return {
        "decision": decision,
        "order_type": ai.get("order_type", action.get("order_type", "market")),
        "price": ai.get("price") or orig_price,
        "size": final_size,
        "reasoning": ai.get("reasoning", ""),
    }

# ── Execution functions ──────────────────────────────────────────────────────

def _execute_buy(client, action_data, ai_decision):
    action = action_data.get("action", {})
    symbol = action.get("symbol", "")
    price = ai_decision.get("price")
    size = ai_decision.get("size")
    order_type = ai_decision.get("order_type", "market")

    # Defensive guard: buy "size" must be an EUR amount to spend (quote_size).
    # If the strategist/AI mistakenly filled a coin quantity instead (common
    # confusion for a €0.0003 BTC line vs €5 EUR line), fall back to the
    # "amount" field which some upstream agents use for the EUR figure.
    # Heuristic: a real EUR buy is never below €1 (min order) — a bare coin
    # quantity for BTC/ETH/etc will almost always be < 1.
    if size is not None and size < 1:
        fallback_amount = action.get("amount")
        if fallback_amount and fallback_amount >= 1:
            log.warning(f"  size={size} looks like a coin quantity, not EUR — using amount={fallback_amount} instead")
            size = fallback_amount

    if not symbol or not size or size < 1:
        log.error(f"  Missing/invalid EUR size for buy (size={size}) — refusing to execute")
        return False

    symbol_quote = symbol if symbol.upper().endswith("-EUR") else f"{symbol}-EUR"
    log.info(f"  BUY {symbol}: size=€{size} price={price or 'market'} type={order_type}")

    try:
        if order_type == "limit":
            result = client.place_order(
                symbol=symbol_quote, side="buy", order_type="limit",
                quote_size=str(size), price=str(price) if price else None,
            )
        else:
            result = client.place_order(
                symbol=symbol_quote, side="buy", order_type="market",
                quote_size=str(size),
            )
        log.info(f"  Result: {json.dumps(result, indent=2)[:200]}")
        return True
    except Exception as e:
        log.error(f"  Buy failed: {e}")
        return False

def _execute_sell(client, action_data, ai_decision):
    action = action_data.get("action", {})
    symbol = action.get("symbol", "")
    sell_all = action.get("sell_all", True)
    sell_qty = action.get("sell_qty")
    price = ai_decision.get("price") or action.get("price")
    size = ai_decision.get("size") or action.get("size")
    order_type = ai_decision.get("order_type", "market")

    if not symbol:
        return False

    # For sell_all, always pull the exact live balance from the exchange
    if sell_all:
        try:
            balances = client.get_balances()
            held = 0.0
            base_currency = symbol.replace("-EUR", "").replace("-USD", "").split("-")[0]
            for entry in balances:
                if entry.get("currency", "") == base_currency:
                    held = float(entry.get("available", 0))
                    break
            if held <= 0:
                log.error(f"  sell_all requested for {symbol} but held balance is {held} — nothing to sell")
                return False
            if held < 0.0001:
                log.info(f"  Dust amount {held:.8f} {symbol} — skipping sell (exchange minimum: 0.0001)")
                return False
            size = held
        except Exception as e:
            log.error(f"  Failed to fetch live balance for sell_all {symbol}: {e}")
            return False
    elif sell_qty and sell_qty > 0:
        # AI specified a fraction to sell — use sell_qty
        size = sell_qty
        log.info(f"  Partial sell: qty={sell_qty} of {symbol}")

    if not sell_all and not size:
        return False

    symbol_quote = symbol if symbol.upper().endswith("-EUR") else f"{symbol}-EUR"
    log.info(f"  SELL {symbol}: sell_all={sell_all} qty={size} price={price or 'market'} type={order_type}")

    try:
        if order_type == "limit":
            result = client.place_order(
                symbol=symbol_quote, side="sell", order_type="limit",
                base_size=str(size) if size else None, price=str(price) if price else None,
            )
        else:
            result = client.place_order(
                symbol=symbol_quote, side="sell", order_type="market",
                base_size=str(size) if size else None,
            )
        log.info(f"  Result: {json.dumps(result, indent=2)[:200]}")
        return True
    except Exception as e:
        log.error(f"  Sell failed: {e}")
        return False

def _execute_bank_withdraw(client, action_data, ai_decision):
    action = action_data.get("action", {})
    amount = ai_decision.get("size") or action.get("amount") or 0
    keep = action.get("keep_on_exchange") or 10
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 0.0
    try:
        keep = float(keep)
    except (TypeError, ValueError):
        keep = 10.0
    log.info(f"  🏦 BANK WITHDRAW: €{amount:.2f} (keep €{keep:.2f}) — manual action needed")
    return True

# ── Pionex execution functions ──────────────────────────────────────────────

def _execute_pionex_buy(pionex_mod, action_data, ai_decision):
    """Execute a Pionex SPOT buy order."""
    action = action_data.get("action", {})
    symbol = action.get("symbol", "")
    side = action.get("side", "buy").upper()
    price = ai_decision.get("price") or action.get("price")
    size = ai_decision.get("size") or action.get("size")
    amount = ai_decision.get("amount") or action.get("amount")
    order_type = ai_decision.get("order_type", "market").upper()

    if not symbol:
        log.error("  Pionex buy: no symbol")
        return False

    log.info(f"  PIONEX BUY {symbol}: amount={amount} size={size} price={price} type={order_type}")

    try:
        result = pionex_mod.place_order(
            symbol=symbol, side=side, order_type=order_type,
            size=str(size) if size else None,
            amount=str(amount) if amount else None,
            price=str(price) if price else None,
        )
        log.info(f"  Result: {json.dumps(result, indent=2)[:200]}")
        return "error" not in result
    except Exception as e:
        log.error(f"  Pionex buy failed: {e}")
        return False

def _execute_pionex_sell(pionex_mod, action_data, ai_decision):
    """Execute a Pionex SPOT sell order."""
    action = action_data.get("action", {})
    symbol = action.get("symbol", "")
    sell_all = action.get("sell_all", True)
    price = ai_decision.get("price") or action.get("price")
    order_type = ai_decision.get("order_type", "market").upper()

    if not symbol:
        log.error("  Pionex sell: no symbol")
        return False

    # For sell_all, pull live balance
    base = symbol.split("_")[0]
    if sell_all:
        try:
            bal = pionex_mod.get_balance()
            balances = bal.get("balances", [])
            held = 0.0
            for b in balances:
                if b.get("coin") == base:
                    held = float(b.get("free", 0))
                    break
            if held <= 0:
                log.error(f"  Pionex sell_all for {symbol}: no {base} balance (held={held})")
                return False
            size = held
            log.info(f"  Pionex sell_all: {size} {base}")
        except Exception as e:
            log.error(f"  Pionex sell_all balance check failed: {e}")
            return False
    else:
        size = ai_decision.get("size") or action.get("size")

    try:
        result = pionex_mod.place_order(
            symbol=symbol, side="SELL", order_type=order_type,
            size=str(size) if size else None,
            price=str(price) if price else None,
        )
        log.info(f"  Result: {json.dumps(result, indent=2)[:200]}")
        return "error" not in result
    except Exception as e:
        log.error(f"  Pionex sell failed: {e}")
        return False

def _execute_hyperliquid(action_data, ai_decision):
    """Execute a Hyperliquid trade (LONG/SHORT/CLOSE)."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hyperliquid_client as hl
    except ImportError as e:
        log.error(f"  Hyperliquid: client import failed — {e}")
        return False
    except Exception as e:
        log.error(f"  Hyperliquid: client init error — {type(e).__name__}: {e}")
        return False

    action = action_data.get("action", {})
    coin = action.get("coin", action.get("symbol", ""))
    direction = action.get("direction", "long")
    size_usd = float(ai_decision.get("size_usd", action.get("size_usd", 0)))
    leverage = int(action.get("leverage", 6))
    entry_price = ai_decision.get("entry_price") or action.get("entry_price")
    action_type = action.get("type", "buy")

    if not coin:
        log.error("  Hyperliquid: no coin")
        return False

    # Get current price
    try:
        mids = hl.get_all_mids()
        mid = float(mids.get(coin, 0))
    except:
        mid = 0

    if mid <= 0:
        log.error(f"  Hyperliquid: no price for {coin}")
        return False

    # Calculate size in base units from USD
    size_base = size_usd / mid if size_usd > 0 else 0
    # Note: market_open handles rounding internally; order() needs pre-rounded size
    # For limit orders, round to 5 decimal places (order() passes to SDK which rounds further)
    size_base = round(size_base, 5)

    # Respect AI's order_type recommendation
    ai_order_type = ai_decision.get("order_type", "")
    use_market = ai_order_type == "market" or not entry_price

    log.info(f"  HYPERLIQUID {action_type} {coin}: dir={direction} size=${size_usd:.2f} ({size_base:.4f} {coin}) lev={leverage}x price={'market' if use_market else 'limit $'+str(entry_price)}")

    try:
        # Set leverage (skip for close actions — no leverage needed)
        if action_type != "close":
            hl.update_leverage(coin, leverage)

        if action_type == "close":
            result = hl.market_close(coin)
        elif direction == "long":
            if use_market:
                result = hl.market_open(coin, True, size_usd)
            else:
                result = hl.order(coin, True, size_base, entry_price, order_type="gtc")
        else:  # short
            if use_market:
                result = hl.market_open(coin, False, size_usd)
            else:
                result = hl.order(coin, False, size_base, entry_price, order_type="gtc")

        if result.get("status") == "ok":
            # Check for errors in response data (HL wraps errors in statuses)
            resp_data = result.get("response", {}).get("data", {})
            statuses = resp_data.get("statuses", [])
            for s in statuses:
                if "error" in s:
                    log.error(f"    ❌ HL rejected: {s['error']}")
                    return False
            if not statuses:
                # market_open returns different format
                pass
            log.info(f"    ✅ Order placed: {result.get('response',{}).get('type','ok')}")

            # ── Record trade for self-learning (close actions only) ──
            if action_type == "close":
                try:
                    entry_price = float(action.get("entry_price", 0))
                    mid_now = float(mids.get(coin, 0)) if 'mids' in dir() else mid
                    reason = action.get("reason", "close")
                    side_str = action.get("side", action.get("direction", "LONG"))
                    from hyperliquid_learner import record_trade, TradeRecord
                    pnl = (mid_now - entry_price) * size_base if side_str == "LONG" else (entry_price - mid_now) * size_base
                    pnl_pct = (mid_now - entry_price) / entry_price * 100 if side_str == "LONG" else (entry_price - mid_now) / entry_price * 100
                    record_trade(TradeRecord(
                        coin=coin, side=side_str, entry_price=entry_price,
                        exit_price=mid_now, pnl=pnl, pnl_pct=pnl_pct,
                        entry_time=time.time() - 3600, exit_time=time.time(),
                        regime=action.get("regime", "sideways"),
                        conviction=action.get("confidence", 0),
                        composite_score=action.get("composite", 0),
                        leverage=action.get("leverage", 3),
                        reason=reason,
                    ))
                    log.info(f"    📚 Recorded close: {coin} PnL=${pnl:.2f} ({pnl_pct:+.1f}%)")
                    # Also feed to performance tracker
                    try:
                        from cvd_engine import record_closed_trade
                        record_closed_trade(coin, pnl, pnl_pct, reason)
                    except Exception:
                        pass
                except Exception as e:
                    log.debug(f"    Trade recording skipped: {type(e).__name__}")

            # ── Place TP/SL trigger orders for protection ──
            if action_type in ("buy", "sell"):
                stop_loss = float(action.get("stop_loss", 0))
                tp_levels = action.get("tp_levels", [])
                is_long = direction == "long"
                if stop_loss > 0 and tp_levels:
                    try:
                        # Determine price decimals for this coin (Hyperliquid wire precision)
                        price_dec = _get_price_decimals(coin)
                        sl_px = round(stop_loss, price_dec)
                        # Place stop-loss trigger
                        sl_sz = size_base
                        sl_result = hl.trigger_order(coin, not is_long, sl_sz, sl_px, "sl", True, True)
                        sl_oid = _extract_oid(sl_result, "sl")
                        tp_count = 0
                        tp_oids = []
                        for tp in tp_levels[:2]:  # Max 2 TP orders
                            tp_px = round(float(tp.get("price", 0)), price_dec)
                            tp_frac = float(tp.get("fraction", 0.33))
                            tp_sz = round(size_base * tp_frac, 5)
                            if tp_px > 0 and tp_sz > 0:
                                tp_result = hl.trigger_order(coin, not is_long, tp_sz, tp_px, "tp", True, True)
                                tp_oid = _extract_oid(tp_result, "tp")
                                if tp_oid:
                                    tp_oids.append(tp_oid)
                                tp_count += 1
                        if tp_count > 0:
                            log.info(f"    🛡️ {tp_count} TP + SL trigger orders placed")
                        # Track orders for fill verification
                        _track_order(coin, {
                            "sl_oid": sl_oid, "tp_oids": tp_oids,
                            "stop_loss": sl_px, "tp_levels": tp_levels,
                            "size": size_base, "side": direction,
                            "placed_at": time.time(),
                        })
                    except Exception as tp_err:
                        log.warning(f"    ⚠️ TP/SL placement failed: {tp_err}")

            return True
        elif "error" in result:
            log.error(f"    ❌ {result}")
            return False
        else:
            log.info(f"    Result: {result}")
            return result.get("status") == "ok"

    except Exception as e:
        import traceback
        tb_short = traceback.format_exc().split('\n')[-3:]
        log.error(f"  Hyperliquid failed: {type(e).__name__}: {e} | {tb_short[-1].strip() if tb_short else ''}")
        return False


def _execute_pionex_futures(pionex_mod, action_data, ai_decision):
    """Execute a Pionex futures PERP order with leverage."""
    action = action_data.get("action", {})
    symbol = action.get("symbol", "")
    side = action.get("side", action.get("type", "").replace("leverage_", "")).upper()
    price = ai_decision.get("price") or action.get("price")
    size = ai_decision.get("size") or action.get("size")
    order_type = ai_decision.get("order_type", "MARKET_QTY").upper()
    leverage = int(action.get("leverage", 20))
    client_order_id = action.get("batch_id", "")

    if not symbol:
        log.error("  Pionex futures: no symbol")
        return False

    # Ensure PERP symbol
    if "_PERP" not in symbol and "_USDT" in symbol:
        symbol = symbol.replace("_USDT", "_USDT_PERP")
    elif "_PERP" not in symbol:
        symbol = f"{symbol}_USDT_PERP"

    log.info(f"  PIONEX FUTURES {side} {symbol}: size={size} leverage={leverage}x type={order_type}")

    # Step 1: Set leverage
    try:
        lr = pionex_mod.set_futures_leverage(symbol, min(leverage, 30))
        log.info(f"    Leverage set: {lr}")
    except Exception as e:
        log.warning(f"    Leverage set warning: {e}")

    # Step 2: Place order
    try:
        side_str = "BUY" if "buy" in side.lower() else "SELL"
        result = pionex_mod.place_futures_order(
            symbol=symbol, side=side_str, order_type=order_type,
            size=str(size) if size else None,
            price=str(price) if price else None,
            position_side="BOTH",
            client_order_id=client_order_id,
        )
        log.info(f"    Result: {json.dumps(result, indent=2)[:300]}")
        if "error" not in result:
            log.info(f"  ✅ Futures {side_str} {symbol} x{leverage} — order placed")
            return True
        log.error(f"  ❌ Futures order failed: {result.get('error', '?')}")
        return False
    except Exception as e:
        log.error(f"  Pionex futures {side} failed: {e}")
        return False


# ── Main loop ────────────────────────────────────────────────────────────────

def _collect_market_snapshot():
    """Collect quick market data for AI context."""
    data = {}
    try:
        if data_enricher_mod:
            fg = data_enricher_mod.get_fear_greed()
            if fg:
                data["fear_greed"] = fg
    except Exception:
        pass
    try:
        import market_regime
        r = market_regime.classify_regime()
        if r:
            data["regime"] = r.get("regime", "?")
    except Exception:
        pass
    return data

def run_executor():
    log.info("=" * 60)
    log.info("⚡ EXECUTOR v2 (AI-powered) STARTING")
    log.info("=" * 60)

    approved = _get_approved_actions()
    if not approved:
        log.info("No approved actions to execute")
        return {"status": "idle", "executed": 0}

    log.info(f"📋 Found {len(approved)} approved action(s)")

    # Connect to Revolut
    client = None
    if rev_client_mod:
        try:
            client = rev_client_mod.RevolutXClient()
        except Exception as e:
            log.error(f"Failed to create Revolut client: {e}")

    # Pionex client (module-level calls, no instance needed) — optional
    try:
        import pionex_client as pionex_mod
        pionex_avail = pionex_mod.is_configured()
    except (ImportError, ModuleNotFoundError):
        pionex_mod = None
        pionex_avail = False

    # Market snapshot for AI context
    market_data = _collect_market_snapshot()

    results = []
    for item in approved:
        action = item.get("action", {})
        action_type = action.get("type", "unknown")
        symbol = action.get("symbol", "?")
        fname = item.get("_file", "")

        log.info(f"  Processing: {action_type} {symbol}")

        # Skip alerts — they waste AI calls and can't be executed
        if action_type == "alert":
            try:
                os.remove(os.path.join(APPROVED_DIR, fname))
            except OSError:
                pass
            continue

        # Close actions bypass AI — always execute directly
        # Buy/sell actions that were already AI-validated also bypass — execute directly
        if action_type in ("close", "buy", "sell"):
            # Extract size/price from action (validator-approved values come from
            # item["modified_size"] or the original action["size"])
            action_size = item.get("modified_size") or action.get("size")
            action_price = item.get("modified_price") or action.get("price")
            order_type = action.get("order_type", "market")
            if item.get("use_limit") and item.get("limit_price"):
                order_type = "limit"
                action_price = item.get("limit_price")
            ai_dec = {
                "decision": "execute",
                "order_type": order_type,
                "price": action_price,
                "size": action_size,
                # Hyperliquid fields
                "size_usd": action.get("size_usd", 0),
                "entry_price": action.get("entry_price"),
                "stop_loss": action.get("stop_loss"),
                "direction": action.get("direction"),
            }
        else:
            ai_dec = _ai_evaluate_action(client, item, market_data)

        if ai_dec["decision"] == "cancel":
            log.info(f"  🚫 AI cancelled: {ai_dec.get('reasoning','')[:100]}")
            results.append({"type": action_type, "symbol": symbol, "success": False, "reason": "ai_cancelled"})
        elif ai_dec["decision"] == "wait":
            log.info(f"  ⏳ AI says wait: {ai_dec.get('reasoning','')[:100]}")
            # Leave it in the queue for next run
            results.append({"type": action_type, "symbol": symbol, "success": False, "reason": "ai_wait"})
            continue  # Don't remove from queue
        else:
            # Step 2: Execute
            success = False
            exchange = action.get("exchange", "revolut")
            if exchange == "pionex" and pionex_avail:
                if action_type == "buy":
                    success = _execute_pionex_buy(pionex_mod, item, ai_dec)
                elif action_type == "sell":
                    success = _execute_pionex_sell(pionex_mod, item, ai_dec)
                elif action_type in ("leverage_buy", "leverage_sell", "leverage"):
                    success = _execute_pionex_futures(pionex_mod, item, ai_dec)
                else:
                    log.warning(f"  Unknown Pionex type: {action_type}")
            elif exchange == "hyperliquid":
                if action_type in ("buy", "sell", "close"):
                    success = _execute_hyperliquid(item, ai_dec)
                else:
                    log.warning(f"  Unknown Hyperliquid type: {action_type}")
            elif action_type == "buy":
                if client:
                    success = _execute_buy(client, item, ai_dec)
                else:
                    log.error("  No Revolut client")
            elif action_type == "sell":
                if client:
                    success = _execute_sell(client, item, ai_dec)
                else:
                    log.error("  No Revolut client")
            elif action_type == "bank_withdraw":
                success = _execute_bank_withdraw(client, item, ai_dec)
            else:
                log.warning(f"  Unknown type: {action_type}")
                success = False

            results.append({"type": action_type, "symbol": symbol, "success": success})

        # Move to completed (unless wait)
        if ai_dec["decision"] != "wait":
            result_entry = {
                "original": item,
                "ai_decision": ai_dec,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "success": results[-1].get("success", False),
            }
            outpath = os.path.join(COMPLETED_DIR, f"done_{fname}")
            with open(outpath, "w") as f:
                json.dump(result_entry, f, indent=2)

            src = os.path.join(APPROVED_DIR, fname)
            if os.path.exists(src):
                os.remove(src)

    log.info(f"✅ Executed {len(results)} actions")
    for r in results:
        status = "✅" if r.get("success") else "⏳" if r.get("reason") == "ai_wait" else "❌"
        log.info(f"  {status} {r['type']} {r['symbol']}")

    return {"status": "ok", "executed": len(results), "results": results}


if __name__ == "__main__":
    result = run_executor()
    print(json.dumps(result, indent=2))