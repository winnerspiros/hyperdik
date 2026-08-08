#!/usr/bin/env python3
"""
Seller Agent — AI-powered sell decision maker.
- Reads strategy from calendar (reviewer + predictor)
- Checks current holdings, P&L, market conditions
- AI decides: what to sell, how much, at what price
- Submits proposed action to pending_actions/ for validator to approve
- Uses standard model (Gemini Flash Lite) for decisions
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SELLER] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "seller.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Seller")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
PENDING_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
MEMORY_PATH = os.path.join(TRADER_DIR, "data", "trade_memory.json")
os.makedirs(PENDING_DIR, exist_ok=True)


class SellerBrain:
    """Standard model for sell decisions."""

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

    def _call(self, system_prompt, user_prompt, temperature=0.4):
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
            "max_tokens": 1200,
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
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
                return {"content": content}
        except Exception as e:
            return {"error": str(e)}


def _load_calendar():
    try:
        with open(CALENDAR_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_entry_prices():
    try:
        with open(MEMORY_PATH) as f:
            data = json.load(f)
        return data.get("entry_prices", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_seller():
    log.info("=" * 60)
    log.info("💸 SELLER AGENT STARTING")
    log.info("=" * 60)

    # Coordination: check reviewer ran recently
    try:
        from agent_coordination import check_upstream, get_reviewer_state, get_pending_count
        if not check_upstream("reviewer", 120, log.info):
            return {"status": "skipped", "reason": "reviewer_stale"}
        reviewer_state = get_reviewer_state()
        log.info(f"📖 Assessment: {reviewer_state['assessment']} | Health: {reviewer_state['health']}")
        log.info(f"📖 Sells pending: {get_pending_count()}")
    except Exception as e:
        log.warning(f"Coordination check failed: {e}")

    cal = _load_calendar()
    reviewer = cal.get("reviewer", {})
    predictor = cal.get("predictor", {})

    strategy = predictor.get("active_strategy", "No strategy")
    outlook = predictor.get("market_outlook", "N/A")
    holdings = reviewer.get("positions", [])
    eur_balance = reviewer.get("eur_balance", 0)
    total_value = reviewer.get("portfolio_value", 0)

    log.info(f"📖 Strategy: {strategy}")
    log.info(f"📖 Holdings: {len(holdings)} positions | EUR: €{eur_balance:.2f} | Value: €{total_value:.2f}")

    if not holdings:
        log.info("No holdings to sell")
        return {"status": "skipped", "reason": "no_holdings"}

    # 2. Load entry prices for P&L
    entry_prices = _load_entry_prices()

    # 3. Build position summary with P&L
    enriched = []
    for p in holdings:
        sym = p["symbol"]
        qty = p["qty"]
        cur_price = p["current_price"]
        value = p["value_eur"]

        # Get entry price
        entries = entry_prices.get(sym, [])
        entry_price = entries[0][0] if entries else None
        pnl_pct = ((cur_price - entry_price) / entry_price * 100) if entry_price and entry_price > 0 else 0

        enriched.append({
            "symbol": sym,
            "qty": qty,
            "current_price": cur_price,
            "value_eur": value,
            "entry_price": entry_price,
            "pnl_pct": round(pnl_pct, 1),
        })
        log.info(f"   {sym}: {qty:.6f} @ €{cur_price:.4f} | value=€{value:.2f} | P&L={pnl_pct:+.1f}%")

    # 4. Ask AI what to sell
    brain = SellerBrain()
    if not brain.api_key:
        log.error("No API key")
        return {"status": "error", "reason": "no_api_key"}

    # Build position table
    pos_lines = []
    for p in enriched:
        pos_lines.append(f"  {p['symbol']:6s} | qty={p['qty']:.6f} | price=€{p['current_price']:.4f} | value=€{p['value_eur']:.2f} | entry=€{p['entry_price'] or '?'} | P&L={p['pnl_pct']:+.1f}%")

    prompt_lines = [
        "You are a crypto SELLER agent. Decide what to sell to raise EUR.",
        "Respond ONLY with JSON. No prose.",
        "",
        "Consider:",
        "- Need EUR for opportunities or preventing losses",
        "- P&L: take profit on winners, cut losses on losers",
        "- Don't sell everything unless necessary",
        "- Selling at a small loss is OK if it frees capital for better use",
        "- If no reason to sell, say 'hold'",
        "",
        f"EUR Balance: €{eur_balance:.2f}",
        f"Total Portfolio: €{total_value:.2f}",
        f"Total Value (EUR + holdings): €{eur_balance + total_value:.2f}",
        f"Current Holdings:",
    ] + pos_lines + [
        f"Strategy: {strategy}",
        f"Market Outlook: {outlook}",
        f"Reviewer Notes: {reviewer.get('notes', '')[:200]}",
        "",
        'Respond with JSON:',
        '```json',
        '{',
        '  "decision": "sell | hold",',
        '  "symbol": "BTC",',
        '  "sell_all": true,',
        '  "sell_qty": null,',
        '  "price": null,',
        '  "order_type": "limit | market",',
        '  "reasoning": "why this decision"',
        '}',
        '```',
        "",
        "IMPORTANT: You can submit ONE sell action per run. If multiple positions need selling, focus on the most urgent one.",
    ]
    user_prompt = "\n".join(prompt_lines)

    result = brain._call(
        "You are a crypto seller. Sell strategically to maximize portfolio health.",
        user_prompt,
    )

    if "error" in result:
        log.error(f"AI call failed: {result['error']}")
        return {"status": "error", "reason": result["error"]}

    content = result["content"]
    json_match = re.search(r'```(?:json)?\s*\n?({.*?})\n?```', content, re.DOTALL)
    if json_match:
        try:
            decision = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            decision = None
    else:
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            decision = None

    if not decision:
        log.warning(f"Could not parse AI response: {content[:200]}")
        return {"status": "error", "reason": "parse_failed"}

    action = decision.get("decision", "hold")
    log.info(f"🤖 AI Decision: {action}")

    if action == "hold":
        log.info("AI says hold — no sell action submitted")
        return {"status": "hold", "reasoning": decision.get("reasoning", "")}

    # 5. Submit sell action
    symbol = decision.get("symbol", "")
    if not symbol:
        log.warning("No symbol in sell decision")
        return {"status": "error", "reason": "no_symbol"}

    sell_all = decision.get("sell_all", True)
    sell_qty = decision.get("sell_qty")
    sell_price = decision.get("price")
    order_type = decision.get("order_type", "limit")

    sell_action = {
        "type": "sell",
        "symbol": symbol,
        "side": "sell",
        "sell_all": sell_all,
        "size": sell_qty,  # consistent key for validator
        "price": sell_price,
        "order_type": order_type,
        "reasoning": decision.get("reasoning", ""),
        "source": "seller_agent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eur_balance_before": eur_balance,
    }

    fname = f"sell_{symbol}_{int(time.time())}.json"
    with open(os.path.join(PENDING_DIR, fname), "w") as f:
        json.dump(sell_action, f, indent=2)

    log.info(f"✅ Sell action submitted: {symbol} sell_all={sell_all} qty={sell_qty} @ €{sell_price or 'market'}")
    log.info(f"   Reasoning: {decision.get('reasoning', '')[:200]}")
    log.info(f"   -> {PENDING_DIR}/{fname} (waiting for validator)")

    return {
        "status": "submitted",
        "symbol": symbol,
        "sell_all": sell_all,
        "qty": sell_qty,
        "price": sell_price,
        "reasoning": decision.get("reasoning", ""),
    }


if __name__ == "__main__":
    result = run_seller()
    print(json.dumps(result, indent=2))