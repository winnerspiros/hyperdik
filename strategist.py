#!/usr/bin/env python3
"""
Strategist — single joint decision-maker for buy/sell/bank actions.

WHY THIS REPLACES buyer_agent.py + seller_agent.py + bank_manager.py:
Those three ran as independent hourly cron jobs, each calling the AI blind to
what the other two were doing. A portfolio decision is inherently joint —
"sell LDO to fund a BTC dip-buy" is one decision, not two independent ones
that happen to both touch the same EUR balance. Splitting them let the seller
and buyer each think they had the full EUR balance available, causing
double-counted spending intent and no way to express "sell A to buy B".

This module makes ONE call to the AI with the full portfolio (all positions +
EUR + market intelligence) and lets it propose a coherent plan: which
positions to sell, what to buy with the proceeds, and whether to move EUR to
the bank — all reasoned about together, with the EUR budget shared correctly
across every action in the plan.

Runs hourly. Reads:
  - market_intelligence.get_snapshot() — canonical world state (10min TTL, shared)
  - strategy_calendar.json — reviewer assessment + predictor outlook
  - trader_db — positions & entry prices (single source of truth for state)

Writes:
  - One or more actions to data/pending_actions/, all sharing a `batch_id` so
    the validator can see they're part of the same plan and reason about
    total EUR exposure across the batch, not just one action in isolation.
  - Reserves EUR via agent_coordination.reserve_eur() so a second strategist
    run (or day_trader/scalper, if they later check the ledger) won't
    double-spend the same cash before the validator/executor clear the batch.
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STRATEGIST] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "strategist.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Strategist")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
PENDING_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
MEMORY_PATH = os.path.join(TRADER_DIR, "data", "trade_memory.json")
os.makedirs(PENDING_DIR, exist_ok=True)

TRADABLE_COINS = [
    "ADA","APT","ARB","ATOM","AVAX","BCH","BNB","BONK","BTC","CRV",
    "DOGE","DOT","ENA","ETH","FIL","HYPE","ICP","INJ","LINK","LTC",
    "NEAR","OP","PEPE","SEI","SHIB","SOL","SUI","TON","TRX","UNI","WIF","XRP",
]


def _import(name):
    try:
        return __import__(name, fromlist=[""])
    except Exception:
        return None


def _get_tradable_symbols():
    """
    Fetch the LIVE tradable symbol set from Revolut instead of relying on the
    static TRADABLE_COINS list above (which is a fallback only). The static
    list goes stale as Revolut adds/removes pairs — RENDER, for example, is
    tradable but was missing from the hardcoded list, silently blocking sells
    of a real held position. Always prefer live data; only fall back to the
    static list if the exchange call fails.
    """
    rev_client_mod = _import("revolut_client")
    if not rev_client_mod:
        return set(TRADABLE_COINS)
    try:
        client = rev_client_mod.RevolutXClient()
        raw = client.get_tickers()
        tickers = raw if isinstance(raw, list) else raw.get("data", [])
        symbols = set()
        for t in tickers:
            if isinstance(t, dict):
                sym = t.get("symbol", "").replace("/USD", "").replace("/EUR", "")
                if sym:
                    symbols.add(sym)
        if symbols:
            return symbols
    except Exception:
        pass
    return set(TRADABLE_COINS)


class StrategistBrain:
    """Standard model (DeepSeek V4 Pro) for joint portfolio decisions — premium AI for data-rich decisions."""

    def __init__(self):
        self.api_key = ""
        for source in [os.environ.get("OPENROUTER_API_KEY"), os.environ.get("OPENAI_API_KEY")]:
            if source:
                self.api_key = source
                break
        if not self.api_key:
            for path in [os.path.expanduser("~/.hermes/.env"), "/home/ubuntu/.hermes/.env"]:
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
        self.model = "google/gemini-2.5-flash-lite"  # Cheap model for portfolio decisions

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
            "max_tokens": 1800,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=40) as resp:
                result = json.loads(resp.read())
                if "choices" in result and result["choices"]:
                    msg = result["choices"][0].get("message", {})
                    content = msg.get("content")
                    if content:
                        return {"content": content.strip()}
                    return {"error": f"empty_content: {json.dumps(msg)[:200]}"}
                return {"error": "no_choices"}
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


def _parse_plan(content):
    """Extract the JSON plan from the AI response, tolerant of markdown fences."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl > 0:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r'({[\s\S]*"actions"[\s\S]*})', cleaned)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def run_strategist():
    log.info("=" * 60)
    log.info("🧭 STRATEGIST STARTING (joint buy/sell/bank plan)")
    log.info("=" * 60)

    # Coordination: check reviewer freshness + existing pending/approved actions
    try:
        from agent_coordination import check_upstream, get_reviewer_state, get_pending_count, get_approved_count
        if not check_upstream("reviewer", 120, log.info):
            return {"status": "skipped", "reason": "reviewer_stale"}
        reviewer_state = get_reviewer_state()
        pending = get_pending_count()
        approved = get_approved_count()
        log.info(f"📖 Assessment: {reviewer_state['assessment']} | Health: {reviewer_state['health']}")
        if pending or approved:
            log.info(f"⏸️ {pending} pending + {approved} approved actions still in flight — skipping to avoid piling on")
            return {"status": "skipped", "reason": "actions_in_flight", "pending": pending, "approved": approved}
    except Exception as e:
        log.warning(f"Coordination check failed: {e}")

    cal = _load_calendar()
    reviewer = cal.get("reviewer", {})
    predictor = cal.get("predictor", {})

    strategy = predictor.get("active_strategy", "No strategy set")
    outlook = predictor.get("market_outlook", "N/A")
    portfolio_health = reviewer.get("portfolio_health", "N/A")
    assessment = reviewer.get("overall_assessment", "N/A")
    holdings = reviewer.get("positions", [])
    eur_balance = reviewer.get("eur_balance", 0)
    total_value = reviewer.get("portfolio_value", 0)

    log.info(f"📖 Strategy: {strategy} | Outlook: {outlook}")
    log.info(f"💰 EUR: €{eur_balance:.2f} | Holdings: {len(holdings)} positions | Value: €{total_value:.2f}")

    if eur_balance < 1 and not holdings:
        log.info("Nothing to work with — no EUR, no holdings")
        return {"status": "skipped", "reason": "empty_portfolio"}

    # Canonical market intelligence (single fetch, shared with predictor/validator)
    mi = _import("market_intelligence")
    snapshot_text = ""
    if mi:
        try:
            snap = mi.get_snapshot(holdings=[p.get("symbol") for p in holdings])
            snapshot_text = mi.format_for_ai(snap)
        except Exception as e:
            log.warning(f"market_intelligence error: {e}")

    # Position P&L table
    entry_prices = _load_entry_prices()
    pos_lines = []
    for p in holdings:
        sym = p.get("symbol", "?")
        qty = p.get("qty", 0)
        cur_price = p.get("current_price", 0)
        value = p.get("value_eur", 0)
        entries = entry_prices.get(sym, [])
        entry_price = entries[0][0] if entries else None
        pnl_pct = ((cur_price - entry_price) / entry_price * 100) if entry_price and entry_price > 0 else 0
        pos_lines.append(
            f"  {sym:6s} | qty={qty:.6f} | price=€{cur_price:.4f} | value=€{value:.2f} | "
            f"entry=€{entry_price or '?'} | P&L={pnl_pct:+.1f}%"
        )

    brain = StrategistBrain()
    if not brain.api_key:
        log.error("No API key")
        return {"status": "error", "reason": "no_api_key"}

    # Live tradable set (falls back to the static list on API failure). Union
    # with currently-held symbols so a sell of an existing position is never
    # blocked just because it's missing from either list — you can always
    # sell what you actually hold.
    tradable = _get_tradable_symbols()
    held_symbols = {p.get("symbol") for p in holdings if p.get("symbol")}
    tradable_for_prompt = sorted(tradable | held_symbols)

    prompt_lines = [
        "You are the STRATEGIST for a crypto portfolio. You make ONE joint plan covering",
        "buys, sells, and bank/EUR management together — not three separate decisions.",
        "Respond ONLY with JSON. No prose.",
        "",
        "Reason jointly: e.g. if you want to buy X, you can decide to sell Y to fund it",
        "in the SAME plan. The total EUR spent across all buy actions must not exceed",
        "current EUR balance PLUS proceeds from any sells in this same plan.",
        "",
        f"EUR Balance: €{eur_balance:.2f}",
        f"Total Portfolio Value: €{total_value:.2f}",
        f"Total (EUR + holdings): €{eur_balance + total_value:.2f}",
        "",
        "Current Holdings:",
    ] + (pos_lines if pos_lines else ["  (none)"]) + [
        "",
        f"Strategy (from predictor): {strategy}",
        f"Market Outlook: {outlook}",
        f"Portfolio Health (from reviewer): {portfolio_health}",
        f"Reviewer Assessment: {assessment}",
        "",
        "Market Intelligence:",
        snapshot_text or "(unavailable)",
        "",
        f"TRADABLE COINS: {', '.join(tradable_for_prompt)}",
        "",
        "Guidance:",
        "- Selling at a small loss is fine if it frees capital for a better opportunity or reduces risk.",
        "- Don't sell everything unless the data clearly calls for it.",
        "- Don't over-concentrate EUR into one buy.",
        "- If EUR balance is comfortably above trading needs, you may propose a bank withdrawal.",
        "- If nothing makes sense right now, propose zero actions — that's a valid plan.",
        "- No hardcoded percentage thresholds — reason from the full context above.",
        "",
        "Respond with JSON:",
        "```json",
        "{",
        '  "plan_summary": "one sentence describing the overall plan",',
        '  "actions": [',
        "    {",
        '      "type": "buy | sell | bank_withdraw",',
        '      "symbol": "BTC",',
        '      "side": "buy | sell",',
        '      "sell_all": false,',
        '      "size": null,',
        '      "price": null,',
        '      "order_type": "limit | market",',
        '      "amount": null,',
        '      "keep_on_exchange": null,',
        '      "reasoning": "why this specific action, and how it relates to the rest of the plan"',
        "    }",
        "  ]",
        "}",
        "```",
        "",
        'If no action is warranted, respond with "actions": [] and explain why in plan_summary.',
        "",
        "FIELD MEANINGS (do not confuse these — this caused bugs before):",
        '  - For type="buy": "size" = EUR amount to spend (e.g. 5.00 means spend €5.00). Do NOT put a coin quantity here.',
        '  - For type="sell": "size" = coin quantity to sell (e.g. 0.001 BTC), OR set "sell_all": true and leave "size" null.',
        '  - For type="bank_withdraw": use "amount" for the EUR amount to withdraw, "size" is unused.',
        '  - "price": only set for order_type="limit". Leave null for order_type="market".',
    ]
    user_prompt = "\n".join(prompt_lines)

    result = brain._call(
        "You are a disciplined portfolio strategist. Think in terms of ONE coherent plan, "
        "not isolated buy/sell/bank decisions. Be conservative with position sizing.",
        user_prompt,
    )

    if "error" in result:
        log.error(f"AI call failed: {result['error']}")
        return {"status": "error", "reason": result["error"]}

    plan = _parse_plan(result["content"])
    if not plan:
        log.warning(f"Could not parse AI response: {result['content'][:200]}")
        return {"status": "error", "reason": "parse_failed"}

    log.info(f"📋 Plan: {plan.get('plan_summary', '')}")
    actions = plan.get("actions", [])

    if not actions:
        log.info("No actions proposed this cycle")
        return {"status": "no_action", "plan_summary": plan.get("plan_summary", "")}

    batch_id = f"batch_{int(time.time())}"
    submitted = []

    # Reserve EUR across the whole batch before writing individual actions,
    # so a concurrent run (or a future day_trader/scalper check) sees the
    # committed total, not just one action at a time.
    total_buy_eur = sum(
        float(a.get("size") or 0) for a in actions if a.get("type") == "buy"
    )
    coord = _import("agent_coordination")
    if coord and hasattr(coord, "reserve_eur"):
        try:
            coord.reserve_eur(batch_id, total_buy_eur, ttl_minutes=15)
        except Exception as e:
            log.warning(f"EUR reservation failed (non-fatal): {e}")

    for i, action in enumerate(actions):
        action_type = action.get("type", "")
        symbol = action.get("symbol", "")
        if action_type in ("buy", "sell") and symbol not in tradable_for_prompt:
            log.warning(f"Skipping invalid symbol: {symbol}")
            continue

        record = {
            "type": action_type,
            "symbol": symbol,
            "side": action.get("side", action_type),
            "sell_all": action.get("sell_all", False),
            "size": action.get("size"),
            "price": action.get("price"),
            "order_type": action.get("order_type", "limit"),
            "amount": action.get("amount"),
            "keep_on_exchange": action.get("keep_on_exchange"),
            "reasoning": action.get("reasoning", ""),
            "source": "strategist",
            "batch_id": batch_id,
            "plan_summary": plan.get("plan_summary", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "eur_balance_at_plan": eur_balance,
        }

        fname = f"{action_type}_{symbol or 'bank'}_{int(time.time())}_{i}.json"
        with open(os.path.join(PENDING_DIR, fname), "w") as f:
            json.dump(record, f, indent=2)

        log.info(f"✅ Submitted [{batch_id}] {action_type} {symbol}: {action.get('reasoning', '')[:120]}")
        submitted.append({"type": action_type, "symbol": symbol, "file": fname})

    return {
        "status": "submitted" if submitted else "no_action",
        "batch_id": batch_id,
        "plan_summary": plan.get("plan_summary", ""),
        "actions": submitted,
    }


if __name__ == "__main__":
    result = run_strategist()
    print(json.dumps(result, indent=2))
