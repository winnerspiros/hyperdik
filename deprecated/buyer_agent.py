#!/usr/bin/env python3
"""
Buyer Agent — AI-powered buy decision maker.
- Reads strategy from calendar (reviewer assessment + predictor outlook)
- Checks current portfolio, EUR balance, market conditions
- AI decides: what to buy, at what price, how much
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
    format="%(asctime)s [BUYER] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "buyer.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Buyer")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
PENDING_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
os.makedirs(PENDING_DIR, exist_ok=True)

# ── Lazy imports ─────────────────────────────────────────────────────────────

def _import(name):
    try:
        return __import__(name, fromlist=[""])
    except Exception as e:
        return None

rev_client_mod = _import("revolut_client")
data_enricher_mod = _import("data_enricher")
market_regime_mod = _import("market_regime")
market_analyzer_mod = _import("market_analyzer")


class BuyerBrain:
    """Standard model for buy decisions."""

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


def run_buyer():
    log.info("=" * 60)
    log.info("💰 BUYER AGENT STARTING")
    log.info("=" * 60)

    # Coordination: check reviewer ran recently
    try:
        from agent_coordination import check_upstream, get_reviewer_state, get_pending_count
        if not check_upstream("reviewer", 120, log.info):
            return {"status": "skipped", "reason": "reviewer_stale"}
        reviewer_state = get_reviewer_state()
        log.info(f"📖 Assessment: {reviewer_state['assessment']} | Health: {reviewer_state['health']}")
        log.info(f"📖 Buys pending: {get_pending_count()}")
    except Exception as e:
        log.warning(f"Coordination check failed: {e}")

    cal = _load_calendar()
    reviewer = cal.get("reviewer", {})
    predictor = cal.get("predictor", {})

    strategy = predictor.get("active_strategy", "No strategy set")
    outlook = predictor.get("market_outlook", "N/A")
    portfolio_health = reviewer.get("portfolio_health", "N/A")
    assessment = reviewer.get("overall_assessment", "N/A")

    log.info(f"📖 Strategy: {strategy}")
    log.info(f"📖 Outlook: {outlook} | Health: {portfolio_health}")

    # 2. Check EUR balance
    eur_balance = reviewer.get("eur_balance", 0)
    log.info(f"💰 EUR available: €{eur_balance:.2f}")

    # If no EUR, skip
    if eur_balance < 1:
        log.info("No EUR to buy with — skipping")
        return {"status": "skipped", "reason": "no_eur"}

    # 3. Get current market data
    market_data = {}
    try:
        if data_enricher_mod:
            market_data["fear_greed"] = data_enricher_mod.get_fear_greed()
            market_data["global_data"] = data_enricher_mod.get_global_data()
            market_data["trending"] = data_enricher_mod.get_trending()
    except Exception as e:
        log.warning(f"Market data error: {e}")

    # 4. Get current holdings to avoid duplicates
    holdings = reviewer.get("positions", [])
    held_symbols = {p["symbol"] for p in holdings}
    log.info(f"Current holdings: {held_symbols}")

    # 5. Ask AI what to buy
    brain = BuyerBrain()
    if not brain.api_key:
        log.error("No API key")
        return {"status": "error", "reason": "no_api_key"}

    fg = market_data.get("fear_greed", {})
    prompt_lines = [
        "You are a crypto BUYER agent. Decide what to buy based on the data.",
        "Respond ONLY with JSON. No prose.",
        "",
        "Consider:",
        "- EUR balance vs portfolio value",
        "- Market regime, Fear & Greed, trends",
        "- Don't over-concentrate in one coin",
        "- If it's a bad time to buy, say 'wait'",
        "- If buying, specify symbol, price, size, and reasoning",
        "",
        f"EUR Balance: €{eur_balance:.2f}",
        f"Available to buy: €{eur_balance * 0.9:.2f} (90% of balance)",
        f"Current holdings: {list(held_symbols) if held_symbols else 'none'}",
        f"Strategy: {strategy}",
        f"Market Outlook: {outlook}",
        f"Portfolio Health: {portfolio_health}",
        f"Reviewer Assessment: {assessment}",
        f"F&G: {fg.get('value', '?')} ({fg.get('classification', '?')})",
        "",
        "TRADABLE COINS: ADA, APT, ARB, ATOM, AVAX, BCH, BNB, BONK, BTC, CRV, DOGE, DOT, ENA, ETH, FIL, HYPE, ICP, INJ, LINK, LTC, NEAR, OP, PEPE, SEI, SHIB, SOL, SUI, TON, TRX, UNI, WIF, XRP",
        "",
        'Respond with JSON:',
        '```json',
        '{',
        '  "decision": "buy | wait | skip",',
        '  "symbol": "BTC",',
        '  "price": null,',
        '  "size": null,',
        '  "order_type": "limit | market",',
        '  "reasoning": "why this decision"',
        '}',
        '```',
    ]
    user_prompt = "\n".join(prompt_lines)

    result = brain._call(
        "You are a crypto buyer. Only buy when conditions are favorable. Be conservative.",
        user_prompt,
    )

    if "error" in result:
        log.error(f"AI call failed: {result['error']}")
        return {"status": "error", "reason": result["error"]}

    # Parse response
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

    action = decision.get("decision", "wait")
    log.info(f"🤖 AI Decision: {action}")

    if action == "wait" or action == "skip":
        log.info("AI says wait — no buy action submitted")
        return {"status": "wait", "reasoning": decision.get("reasoning", "")}

    # 6. Submit buy action to pending_actions
    symbol = decision.get("symbol", "")
    if not symbol or symbol not in [
        "ADA","APT","ARB","ATOM","AVAX","BCH","BNB","BONK","BTC","CRV",
        "DOGE","DOT","ENA","ETH","FIL","HYPE","ICP","INJ","LINK","LTC",
        "NEAR","OP","PEPE","SEI","SHIB","SOL","SUI","TON","TRX","UNI","WIF","XRP"
    ]:
        log.warning(f"Invalid symbol: {symbol}")
        return {"status": "error", "reason": f"invalid_symbol_{symbol}"}

    buy_action = {
        "type": "buy",
        "symbol": symbol,
        "side": "buy",
        "price": decision.get("price"),
        "size": decision.get("size"),
        "order_type": decision.get("order_type", "limit"),
        "reasoning": decision.get("reasoning", ""),
        "source": "buyer_agent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eur_balance": eur_balance,
    }

    fname = f"buy_{symbol}_{int(time.time())}.json"
    with open(os.path.join(PENDING_DIR, fname), "w") as f:
        json.dump(buy_action, f, indent=2)

    log.info(f"✅ Buy action submitted: {symbol} @ {decision.get('price', '?')} size={decision.get('size', '?')}")
    log.info(f"   Reasoning: {decision.get('reasoning', '')[:200]}")
    log.info(f"   -> {PENDING_DIR}/{fname} (waiting for validator)")

    return {
        "status": "submitted",
        "symbol": symbol,
        "price": decision.get("price"),
        "size": decision.get("size"),
        "reasoning": decision.get("reasoning", ""),
    }


if __name__ == "__main__":
    result = run_buyer()
    print(json.dumps(result, indent=2))