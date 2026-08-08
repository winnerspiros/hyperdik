#!/usr/bin/env python3
"""
Bank Manager — AI-powered EUR balance manager.
- Reads calendar for portfolio state
- Checks EUR balance, total portfolio value, recent P&L
- AI decides: hold EUR, withdraw to bank, or signal for more buying
- Submits bank actions to pending_actions/ for validator to approve
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BANK] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bank.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Bank")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
PENDING_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
os.makedirs(PENDING_DIR, exist_ok=True)


class BankBrain:
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

    def _call(self, system_prompt, user_prompt, temperature=0.3):
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


def _check_recent_profits():
    """Quick estimate of recent trading profits from DB."""
    try:
        import trader_db
        conn = trader_db.get_db()
        rows = conn.execute(
            "SELECT SUM(pnl) FROM trades WHERE exit_time > datetime('now', '-7 days')"
        ).fetchone()
        conn.close()
        return rows[0] if rows and rows[0] else 0
    except Exception:
        return 0


def run_bank():
    log.info("=" * 60)
    log.info("🏦 BANK MANAGER STARTING")
    log.info("=" * 60)

    # Coordination: check reviewer and predictor
    try:
        from agent_coordination import check_upstream, minutes_since
        if not check_upstream("reviewer", 240, log.info):
            return {"status": "skipped", "reason": "reviewer_stale"}
    except Exception:
        pass

    cal = _load_calendar()
    reviewer = cal.get("reviewer", {})
    predictor = cal.get("predictor", {})

    eur_balance = reviewer.get("eur_balance", 0)
    total_portfolio = reviewer.get("portfolio_value", 0)
    total_value = reviewer.get("total_value", eur_balance + total_portfolio)
    assessment = reviewer.get("overall_assessment", "N/A")
    outlook = predictor.get("market_outlook", "N/A")

    log.info(f"💰 EUR: €{eur_balance:.2f} | Portfolio: €{total_portfolio:.2f} | Total: €{total_value:.2f}")

    # Check recent profits
    recent_pnl = _check_recent_profits()
    log.info(f"📊 Recent P&L (7d): €{recent_pnl:+.2f}")

    # No EUR? Skip
    if eur_balance < 1:
        log.info("No EUR to manage — skipping")
        return {"status": "skipped", "reason": "no_eur"}

    brain = BankBrain()
    if not brain.api_key:
        log.error("No API key")
        return {"status": "error", "reason": "no_api_key"}

    prompt_lines = [
        "You are a BANK MANAGER for a crypto trading fund.",
        "Decide what to do with the EUR balance.",
        "Respond ONLY with JSON. No prose.",
        "",
        "Options:",
        '  "hold" — keep EUR on Revolut for trading opportunities',
        '  "withdraw" — move some EUR to bank (safety)',
        "Consider:",
        "- If portfolio is growing, withdraw profits",
        "- If market is bullish, keep more EUR for buying dips",
        "- If EUR is too high vs portfolio, rebalance",
        "- Leave at least €10 on Revolut for trading",
        "",
        f"EUR Balance: €{eur_balance:.2f}",
        f"Portfolio Value: €{total_portfolio:.2f}",
        f"Total: €{total_value:.2f}",
        f"Recent P&L (7d): €{recent_pnl:+.2f}",
        f"Market Assessment: {assessment}",
        f"Market Outlook: {outlook}",
        "",
        'Respond with JSON:',
        '```json',
        '{',
        '  "decision": "hold | withdraw",',
        '  "withdraw_amount": null,',
        '  "keep_on_exchange": null,',
        '  "reasoning": "why this decision"',
        '}',
        '```',
    ]
    user_prompt = "\n".join(prompt_lines)

    result = brain._call(
        "You are a conservative bank manager. Prioritize capital preservation.",
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
        log.warning(f"Could not parse: {content[:200]}")
        return {"status": "error", "reason": "parse_failed"}

    action = decision.get("decision", "hold")
    log.info(f"🤖 AI Decision: {action}")

    if action == "hold":
        log.info(f"  Keep EUR on exchange: {decision.get('reasoning', '')[:200]}")
        return {"status": "hold", "reasoning": decision.get("reasoning", "")}

    # Submit withdraw action
    withdraw_amount = decision.get("withdraw_amount")
    keep_exchange = decision.get("keep_on_exchange", 10)

    withdraw_action = {
        "type": "bank_withdraw",
        "side": "withdraw",
        "symbol": "EUR",
        "amount": withdraw_amount or max(0, eur_balance - keep_exchange),
        "keep_on_exchange": keep_exchange,
        "reasoning": decision.get("reasoning", ""),
        "source": "bank_manager",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eur_balance_before": eur_balance,
    }

    fname = f"withdraw_{int(time.time())}.json"
    with open(os.path.join(PENDING_DIR, fname), "w") as f:
        json.dump(withdraw_action, f, indent=2)

    log.info(f"✅ Withdraw action submitted: €{withdraw_action['amount']:.2f} (keep €{keep_exchange:.2f})")
    log.info(f"   Reasoning: {decision.get('reasoning', '')[:200]}")

    return {
        "status": "submitted",
        "amount": withdraw_action["amount"],
        "keep_on_exchange": keep_exchange,
        "reasoning": decision.get("reasoning", ""),
    }


if __name__ == "__main__":
    result = run_bank()
    print(json.dumps(result, indent=2))