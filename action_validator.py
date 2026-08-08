#!/usr/bin/env python3
"""
Action Validator — Premium model validates ALL Revolut X actions before execution.
- Reads proposed action (buy/sell/withdraw) from calendar
- Checks current market context + portfolio state
- Approves/Rejects/Modifies with reasoning
|- Uses Gemini Flash Lite (cheap) — data-intensive decisions use premium in strategist
- Only writes to the "approved_actions" queue — separate executors pick up
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VALIDATOR] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "validator.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Validator")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
ACTIONS_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
APPROVED_DIR = os.path.join(TRADER_DIR, "data", "approved_actions")
os.makedirs(ACTIONS_DIR, exist_ok=True)
os.makedirs(APPROVED_DIR, exist_ok=True)

# ── Premium model ────────────────────────────────────────────────────────────

class ValidatorBrain:
    """Premium LLM — validates every action before Revolut execution."""

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
        self.model = "google/gemini-2.5-flash-lite"  # Cheap model for all validations

    def _call(self, system_prompt, user_prompt, temperature=0.3, max_retries=2):
        import urllib.request, ssl
        if not self.api_key:
            return {"error": "no_api_key"}
        
        last_error = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait = 2 ** attempt  # 2s, 4s backoff
                time.sleep(wait)
            try:
                ctx = ssl.create_default_context()
                data = json.dumps({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": temperature,
                }).encode()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-OpenRouter-Cache": "true",
                    },
                )
                with urllib.request.urlopen(req, context=ctx, timeout=45) as resp:
                    raw_bytes = resp.read()
                    result = json.loads(raw_bytes)
                    if "choices" in result and len(result["choices"]) > 0:
                        msg = result["choices"][0].get("message", {})
                        content = msg.get("content")
                        if content:
                            return {"content": content.strip()}
                        else:
                            return {"error": f"empty_content: {json.dumps(msg)[:200]}"}
                    else:
                        return {"error": f"no_choices: {json.dumps(result)[:200]}"}
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code in (402, 429, 502, 503, 504):
                    if attempt < max_retries:
                        continue  # Retry on rate-limit/server errors
                return {"error": last_error}
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    continue
                return {"error": last_error}
        
        return {"error": last_error or "retry_exhausted"}


def _load_calendar():
    try:
        with open(CALENDAR_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"version": 1, "last_updated": None, "reviewer": {}, "predictor": {}, "validator": {}}


def _get_pending_actions():
    """Read all pending action files (submitted by buyer, seller, bank agents)."""
    actions = []
    for fname in os.listdir(ACTIONS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(ACTIONS_DIR, fname)) as f:
                    action = json.load(f)
                    action["_file"] = fname
                    actions.append(action)
            except Exception as e:
                log.warning(f"Bad action file {fname}: {e}")
    return actions


def _save_approved(action, verdict):
    """Move approved action to approved queue with validator's reasoning."""
    verdict_str = verdict.get("verdict", "REJECTED")
    
    # Only move to approved if actually APPROVED
    if verdict_str != "APPROVE":
        log.info(f"  {verdict_str}: {action.get('type', '?')} {action.get('symbol', '?')} — {verdict.get('reasoning', '')[:100]}")
        # Remove from pending without approving
        fname = action.get("_file", "")
        if fname:
            src = os.path.join(ACTIONS_DIR, fname)
            if os.path.exists(src):
                os.remove(src)
        return {"verdict": verdict_str, "reasoning": verdict.get("reasoning", "")}

    approved = {
        "action": action,
        "validator_verdict": verdict_str,
        "validator_reasoning": verdict.get("reasoning", ""),
        "modified_price": verdict.get("modified_price"),
        "modified_size": verdict.get("modified_size"),
        "use_limit": verdict.get("use_limit", False),
        "limit_price": verdict.get("limit_price"),
        "validator_confidence": verdict.get("validator_confidence", 50),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Write to approved dir
    fname = action.get("_file", f"action_{int(time.time())}.json")
    outpath = os.path.join(APPROVED_DIR, f"approved_{fname}")
    with open(outpath, "w") as f:
        json.dump(approved, f, indent=2)

    # Remove from pending
    src = os.path.join(ACTIONS_DIR, fname)
    if os.path.exists(src):
        os.remove(src)

    log.info(f"  {'APPROVED' if verdict.get('verdict') == 'APPROVE' else 'REJECTED'}: {action.get('type', '?')} {action.get('symbol', '?')} — {verdict.get('reasoning', '')[:100]}")
    return approved


def validate_action(action, calendar, market_data):
    """Validate a single action with the premium model."""
    brain = ValidatorBrain()
    if not brain.api_key:
        log.error("❌ No API key for validator")
        return {"verdict": "ERROR", "reasoning": "no_api_key"}

    action_type = action.get("type", "unknown")
    symbol = action.get("symbol", "?")
    side = action.get("side", "CLOSE" if action_type == "close" else "?")
    price = action.get("price", "?")
    size = action.get("size", "?")
    reasoning = action.get("reasoning", action.get("reason", "none given"))

    # Build compact prompt for validator
    reviewer = calendar.get("reviewer", {})
    predictor = calendar.get("predictor", {})
    lines = [
        f"VAL:{action_type} {symbol} {side} price={price} size={size}",
        f"AGENT:{reasoning[:80]}",
        f"REVIEWER:{reviewer.get('overall_assessment','?')} health={reviewer.get('portfolio_health','?')} mkt={reviewer.get('market_trend','?')} concern={reviewer.get('top_concern','')[:60]}",
        f"PREDICTOR:{predictor.get('market_outlook','?')} strat={predictor.get('active_strategy','')[:60]} risk={predictor.get('biggest_risk','')[:60]}",
    ]
    if market_data:
        md_parts = [f"{k}={str(v)[:40]}" for k, v in market_data.items() if v]
        if md_parts:
            lines.append(f"MKT:{' '.join(md_parts[:6])}")
    lines.append('JSON:{"v":"APPROVE|REJECT|MODIFY","conf":0-100,"rf":["wrong_side","bad_price","absurd_size","ok","stale_price","low_liq"],"mp":null,"ms":null,"limit":false,"lp":null}')
    lines.append("limit=true → use limit order at lp price (better fill). mp/ms=modify price/size. conf=your certainty.")
    prompt = "\n".join(lines)
    result = brain._call(
        "Validate trades. Default APPROVE unless clear error. Output ONLY valid JSON.",
        prompt,
    )

    if "error" in result:
        log.error(f"Validator call failed: {result['error']}")
        return {"verdict": "ERROR", "reasoning": result["error"]}

    # Parse JSON from response — multiple fallback strategies
    content = result["content"]
    verdict = None
    
    # Strategy 1: JSON in code block (```json ... ``` or ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        block = json_match.group(1).strip()
        try:
            verdict = json.loads(block)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Raw JSON anywhere in response
    if verdict is None:
        # Find the outermost {...} JSON object
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                verdict = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    
    # Strategy 3: Try to fix common AI formatting issues
    if verdict is None:
        try:
            # Remove trailing commas, fix unquoted keys (common LLM mistakes)
            cleaned = re.sub(r',\s*}', '}', content)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                verdict = json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    
    if verdict:
        # Normalize compact keys (v,rf,mp,ms) to verbose (verdict,reasoning,modified_price,modified_size)
        if "v" in verdict and "verdict" not in verdict:
            verdict["verdict"] = verdict.pop("v")
        if "rf" in verdict and "reasoning" not in verdict:
            rf_val = verdict.pop("rf")
            verdict["reasoning"] = ",".join(rf_val) if isinstance(rf_val, list) else str(rf_val)
        if "mp" in verdict and "modified_price" not in verdict:
            verdict["modified_price"] = verdict.pop("mp")
        if "ms" in verdict and "modified_size" not in verdict:
            verdict["modified_size"] = verdict.pop("ms")
        if "limit" in verdict:
            verdict["use_limit"] = verdict.get("limit", False)
            verdict["limit_price"] = verdict.get("lp")
        if "conf" in verdict:
            verdict["validator_confidence"] = verdict.get("conf", 50)
        return verdict
    
    # All strategies failed — default to APPROVE for alerts (non-trade actions)
    # Better to let an alert through than silently reject it
    action_type = action.get("type", "")
    if action_type in ("alert", "notify"):
        log.warning(f"Could not parse validator response for {action_type}, defaulting to APPROVE: {content[:120]}")
        return {"verdict": "APPROVE", "reasoning": "parse_recovery: non-trade action, default approve"}
    
    log.warning(f"Could not parse validator response: {content[:200]}")
    return {"verdict": "REJECTED", "reasoning": "parse_failed", "modified_price": None, "modified_size": None}


def run_validation():
    """Main loop: check pending actions, validate each, move to approved."""
    log.info("=" * 60)
    log.info("✅ ACTION VALIDATOR STARTING")
    log.info("=" * 60)

    # Check for pending actions
    pending = _get_pending_actions()
    if not pending:
        log.info("No pending actions to validate")
        return {"status": "idle", "validated": 0}

    log.info(f"📋 Found {len(pending)} pending action(s)")

    # Load calendar for context
    cal = _load_calendar()

    # Collect market data snapshot
    market_data = {}
    try:
        import data_enricher
        fg = data_enricher.get_fear_greed()
        if fg:
            market_data["fear_greed"] = fg
        global_data = data_enricher.get_global_data()
        if global_data:
            market_data["global"] = global_data
    except Exception:
        pass

    results = []
    for action in pending:
        log.info(f"  Validating: {action.get('type', '?')} {action.get('side', '?')} {action.get('symbol', '?')} @ {action.get('price', '?')}")
        verdict = validate_action(action, cal, market_data)
        approved = _save_approved(action, verdict)
        results.append(approved)

    log.info(f"✅ Validated {len(results)} actions")
    return {"status": "ok", "validated": len(results), "results": results}


if __name__ == "__main__":
    result = run_validation()
    print(json.dumps(result, indent=2, default=str))