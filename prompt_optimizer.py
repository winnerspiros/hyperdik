#!/usr/bin/env python3
"""
HYPERDIK Prompt Optimizer — automatic LLM prompt improvement.

HOW IT WORKS:
  1. Collects (prompt_template → AI_response → trade_outcome) triples
  2. Evaluates which prompt patterns produce winning trades
  3. Uses the AI itself to suggest better prompts (meta-prompting)
  4. A/B tests variants: runs old vs new prompt, tracks PnL difference
  5. Graduates winning prompts to production

No external dependencies beyond what we already have (OpenRouter API).
"""

import json, os, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).parent
PROMPT_LOG = ROOT / "data" / "prompt_performance.json"


def load_prompt_log() -> dict:
    """Load prompt performance tracking."""
    if PROMPT_LOG.exists():
        try:
            return json.loads(PROMPT_LOG.read_text())
        except Exception:
            pass
    return {"versions": {}, "trials": [], "current": "v1"}


def save_prompt_log(log: dict):
    """Persist prompt performance data."""
    PROMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_LOG.write_text(json.dumps(log, indent=2, default=str))


def record_prompt_outcome(prompt_hash: str, coin: str, pnl_pct: float, 
                          regime: str, confidence: int, side: str):
    """Record that a trade made with this prompt version had this outcome."""
    log = load_prompt_log()
    log["trials"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": prompt_hash,
        "coin": coin, "side": side,
        "pnl_pct": round(pnl_pct, 3),
        "regime": regime, "confidence": confidence,
    })
    # Keep last 500 trials
    if len(log["trials"]) > 500:
        log["trials"] = log["trials"][-500:]
    save_prompt_log(log)


def get_prompt_stats(prompt_hash: str) -> dict:
    """Get win rate and avg PnL for a prompt version."""
    log = load_prompt_log()
    trials = [t for t in log["trials"] if t["version"] == prompt_hash]
    if not trials:
        return {"trials": 0, "win_rate": 0, "avg_pnl": 0}
    wins = [t for t in trials if t["pnl_pct"] > 0]
    return {
        "trials": len(trials),
        "win_rate": round(len(wins) / len(trials) * 100, 1),
        "avg_pnl": round(sum(t["pnl_pct"] for t in trials) / len(trials), 3),
    }


def hash_prompt(prompt_text: str) -> str:
    """Short hash for prompt versioning."""
    return f"v{hashlib.sha256(prompt_text.encode()).hexdigest()[:8]}"


def build_meta_prompt(current_prompt: str, stats: dict, 
                      recent_results: list) -> str:
    """Build a prompt that asks the AI to IMPROVE another prompt.
    
    This is meta-prompting: the AI analyzes what prompts work best
    and generates optimized versions.
    """
    return f"""You are a prompt engineer optimizing a crypto trading AI.

The AI uses this prompt to pick coins:
===== CURRENT PROMPT =====
{current_prompt[:3000]}
===== END PROMPT =====

Performance with this prompt:
  Trials: {stats.get('trials', 0)}
  Win rate: {stats.get('win_rate', 0)}%
  Avg PnL: {stats.get('avg_pnl', 0)}%

Recent results (last 20):
{json.dumps(recent_results[-20:] if recent_results else [], indent=2)[:3000]}

TASK: Rewrite the prompt to improve win rate and total PnL.
- Focus on coin selection criteria, conviction thresholds, regime awareness
- Add specific rules that would have caught winners and avoided losers
- Make the prompt MORE aggressive — prioritize catching trends early
- NEVER make it more conservative — more trades = more profit

Return ONLY the improved prompt text, no JSON wrapper, no commentary.
The prompt must be under 3000 chars to fit in the trading bot's token budget."""


def optimize_prompt_via_ai(current_prompt: str, api_key: str) -> str | None:
    """Use the AI to generate an improved prompt. Returns new prompt or None."""
    if not api_key:
        return None
    
    log = load_prompt_log()
    version = hash_prompt(current_prompt)
    stats = get_prompt_stats(version)
    recent = log["trials"][-50:]
    
    meta = build_meta_prompt(current_prompt, stats, recent)
    
    import urllib.request
    payload = json.dumps({
        "model": "qwen/qwen3-235b-a22b-2507",
        "messages": [
            {"role": "system", "content": "Expert prompt engineer. Output ONLY the improved prompt text, no JSON, no markdown wrapper."},
            {"role": "user", "content": meta}
        ],
        "temperature": 0.5,
        "max_tokens": 1500,
        "plugins": [{"id": "web", "max_results": 3}],
        "provider": {"order": ["Groq", "DeepInfra", "Together"], "allow_fallbacks": True},
    }).encode()
    
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=payload, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "X-Title": "hyperdik-optimizer",
            })
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            content = (resp["choices"][0]["message"]["content"]).strip()
            # Strip any markdown fences
            if content.startswith("```"):
                nl = content.find("\n")
                content = content[nl+1:] if nl > 0 else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()
    except Exception as e:
        print(f"Prompt optimization failed: {e}")
        return None


def graduate_prompt(new_prompt: str, old_hash: str) -> str:
    """Promote a prompt to production. Returns new version hash."""
    log = load_prompt_log()
    new_hash = hash_prompt(new_prompt)
    
    log["versions"][new_hash] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "parent": old_hash,
        "prompt": new_prompt[:5000],
    }
    log["current"] = new_hash
    save_prompt_log(log)
    return new_hash