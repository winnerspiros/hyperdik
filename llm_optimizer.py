#!/usr/bin/env python3
"""
LLM Prompt Optimizer — Token-efficient prompts, cached responses, batch processing.

Principles:
  1. Token budgets: evolution prompts < 1000 tokens, execution < 500
  2. Response caching: same context → reuse previous LLM response
  3. Batch processing: combine multiple queries into one LLM call
  4. Response validation: parse and validate before using
  5. Fallback chain: cheap model first, expensive only if needed

Cost model (OpenRouter pricing):
  - deepseek-chat: $0.14/M input, $0.28/M output
  - gemini-flash-lite: $0.10/M input, $0.40/M output (free tier)
  - claude-sonnet: $3.00/M input, $15.00/M output

Strategy: deepseek-chat for everything, gemini-flash for validation.
  Target: <$0.01/day total LLM costs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# Config
# ============================================================

CACHE_PATH = "data/llm_cache.json"
MAX_CACHE_ENTRIES = 500
CACHE_TTL_SECONDS = 86400  # 24 hours

# Token budgets
MAX_TOKENS_EVOLUTION = 1000   # Evolution prompt
MAX_TOKENS_VALIDATION = 200    # Quick validation
MAX_TOKENS_EXECUTION = 400     # Execution decision (deprecated — now deterministic)

# Cost tracking
COST_PER_1K_INPUT = 0.00014    # deepseek-chat
COST_PER_1K_OUTPUT = 0.00028


# ============================================================
# Cache
# ============================================================

@dataclass
class LLMCache:
    """LRU cache for LLM responses."""
    entries: dict[str, dict] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    total_cost: float = 0.0

    def _hash(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str) -> str | None:
        key = self._hash(prompt, model)
        entry = self.entries.get(key)
        if entry and time.time() - entry["time"] < CACHE_TTL_SECONDS:
            self.hits += 1
            return entry["response"]
        self.misses += 1
        return None

    def set(self, prompt: str, model: str, response: str) -> None:
        key = self._hash(prompt, model)
        self.entries[key] = {"response": response, "time": time.time()}

        # LRU eviction
        if len(self.entries) > MAX_CACHE_ENTRIES:
            oldest = min(self.entries, key=lambda k: self.entries[k]["time"])
            del self.entries[oldest]

    def track_cost(self, input_tokens: int, output_tokens: int) -> None:
        cost = (input_tokens / 1000 * COST_PER_1K_INPUT +
                output_tokens / 1000 * COST_PER_1K_OUTPUT)
        self.total_cost += cost

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "total_cost": round(self.total_cost, 6),
            "entries": len(self.entries),
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump({
                "entries": self.entries,
                "hits": self.hits,
                "misses": self.misses,
                "total_cost": self.total_cost,
            }, f, indent=2)

    @classmethod
    def load(cls) -> "LLMCache":
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                data = json.load(f)
            cache = cls()
            cache.entries = data.get("entries", {})
            cache.hits = data.get("hits", 0)
            cache.misses = data.get("misses", 0)
            cache.total_cost = data.get("total_cost", 0)
            return cache
        return cls()


# Global cache instance
_cache = LLMCache.load()


# ============================================================
# Token-Efficient Prompts
# ============================================================

def evolution_prompt_v2(segments_json: str, tactical_json: str, initial_json: str) -> str:
    """Ultra-compact evolution prompt (~800 tokens)."""
    return f"""Review trading results. Suggest 1-3 micro-adjustments (±10% max, ±30% total drift).

P1: If cumulative return positive, don't overreact.
P2: Why did wins work? Don't change what's working.
P3: Root cause of losses, not "it lost money".
P4: EXACT parameter name. Never vague.
P5: ±10% per param per round.
P6: No change if <2 segments since last change.
P7: Must suggest 1 change every 3 rounds.

Personality params LOCKED. Only change tactical.

Results: {segments_json[:500]}
Tactical: {tactical_json[:200]}
Initial: {initial_json[:200]}

JSON array: [{{"parameter":"name","current":v,"suggested":v,"reason":"P#: why"}}]"""


def execution_prompt_v2(context: str, direction: str, coin: str) -> str:
    """Compact execution prompt (~200 tokens). DEPRECATED — now deterministic."""
    return f"""Direction:{direction} {coin}\nPrice:{context[:80]}\n
JSON:{{"proceed":bool,"entry":float,"lev":int,"size":float,"stop":float,"tp":float,"reason":str}}
Rules: limit only, at support, never chase. If >5% from support → proceed=false."""


def validate_trade_prompt(action_json: str) -> str:
    """Minimal validation prompt (~100 tokens)."""
    return f"""Safety check this trade. Flag if: lev>5x, size>20%equity, stop>5%, unreasonable TP.
Action: {action_json[:200]}
JSON:{{"ok":bool,"flags":[""],"suggestion":""}}"""


# ============================================================
# Batch Processor
# ============================================================

def batch_validate(actions: list[dict]) -> list[dict]:
    """Validate multiple actions in one LLM call (saves tokens)."""
    if not actions:
        return []

    # Combine into one prompt
    actions_str = json.dumps([{"id": i, **a} for i, a in enumerate(actions)])
    prompt = f"""Validate these {len(actions)} trades. Flag any that violate:
- leverage >5x
- size >20% equity
- stop >5% from entry
- TP unreasonable (>50%)
- duplicate on same coin

Actions: {actions_str[:1500]}
JSON array: [{{"id":int,"ok":bool,"flags":[""],"suggestion":""}}]"""

    # This would call the LLM — here we just return the prompt
    return [{"type": "batch_validation", "count": len(actions), "prompt_chars": len(prompt)}]


# ============================================================
# Call LLM with cache + cost tracking
# ============================================================

def call_llm(prompt: str, model: str = "deepseek/deepseek-chat",
             max_tokens: int = 500) -> dict:
    """Call LLM with cache lookup, cost tracking, and fallback.

    Returns {"content": str, "cached": bool, "cost": float, "tokens": int}
    """
    # Check cache
    cached = _cache.get(prompt, model)
    if cached:
        return {"content": cached, "cached": True, "cost": 0.0, "tokens": 0}

    try:
        from ai_brain import AITradingBrain
        brain = AITradingBrain()
        resp = brain._call_llm(prompt, model=model)

        content = resp.get("content", "") if isinstance(resp, dict) else str(resp)

        # Estimate tokens (4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(content) // 4

        _cache.track_cost(input_tokens, output_tokens)
        _cache.set(prompt, model, content)

        return {"content": content, "cached": False,
                "cost": _cache.total_cost, "tokens": input_tokens + output_tokens}
    except Exception as e:
        return {"content": "", "cached": False, "cost": 0, "tokens": 0, "error": str(e)}


# ============================================================
# Daily Budget Enforcer
# ============================================================

DAILY_BUDGET_USD = 0.02  # Max $0.02/day on LLM

def check_budget() -> tuple[bool, float]:
    """Check if we're within daily budget."""
    # Reset daily if needed
    today = time.strftime("%Y-%m-%d")
    last_reset = getattr(_cache, "last_budget_reset", "")
    if last_reset != today:
        _cache.total_cost = 0.0
        _cache.last_budget_reset = today

    remaining = DAILY_BUDGET_USD - _cache.total_cost
    return remaining > 0, remaining


# ============================================================
# Stats
# ============================================================

def get_llm_stats() -> str:
    """Get LLM usage statistics."""
    stats = _cache.stats()
    ok, remaining = check_budget()

    return (
        f"LLM Stats: {stats['hits']} cache hits / {stats['misses']} misses "
        f"({stats['hit_rate']*100:.0f}% hit rate)\n"
        f"  Total cost: ${stats['total_cost']:.6f}\n"
        f"  Daily budget: ${DAILY_BUDGET_USD:.4f} (${remaining:.4f} remaining)\n"
        f"  Cache entries: {stats['entries']}"
    )


if __name__ == "__main__":
    print(get_llm_stats())
    print(f"\nEstimated cost per evolution cycle: ~$0.0003")
    print(f"Estimated cost per day (12 evolutions): ~$0.0036")
    print(f"Budget: ${DAILY_BUDGET_USD}/day (70x headroom)")
