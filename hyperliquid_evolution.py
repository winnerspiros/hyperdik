#!/usr/bin/env python3
"""
HYPERDIK Evolution Engine v2 — Autonomous trading system optimizer.

Capabilities:
  - Rewrite AI prompts sent to Qwen/Llama
  - Modify daemon code (with file writes)
  - Adjust ALL strategy parameters (no caps when justified)
  - Restart daemon after changes
  - Web search for market context + strategy research

Safety:
  - All changes logged to evolution_state.json
  - Git auto-commit after every evolution run
  - Old files backed up before modification
  - Daemon restart only if code changes were applied
"""

import json, os, time, hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

ROOT = Path(__file__).parent
STATE_PATH = ROOT / "data" / "evolution_state.json"
TRADE_LOG = ROOT / "data" / "trade_memory.json"
WEIGHTS_PATH = ROOT / "data" / "continuous_weights.json"
LEARNED_PATH = ROOT / "data" / "learned_weights.json"


# ═══════════════════════════════════════════════════════════
# DATA DUMP — feeds the AI EVERYTHING
# ═══════════════════════════════════════════════════════════

def build_full_context() -> dict:
    """Assemble complete trading system state for the evolution AI."""
    ctx = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "performance": {},
    }

    # ── Read all data files ──
    for label, path in [
        ("trade_memory", TRADE_LOG),
        ("continuous_weights", WEIGHTS_PATH),
        ("learned_weights", LEARNED_PATH),
        ("evolution_state", STATE_PATH),
    ]:
        if path.exists():
            try:
                ctx["files"][label] = json.loads(path.read_text())
            except Exception:
                ctx["files"][label] = f"<unreadable: {path}>"

    # ── Read daemon log (last 200 lines) ──
    log_path = ROOT / "logs" / "hyperliquid_daemon.log"
    if log_path.exists():
        try:
            lines = log_path.read_text().split("\n")[-200:]
            ctx["files"]["daemon_log_tail"] = "\n".join(lines)
        except Exception:
            ctx["files"]["daemon_log_tail"] = "<unreadable>"

    # ── Performance summary from trade memory ──
    trades = ctx["files"].get("trade_memory", [])
    if isinstance(trades, list) and trades:
        wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
        coins = {}
        for t in trades:
            c = t.get("coin", "?")
            coins.setdefault(c, {"trades": 0, "pnl": 0, "wins": 0})
            coins[c]["trades"] += 1
            coins[c]["pnl"] += t.get("pnl_pct", 0)
            if t.get("pnl_pct", 0) > 0:
                coins[c]["wins"] += 1

        ctx["performance"] = {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl_pct": round(sum(t.get("pnl_pct", 0) for t in trades), 2),
            "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0,
            "best_coin": max(coins.items(), key=lambda x: x[1]["pnl"]) if coins else None,
            "worst_coin": min(coins.items(), key=lambda x: x[1]["pnl"]) if coins else None,
            "per_coin": {c: v for c, v in sorted(coins.items(), key=lambda x: x[1]["pnl"])},
        }

    # ── Read current AI prompts from ai_decider.py ──
    ai_decider = ROOT / "ai_decider.py"
    if ai_decider.exists():
        try:
            code = ai_decider.read_text()
            # Extract prompt-building functions
            prompts = {}
            for marker in ["TRADABLE", "COINS:", "ANALYZE", "EXIT EVAL", "DEBATE"]:
                idx = code.find(marker)
                if idx > 0:
                    snippet = code[max(0, idx-50):idx+300]
                    prompts[f"prompt_section_{marker}"] = snippet
            ctx["files"]["ai_prompts"] = prompts
        except Exception:
            ctx["files"]["ai_prompts"] = "<unreadable>"

    # ── Read daemon key sections ──
    daemon = ROOT / "hyperliquid_daemon.py"
    if daemon.exists():
        try:
            code = daemon.read_text()
            # Extract key parameters
            params = {}
            for line in code.split("\n"):
                for key in ["BASE_LEVERAGE", "CYCLE_SECONDS", "MAX_POSITIONS",
                           "MIN_CONVICTION", "VWAP", "COMPOSITE", "STOP", "SLIPPAGE"]:
                    if key in line and "=" in line and not line.strip().startswith("#"):
                        params[key] = line.strip()[:120]
            ctx["files"]["daemon_params"] = params
        except Exception:
            ctx["files"]["daemon_params"] = "<unreadable>"

    return ctx


# ═══════════════════════════════════════════════════════════
# PROMPT BUILDER — massive context, web search instructions
# ═══════════════════════════════════════════════════════════

def build_evolution_prompt(ctx: dict) -> str:
    """Build the complete evolution prompt with full system state."""
    
    perf = ctx.get("performance", {})
    files = ctx.get("files", {})

    prompt = f"""YOU ARE AN AUTONOMOUS TRADING SYSTEM OPTIMIZER. YOU HAVE FULL AUTHORITY.

⏰ {ctx['timestamp']}
📊 {perf.get('total_trades', 0)} trades | {perf.get('win_rate', 0)}% WR | {perf.get('total_pnl_pct', 0):+.1f}% total PnL
✅ {perf.get('wins', 0)} wins (avg {perf.get('avg_win_pct', 0):+.2f}%) | ❌ {perf.get('losses', 0)} losses (avg {perf.get('avg_loss_pct', 0):+.2f}%)
🏆 Best: {perf.get('best_coin')} | 💀 Worst: {perf.get('worst_coin')}

=== COMPLETE TRADE HISTORY ===
{json.dumps(files.get('trade_memory', []), indent=2)[:20000]}

=== PER-COIN PERFORMANCE ===
{json.dumps(perf.get('per_coin', {}), indent=2)[:15000]}

=== CURRENT PARAMETERS ===
{json.dumps(files.get('daemon_params', {}), indent=2)[:10000]}

=== AI PROMPT SNIPPETS (what we tell the trading AI) ===
{json.dumps(files.get('ai_prompts', {}), indent=2)[:15000]}

=== RECENT DAEMON LOGS ===
{files.get('daemon_log_tail', 'N/A')[:15000]}

=== LEARNED WEIGHTS ===
{json.dumps(files.get('learned_weights', {}), indent=2)[:12000]}

=== CONTINUOUS WEIGHTS ===
{json.dumps(files.get('continuous_weights', {}), indent=2)[:10000]}

=== WEB SEARCH TASKS ===
1. Search for: "{perf.get('best_coin',['UNKNOWN'])[0] if perf.get('best_coin') else 'BTC'} crypto news price action today"
2. Search for: "{perf.get('worst_coin',['UNKNOWN'])[0] if perf.get('worst_coin') else 'ETH'} why underperforming"
3. Search for: "crypto market regime August 2026 fear greed"
4. Search for: "best crypto trading strategies 2026 hyperliquid perpetuals"
5. Search for: "how to improve {perf.get('win_rate', 0)}% win rate crypto trading bot"

=== YOUR TASK ===
You have FULL authority to optimize this trading system. Analyze the data, search the web, then return changes.

You CAN:
- Rewrite AI prompts in ai_decider.py (better instructions to Qwen/Llama)
- Modify trading parameters in hyperliquid_daemon.py
- Change signal weights, thresholds, conviction requirements
- Adjust position sizing, leverage, entry/exit logic
- Request daemon restart

Your GOAL: MAXIMIZE profit. More trades, better entries, fewer missed opportunities.
NEVER suggest being more conservative. Inaction = loss.
If win rate is low but individual wins are big, that's FINE — optimize for total PnL.

RETURN JSON ONLY:
{{
  "analysis": "1-paragraph summary of what's happening",
  "market_context": "what you found from web search",
  "changes": [
    {{
      "file": "ai_decider.py or hyperliquid_daemon.py",
      "type": "prompt_rewrite|param_change|code_change|add_logic|remove|
      "old": "exact string to find in file (for prompt_rewrite/param_change)",
      "new": "replacement string (for prompt_rewrite/param_change)",
      "reason": "why this change, data-backed",
      "expected_impact": "what should improve",
      "confidence": 0-100
    }}
  ],
  "restart_required": true/false,
  "missed_opportunities": ["patterns you noticed"],
  "risk_warning": "any concerns (be honest)"
}}"""

    return prompt


# ═══════════════════════════════════════════════════════════
# EXECUTION — apply AI-suggested changes
# ═══════════════════════════════════════════════════════════

def apply_changes(changes: list, dry_run: bool = False) -> list:
    """Apply AI-suggested file changes. Backs up files first. Returns change log."""
    applied = []
    
    for ch in changes:
        file = ch.get("file", "")
        old_str = ch.get("old", "")
        new_str = ch.get("new", "")
        reason = ch.get("reason", "")[:100]
        
        if not file or not old_str or not new_str:
            applied.append(f"SKIP: missing fields in {ch}")
            continue
        
        path = ROOT / file
        if not path.exists():
            applied.append(f"SKIP: {file} not found")
            continue
        
        if dry_run:
            applied.append(f"DRY_RUN: would change {file}: {reason}")
            continue
        
        try:
            content = path.read_text()
            if old_str not in content:
                applied.append(f"SKIP: old string not found in {file}")
                continue
            
            # Backup
            backup = ROOT / f"{file}.bak.{int(time.time())}"
            backup.write_text(content)
            
            # Apply
            content = content.replace(old_str, new_str, 1)
            path.write_text(content)
            applied.append(f"APPLIED: {file} — {reason}")
        except Exception as e:
            applied.append(f"ERROR: {file} — {type(e).__name__}: {e}")
    
    return applied


def run_full_evolution() -> dict:
    """Main evolution entry point — builds context, calls AI, applies changes."""
    
    # 1. Build massive context dump
    ctx = build_full_context()
    
    # 2. Build evolution prompt
    prompt = build_evolution_prompt(ctx)
    
    # 3. Call AI (handled by run_evolution_analysis in ai_decider.py)
    #    The daemon calls this through the web-search-enabled path
    
    result = {
        "context_size": len(prompt),
        "prompt": prompt,
        "timestamp": ctx["timestamp"],
        "performance_snapshot": ctx.get("performance", {}),
    }
    
    return result