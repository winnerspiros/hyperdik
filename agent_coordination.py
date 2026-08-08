#!/usr/bin/env python3
"""
Agent coordination utilities — shared across all agents.
Each agent checks if upstream agents have run recently before acting.
"""
import json, os, time
from datetime import datetime, timezone

CALENDAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "strategy_calendar.json")

def _load_calendar():
    try:
        with open(CALENDAR_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def minutes_since(agent_name):
    """Minutes since a given agent last ran. Returns None if never ran."""
    cal = _load_calendar()
    section = cal.get(agent_name, {})
    last_run = section.get("last_run")
    if not last_run:
        return None
    try:
        dt = datetime.fromisoformat(last_run)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None

def check_upstream(agent_name, max_minutes, log_fn=None):
    """Check if upstream agent ran recently. Returns True if OK, False if stale."""
    mins = minutes_since(agent_name)
    if mins is None:
        msg = f"  ⏸️ {agent_name} has never run — skipping"
        if log_fn:
            log_fn(msg)
        return False
    if mins > max_minutes:
        msg = f"  ⏸️ {agent_name} last ran {mins:.0f}m ago (max {max_minutes}m) — skipping"
        if log_fn:
            log_fn(msg)
        return False
    return True

def get_reviewer_state():
    """Get the latest reviewer assessment summary."""
    cal = _load_calendar()
    r = cal.get("reviewer", {})
    return {
        "assessment": r.get("overall_assessment", "UNKNOWN"),
        "health": r.get("portfolio_health", "UNKNOWN"),
        "eur": r.get("eur_balance", 0),
        "portfolio_value": r.get("portfolio_value", 0),
        "total_value": r.get("total_value", 0),
        "positions": r.get("positions", []),
        "last_run": r.get("last_run", "never"),
    }

def get_predictor_state():
    """Get the latest predictor outlook."""
    cal = _load_calendar()
    p = cal.get("predictor", {})
    return {
        "outlook": p.get("market_outlook", "UNCERTAIN"),
        "strategy": p.get("active_strategy", "N/A"),
        "last_run": p.get("last_run", "never"),
    }

def get_pending_count():
    """Count pending actions."""
    pending_dir = os.path.join(os.path.dirname(CALENDAR_PATH), "pending_actions")
    try:
        return len([f for f in os.listdir(pending_dir) if f.endswith(".json")])
    except FileNotFoundError:
        return 0

def get_approved_count():
    """Count approved actions waiting to be executed."""
    approved_dir = os.path.join(os.path.dirname(CALENDAR_PATH), "approved_actions")
    try:
        return len([f for f in os.listdir(approved_dir) if f.endswith(".json")])
    except FileNotFoundError:
        return 0


# ── EUR reservation ledger ───────────────────────────────────────────────────
# Prevents two concurrent proposals (or a proposal + a stale one) from both
# assuming the same EUR is free to spend. A reservation is a soft claim with
# a TTL — if the batch never gets executed (validator rejects everything, or
# the executor crashes), the reservation expires on its own rather than
# permanently locking cash.

RESERVATIONS_PATH = os.path.join(os.path.dirname(CALENDAR_PATH), "eur_reservations.json")


def _load_reservations():
    try:
        with open(RESERVATIONS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_reservations(data):
    with open(RESERVATIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _prune_expired(reservations):
    now = time.time()
    return {k: v for k, v in reservations.items() if v.get("expires_at", 0) > now}


def reserve_eur(batch_id, amount, ttl_minutes=15):
    """Reserve `amount` EUR under `batch_id` for `ttl_minutes`. Returns the reservation."""
    reservations = _prune_expired(_load_reservations())
    reservations[batch_id] = {
        "amount": amount,
        "reserved_at": time.time(),
        "expires_at": time.time() + ttl_minutes * 60,
    }
    _save_reservations(reservations)
    return reservations[batch_id]


def release_eur(batch_id):
    """Release a reservation early (e.g. once the batch is fully executed or rejected)."""
    reservations = _prune_expired(_load_reservations())
    reservations.pop(batch_id, None)
    _save_reservations(reservations)


def get_reserved_eur():
    """Total EUR currently reserved across all live (non-expired) batches."""
    reservations = _prune_expired(_load_reservations())
    _save_reservations(reservations)  # persist pruning
    return sum(v.get("amount", 0) for v in reservations.values())


def get_available_eur(actual_balance):
    """EUR balance minus anything already reserved by an in-flight batch."""
    return max(0.0, actual_balance - get_reserved_eur())


# ── Symbol locks ─────────────────────────────────────────────────────────────
# Prevents two agents from proposing conflicting actions on the same symbol
# in the same window (e.g. strategist proposes SELL LDO while a stale queue
# item still proposes BUY LDO).

LOCKS_PATH = os.path.join(os.path.dirname(CALENDAR_PATH), "symbol_locks.json")


def _load_locks():
    try:
        with open(LOCKS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_locks(data):
    with open(LOCKS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def is_symbol_locked(symbol, ttl_minutes=10):
    locks = _load_locks()
    entry = locks.get(symbol)
    if not entry:
        return False
    age_min = (time.time() - entry.get("locked_at", 0)) / 60.0
    return age_min < ttl_minutes


def lock_symbol(symbol, owner, ttl_minutes=10):
    locks = _load_locks()
    locks[symbol] = {"locked_at": time.time(), "owner": owner, "ttl_minutes": ttl_minutes}
    _save_locks(locks)


def unlock_symbol(symbol):
    locks = _load_locks()
    locks.pop(symbol, None)
    _save_locks(locks)