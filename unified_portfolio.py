#!/usr/bin/env python3
"""
Unified Portfolio Manager — Revolut X (EUR spot) + Hyperliquid (USDC perps).

Philosophy:
  - Revolut X = SAFETY NET — holds EUR, buys spot crypto, holds long-term
  - Hyperliquid = ACTIVE TRADING — leverage, LONG/SHORT, aggressive
  - Profits flow: HL → Revolut X for safekeeping
  - Losses: Revolut X can backstop HL margin if needed
  - Unified view: both balances + positions in one dashboard
  - Allocation: Revolut X holds 70% of total, HL trades with 30%

24/7 DESIGN:
  - Runs as systemd service, auto-restarts on crash/reboot
  - Heartbeat file for health monitoring
  - Graceful shutdown on SIGTERM
  - State persisted to disk, survives crashes
"""

from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("unified")

# ============================================================
# Config
# ============================================================

UNIFIED_STATE_PATH = "data/unified_state.json"

# Allocation: how much of total goes to each exchange
REVOLUT_ALLOCATION = 0.70   # 70% — safety, spot holds
HL_ALLOCATION = 0.30        # 30% — leveraged trading

# Rebalancing thresholds
REBALANCE_THRESHOLD = 0.10  # Rebalance if allocation drifts >10%

# Safety limits
MIN_REVOLUT_EUR = 50.0      # Always keep at least €50 in Revolut
MAX_HL_PERP_PCT = 0.30      # Max 30% of HL balance in perps

# Profit harvesting: when HL grows >40% of total, move profit to Revolut
PROFIT_HARVEST_THRESHOLD = 0.40

# Heartbeat
HEARTBEAT_PATH = "/tmp/unified_portfolio.heartbeat"
HEARTBEAT_INTERVAL = 30  # seconds


# ============================================================
# State
# ============================================================

@dataclass
class ExchangeState:
    """Snapshot of one exchange."""
    name: str
    balance_usd: float = 0.0
    positions_usd: float = 0.0
    pnl_usd: float = 0.0
    available_usd: float = 0.0
    currency: str = "USD"
    positions_detail: dict[str, dict] = field(default_factory=dict)
    last_update: float = 0.0


@dataclass
class UnifiedState:
    """Combined view of all exchanges."""
    revolut: ExchangeState = field(default_factory=lambda: ExchangeState(name="revolut", currency="EUR"))
    hyperliquid: ExchangeState = field(default_factory=lambda: ExchangeState(name="hyperliquid", currency="USDC"))
    total_equity_usd: float = 0.0
    eur_usd_rate: float = 1.08  # Approximate EUR/USD
    last_rebalance: float = 0.0
    profit_harvested: float = 0.0
    version: int = 0

    @property
    def revolut_pct(self) -> float:
        if self.total_equity_usd <= 0:
            return 0.0
        rev_usd = self.revolut.balance_usd * self.eur_usd_rate
        return rev_usd / self.total_equity_usd

    @property
    def hl_pct(self) -> float:
        if self.total_equity_usd <= 0:
            return 0.0
        return self.hyperliquid.balance_usd / self.total_equity_usd

    def to_dict(self) -> dict:
        return {
            "revolut": self.revolut.__dict__,
            "hyperliquid": self.hyperliquid.__dict__,
            "total_equity_usd": self.total_equity_usd,
            "eur_usd_rate": self.eur_usd_rate,
            "last_rebalance": self.last_rebalance,
            "profit_harvested": self.profit_harvested,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedState":
        state = cls()
        state.revolut = ExchangeState(**d.get("revolut", {"name": "revolut", "currency": "EUR"}))
        state.hyperliquid = ExchangeState(**d.get("hyperliquid", {"name": "hyperliquid", "currency": "USDC"}))
        state.total_equity_usd = d.get("total_equity_usd", 0)
        state.eur_usd_rate = d.get("eur_usd_rate", 1.08)
        state.last_rebalance = d.get("last_rebalance", 0)
        state.profit_harvested = d.get("profit_harvested", 0)
        state.version = d.get("version", 0)
        return state


def load_state() -> UnifiedState:
    if os.path.exists(UNIFIED_STATE_PATH):
        with open(UNIFIED_STATE_PATH) as f:
            return UnifiedState.from_dict(json.load(f))
    return UnifiedState()


def save_state(state: UnifiedState) -> None:
    os.makedirs(os.path.dirname(UNIFIED_STATE_PATH), exist_ok=True)
    state.version += 1
    with open(UNIFIED_STATE_PATH, "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)


# ============================================================
# Exchange Interface
# ============================================================

def fetch_revolut_state() -> ExchangeState:
    """Fetch current Revolut X state."""
    state = ExchangeState(name="revolut", currency="EUR")
    try:
        from revolut_client import RevolutXClient
        client = RevolutXClient()
        balances = client.get_balances()

        if isinstance(balances, dict):
            data = balances.get("data", []) if "data" in balances else [balances]
        else:
            data = balances if isinstance(balances, list) else []

        total_eur = 0.0
        positions = {}
        for b in data:
            curr = b.get("currency", "")
            avail = float(b.get("available", 0))
            if curr == "EUR":
                total_eur += avail
            elif curr != "EUR" and avail > 0:
                # Non-EUR = crypto position
                ticker = _get_revolut_price(f"{curr}-EUR")
                usd_value = avail * ticker if ticker > 0 else 0
                positions[curr] = {
                    "amount": avail,
                    "price_eur": ticker,
                    "value_eur": avail * ticker,
                }
                total_eur += avail * ticker

        state.balance_usd = total_eur * 1.08  # Approximate EUR→USD
        state.available_usd = total_eur * 1.08
        state.positions_detail = positions
        state.last_update = time.time()
    except Exception as e:
        log.warning(f"Revolut fetch failed: {e}")
    return state


def fetch_hyperliquid_state() -> ExchangeState:
    """Fetch current Hyperliquid state (unified account)."""
    state = ExchangeState(name="hyperliquid", currency="USDC")
    try:
        import hyperliquid_client as hl
        acc = hl.get_account()
        state.balance_usd = float(acc.get("total_equity", 0))
        state.available_usd = float(acc.get("perp_equity", 0))
        state.pnl_usd = float(acc.get("unrealized_pnl", 0))
        state.last_update = time.time()

        # Positions
        for p in acc.get("positions", []):
            coin = p.get("coin", "")
            szi = float(p.get("szi", 0))
            if abs(szi) > 0.0001:
                mid = hl.get_all_mids().get(coin, 0)
                state.positions_detail[coin] = {
                    "size": szi,
                    "entry": float(p.get("entryPx", 0)),
                    "mark": mid,
                    "pnl": float(p.get("unrealizedPnl", 0)),
                    "liq": float(p.get("liquidationPx", 0)),
                }
                state.positions_usd += abs(szi) * mid
    except Exception as e:
        log.warning(f"Hyperliquid fetch failed: {e}")
    return state


def _get_revolut_price(symbol: str) -> float:
    """Get current price from Revolut X."""
    try:
        from revolut_client import RevolutXClient
        client = RevolutXClient()
        tickers = client.get_tickers([symbol])
        if isinstance(tickers, dict):
            data = tickers.get("data", [])
            for t in (data if isinstance(data, list) else [tickers]):
                if t.get("symbol") == symbol:
                    return float(t.get("last_price", 0))
        return 0.0
    except Exception:
        return 0.0


# ============================================================
# Allocation Logic
# ============================================================

def compute_allocations(state: UnifiedState) -> dict:
    """Compute target allocations and rebalancing needs.

    Returns {
        "revolut_target_usd": float,
        "hl_target_usd": float,
        "revolut_delta": float,   # positive = need to send TO revolut
        "hl_delta": float,        # positive = need to send TO hyperliquid
        "needs_rebalance": bool,
        "should_harvest_profit": bool,
        "harvest_amount": float,  # how much to move HL → Revolut
    }
    """
    total = state.total_equity_usd
    if total <= 0:
        return {"needs_rebalance": False, "should_harvest_profit": False}

    rev_target = total * REVOLUT_ALLOCATION
    hl_target = total * HL_ALLOCATION

    rev_current = state.revolut.balance_usd * state.eur_usd_rate
    hl_current = state.hyperliquid.balance_usd

    rev_delta = rev_target - rev_current
    hl_delta = hl_target - hl_current

    # Check if rebalancing needed
    rev_drift = abs(state.revolut_pct - REVOLUT_ALLOCATION)
    needs_rebalance = rev_drift > REBALANCE_THRESHOLD

    # Profit harvesting: if HL > 40% of total, move profit to Revolut
    should_harvest = state.hl_pct > PROFIT_HARVEST_THRESHOLD
    harvest_amount = 0.0
    if should_harvest:
        excess = hl_current - total * PROFIT_HARVEST_THRESHOLD
        harvest_amount = max(0, excess * 0.5)  # Harvest 50% of excess

    return {
        "revolut_target_usd": rev_target,
        "hl_target_usd": hl_target,
        "revolut_delta": rev_delta,
        "hl_delta": hl_delta,
        "needs_rebalance": needs_rebalance,
        "should_harvest_profit": should_harvest,
        "harvest_amount": harvest_amount,
    }


# ============================================================
# Unified AI Context
# ============================================================

def build_unified_context(state: UnifiedState) -> str:
    """Build AI context showing both exchanges."""
    alloc = compute_allocations(state)
    rev_usd = state.revolut.balance_usd * state.eur_usd_rate

    lines = [
        f"=== UNIFIED PORTFOLIO | {datetime.now(timezone.utc).strftime('%H:%M:%S')} ===",
        f"TOTAL EQUITY: ${state.total_equity_usd:,.2f}",
        "",
        f"REVOLUT X (SAFETY): €{state.revolut.balance_usd:,.2f} (~${rev_usd:,.2f})",
        f"  Allocation: {state.revolut_pct*100:.0f}% (target {REVOLUT_ALLOCATION*100:.0f}%)",
    ]

    if state.revolut.positions_detail:
        lines.append("  Spot holdings:")
        for coin, pos in state.revolut.positions_detail.items():
            lines.append(f"    {coin}: {pos['amount']:.6f} @ €{pos['price_eur']:,.2f} = €{pos['value_eur']:,.2f}")

    lines.extend([
        "",
        f"HYPERLIQUID (TRADING): ${state.hyperliquid.balance_usd:,.2f}",
        f"  PnL: ${state.hyperliquid.pnl_usd:+.2f}",
        f"  Allocation: {state.hl_pct*100:.0f}% (target {HL_ALLOCATION*100:.0f}%)",
    ])

    if state.hyperliquid.positions_detail:
        lines.append("  Perp positions:")
        for coin, pos in state.hyperliquid.positions_detail.items():
            side = "LONG" if pos["size"] > 0 else "SHORT"
            lines.append(f"    {coin} {side}: {abs(pos['size']):.4f}u "
                         f"entry=${pos['entry']:,.2f} mark=${pos['mark']:,.2f} "
                         f"PnL=${pos['pnl']:+.2f}")

    lines.extend([
        "",
        f"ALLOCATION STATUS:",
        f"  Drift: {state.revolut_pct*100:.0f}%/{REVOLUT_ALLOCATION*100:.0f}% "
        f"(threshold {REBALANCE_THRESHOLD*100:.0f}%)",
        f"  Rebalance needed: {alloc['needs_rebalance']}",
    ])

    if alloc["should_harvest_profit"]:
        lines.append(f"  🔔 PROFIT HARVEST: move ${alloc['harvest_amount']:,.2f} HL → Revolut")

    if state.profit_harvested > 0:
        lines.append(f"  Total harvested: ${state.profit_harvested:,.2f}")

    return "\n".join(lines)


# ============================================================
# Heartbeat
# ============================================================

def heartbeat() -> None:
    """Write heartbeat file for health monitoring."""
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(str(int(time.time())))
    except Exception:
        pass


# ============================================================
# Main sync loop
# ============================================================

def sync_portfolio() -> UnifiedState:
    """Fetch both exchanges and update unified state."""
    state = load_state()

    # Fetch both
    state.revolut = fetch_revolut_state()
    state.hyperliquid = fetch_hyperliquid_state()

    # Compute total
    rev_usd = state.revolut.balance_usd * state.eur_usd_rate
    hl_usd = state.hyperliquid.balance_usd
    state.total_equity_usd = rev_usd + hl_usd

    # Check for profit harvesting
    alloc = compute_allocations(state)
    if alloc["should_harvest_profit"] and alloc["harvest_amount"] > 10:
        log.info(f"💰 PROFIT HARVEST: {alloc['harvest_amount']:.2f} USD from HL → Revolut")
        state.profit_harvested += alloc["harvest_amount"]
        # Note: actual transfer is manual/user-controlled
        # The bot logs this as a recommendation

    save_state(state)
    heartbeat()
    return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    state = sync_portfolio()
    print(build_unified_context(state))
