#!/usr/bin/env python3
"""
Portfolio Bridge — links Revolut X and Hyperliquid into one unified view.

Wires both exchanges together for:
- Unified equity (shared between daemons)
- Profit harvesting signals (HL→Revolut when HL>40%)
- Allocation drift monitoring
- Risk-aware sizing (reduce leverage if safety vault is low)

Lightweight: no extra process. Each daemon imports and calls get_unified().
State cached to data/unified_portfolio.json, refreshed every 60s.
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
STATE_PATH = ROOT / "data" / "unified_portfolio.json"

log = logging.getLogger("bridge")

# ── Allocation ──
REVOLUT_ALLOCATION = 0.70
HL_ALLOCATION = 0.30
PROFIT_HARVEST_THRESHOLD = 0.40  # Harvest when HL > 40% of total
REBALANCE_THRESHOLD = 0.10       # Flag when allocation drifts >10%

CACHE_TTL = 60  # Refresh balances every 60s


@dataclass
class UnifiedState:
    """Single source of truth for both exchanges."""
    timestamp: float = 0.0

    # Revolut X
    revolut_eur: float = 0.0
    revolut_crypto_eur: float = 0.0
    revolut_total_eur: float = 0.0

    # Hyperliquid
    hl_perp_usd: float = 0.0
    hl_spot_usd: float = 0.0
    hl_total_usd: float = 0.0
    hl_margin_used: float = 0.0
    hl_positions: int = 0
    hl_position_coins: str = ""

    # Unified
    total_euro_equivalent: float = 0.0  # Everything in EUR equivalent
    revolut_pct: float = 0.0
    hl_pct: float = 0.0
    allocation_drift: float = 0.0       # How far from 70/30 target

    # Signals
    profit_harvest_signal: str = ""      # "harvest", "warning", "ok"
    profit_harvest_amount_eur: float = 0.0
    revolut_low_warning: bool = False    # Safety vault too small

    def to_dict(self) -> dict:
        return {
            "ts": self.timestamp,
            "revolut": {
                "eur": round(self.revolut_eur, 2),
                "crypto_eur": round(self.revolut_crypto_eur, 2),
                "total_eur": round(self.revolut_total_eur, 2),
            },
            "hyperliquid": {
                "perp_usd": round(self.hl_perp_usd, 2),
                "spot_usd": round(self.hl_spot_usd, 2),
                "total_usd": round(self.hl_total_usd, 2),
                "margin_used": round(self.hl_margin_used, 2),
                "positions": self.hl_positions,
                "coins": self.hl_position_coins,
            },
            "unified": {
                "total_eur": round(self.total_euro_equivalent, 2),
                "revolut_pct": round(self.revolut_pct * 100, 1),
                "hl_pct": round(self.hl_pct * 100, 1),
                "drift_pct": round(self.allocation_drift * 100, 1),
            },
            "signals": {
                "profit_harvest": self.profit_harvest_signal,
                "harvest_amount_eur": round(self.profit_harvest_amount_eur, 2),
                "revolut_low": self.revolut_low_warning,
            },
        }


# ── Module-level cache ──
_cached_state: UnifiedState | None = None
_cache_ts: float = 0.0


def _fetch_revolut() -> dict:
    """Fetch Revolut X balances. Returns {eur, crypto_eur, total_eur}."""
    try:
        from revolut_client import RevolutXClient
        client = RevolutXClient()
        balances = client.get_balances()

        eur = 0.0
        crypto_eur = 0.0
        for b in (balances if isinstance(balances, list) else []):
            curr = b.get("currency", "?")
            bal = float(b.get("balance", 0)) if b.get("balance") else 0.0
            if curr == "EUR":
                eur += bal
            elif bal > 0.0001:
                # Estimate crypto value in EUR (approximate)
                crypto_eur += bal * 1.0  # placeholder — would need ticker prices

        # Try to get actual crypto values via tickers
        try:
            coins_with_balance = [
                b.get("currency") for b in (balances if isinstance(balances, list) else [])
                if b.get("currency") not in ("EUR", "USD")
                and float(b.get("balance", 0)) > 0.0001
            ]
            if coins_with_balance:
                ticker_data = client.get_tickers([f"{c}-EUR" for c in coins_with_balance[:5]])
                tickers = {}
                if isinstance(ticker_data, dict):
                    ticker_list = ticker_data.get("data", [ticker_data])
                    for t in (ticker_list if isinstance(ticker_list, list) else []):
                        sym = t.get("symbol", "")
                        if "-EUR" in sym:
                            tickers[sym.split("-")[0]] = float(t.get("last_price", 0))
                crypto_eur = 0.0
                for b in (balances if isinstance(balances, list) else []):
                    curr = b.get("currency", "")
                    bal = float(b.get("balance", 0)) if b.get("balance") else 0.0
                    if curr in tickers and bal > 0:
                        crypto_eur += bal * tickers[curr]
        except Exception:
            pass

        return {"eur": eur, "crypto_eur": crypto_eur, "total_eur": eur + crypto_eur}
    except Exception as e:
        log.debug(f"Revolut fetch failed (benign if not funded): {e}")
        return {"eur": 0.0, "crypto_eur": 0.0, "total_eur": 0.0}


def _fetch_hyperliquid() -> dict:
    """Fetch Hyperliquid balances."""
    try:
        from hyperliquid_client import get_account
        acc = get_account()
        positions = acc.get("positions", [])
        coins = []
        for p in positions:
            pp = p.get("position", p)
            coin = pp.get("coin", "")
            if coin:
                coins.append(coin)
        return {
            "perp_usd": float(acc.get("perp_equity", 0)),
            "spot_usd": float(acc.get("spot_usdc", 0)),
            "total_usd": float(acc.get("total_equity", 0)),
            "margin_used": float(acc.get("margin_used", 0)),
            "positions": len(coins),
            "coins": ",".join(coins),
        }
    except Exception as e:
        log.debug(f"HL fetch failed: {e}")
        return {"perp_usd": 0, "spot_usd": 0, "total_usd": 0,
                "margin_used": 0, "positions": 0, "coins": ""}


def get_unified(force_refresh: bool = False) -> UnifiedState:
    """Get unified portfolio state. Cached for 60s by default."""
    global _cached_state, _cache_ts

    now = time.time()
    if not force_refresh and _cached_state and (now - _cache_ts) < CACHE_TTL:
        return _cached_state

    rev = _fetch_revolut()
    hl = _fetch_hyperliquid()

    state = UnifiedState(timestamp=now)
    state.revolut_eur = rev["eur"]
    state.revolut_crypto_eur = rev["crypto_eur"]
    state.revolut_total_eur = rev["total_eur"]

    state.hl_perp_usd = hl["perp_usd"]
    state.hl_spot_usd = hl["spot_usd"]
    state.hl_total_usd = hl["total_usd"]
    state.hl_margin_used = hl["margin_used"]
    state.hl_positions = hl["positions"]
    state.hl_position_coins = hl["coins"]

    # Convert to EUR equivalent (approximate: USD≈EUR)
    revolut_eur = state.revolut_total_eur
    hl_eur = state.hl_total_usd * 0.92  # USD→EUR approximate
    state.total_euro_equivalent = revolut_eur + hl_eur

    if state.total_euro_equivalent > 0:
        state.revolut_pct = revolut_eur / state.total_euro_equivalent
        state.hl_pct = hl_eur / state.total_euro_equivalent
        state.allocation_drift = abs(state.revolut_pct - REVOLUT_ALLOCATION)
    else:
        state.revolut_pct = 0.0
        state.hl_pct = 1.0
        state.allocation_drift = 0.0

    # ── Profit harvesting signal ──
    if state.hl_pct > PROFIT_HARVEST_THRESHOLD and state.total_euro_equivalent > 50:
        excess_pct = state.hl_pct - PROFIT_HARVEST_THRESHOLD
        harvest_amount = state.hl_total_usd * excess_pct * 0.50  # Move 50% of excess
        state.profit_harvest_signal = "harvest"
        state.profit_harvest_amount_eur = harvest_amount
    elif state.hl_pct > PROFIT_HARVEST_THRESHOLD * 0.8:
        state.profit_harvest_signal = "warning"
    else:
        state.profit_harvest_signal = "ok"

    # ── Revolut low warning ──
    state.revolut_low_warning = state.revolut_total_eur < 10 and state.hl_total_usd > 20

    # Persist
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state.to_dict(), indent=2, default=str))
    except Exception:
        pass

    _cached_state = state
    _cache_ts = now
    return state


def get_unified_equity() -> float:
    """Quick access: total unified equity in USD equivalent. For position sizing."""
    state = get_unified()
    return state.total_euro_equivalent


def get_profit_harvest_signal() -> tuple[str, float]:
    """Returns (signal, amount_eur). Signal: 'harvest', 'warning', 'ok'."""
    state = get_unified()
    return state.profit_harvest_signal, state.profit_harvest_amount_eur
