#!/usr/bin/env python3
"""
Market Intelligence — single canonical world-state snapshot for the strategic
agent tier (strategist, predictor, validator, executor).

WHY THIS EXISTS:
Before this module, portfolio_reviewer, market_predictor, buyer_agent,
seller_agent, and bank_manager each independently called data_enricher,
market_regime, get_tickers(), get_balances() etc — 5 separate fetch chains
producing 5 slightly different views of "now" within the same hour, each
paying the CoinGecko/ticker latency cost separately.

This module fetches ONCE, writes ONE snapshot file, and every strategic-tier
agent reads that file. TTL-gated: if the snapshot is fresh (<10min old),
agents skip re-fetching entirely and just read.

Does NOT touch day_trader.py / scalper.py — they run on their own faster
cadence (5min / 60s) with their own data_enricher.build_rich_context() calls,
which already have a 60s cache. Rewiring the live daemons is a separate,
higher-risk decision reserved for later.
"""
import sys, os, json, time
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

SNAPSHOT_PATH = os.path.join(TRADER_DIR, "data", "market_snapshot.json")
SNAPSHOT_TTL_SECONDS = 600  # 10 minutes — strategic tier doesn't need fresher than this

os.makedirs(os.path.join(TRADER_DIR, "data"), exist_ok=True)


def _import(name):
    try:
        return __import__(name, fromlist=[""])
    except Exception:
        return None


def _load_snapshot():
    try:
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _snapshot_age_seconds(snap):
    if not snap or "fetched_at" not in snap:
        return float("inf")
    try:
        dt = datetime.fromisoformat(snap["fetched_at"])
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return float("inf")


def get_snapshot(force_refresh=False, holdings=None):
    """
    Return the canonical market snapshot. Refreshes if stale or force_refresh=True.
    holdings: optional list of symbols to fetch prices for (in addition to core set).
    """
    existing = _load_snapshot()
    if not force_refresh and existing and _snapshot_age_seconds(existing) < SNAPSHOT_TTL_SECONDS:
        return existing

    snap = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fear_greed": None,
        "global": None,
        "trending": [],
        "regime": None,
        "funding": None,
        "onchain": None,
        "defi": None,
        "mempool": None,
        "tickers": {},
    }

    data_enricher = _import("data_enricher")
    market_regime_mod = _import("market_regime")
    funding_mod = _import("funding_signals")
    onchain_mod = _import("onchain_metrics")
    defi_mod = _import("defi_monitor")
    mempool_mod = _import("mempool_monitor")
    rev_client_mod = _import("revolut_client")

    # Core macro data
    try:
        if data_enricher:
            snap["fear_greed"] = data_enricher.get_fear_greed()
            snap["global"] = data_enricher.get_global_data()
            snap["trending"] = data_enricher.get_trending()
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"data_enricher: {e}"]

    # Regime (needs BTC change + F&G — best-effort)
    try:
        if market_regime_mod and snap.get("global"):
            btc_change = None
            fg_val = None
            if snap.get("fear_greed"):
                try:
                    fg_val = int(snap["fear_greed"].get("value", 0))
                except (ValueError, TypeError):
                    fg_val = None
            regime = market_regime_mod.classify_regime(btc_24h_change=btc_change, fear_greed=fg_val)
            snap["regime"] = {
                "regime": regime.get("regime"),
                "action_bias": regime.get("action_bias"),
                "description": regime.get("description"),
            }
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"regime: {e}"]

    # Funding rates (Kraken + OKX — Binance/Bybit blocked from this server)
    try:
        if funding_mod and hasattr(funding_mod, "get_funding_data"):
            snap["funding"] = funding_mod.get_funding_data()
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"funding: {e}"]

    # On-chain (Mayer, MVRV, SOPR, NUPL)
    try:
        if onchain_mod and hasattr(onchain_mod, "get_combined_onchain_summary"):
            snap["onchain"] = onchain_mod.get_combined_onchain_summary()
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"onchain: {e}"]

    # DeFi TVL
    try:
        if defi_mod and hasattr(defi_mod, "format_defi_for_ai"):
            snap["defi"] = defi_mod.format_defi_for_ai()
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"defi: {e}"]

    # Mempool (BTC fee pressure — leading volatility indicator)
    try:
        if mempool_mod and hasattr(mempool_mod, "format_mempool_for_ai"):
            snap["mempool"] = mempool_mod.format_mempool_for_ai()
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"mempool: {e}"]

    # Live tickers from Revolut (source of truth for tradeable prices)
    try:
        if rev_client_mod:
            client = rev_client_mod.RevolutXClient()
            raw = client.get_tickers()
            tickers = raw if isinstance(raw, list) else raw.get("data", [])
            for t in tickers:
                if not isinstance(t, dict):
                    continue
                sym = t.get("symbol", "").replace("/USD", "")
                try:
                    price = float(t.get("last_price", 0) or t.get("mid", 0) or 0)
                except (ValueError, TypeError):
                    price = 0
                if sym and price:
                    snap["tickers"][sym] = price
    except Exception as e:
        snap["_errors"] = snap.get("_errors", []) + [f"tickers: {e}"]

    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snap, f, indent=2, default=str)

    return snap


def format_for_ai(snap, max_chars=1500):
    """Compact human-readable block for injecting into an AI prompt."""
    lines = []
    fg = snap.get("fear_greed") or {}
    if fg:
        lines.append(f"Fear & Greed: {fg.get('value')} ({fg.get('classification')})")
    g = snap.get("global") or {}
    if g:
        cap = g.get("total_market_cap")
        btc_dom = g.get("btc_dominance")
        if cap:
            lines.append(f"Total mcap: €{cap:,.0f} | BTC dominance: {btc_dom:.1f}%" if btc_dom else f"Total mcap: €{cap:,.0f}")
    r = snap.get("regime") or {}
    if r:
        lines.append(f"Regime: {r.get('regime')} (bias: {r.get('action_bias')}) — {r.get('description')}")
    if snap.get("trending"):
        names = ", ".join(c.get("symbol", "?") for c in snap["trending"][:5])
        lines.append(f"Trending: {names}")
    age = _snapshot_age_seconds(snap)
    lines.append(f"[snapshot age: {age/60:.0f}min]")
    text = "\n".join(lines)
    return text[:max_chars]


if __name__ == "__main__":
    snap = get_snapshot(force_refresh=True)
    print(json.dumps(snap, indent=2, default=str)[:2000])
    print("\n--- AI-formatted ---")
    print(format_for_ai(snap))
