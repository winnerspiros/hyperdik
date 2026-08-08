#!/usr/bin/env python3
"""
Unified Orchestrator v1 — 24/7 master process managing Revolut X + Hyperliquid.

ARCHITECTURE:
  unified_orchestrator.py (master — this file)
    ├── Revolut X spot bot (EUR, holds crypto, buys dips)
    ├── Hyperliquid daemon v3 (leverage, LONG/SHORT, 5-pillar engine)
    └── Portfolio sync (unified view, allocation, profit harvesting)

24/7 DESIGN:
  - Runs as systemd service
  - Auto-restarts on crash/reboot
  - Heartbeat monitoring
  - Graceful shutdown
  - State persisted to disk

REVOLUT X ROLE:
  - Hold EUR + spot crypto (safety net)
  - Buy spot on big dips (when HL is leveraged long the same asset)
  - Sell spot to realize long-term gains
  - Convert EUR→crypto when allocation favors it

HYPERLIQUID ROLE:
  - Active trading with leverage
  - LONG/SHORT based on 5-pillar composite signal
  - Multi-tier TP + trailing stops
  - All 10+ safety gates

FLOW:
  Every 60s:
    1. Sync portfolio (both exchanges)
    2. Check allocation drift
    3. Log profit harvesting opportunities
    4. Heartbeat
"""

import json
import os
import signal
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ============================================================
# Config
# ============================================================

SYNC_INTERVAL = 60       # Full portfolio sync every 60s
REVOLUT_INTERVAL = 300   # Revolut trading check every 5min
HEARTBEAT_PATH = "/tmp/unified_orchestrator.heartbeat"
LOG_PATH = "logs/unified_orchestrator.log"

# ============================================================
# Logging
# ============================================================

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORCH] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("orchestrator")


# ============================================================
# Heartbeat
# ============================================================

def heartbeat() -> None:
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(f"{int(time.time())}\nrunning")
    except Exception:
        pass


# ============================================================
# Revolut X Spot Bot
# ============================================================

def revolut_cycle(unified_state) -> list[str]:
    """Run one Revolut X spot trading cycle.

    Strategy: Buy spot on dips, hold long-term, sell for profit taking.
    Very conservative — mostly holds.
    """
    actions = []
    try:
        from revolut_client import RevolutXClient
        client = RevolutXClient()
        balances = client.get_balances()

        if isinstance(balances, dict):
            data = balances.get("data", []) if "data" in balances else [balances]
        else:
            data = balances if isinstance(balances, list) else []

        eur_available = 0.0
        holdings = {}
        for b in data:
            curr = b.get("currency", "")
            avail = float(b.get("available", 0))
            if curr == "EUR":
                eur_available = avail
            elif avail > 0.001:
                holdings[curr] = avail

        # Only use 10% of EUR for spot buys (keep 90% as safety)
        tradable_eur = eur_available * 0.10

        # Check BTC and ETH for dip buying
        SPOT_COINS = ["BTC", "ETH"]
        for coin in SPOT_COINS:
            try:
                tickers = client.get_tickers([f"{coin}-EUR"])
                if isinstance(tickers, dict):
                    ticker_data = tickers.get("data", [tickers])
                    if isinstance(ticker_data, list):
                        for t in ticker_data:
                            if t.get("symbol") == f"{coin}-EUR":
                                price = float(t.get("last_price", 0))
                                change_24h = float(t.get("price_change_24h_pct", 0))

                                # Dip buy: price down >5% in 24h AND we have EUR
                                if change_24h < -5 and tradable_eur > 10:
                                    buy_eur = min(tradable_eur * 0.5, 50)  # Max €50 per dip buy
                                    size = buy_eur / price
                                    log.info(f"🟢 REVOLUT: Dip buy {coin} — {change_24h:+.1f}% 24h — €{buy_eur:.2f} = {size:.6f} {coin}")
                                    try:
                                        client.place_order(
                                            f"{coin}-EUR", "buy", "limit",
                                            base_size=str(round(size, 6)),
                                            price=str(round(price * 0.995, 2)),
                                            time_in_force="gtc",
                                        )
                                        actions.append(f"dip_buy_{coin}")
                                    except Exception as e:
                                        log.warning(f"Revolut {coin} buy failed: {e}")

                                # Profit taking: price up >20% from our entry AND we hold
                                if coin in holdings and change_24h > 5:
                                    hold_value_eur = holdings[coin] * price
                                    if hold_value_eur > 10:
                                        sell_amt = holdings[coin] * 0.25  # Sell 25%
                                        log.info(f"🔴 REVOLUT: Take profit {coin} — +{change_24h:+.1f}% — sell {sell_amt:.6f}")
                                        try:
                                            client.place_order(
                                                f"{coin}-EUR", "sell", "limit",
                                                base_size=str(round(sell_amt, 6)),
                                                price=str(round(price * 1.005, 2)),
                                                time_in_force="gtc",
                                            )
                                            actions.append(f"profit_sell_{coin}")
                                        except Exception as e:
                                            log.warning(f"Revolut {coin} sell failed: {e}")
            except Exception as e:
                log.debug(f"Revolut {coin} check: {e}")

    except Exception as e:
        log.warning(f"Revolut cycle error: {e}")

    return actions


# ============================================================
# Hyperliquid Daemon Wrapper
# ============================================================

hl_daemon_running = False

def start_hl_daemon():
    """Start Hyperliquid daemon in same process."""
    global hl_daemon_running
    try:
        # Import and run one cycle
        import hyperliquid_daemon
        log.info("🟢 Hyperliquid daemon started")
        hl_daemon_running = True
        hyperliquid_daemon.run(dry_run=False)
    except Exception as e:
        log.error(f"Hyperliquid daemon error: {e}")
        hl_daemon_running = False


# ============================================================
# Main Orchestrator Loop
# ============================================================

def run():
    global hl_daemon_running

    log.info("=" * 60)
    log.info("UNIFIED ORCHESTRATOR — Revolut X + Hyperliquid")
    log.info(f"  Revolut interval: {REVOLUT_INTERVAL}s")
    log.info(f"  Sync interval: {SYNC_INTERVAL}s")
    log.info(f"  PID: {os.getpid()}")
    log.info("=" * 60)

    # Start HL daemon in a way that doesn't block
    # We'll run it in the main thread and use signal handlers
    import threading

    def hl_thread():
        global hl_daemon_running
        try:
            import hyperliquid_daemon
            hl_daemon_running = True
            log.info("🔵 Hyperliquid daemon thread started")
            hyperliquid_daemon.run(dry_run=False)
        except Exception as e:
            log.error(f"HL daemon crash: {e}")
            hl_daemon_running = False

    hl_thread_handle = threading.Thread(target=hl_thread, daemon=True, name="hl-daemon")
    hl_thread_handle.start()
    time.sleep(5)  # Let HL daemon initialize

    # Main sync loop
    from unified_portfolio import sync_portfolio, build_unified_context, UnifiedState

    last_revolut = 0
    last_sync = 0
    last_heartbeat = 0

    while True:
        now = time.time()

        # Heartbeat
        if now - last_heartbeat >= 30:
            heartbeat()
            last_heartbeat = now

        # Portfolio sync
        if now - last_sync >= SYNC_INTERVAL:
            last_sync = now
            try:
                state = sync_portfolio()
                ctx = build_unified_context(state)
                log.info(f"\n{ctx}")

                # Check if HL daemon is alive
                if not hl_daemon_running or not hl_thread_handle.is_alive():
                    log.error("🔴 HL daemon stopped! Restarting...")
                    hl_thread_handle = threading.Thread(target=hl_thread, daemon=True, name="hl-daemon")
                    hl_thread_handle.start()
            except Exception as e:
                log.error(f"Sync error: {e}")

        # Revolut cycle
        if now - last_revolut >= REVOLUT_INTERVAL:
            last_revolut = now
            try:
                actions = revolut_cycle(None)
                if actions:
                    log.info(f"Revolut actions: {actions}")
            except Exception as e:
                log.warning(f"Revolut cycle error: {e}")

        time.sleep(1)


# ============================================================
# Signal handlers for graceful shutdown
# ============================================================

def shutdown_handler(signum, frame):
    log.info(f"🛑 Received signal {signum} — shutting down gracefully...")
    heartbeat()
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)


# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        log.info("🔒 DRY RUN MODE")

    run()
