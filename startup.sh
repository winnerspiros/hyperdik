#!/bin/bash
# Hyperliquid Daemon Startup — no Revolut X, Hyperliquid only
# Cleans stale action files and starts the HL daemon.

TRADER_DIR="/home/ubuntu/hyperliquid-trader"
LOG="$TRADER_DIR/logs/boot.log"

echo "--- BOOT $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG"
sleep 10  # Wait for network

# Clean stale action files from pre-reboot
rm -f "$TRADER_DIR/data/pending_actions/"*.json 2>/dev/null
rm -f "$TRADER_DIR/data/approved_actions/"*.json 2>/dev/null
rm -f "$TRADER_DIR/data/completed_actions/"*.json 2>/dev/null
echo "Cleaned action queues" >> "$LOG"

cd "$TRADER_DIR"
export PYTHONOPTIMIZE=2
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536

# Start Hyperliquid daemon (the only trading engine)
nohup python3 hyperliquid_daemon.py >> "$TRADER_DIR/logs/hyperliquid_daemon.log" 2>> "$TRADER_DIR/logs/hyperliquid_daemon_error.log" &
echo "Hyperliquid daemon PID=$!" >> "$LOG"

echo "--- BOOT COMPLETE ---" >> "$LOG"
