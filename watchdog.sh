#!/bin/bash
# Watchdog v9 — Hyperliquid daemon monitor
# Systemd is the primary manager. This is a fallback only.
# systemd handles restarts with proper delay. Don't interfere.
TRADER_DIR="/home/ubuntu/hyperliquid-trader"
LOG="$TRADER_DIR/logs/watchdog.log"
LOCK_FILE="/tmp/hyperliquid_daemon.lock"
PIDFILE="/tmp/hyperliquid_daemon.pid"

# If systemd is active for this unit, let it manage the daemon — exit
if systemctl --user is-active hyperliquid-daemon.service &>/dev/null; then
    exit 0
fi

# Systemd is not running — fallback mode
COUNT=$(pgrep -f "python3.*hyperliquid_daemon.py" 2>/dev/null | wc -l)

if [ "$COUNT" -gt 0 ]; then
    pgrep -f "python3.*hyperliquid_daemon.py" | tail -1 > "$PIDFILE"
    exit 0
fi

# Daemon is dead and systemd is down — restart manually
echo "$(date) hyperliquid_daemon dead (no systemd), manual restart" >> "$LOG"

LAST_RESTART=$(cat /tmp/hl_last_restart 2>/dev/null || echo 0)
NOW=$(date +%s)
GAP=$((NOW - LAST_RESTART))
if [ "$GAP" -lt 90 ]; then
    DELAY=$((90 - GAP))
    echo "$(date) Rate-limit cooldown: sleeping ${DELAY}s" >> "$LOG"
    sleep "$DELAY"
fi
date +%s > /tmp/hl_last_restart

cd "$TRADER_DIR"
rm -f "$LOCK_FILE"
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

API_KEY=$(grep "^OPENROUTER_API_KEY=" ~/.hermes/.env | head -1 | cut -d= -f2)
export OPENROUTER_API_KEY="$API_KEY"
export HL_STARTUP_DELAY=45

nohup python3 -B hyperliquid_daemon.py >> "$TRADER_DIR/logs/hyperliquid_daemon.log" 2>> "$TRADER_DIR/logs/hyperliquid_daemon_error.log" &
NEW_PID=$!
echo "$NEW_PID" > "$PIDFILE"
echo "$(date) Restarted hyperliquid_daemon PID=$NEW_PID" >> "$LOG"
