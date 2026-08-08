#!/bin/bash
# Revolut X 24/7 Trader Wrapper — auto-restarts on crash
LOGFILE="/tmp/revolut_trader_wrapper.log"
VENV="/tmp/trading-venv"
TRADER_DIR="/home/ubuntu/revolut-x-trader"

echo "$(date) Starting Revolut X Trader wrapper" >> "$LOGFILE"

cd "$TRADER_DIR"
source "$VENV/bin/activate"

# If first arg is --live, pass it through
LIVE_FLAG=""
if [ "$1" = "--live" ]; then
    LIVE_FLAG="--live"
    echo "$(date) *** LIVE MODE ***" >> "$LOGFILE"
fi

while true; do
    echo "$(date) Starting trading daemon..." >> "$LOGFILE"
    python3 daemon.py --interval 15 $LIVE_FLAG
    EXIT_CODE=$?
    echo "$(date) Daemon exited with code $EXIT_CODE, restarting in 10s..." >> "$LOGFILE"
    sleep 10
done
