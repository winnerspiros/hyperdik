#!/bin/bash
# Run action validator with 4-min timeout (cron runs every 5min)
cd /home/ubuntu/hyperliquid-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
timeout 240 python3 action_validator.py 2>&1
