#!/bin/bash
# Run action executor with 4-min timeout
cd /home/ubuntu/hyperliquid-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
timeout 240 python3 action_executor.py 2>&1
