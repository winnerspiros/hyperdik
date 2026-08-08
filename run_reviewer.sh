#!/bin/bash
# Run portfolio reviewer only (every 30min)
cd /home/ubuntu/hyperliquid-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
python3 portfolio_reviewer.py 2>&1