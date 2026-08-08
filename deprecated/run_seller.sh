#!/bin/bash
# Run seller agent (decides what to sell)
cd /home/ubuntu/revolut-x-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
python3 seller_agent.py 2>&1