#!/bin/bash
# Run buyer agent (decides what to buy)
cd /home/ubuntu/revolut-x-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
python3 buyer_agent.py 2>&1