#!/bin/bash
# Run strategist (joint buy/sell/bank decision-maker — replaces buyer/seller/bank)
cd /home/ubuntu/hyperliquid-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
python3 strategist.py 2>&1