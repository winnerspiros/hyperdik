#!/bin/bash
# Run market predictor only (every 2h)
cd /home/ubuntu/hyperliquid-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
python3 market_predictor.py 2>&1