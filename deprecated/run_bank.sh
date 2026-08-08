#!/bin/bash
# Run bank manager (manages EUR balance)
cd /home/ubuntu/revolut-x-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true
python3 bank_manager.py 2>&1