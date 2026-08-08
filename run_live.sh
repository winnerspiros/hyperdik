#!/bin/bash
# Launch daemon in LIVE mode — survives session exit via `at`
cd /home/ubuntu/revolut-x-trader
source /tmp/trading-venv/bin/activate
python3 daemon.py --live --interval 15 >> /home/ubuntu/revolut-x-trader/logs/daemon.log 2>&1