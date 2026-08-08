#!/bin/bash
# Revolut X 24/7 Trader — Direct launcher (no wrapper, for systemd/cron direct use)
# This script directly starts the daemon and survives session exit

cd /home/ubuntu/revolut-x-trader
export PYTHONOPTIMIZE=2
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
# Use system Python (hyperliquid and deps are installed there)
exec python3 daemon.py --interval 15 >> /home/ubuntu/revolut-x-trader/logs/daemon_$(date +%Y%m%d).log 2>&1