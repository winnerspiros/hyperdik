#!/bin/bash
# Start both bot processes cleanly
export OPENROUTER_API_KEY=$(grep "^OPENROUTER_API_KEY=" ~/.hermes/.env | head -1 | cut -d= -f2)
cd /home/ubuntu/revolut-x-trader
export PYTHONOPTIMIZE=2
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
source /tmp/trading-venv/bin/activate || exit 1

# Kill any existing instances first
pkill -f "python3 day_trader.py --interval 5" 2>/dev/null
pkill -f "python3 scalper.py" 2>/dev/null
sleep 2

# Start day trader
nohup python3 day_trader.py --interval 5 >> /home/ubuntu/revolut-x-trader/logs/day_trader.log 2>&1 &
echo $! > /tmp/day_trader.pid
echo "Day trader PID=$!"

# Start scalper
nohup python3 scalper.py >> /home/ubuntu/revolut-x-trader/logs/scalper.log 2>&1 &
echo $! > /tmp/scalper.pid
echo "Scalper PID=$!"

echo "Both started. Check with: ps aux | grep python3"