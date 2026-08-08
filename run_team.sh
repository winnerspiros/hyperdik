#!/bin/bash
# Full pipeline test (v2, 2026-07-13 redesign): review → strategist → validate → execute → predict
set -e
cd /home/ubuntu/revolut-x-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] === FULL PIPELINE TEST (v2) ==="

echo "📋 Step 1: Portfolio Reviewer"
python3 portfolio_reviewer.py 2>&1 | tail -10

echo "🧭 Step 2: Strategist (joint buy/sell/bank plan)"
python3 strategist.py 2>&1 | tail -15

echo "✅ Step 3: Action Validator (premium model)"
python3 action_validator.py 2>&1 | tail -15

echo "⚡ Step 4: Action Executor (AI-powered)"
python3 action_executor.py 2>&1 | tail -15

echo "🔮 Step 5: Market Predictor"
python3 market_predictor.py 2>&1 | tail -15

echo "✅ Full pipeline complete"
