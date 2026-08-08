#!/bin/bash
# Run the full review → predict pipeline
set -e
cd /home/ubuntu/revolut-x-trader
source /tmp/trading-venv/bin/activate 2>/dev/null || true

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] 🚀 Starting Review → Predict pipeline"

# Step 1: Review (read-only portfolio + market analysis)
echo "──────────────────────────────────────────────"
echo "📋 Step 1: Portfolio Review"
python3 portfolio_reviewer.py 2>&1
REVIEW_EXIT=$?
if [ $REVIEW_EXIT -ne 0 ]; then
    echo "⚠️  Reviewer exited with code $REVIEW_EXIT — continuing anyway"
fi

# Step 2: Predict (forecast based on reviewer output)
echo ""
echo "──────────────────────────────────────────────"
echo "🔮 Step 2: Market Prediction"
python3 market_predictor.py 2>&1
PREDICT_EXIT=$?
if [ $PREDICT_EXIT -ne 0 ]; then
    echo "⚠️  Predictor exited with code $PREDICT_EXIT"
fi

echo ""
echo "──────────────────────────────────────────────"
echo "✅ Pipeline complete (review=$REVIEW_EXIT predict=$PREDICT_EXIT)"
echo "Calendar: data/strategy_calendar.json"
echo "Advisories: data/predictions/*.json"