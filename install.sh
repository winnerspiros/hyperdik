#!/bin/bash
# install.sh — Install Unified Trading System (Revolut X + Hyperliquid)
# Run once to set up 24/7 trading with auto-restart on reboot
set -e

echo "================================================"
echo " UNIFIED TRADING SYSTEM INSTALLER"
echo " Revolut X (EUR spot) + Hyperliquid (USDC perps)"
echo "================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="unified-orchestrator"
SERVICE_FILE="${SCRIPT_DIR}/unified-orchestrator.service"

# Check requirements
echo "[1/6] Checking requirements..."
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required"; exit 1; }
echo "  python3: $(python3 --version)"

# Check config files
if [ ! -f "$HOME/.config/revolut-x/config.json" ]; then
    echo "  WARNING: Revolut X config not found at ~/.config/revolut-x/config.json"
    echo "  The bot will run but Revolut X integration will fail"
fi

if [ ! -f "$HOME/.hyperliquid/config.json" ]; then
    echo "  WARNING: Hyperliquid config not found at ~/.hyperliquid/config.json"
    echo "  The bot will run but Hyperliquid integration will fail"
fi

# Create directories
echo "[2/6] Creating directories..."
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/data/pending_actions"
mkdir -p "${SCRIPT_DIR}/data/approved_actions"
mkdir -p "${SCRIPT_DIR}/data/completed_actions"
echo "  Directories created"

# Check Python deps
echo "[3/6] Checking Python dependencies..."
python3 -c "import pandas, numpy, json, time" 2>/dev/null || {
    echo "  Installing pandas, numpy..."
    pip3 install --break-system-packages pandas numpy 2>/dev/null || \
    pip3 install pandas numpy 2>/dev/null || \
    echo "  WARNING: Could not install pandas/numpy — strategy engine needs these"
}

python3 -c "import hyperliquid" 2>/dev/null || {
    echo "  WARNING: hyperliquid-python-sdk not installed"
    echo "  Install: pip3 install --break-system-packages hyperliquid-python-sdk"
}

python3 -c "from cryptography.hazmat.primitives.asymmetric import ed25519" 2>/dev/null || {
    echo "  WARNING: cryptography not installed (needed for Revolut X)"
    echo "  Install: pip3 install --break-system-packages cryptography"
}
echo "  Dependencies OK"

# Syntax check all modules
echo "[4/6] Syntax checking modules..."
cd "$SCRIPT_DIR"
ERRORS=0
for f in hyperliquid_strategy.py hyperliquid_risk.py hyperliquid_execution.py \
         hyperliquid_daemon.py unified_portfolio.py unified_orchestrator.py \
         hyperliquid_whale.py hyperliquid_funding_sniper.py hyperliquid_delta_neutral.py \
         hyperliquid_evolution.py; do
    if [ -f "$f" ]; then
        python3 -c "compile(open('$f').read(), '$f', 'exec')" 2>/dev/null && echo "  ✓ $f" || { echo "  ✗ $f — SYNTAX ERROR"; ERRORS=$((ERRORS+1)); }
    fi
done

if [ $ERRORS -gt 0 ]; then
    echo "  WARNING: $ERRORS module(s) have syntax errors"
fi

# Install systemd service
echo "[5/6] Installing systemd service..."
if command -v systemctl >/dev/null 2>&1; then
    sudo cp "$SERVICE_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
    echo "  Service installed and enabled"
    echo "  Start: sudo systemctl start ${SERVICE_NAME}"
    echo "  Stop:  sudo systemctl stop ${SERVICE_NAME}"
    echo "  Logs:  sudo journalctl -u ${SERVICE_NAME} -f"
else
    echo "  systemctl not found — skipping service install"
    echo "  Run manually: python3 unified_orchestrator.py"
fi

# Health check
echo "[6/6] Running health check..."
python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from unified_portfolio import UnifiedState
state = UnifiedState()
print(f'  Unified portfolio engine OK')
from hyperliquid_strategy import MarketRegime
print(f'  Strategy engine OK ({len(MarketRegime.__members__)} regimes)')
from hyperliquid_risk import HSLState, compute_portfolio_heat
hsl = HSLState(); hsl.update(10000)
print(f'  Risk engine OK (HSL={hsl.tier.value})')
" 2>&1 || echo "  Health check had issues — check logs"

echo ""
echo "================================================"
echo " INSTALL COMPLETE"
echo "================================================"
echo ""
echo "QUICK START:"
echo "  1. Start bot:     sudo systemctl start ${SERVICE_NAME}"
echo "  2. Check status:  sudo systemctl status ${SERVICE_NAME}"
echo "  3. View logs:     tail -f ${SCRIPT_DIR}/logs/orchestrator.log"
echo "  4. Dry run first: python3 unified_orchestrator.py --dry-run"
echo ""
echo "The bot auto-starts on VM reboot. Services are independent:"
echo "  - Revolut X: checks every 5min, buys dips, takes profit"
echo "  - Hyperliquid: runs every 60s, 5-pillar composite engine"
echo "  - Portfolio: syncs every 60s, monitors allocation"
echo ""
echo "Heartbeat: cat /tmp/unified_orchestrator.heartbeat"
echo "State:     cat ${SCRIPT_DIR}/data/unified_state.json"
echo ""
