# 🤖 HYPERDIK — Hyperliquid AI Trading Bot

<p align="center">
  <img src="assets/hyperdik-logo.jpg" alt="HYPERDIK" width="384">
</p>

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Hyperliquid](https://img.shields.io/badge/Exchange-Hyperliquid-green)](https://hyperliquid.xyz)
[![AI](https://img.shields.io/badge/AI-Qwen_235B_%2B_Llama_4-purple)](https://openrouter.ai)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen)]()

> *"stupid name till i think of something better"*

## 🧠 What It Does

A fully autonomous trading bot that scans 177 perpetual futures markets every 5 seconds, picks the best trades using three AI models, and executes with microsecond-precision entry timing and a multi-layer exit system.

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 📡 DATA  │───▶│ 🧠 AI    │───▶│ 📊 SIGNAL│───▶│ ⚡ TRADE │
  │ 177 mids │    │ Qwen 235B│    │ 5 layers │    │ micro-peak│
  │ WS ticks │    │ Maverick │    │ 7 gates  │    │ 6x-12x   │
  │ OI·CVD   │    │ Scout    │    │ ensemble │    │ TP/SL     │
  └──────────┘    └──────────┘    └──────────┘    └─────┬────┘
                                                        │
                      ┌─────────────────────────────────┘
                      ▼
               ┌──────────┐    ┌──────────┐
               │ 🛡️ EXIT  │───▶│ 📈 LEARN │
               │ breakeven │    │ outcomes │
               │ profit-lock│   │ weights  │
               │ trend-kill│    │ memory   │
               └──────────┘    └──────────┘
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/winnerspiros/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install hyperliquid-python-sdk eth-account xgboost numpy scikit-learn pandas

# Wallet (required)
mkdir -p ~/.hyperliquid
cat > ~/.hyperliquid/config.json << 'EOF'
{
  "api_wallet": "0xYOUR_API_WALLET",
  "api_private_key": "0xYOUR_PRIVATE_KEY",
  "main_wallet": "0xYOUR_MAIN_WALLET"
}
EOF

# AI (optional — enables coin selection)
export OPENROUTER_API_KEY=your_key_here

# Go
python3 -B hyperliquid_daemon.py
```

---

## 🤖 AI Models

| Model | Role | Cost |
|---|---|---|
| **Qwen 235B MoE** | Decisions, sizing, exits | $0.64/M tok |
| **Llama 4 Maverick** | Coin picks, market read | $1.00/M tok |
| **Llama 4 Scout** | Fallback (rate-limit recovery) | $0.40/M tok |

All via [OpenRouter](https://openrouter.ai) — one API key, three models. Scout only activates when Qwen gets rate-limited.

---

## ⚙️ How Trades Happen

### 📥 Entry

1. **AI Scan** — 40 coins per cycle (rotating window), picks top 2-3 with direction, confidence, target, stop
2. **5 Signal Layers** vote: composite · enriched · unified ML · funding · whale
3. **7 Quality Gates**: VWAP extremes · ML contradiction · composite floor · EV gate · regime guard · survival · chase block
4. **Micro-Peak Timing** — samples WebSocket mids at 150ms for 4s, catches the best tick before entering

### 📤 Exit

| Trigger | Action |
|---|---|
| 🔒 Breakeven lock | Net ≥ fees → SL to entry, can't lose |
| 💰 Profit lock | 0.25% · 0.50% · 1.50% · 3.00% peaks → partial close |
| 🧠 AI re-check | Flat 5min → ask AI before kill (extends to 10min) |
| 📉 Trend-kill | 4h strongly against → immediate |
| 🩸 Bleed | Peak dropping + momentum turning → cut |
| 🚨 Liquidation | <1% distance → force escape |

---

## 📦 Modules

### 🧠 AI & Decision
| Module | Purpose |
|---|---|
| `ai_decider.py` | Coin selection, exits, sizing, AI debate |
| `ai_validator.py` | Second opinion on every prediction |
| `market_intel.py` | Fear & Greed · trending coins · BTC dom · news |

### 🔮 Prediction
| Module | Purpose |
|---|---|
| `continuous_predictor.py` | **Primary** — 15 layers, 6 horizons |
| `unified_predictor.py` | ML + structure + flow + macro |
| `perfect_predictor.py` | 3-signal: funding · OB · multi-TF |
| `ml_predictor.py` | XGBoost + LSTM models |
| `cryptogat_predictor.py` | Graph Attention Network |
| `kronos_predictor.py` | Transformer trajectory |

### 📊 Signals
`hyperliquid_strategy.py` · `unified_direction.py` · `vwap_signal.py` · `oi_delta.py` · `cvd_engine.py` · `cvd_divergence.py` · `taker_ratio.py` · `volume_delta.py` · `imbalance_trend.py` · `market_regime.py` · `market_structure.py` · `hurst_detector.py` · `chart_patterns.py` · `funding_acceleration.py` · `hyperliquid_funding_sniper.py`

### 📡 Data
`context_enricher.py` · `market_analyzer.py` · `price_extremes.py` · `correlation_tracker.py` · `cross_exchange.py` · `sector_rotation.py` · `economic_calendar.py`

### 🛡️ Risk
`hyperliquid_risk.py` · `hrp_sizing.py` · `ev_gate.py`

### ⚡ Execution
`hyperliquid_client.py` · `hyperliquid_execution.py` · `action_executor.py` · `hyperliquid_ws.py`

### 🔧 Extensions
`hyperliquid_delta_neutral.py` · `hyperliquid_evolution.py` · `hyperliquid_learner.py` · `hyperliquid_whale.py`

### 💀 Liquidation
`liquidation_zones.py` · `liquidation_monitor.py`

### ⛰️ Peak Detection
`peak_exhaustion_detector.py` · `pump_detector.py`

### 🏠 Main
`hyperliquid_daemon.py` — the brain stem (5,900 lines)

> 📖 Full details in [`STRUCTURE.md`](STRUCTURE.md)

---

## 🧰 Tools (`tools/`)

Standalone utilities — run separately, not imported by the daemon:

| Tool | What |
|---|---|
| `auto_retrain_ml.py` | Retrain 177 XGBoost models |
| `hyperliquid_backtest.py` | Full pipeline backtesting |
| `market_predictor.py` | LLM strategic forecaster |
| `sim_fast.py` | Parameter simulation |
| `optimize_grid.py` | Grid search 200+ combos |
| `performance_metrics.py` | Sharpe · drawdown · win rate |
| `prediction_memory.py` | Accuracy tracking |
| `_analyze_trades.py` | Log-based trade review |
| `train_ml_models.py` | Manual model training |
| `onchain_metrics.py` | Mayer Multiple · on-chain |
| `kraken_client.py` | Kraken Futures API |

---

## 🎯 Parameters

| What | Value |
|---|---|
| Leverage | 6× default, 10-12× on confluence |
| Position size | 20% equity, 35% on 4/4 agreement |
| Entry | Market IOC 0.5% slippage · micro-peak 4s @ 150ms |
| Max positions | 3 |
| Coins scanned | 40/cycle from 177 |
| Cycle time | 5 seconds |

---

## 💡 Approach

> 🧠 **AI is the brain** — send maximum data, trust its decisions over mechanical gates
>
> 🎯 **Better model > tighter gates** — a smarter AI fixes bad decisions, not stricter thresholds
>
> 📈 **Catch trends early** — don't chase pumps, ride the move from the start
>
> 🪙 **Trade the alts** — BTC/ETH for macro context only
>
> ⚡ **Every basis point counts** — micro-peak entry for better prices
>
> 🔄 **Stay active** — inaction is also a loss
