# Hyperliquid AI Trading Bot

AI-driven perpetual futures trading on [Hyperliquid](https://hyperliquid.xyz). Multi-model ensemble (Qwen 235B + Llama 4 Maverick + Llama 4 Scout fallback) with zero-loss exit system, micro-peak entry timing, and live market intelligence.

**Live** · 177 coins · 5s cycles · 47 modules · 11 tools

---

## Quick Start

```bash
git clone https://github.com/winnerspiros/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot
python3 -m venv .venv && source .venv/bin/activate
pip install hyperliquid-python-sdk eth-account xgboost numpy scikit-learn pandas

# API keys (required)
mkdir -p ~/.hyperliquid
cat > ~/.hyperliquid/config.json << 'EOF'
{"api_wallet": "0xYOUR_API", "api_private_key": "0xYOUR_KEY", "main_wallet": "0xYOUR_MAIN"}
EOF

# AI keys (optional — enables AI coin selection)
export OPENROUTER_API_KEY=your_key

# Run
python3 -B hyperliquid_daemon.py
```

---

## How It Works

```
 ┌─ MAIN CYCLE (5s) ──────────────────────────────────────────┐
 │                                                            │
 │  Account → Mids → OI → AI Market Intel → Sector Scan       │
 │     ↓                                                      │
 │  Coin Rotation (40/177) → AI Picks Top 3                   │
 │     ↓                                                      │
 │  Per Coin: Enrich → Composite → Unified → ML → Gates       │
 │     ↓                                                      │
 │  AI Sizing → Micro-Peak Entry → Execute                    │
 │     ↓                                                      │
 │  Monitor → Exit Eval → TP/SL → Breakeven Lock              │
 └────────────────────────────────────────────────────────────┘
```

### Entry
1. **AI scans** 40 coins/cycle (rotating window), picks top 2-3 with confidence + direction + target
2. **5 signal layers** vote: composite, enriched, unified ML, funding, whale — weighted ensemble
3. **Gates**: VWAP extreme block, ML contradiction, composite floor, EV gate, regime guard
4. **Micro-peak timing**: samples WebSocket mids at 150ms for up to 4s, enters at best price

### Exit
- **Breakeven lock**: net ≥ fees → SL moved to entry, trade can't lose
- **Tiered profit lock**: 0.25% / 0.50% / 1.50% / 3.00% peak → escalating partial closes
- **AI re-check**: flat 5min → AI asked "still confident?" before killing (extends to 10min if yes)
- **Trend-kill**: 4h strongly against → immediate close
- **Bleed detection**: peak dropping + momentum turning → close
- **Liquidation survival**: <1% from liquidation → force close

---

## AI Models

| Model | Role | Cost |
|---|---|---|
| **Qwen 235B MoE** | Decisions, sizing, exit evaluation | $0.64/M tok |
| **Llama 4 Maverick** (free) | Coin picks, market assessment | Free |
| **Llama 4 Scout** | Auto-fallback on errors | Free |

All via OpenRouter. Ban list: Gemini, DeepSeek, OpenAI/GPT.

---

## Module Map

### AI & Decision
| Module | Purpose |
|---|---|
| `ai_decider.py` | Coin selection, exit evaluation, sizing, debate |
| `ai_validator.py` | Second-opinion validation on every prediction |
| `market_intel.py` | Fear & Greed, trending coins, BTC dominance, news |

### Prediction
| Module | Purpose |
|---|---|
| `continuous_predictor.py` | **Primary** — 15-layer ensemble, 6 horizons |
| `unified_predictor.py` | ML + structure + flow + macro |
| `perfect_predictor.py` | 3-signal (funding, OB, multi-TF) |
| `ml_predictor.py` | XGBoost + LSTM direction models |
| `cryptogat_predictor.py` | Graph Attention Network cross-coin |
| `kronos_predictor.py` | Transformer trajectory prediction |

### Signals
| Module | Purpose |
|---|---|
| `hyperliquid_strategy.py` | 5-pillar composite + ensemble signal |
| `unified_direction.py` | Directional signal from all layers |
| `vwap_signal.py` | VWAP deviation, sigma bands, mean reversion |
| `oi_delta.py` | Open Interest delta + divergence |
| `cvd_engine.py` | Cumulative Volume Delta (real trades) |
| `cvd_divergence.py` | CVD/price divergence |
| `taker_ratio.py` | Taker buy/sell ratio |
| `volume_delta.py` | Bar-level volume delta |
| `imbalance_trend.py` | Order book imbalance + trend |
| `market_regime.py` | Regime classification |
| `market_structure.py` | Swing points, BOS/CHOCH |
| `hurst_detector.py` | Hurst exponent |
| `chart_patterns.py` | Classical chart patterns |
| `funding_acceleration.py` | Funding rate acceleration |
| `hyperliquid_funding_sniper.py` | Full funding rate context |

### Data & Context
| Module | Purpose |
|---|---|
| `context_enricher.py` | Rich AI context from all sources |
| `market_analyzer.py` | Market-wide heat, regime, breadth |
| `price_extremes.py` | Multi-TF range position, TP targets |
| `correlation_tracker.py` | Rolling correlations |
| `cross_exchange.py` | Binance/HL price divergence |
| `sector_rotation.py` | Sector momentum tracking |
| `economic_calendar.py` | FOMC, CPI, etc. |

### Risk & Sizing
| Module | Purpose |
|---|---|
| `hyperliquid_risk.py` | WEL/TWEL, HSL, position limits |
| `hrp_sizing.py` | Hierarchical Risk Parity sizing |
| `ev_gate.py` | Expected Value gate |

### Execution
| Module | Purpose |
|---|---|
| `hyperliquid_client.py` | SDK wrapper, adaptive rate limiter |
| `hyperliquid_execution.py` | Order placement, fills, TP/SL |
| `action_executor.py` | Action pipeline executor |
| `hyperliquid_ws.py` | WebSocket real-time feed |

### Strategy Extensions
| Module | Purpose |
|---|---|
| `hyperliquid_delta_neutral.py` | Delta-neutral balancing |
| `hyperliquid_evolution.py` | Parameter evolution |
| `hyperliquid_learner.py` | Self-learning from outcomes |
| `hyperliquid_whale.py` | Whale wallet tracking |

### Liquidation
| Module | Purpose |
|---|---|
| `liquidation_zones.py` | OB depth clusters, heatmap zones |
| `liquidation_monitor.py` | Real WS fill + cascade detection |

### Peak Detection
| Module | Purpose |
|---|---|
| `peak_exhaustion_detector.py` | Candle-level RSI divergence + tick-level OB thinning |
| `pump_detector.py` | Pump & dump risk warnings |

### Main
| Module | Purpose |
|---|---|
| `hyperliquid_daemon.py` | Main loop — entry, exit, monitor, signal scan |

**See `STRUCTURE.md`** for full details on each module's role.

---

## Tools (`tools/`)

Standalone utilities — not imported by the daemon:

| Script | Purpose |
|---|---|
| `market_predictor.py` | LLM strategic forecaster |
| `auto_retrain_ml.py` | Auto-retrain 177 XGBoost models |
| `train_ml_models.py` | Manual ML model training |
| `hyperliquid_backtest.py` | Backtesting engine |
| `performance_metrics.py` | Trade performance analytics |
| `prediction_memory.py` | Prediction accuracy tracking |
| `_analyze_trades.py` | Log-based trade analysis |
| `onchain_metrics.py` | Mayer Multiple, on-chain data |
| `kraken_client.py` | Kraken Futures API client |
| `optimize_grid.py` | Parameter grid search |
| `sim_fast.py` | Fast simulation runner |

---

## Configuration

### Risk Limits
- TWEL (Total Weighted Exposure Limit): 100% of equity
- WEL (Weighted Exposure Limit): 40% per direction
- HSL (Historical Stress Limit): dynamic drawdown protection
- Max positions: 3 (scales with equity)

### Entry Parameters
- Default leverage: 6x (10-12x on 4/4 confluence)
- Position size: 20% equity (35% on confluence)
- Slippage: 0.5% (market IOC)
- Micro-peak window: 4s at 150ms samples

### Exit Parameters
- Breakeven lock: net ≥ 0.42% (roundtrip fee × leverage)
- Profit lock tiers: 0.25% / 0.50% / 1.50% / 3.00%
- NEVER-GREEN timeout: 300s → AI re-check → 600s
- Trend-kill: 4h <1.5% from extreme
- Bleed: >0.3% drop + momentum turning

---

## Philosophy

> AI is the brain — send max data, trust output over mechanical gates.
> Zero losses: every close must be net ≥ 0%. Inaction is also a loss.
> Miss pumps, catch trends early, trade more.
> Coin variety — BTC/ETH context-only, trade the alts.
> Better model fixes bad decisions, not stricter gates.