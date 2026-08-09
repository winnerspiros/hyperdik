# Hyperliquid AI Trading Bot

AI-driven perpetual futures trading on [Hyperliquid](https://hyperliquid.xyz) — 177 coins, 5-second cycles, 47 modules.

**Models**: Qwen 235B MoE (decisions) + Llama 4 Maverick (picks) + Llama 4 Scout (fallback) — all via OpenRouter.

---

## Quick Start

```bash
git clone https://github.com/winnerspiros/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot

python3 -m venv .venv && source .venv/bin/activate
pip install hyperliquid-python-sdk eth-account xgboost numpy scikit-learn pandas

# Hyperliquid wallet (required)
mkdir -p ~/.hyperliquid
cat > ~/.hyperliquid/config.json << 'EOF'
{
  "api_wallet": "0xYOUR_API_WALLET",
  "api_private_key": "0xYOUR_PRIVATE_KEY",
  "main_wallet": "0xYOUR_MAIN_WALLET"
}
EOF

# OpenRouter key (optional — enables AI coin selection)
export OPENROUTER_API_KEY=your_key_here

# Run
python3 -B hyperliquid_daemon.py
```

---

## How It Works

```
  Account  →  Mids  →  OI  →  Market Intel  →  Sector Scan
     │
     ▼
  Coin Rotation (40 of 177)  →  AI Picks Top 2-3
     │
     ▼
  Enrich  →  Composite  →  Unified  →  ML  →  Gates
     │
     ▼
  AI Sizing  →  Micro-Peak Entry  →  Execute
     │
     ▼
  Monitor  →  Exit Eval  →  TP/SL  →  Breakeven Lock
```

**Entry** — AI scans rotating 40-coin window each cycle, picks 2-3 candidates with direction, confidence, target, and stop. Five signal layers vote (composite, enriched, unified ML, funding, whale). Trades pass quality gates (VWAP extremes, ML contradiction, composite floor, EV gate). Before market entry, samples WebSocket mids at 150ms for up to 4s to catch a better price.

**Exit** — Breakeven lock: once net PnL covers fees, exchange SL moves to entry price. Tiered profit locks at 0.25%, 0.50%, 1.50%, and 3.00% peaks. Flat positions get an AI re-check at 5 minutes before killing. Positions trending strongly against are cut immediately. Liquidation survival triggers at <1% distance.

---

## Modules

### AI
| Module | Role |
|---|---|
| `ai_decider.py` | Coin selection, exit evaluation, sizing, debate |
| `ai_validator.py` | Second-opinion on every AI decision |
| `market_intel.py` | Fear & Greed index, trending coins, BTC dominance, news |

### Prediction
| Module | Role |
|---|---|
| `continuous_predictor.py` | 15-layer ensemble, 6 time horizons |
| `unified_predictor.py` | ML + market structure + liquidity flow |
| `perfect_predictor.py` | 3-signal approach — funding, order book, multi-TF |
| `ml_predictor.py` | XGBoost + LSTM models |
| `cryptogat_predictor.py` | Graph Attention Network — cross-coin signals |
| `kronos_predictor.py` | Transformer trajectory prediction |

### Signals
| Module | Role |
|---|---|
| `hyperliquid_strategy.py` | 5-pillar composite signal + ensemble |
| `unified_direction.py` | Directional consensus from all layers |
| `vwap_signal.py` | VWAP deviation, sigma bands, mean reversion |
| `oi_delta.py` | Open Interest delta + price/OI divergence |
| `cvd_engine.py` | Cumulative Volume Delta from real trades |
| `cvd_divergence.py` | CVD/price divergence detection |
| `taker_ratio.py` | Taker buy/sell ratio |
| `volume_delta.py` | Bar-level buy vs sell volume |
| `imbalance_trend.py` | Order book imbalance trending |
| `market_regime.py` | Regime — trending, ranging, volatile, extreme |
| `market_structure.py` | Swing points, BOS/CHOCH, structure breaks |
| `hurst_detector.py` | Hurst exponent — mean reversion vs trending |
| `chart_patterns.py` | Classical chart pattern recognition |
| `funding_acceleration.py` | Funding rate change velocity |
| `hyperliquid_funding_sniper.py` | Full funding rate context + prediction |

### Data
| Module | Role |
|---|---|
| `context_enricher.py` | Assembles rich AI context from all sources |
| `market_analyzer.py` | Market-wide heat, breadth, regime |
| `price_extremes.py` | Multi-TF range position, dynamic TP targets |
| `correlation_tracker.py` | Rolling correlations between positions |
| `cross_exchange.py` | Binance/HL price divergence |
| `sector_rotation.py` | Sector momentum tracking |
| `economic_calendar.py` | Upcoming events — FOMC, CPI, etc. |

### Risk
| Module | Role |
|---|---|
| `hyperliquid_risk.py` | WEL/TWEL limits, HSL, position caps |
| `hrp_sizing.py` | Hierarchical Risk Parity sizing |
| `ev_gate.py` | Expected Value gate — blocks negative-EV trades |

### Execution
| Module | Role |
|---|---|
| `hyperliquid_client.py` | SDK wrapper, adaptive rate limiter, all API calls |
| `hyperliquid_execution.py` | Order placement, fills, TP/SL management |
| `action_executor.py` | Pipeline-based action execution |
| `hyperliquid_ws.py` | WebSocket — real-time trades, books, mids |

### Extensions
| Module | Role |
|---|---|
| `hyperliquid_delta_neutral.py` | Delta-neutral position balancing |
| `hyperliquid_evolution.py` | Strategy parameter optimization |
| `hyperliquid_learner.py` | Self-learning from trade outcomes |
| `hyperliquid_whale.py` | Whale wallet activity tracking |

### Liquidation
| Module | Role |
|---|---|
| `liquidation_zones.py` | Order book depth clusters, heatmap zones |
| `liquidation_monitor.py` | Real-time WS fill monitoring + cascade detection |

### Peak Detection
| Module | Role |
|---|---|
| `peak_exhaustion_detector.py` | Candle-level RSI divergence + tick-level order book thinning |
| `pump_detector.py` | Pump & dump detection for AI risk context |

### Main
| Module | Role |
|---|---|
| `hyperliquid_daemon.py` | Main loop — entry, exit, position monitor, signal scan |

---

## Tools (`tools/`)

Standalone utilities — not imported by the daemon:

| Script | Purpose |
|---|---|
| `auto_retrain_ml.py` | Auto-retrain 177 XGBoost models every 6h |
| `train_ml_models.py` | Manual ML model training |
| `hyperliquid_backtest.py` | Full pipeline backtesting |
| `market_predictor.py` | LLM strategic market forecaster |
| `sim_fast.py` | Fast parameter simulation |
| `optimize_grid.py` | Grid search over 200+ parameter combos |
| `performance_metrics.py` | Trade analytics — Sharpe, drawdown, win rate |
| `prediction_memory.py` | Track prediction accuracy over time |
| `_analyze_trades.py` | Parse daemon logs for trade review |
| `onchain_metrics.py` | Mayer Multiple, on-chain summaries |
| `kraken_client.py` | Kraken Futures API (cross-exchange) |

---

## Key Parameters

**Entry**: 6x default leverage (10-12x on 4/4 confluence), 20% equity per position (35% on confluence), 0.5% slippage market IOC, 4s micro-peak window at 150ms samples.

**Exit**: Breakeven lock at net ≥ 0.42%, profit lock tiers at 0.25/0.50/1.50/3.00%, flat-position AI re-check at 300s (extends to 600s if AI confirms), trend-kill at 4h <1.5% from extreme, bleed detection at >0.3% drop with momentum turning.

**Risk**: TWEL 100%, WEL 40% per direction, HSL dynamic drawdown, max 3 positions.

---

## Approach

- Trust AI decisions over mechanical gates — send max data, let the model decide
- Better model quality fixes bad decisions better than stricter thresholds
- BTC/ETH for macro context only — trade the alt coins
- Don't chase pumps, catch trends early, stay active
- Every basis point on entry matters — micro-peak timing for better prices