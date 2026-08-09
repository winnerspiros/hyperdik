# HYPERDIK — Module Reference

Complete map of every module in the system. See [README.md](README.md) for architecture and quick start.


## AI & Decision

| File | Purpose |
|:---|:---|
| `ai_decider.py` | Coin selection, exit evaluation, sizing, AI debate — Qwen 235B + Llama 4 Maverick |
| `ai_validator.py` | Second-opinion validation on every AI prediction |
| `market_intel.py` | Fear & Greed index, trending coins, BTC dominance, news headlines |

## Prediction

| File | Purpose |
|:---|:---|
| `continuous_predictor.py` | **Primary** — 15-layer weighted ensemble, 6 time horizons |
| `unified_predictor.py` | ML + market structure + liquidity flow + macro |
| `perfect_predictor.py` | 3-signal: funding rate · order book · multi-timeframe |
| `ml_predictor.py` | XGBoost + LSTM direction models |
| `cryptogat_predictor.py` | Graph Attention Network — cross-coin signal propagation |
| `kronos_predictor.py` | Transformer trajectory prediction |

## Signals

| File | Purpose |
|:---|:---|
| `hyperliquid_strategy.py` | 5-pillar composite signal + weighted ensemble |
| `unified_direction.py` | Directional consensus from all prediction layers |
| `vwap_signal.py` | VWAP deviation, sigma bands, mean reversion detection |
| `oi_delta.py` | Open Interest delta + price/OI divergence |
| `cvd_engine.py` | Cumulative Volume Delta from real WebSocket trades |
| `cvd_divergence.py` | CVD/price divergence signals |
| `taker_ratio.py` | Taker buy/sell ratio from trade tape |
| `volume_delta.py` | Bar-level buy vs sell volume + derived features |
| `imbalance_trend.py` | Order book imbalance trending + momentum |
| `market_regime.py` | Regime classification — trending, ranging, volatile, extreme |
| `market_structure.py` | Swing points, BOS/CHOCH, structure breaks |
| `hurst_detector.py` | Hurst exponent — mean-reversion vs trending classification |
| `chart_patterns.py` | Classical chart pattern recognition |
| `funding_acceleration.py` | Funding rate change velocity tracking |
| `hyperliquid_funding_sniper.py` | Full funding rate context + arbitrage windows |

## Data & Context

| File | Purpose |
|:---|:---|
| `context_enricher.py` | Assembles rich AI context from all data sources |
| `market_analyzer.py` | Market-wide heat, breadth, regime overview |
| `price_extremes.py` | Multi-timeframe range position, dynamic TP targets |
| `correlation_tracker.py` | Rolling correlation matrix between positions |
| `cross_exchange.py` | Binance/HL price divergence detection |
| `sector_rotation.py` | Sector-level momentum tracking |
| `economic_calendar.py` | Upcoming events — FOMC, CPI, etc. |

## Risk & Sizing

| File | Purpose |
|:---|:---|
| `hyperliquid_risk.py` | WEL/TWEL limits, HSL drawdown protection, position caps |
| `hrp_sizing.py` | Hierarchical Risk Parity position sizing |
| `ev_gate.py` | Expected Value gate — blocks negative-EV trades |

## Execution

| File | Purpose |
|:---|:---|
| `hyperliquid_client.py` | Hyperliquid SDK wrapper, adaptive rate limiter, all API calls |
| `hyperliquid_execution.py` | Order placement, fills, multi-tier TP/SL management |
| `action_executor.py` | Pipeline-based action execution engine |
| `hyperliquid_ws.py` | WebSocket — real-time trades, order books, mids |

## Evolution & Learning

| File | Purpose |
|:---|:---|
| `hyperliquid_evolution.py` | GPT‑4.1 autonomous optimizer — rewrites code, params, prompts |
| `prompt_optimizer.py` | Meta-prompt improvement — A/B tests, graduates winners |
| `hyperliquid_learner.py` | Self-learning from trade outcomes + weight adjustment |

## Strategy Extensions

| File | Purpose |
|:---|:---|
| `hyperliquid_delta_neutral.py` | Delta-neutral position balancing |
| `hyperliquid_whale.py` | Whale wallet activity tracking |

## Liquidation

| File | Purpose |
|:---|:---|
| `liquidation_zones.py` | Order book depth clustering, heatmap zones |
| `liquidation_monitor.py` | Real-time WebSocket fill monitoring + cascade detection |

## Peak Detection

| File | Purpose |
|:---|:---|
| `peak_exhaustion_detector.py` | Candle-level RSI divergence + tick-level OB thinning |
| `pump_detector.py` | Pump & dump detection for AI risk context |

## Main Loop

| File | Purpose |
|:---|:---|
| `hyperliquid_daemon.py` | ~6,000 lines — entry, exit, position monitor, signal scan, cleanup |


## Data Flow

```
 ┌──────────┐     ┌──────────┐     ┌──────────┐
 │ Predict  │────▶│ Signals  │────▶│  AI      │
 │ 6 models │     │ 15 files │     │ Decider  │
 └──────────┘     └──────────┘     └────┬─────┘
                                        │
    ┌───────────────────────────────────┘
    ▼
 ┌──────────┐     ┌──────────┐     ┌──────────┐
 │  Risk    │────▶│ Execute  │────▶│  Exit    │
 │ 3 files  │     │ 4 files  │     │ Monitor  │
 └──────────┘     └──────────┘     └────┬─────┘
                                        │
              ┌─────────────────────────┘
              ▼
       ┌────────────┐     ┌────────────┐
       │ Evolution  │     │  Learner   │
       │ GPT‑4.1    │     │ Weights    │
       └────────────┘     └────────────┘
```


## Tools (`tools/`)

Standalone utilities — not imported by the daemon:

| File | Purpose |
|:---|:---|
| `auto_retrain_ml.py` | Retrain 177 XGBoost models every 6h |
| `hyperliquid_backtest.py` | Full pipeline backtesting engine |
| `market_predictor.py` | LLM strategic market forecaster |
| `sim_fast.py` | Fast parameter sweep simulation |
| `optimize_grid.py` | Grid search over 200+ parameter combos |
| `performance_metrics.py` | Sharpe · drawdown · win rate analytics |
| `prediction_memory.py` | Prediction accuracy tracking over time |
| `_analyze_trades.py` | Parse daemon logs for trade review |
| `train_ml_models.py` | Manual ML model training |
| `onchain_metrics.py` | Mayer Multiple, on-chain summaries |
| `kraken_client.py` | Kraken Futures API (cross-exchange) |