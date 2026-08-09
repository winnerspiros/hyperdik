# HYPERDIK — Module Map

## AI / Decision Making
ai_decider.py              — AI coin selection, exit evaluation, debate (Qwen 235B + Llama 4 Maverick)
ai_validator.py            — AI second-opinion validation on every prediction
market_intel.py            — Fear & Greed, trending coins, BTC dominance, news headlines

## Prediction Engines
continuous_predictor.py    — 15-layer weighted ensemble, 6 horizons (PRIMARY)
unified_predictor.py       — ML + exhaustion + structure + flow + macro
perfect_predictor.py       — 3-signal approach (funding, order book, multi-TF)
ml_predictor.py            — XGBoost + LSTM direction prediction (library)
cryptogat_predictor.py     — Graph Attention Network cross-coin prediction
kronos_predictor.py        — Transformer trajectory prediction (Kronos-mini)

## Data & Context
context_enricher.py        — Assemblies rich context for AI from all data sources
market_analyzer.py         — Market-wide analysis (heat, regime, breadth)
price_extremes.py          — Multi-TF range position, dynamic TP targets
correlation_tracker.py     — Rolling correlations between held coins
cross_exchange.py          — Binance/HL price divergence detection
sector_rotation.py         — Sector momentum tracking
economic_calendar.py       — Upcoming economic events (FOMC, CPI, etc.)

## Risk & Sizing
hyperliquid_risk.py        — WEL/TWEL, HSL, position limits, exposure
hrp_sizing.py              — Hierarchical Risk Parity position sizing
ev_gate.py                 — Expected Value gate + portfolio heat

## Execution
hyperliquid_client.py      — Hyperliquid SDK wrapper, rate limiter, order ops
hyperliquid_execution.py   — Round price/size, order placement, fills
action_executor.py         — File-based action pipeline executor
hyperliquid_ws.py          — WebSocket real-time feed (trades, books, mids)

## Signals (order flow, market structure)
oi_delta.py                — Open Interest delta + price/OI divergence
vwap_signal.py             — VWAP deviation, sigma bands, mean reversion
cvd_engine.py              — Cumulative Volume Delta engine (real trades)
cvd_divergence.py          — CVD/price divergence signals
taker_ratio.py             — Taker buy/sell ratio
volume_delta.py            — Bar-level volume delta + features
imbalance_trend.py         — Order book imbalance + trend detection
funding_acceleration.py    — Funding rate acceleration tracking
hyperliquid_funding_sniper.py — Full funding rate context + prediction
market_regime.py           — Regime classification (trending/ranging/volatile)
market_structure.py        — Swing points, BOS/CHOCH, structure breaks
hurst_detector.py          — Hurst exponent for mean-reversion vs trending
chart_patterns.py          — Classical chart pattern detection

## Liquidation
liquidation_zones.py       — OB depth clustering, liquidation heatmap zones
liquidation_monitor.py     — Real WS fill monitoring + cascade detection

## Strategy
hyperliquid_strategy.py    — 5-pillar composite + ensemble signal builder
unified_direction.py       — Unified directional signal from all layers
hyperliquid_delta_neutral.py — Delta-neutral position balancing
hyperliquid_evolution.py   — Strategy parameter evolution/optimization
hyperliquid_learner.py     — Self-learning from trade outcomes
hyperliquid_whale.py       — Whale wallet tracking

## Peak Detection
peak_exhaustion_detector.py — Candle-level RSI divergence + tick-level OB thinning
pump_detector.py            — Pump & dump detection for AI risk warnings

## Main
hyperliquid_daemon.py      — 5,907 lines: main loop, entry, exit, monitor, signal scan
