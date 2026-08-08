# Top 10 Actionable Techniques for Predicting Crypto Tops & Bottoms
## Research compiled 2026-07-16 — Hyperliquid Perpetual Futures Context

---

## 1. Real-Time CVD from Trade-Level Delta (NOT OHLCV proxy)

**Source:** [alpha-engine by atg0dd](https://github.com/atg0dd/alpha-engine) ⭐ — institutional-grade quantitative framework  
**What it does:** Computes Cumulative Volume Delta from actual aggressive buy/sell trade ticks in O(1) per update, distinguishing taker buys (+volume) from taker sells (-volume). Detects divergence between price and delta — the #1 leading reversal signal.  
**Our current gap:** `cvd_divergence.py` uses close-vs-open as a proxy (L84-103). This is a lagging, noisy approximation. Real trade-level CVD from the Hyperliquid WebSocket `trades` channel gives you the actual aggressor side.

**How to integrate TODAY:**
1. The Hyperliquid WebSocket already sends `trades` with a `side` field. Your `hyperliquid_ws.py` can forward trade ticks to a new `CVDEngine` class.
2. Copy the 20-line `HighFrequencyMetrics.update_cvd()` from `alpha-engine/src/features/metrics.py` — it's exactly what you need.
3. Feed the stream into your existing `tick_peak_detector.py` to replace the OHLCV-based delta calculation.
4. Add divergence detection: compare CVD swing highs/lows vs price swing highs/lows.

**Effort:** 3-4 hours  
**Key file to study:** `alpha-engine/src/features/metrics.py` — the `HighFrequencyMetrics` class (57 lines, pure gold)

---

## 2. Volume Profile: POC Migration for Support/Resistance Zones

**Source:** [srl-python-indicators](https://github.com/srlcarlg/srl-python-indicators) — `volume_profile.py`, `tpo_profile.py`  
**What it does:** Computes Volume-at-Price (VPVR), identifies Point of Control (POC), Value Area High (VAH), Value Area Low (VAL). POC migration up = bullish trend shift. Price breaking VAH/VAL with volume = breakout confirmation. Unfilled auctions (low-volume nodes) = magnetic price targets.  
**Our current gap:** No volume profile analysis in the prediction stack.

**How to integrate TODAY:**
1. Use the Hyperliquid SDK's `info.candles_snapshot()` to pull 1h candles for rolling 24h/7d windows.
2. Build a `VolumeProfile` class: bin prices into N levels, sum volume per bin, find POC (max), VAH/VAL (68% value area).
3. Feed POC migration direction (up/down/flat) into `unified_predictor.py` as a new feature in the "Market Structure" category.
4. Add unfilled auction detection: identify price gaps with <10% of average volume — these act as magnets.

**Effort:** 4-6 hours  
**Key concept:** "POC migration" — when POC shifts up candle-over-candle, it's the clearest sign of institutional accumulation. When it stalls, distribution is happening. This is the single most reliable volume-based signal.

---

## 3. VPIN (Volume-Synchronized Probability of Informed Trading) — Anticipate Adverse Selection

**Source:** [lubluniky/vpin-opensource](https://github.com/lubluniky/vpin-opensource) | Paper: Easley, López de Prado, O'Hara  
**What it does:** VPIN measures the probability that informed traders are active by tracking volume imbalance within equal-volume buckets (not time buckets). VPIN spikes BEFORE volatility — it's a leading indicator of toxic order flow and impending reversals. CDF of VPIN > 0.8 = high probability of sharp move.  
**Our current gap:** No order flow toxicity measurement.

**How to integrate TODAY:**
1. VPIN needs trade-level data (side + size) — use the `trades` WebSocket subscription you already have.
2. Bucket trades into volume groups of 50 equal-sized buckets per day. For each bucket, compute `VPIN = |V_buy - V_sell| / V_total`.
3. Compute rolling CDF — when VPIN crosses 80th percentile of its own distribution, it signals informed trading.
4. Feed VPIN into your `peak_exhaustion_detector.py` as an early warning: VPIN spikes → exhaustion is near but hasn't happened yet.
5. The key insight: VPIN predicts VOLATILITY, not direction. Combine with CVD for direction.

**Effort:** 5-8 hours  
**Key concept:** "Volume bucketing" — you don't use time buckets. You fill volume buckets and measure VPIN per bucket. This makes the metric robust to varying trading activity across day/night cycles.

---

## 4. Hurst Exponent for Adaptive Mean Reversion Detection

**Source:** [pairscan/ratio-mean-reversion](https://github.com/pairscan/ratio-mean-reversion) | [FilippoMB/python-time-series-handbook](https://github.com/FilippoMB/python-time-series-handbook)  
**What it does:** The Hurst exponent H tells you whether a time series is trending (H>0.5), mean-reverting (H<0.5), or random (H=0.5). This is CRITICAL because your mean-reversion strategies (funding sniper, scalper) are worthless when H>0.5.  
**Our current gap:** The funding sniper and scalper assume mean reversion always works. They need a regime filter.

**How to integrate TODAY:**
1. Compute Hurst exponent on a rolling window (e.g., 100 data points) using R/S analysis or DFA.
2. Feed into every strategy that relies on mean reversion:
   - H < 0.4: strong mean reversion → increase position size
   - 0.4 ≤ H ≤ 0.6: random → normal sizing
   - H > 0.6: trending → disable mean-reversion strategies entirely, switch to momentum
3. Add to `hyperliquid_funding_sniper.py` as a regime gate
4. Add to `scalper.py` as a strategy selector

**Effort:** 2-3 hours  
**Key concept:** R/S analysis: `log(R/S) = H * log(n) + c`. Compute over multiple window sizes n, slope = H. Python implementation is ~20 lines.

---

## 5. Liquidation Cascade Real-Time Monitor (Hyperliquid-native WebSocket)

**Source:** [hyperliquid-realtime-data by bwroniszewski](https://github.com/bwroniszewski/hyperliquid-realtime-data) ⭐8 — Python | [Dwellir liquidation tracker tutorial](https://www.dwellir.com/blog/building-real-time-hyperliquid-liquidation-tracker)  
**What it does:** Subscribes to Hyperliquid WebSocket for real-time L2 order book, trades, candles, AND asset context (oracle prices, funding rates, open interest). The Dwellir tutorial shows how to extract liquidation events from the gRPC `StreamFills` endpoint — every on-chain liquidation is immediately visible.  
**Our current gap:** `liquidation_cascade.py` uses API polling with 30s cache. The WebSocket approach is real-time and more complete.

**How to integrate TODAY:**
1. The `hyperliquid-realtime-data` repo already has the exact WebSocket subscription code you need (see `realtime_data_HL.py` lines for `l2Book`, `trades`, `candle`, `assetCtx` subscriptions).
2. Subscribe to `trades` channel on Hyperliquid WS — liquidation fills are tagged differently.
3. Alternatively, use the Hyperliquid L1 gRPC `StreamFills` endpoint to get raw on-chain fills and filter for liquidation field.
4. Feed real-time liquidation events into your existing `liquidation_cascade.py` — it already has the cascade detection logic (Hawkes process, LAI asymmetry index).
5. Build a rolling window (5-15 min) of liquidation sizes by side. When one side dominates 3:1+, a reversal is imminent.

**Effort:** 4-6 hours  
**Key file:** `bwroniszewski/hyperliquid-realtime-data/realtime_data_HL.py` — the WebSocket subscription pattern works TODAY with your existing Hyperliquid credentials.

---

## 6. Deep Order Flow Imbalance (OFI) from L2 Order Book Snapshots

**Source:** [jaefit/deep-ofi](https://github.com/jaefit/deep-ofi) | Cont-Kukanov-Stoikov paper  
**What it does:** Computes multi-level Order Flow Imbalance (OFI) from consecutive L2 order book snapshots. The key finding: **raw queue imbalance (state) beats OFI (flow)** for short-horizon prediction, but **OFI + raw combined beats either alone**. This is directly actionable for your tick-level prediction.  
**Our current gap:** `tick_peak_detector.py` has "order book thinning" but doesn't compute proper OFI.

**How to integrate TODAY:**
1. Hyperliquid WS sends `l2Book` snapshots. Track consecutive snapshots (100ms apart).
2. OFI per level i: `Δbid_size_i - Δask_size_i` (change in resting order sizes).
3. Multi-level OFI: sum over top N levels (5-10 levels). Window it over 10-50 snapshots.
4. Feed windowed OFI into `tick_peak_detector.py` alongside the existing order book thinning signal.
5. The deep-ofi repo's key finding: queue imbalance (bid_size/(bid_size+ask_size) at best quote) is the SINGLE strongest predictor. Start there, then add OFI on top.

**Effort:** 3-5 hours  
**Key concept:** "Queue imbalance" = `bid_qty / (bid_qty + ask_qty)` at best bid/ask. When > 0.65, price is about to rise. When < 0.35, price is about to fall. This is the #1 microstructure predictor per the deep-ofi research.

---

## 7. PatchTST + N-BEATS for Multi-Horizon Price Prediction

**Source:** [PatchTST](https://github.com/yuqinie98/PatchTST) ⭐2,653 (ICLR 2023) | [N-BEATS](https://github.com/philipperemy/n-beats) | [Crypto Forecasting Benchmark](https://github.com/StephanAkkerman/crypto-forecasting-benchmark)  
**What it does:** PatchTST divides time series into patches (subseries tokens) and achieves SOTA on long-term forecasting with 21% MSE reduction. N-BEATS is an interpretable MLP-based architecture that decomposes predictions into trend + seasonality — perfect for identifying cyclical tops/bottoms. The crypto benchmark tested these on 21 coins across market caps.  
**Our current gap:** XGBoost+LSTM is solid but outdated. These transformers capture multi-scale patterns.

**How to integrate TODAY:**
1. **Quick win:** Install `nbeats-pytorch` or `neuralforecast` (which includes PatchTST, N-BEATS, TFT). Use the Nixtla `neuralforecast` library — it has a unified API.
2. For PatchTST: feed 512 lookback candles → predict 96 steps forward. Use channel-independence mode (treat each feature as univariate).
3. For N-BEATS: use the interpretable configuration (trend + seasonality stacks). The trend component IS your top/bottom signal — when trend forecast reverses, you're at an extreme.
4. The crypto benchmark repo tested N-BEATS against XGBoost/LSTM/GRU/TCN/TFT on 21 coins — their config.py has the exact model parameters you need.
5. Start with the `neuralforecast` library (pip install) and run N-BEATS on your 15m candle data. Compare to your existing LSTM.

**Effort:** 8-12 hours (significant but highest ROI)  
**Key concept:** PatchTST's "patching" — instead of point-by-point attention, it groups 16-64 time steps into one token. This captures LOCAL patterns (the exact shape of a top or bottom) that point-wise models miss. This is THE architecture for top/bottom detection.

---

## 8. Funding Rate Prediction via Ornstein-Uhlenbeck Process

**Source:** [Yosri-Ben-Halima/Modeling-Funding-Rates](https://github.com/Yosri-Ben-Halima/Modeling-Funding-Rates-Using-Stochastic-Models-and-Quantifying-Risk-for-BTC-Prepetuals) ⭐10 | Paper: [arxiv.org/abs/2506.08573](https://arxiv.org/abs/2506.08573)  
**What it does:** Models funding rates as a mean-reverting Ornstein-Uhlenbeck process with Merton jump diffusion for the underlying price. Predicts when funding will revert (entry timing) AND estimates expected time-to-liquidation (risk management).  
**Our current gap:** `hyperliquid_funding_sniper.py` uses static Z-scores. An OU process gives you a dynamic prediction of WHERE and WHEN funding will revert.

**How to integrate TODAY:**
1. Fit an OU process to each asset's funding rate history: `dX_t = θ(μ - X_t)dt + σdW_t`. Parameters: θ (mean reversion speed), μ (long-term mean), σ (volatility).
2. The half-life of mean reversion = `ln(2)/θ` — this tells you how long to hold the position.
3. Use `statsmodels` or `numpy` to calibrate θ, μ, σ from historical funding rates (Hyperliquid API gives you 8h funding history per asset).
4. Feed the OU-predicted funding rate (1h, 4h, 8h ahead) into your funding sniper — replace the static Z-score threshold with a dynamic entry signal.
5. Key: when `|current_rate - μ| / σ > 2` AND `θ` is high (fast reversion), that's your entry.

**Effort:** 4-6 hours  
**Key concept:** OU half-life. If half-life is 2 hours, you know the position should be profitable within 2 hours. If it's 72 hours, it's too slow for a small account.

---

## 9. DeepLOB: CNN+LSTM for Price Movement from Raw Order Book

**Source:** [DeepLOB](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) | Paper: [arxiv.org/abs/1808.03668](https://arxiv.org/abs/1808.03668)  
**What it does:** A convolutional neural network that takes raw L2 order book data (40 levels of bid/ask prices and volumes) and predicts price movement direction. Achieved 71% accuracy on 2-second horizons. Uses CNN filters to capture spatial structure of the book + LSTM for temporal dynamics.  
**Our current gap:** No model that directly ingests raw order book state. Your ML uses derived features.

**How to integrate TODAY:**
1. The Hyperliquid `l2Book` WS subscription gives you exactly the input DeepLOB needs: bids[0:40] and asks[0:40] with prices and sizes.
2. The architecture is straightforward: CNN filters over the order book levels (capturing wall thickness, gaps, clusters) → LSTM over time → output layer.
3. Build a lightweight version: 10 levels instead of 40 (Hyperliquid L2Book gives up to 200 levels, but 10 is enough for short horizons).
4. Feed the DeepLOB prediction into your `tick_peak_detector.py` — replace or complement the existing "order book thinning" logic.
5. PyTorch implementation is ~100 lines. The paper provides the exact architecture.

**Effort:** 6-10 hours  
**Key concept:** The CNN captures the "shape" of the order book — wall steepness, spread width, volume clustering at specific levels. These spatial patterns precede price moves but are invisible to feature-based models.

---

## 10. Hawkes Process for Self-Exciting Liquidation Cascades

**Source:** [arxiv.org/abs/2312.16190](https://arxiv.org/abs/2312.16190) — "Hawkes-based cryptocurrency forecasting via Limit Order Book data" | [arxiv.org/abs/2212.07306](https://arxiv.org/abs/2212.07306) — "Toxic Liquidation Spirals"  
**What it does:** Hawkes processes model self-exciting events — each liquidation INCREASES the probability of more liquidations. The branching ratio tells you if you're in a cascade (ratio > 0.7) or isolated event. The "Toxic Liquidation Spirals" paper models how cascades propagate through lending platforms — directly applicable to perp liquidations.  
**Our current gap:** `liquidation_cascade.py` mentions Hawkes but doesn't implement one.

**How to integrate TODAY:**
1. The Hawkes intensity: `λ(t) = μ + α * Σ exp(-β(t - t_i))` where α is the excitation strength (how much each liquidation increases future probability) and β is the decay rate.
2. Fit α, β, μ from the last N liquidation events (from your WebSocket liquidation monitor — see #5).
3. The branching ratio `n = α/β`. When n < 1: subcritical (cascade dies). When n ≈ 1: critical (cascade sustains). This is your cascade probability signal.
4. When n > 0.7 AND liquidations are one-sided (long-only or short-only), enter in the opposite direction after 3+ liquidations within 60 seconds.
5. The Hawkes paper on LOB data shows how to estimate parameters in real-time — MLE is tractable.

**Effort:** 5-8 hours  
**Key concept:** Branching ratio n = α/β. This single number tells you whether a cascade is accelerating (n → 1) or dying (n → 0). When n crosses 0.7 from below, you're entering a cascade. That's your entry trigger.

---

## Bonus: Hyperliquid-Specific Rapid Wins (1-2 hours each)

### A. Hypertracker.io Whale Monitor
**Source:** [hypertracker.io](https://hypertracker.io/)  
Track whale wallet movements on Hyperliquid in real-time. Their API lets you monitor specific wallets or detect large position changes. Feed whale accumulation into your prediction stack.

### B. CoinGlass Hyperliquid Liquidation Map
**Source:** [coinglass.com/hyperliquid-liquidation-map](https://www.coinglass.com/hyperliquid-liquidation-map)  
Real-time visualization of liquidation clusters at price levels. Use this to validate your cascade predictions — if CoinGlass shows heavy liquidation clusters at a level, your model should be predicting a bounce there.

### C. Loris Tools Hyperliquid Funding Rates
**Source:** [loris.tools/funding/exchange/hyperliquid](https://loris.tools/funding/exchange/hyperliquid)  
Normalized 8h funding rates for all Hyperliquid markets. Use as a cross-reference for your OU model calibration.

---

## Priority Implementation Order (Smallest → Highest Effort)

| # | Technique | Hours | Impact | Prerequisite |
|---|-----------|-------|--------|--------------|
| 1 | Hurst exponent regime filter | 2-3 | HIGH | None |
| 2 | Real-time CVD from trade delta | 3-4 | HIGH | WS trades channel |
| 3 | Queue imbalance (L1 OFI) | 3-5 | VERY HIGH | WS l2Book channel |
| 4 | Liquidation WS monitor | 4-6 | HIGH | WS + gRPC |
| 5 | OU funding rate model | 4-6 | MEDIUM | Historical funding |
| 6 | Volume profile (POC/VAH/VAL) | 4-6 | MEDIUM | 1h candles |
| 7 | VPIN order flow toxicity | 5-8 | HIGH | WS trades channel |
| 8 | Hawkes cascade model | 5-8 | HIGH | #4 liquidation data |
| 9 | DeepLOB CNN+LSTM | 6-10 | VERY HIGH | WS l2Book + GPU |
| 10 | PatchTST / N-BEATS | 8-12 | VERY HIGH | GPU |

**Total: ~50-70 hours to implement all 10. Start with #1-4 (14-18 hours) for maximum immediate impact.**
