# Crypto Tops & Bottoms Prediction — Comprehensive Research

> Compiled 2026-07-16 for Hyperliquid perpetual futures trading bot.
> Focus: Predicting tops/bottoms at EVERY moment across ALL timeframes.
> Current weakness: Stack only fires at extremes, misses slow moves and ranging markets.

---

## EXECUTIVE SUMMARY: TOP 10 RECOMMENDATIONS

### 1. flowsurface (Open-Source Order Flow Platform — Hyperliquid Native)
- **URL**: https://github.com/flowsurface-rs/flowsurface
- **Stars**: 1,624 · Rust · MIT
- **What it does**: Native desktop charting with Heatmap (Historical DOM), Footprint charts, CVD, Volume Profile, Depth of Market, Time & Sales. **Supports Hyperliquid natively** (already built-in exchange connector). Real-time L2 orderbook + trade stream analysis.
- **Why it improves our stack**: Our CVD computation (`volume_delta.py`) and `cvd_divergence.py` rely on OHLCV-only estimation (close-vs-open proxy). flowsurface gets REAL tick-level delta from trade data. The Hyperliquid WebSocket connector is already battle-tested in Rust — we can port the delta/CVD/absorption logic to Python.
- **Effort**: **MEDIUM**. Port core Rust order-flow logic into Python, use our existing `hyperliquid_ws.py` WebSocket feed. The Hyperliquid connector code is in `flowsurface-rs/flowsurface` — look at the exchange adapter.
- **Key files to examine**: 
  - Exchange adapter for Hyperliquid (Rust source)
  - Heatmap/DOM chart logic (real tick delta, not estimated)
  - Footprint chart imbalance calculation
- **Integration path**: Extract tick-aggregation logic → feed into `tick_peak_detector.py` and `cvd_divergence.py` as `use_real_delta=True` mode. Replace the OHLCV proxy in `volume_delta.py`.

### 2. Crypto Microstructure Alpha Lab (LOB Prediction Framework)
- **URL**: https://github.com/nimamot/Crypto-Microstructure-Alpha-Lab
- **Stars**: Low but high-quality research repo · Python
- **What it does**: 13 engineered microstructure features from Binance LOB data. Minute-level granularity short-horizon price prediction. Covers feature engineering, label engineering, model development specifically for crypto microstructure.
- **Why it improves our stack**: Our `order_book_imbalance.py` only computes simple bid/ask volume ratio. This repo provides PROVEN microstructure features:
  - Multi-level order flow imbalance (OFI) across L2 depth levels
  - Order book slope and curvature (shape features)
  - Trade-flow features (aggressor side classification)
  - Cross-sectional features (correlated coins)
- **Effort**: **MEDIUM-LOW**. Pure Python, read feature code and port 5-6 key microstructure features into `order_book_imbalance.py` and `tick_peak_detector.py`.
- **Key files**: Feature engineering code, label engineering for rolling forward returns
- **Integration path**: Add `microstructure_features.py` that extracts L2 features from Hyperliquid WebSocket orderbook data. Feed into unified predictor's `compute_flow_score()`.

### 3. LiT — Limit Order Book Transformer (State-of-the-Art LOB Model)
- **URL**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- **Paper**: Published Oct 2025 in Frontiers in AI
- **GitHub**: https://github.com/nicolas-steinmann/LiT-LimitOrderBookTransformer (limited README)
- **What it does**: Novel transformer architecture specifically designed for LOB mid-price movement prediction. Outperforms DeepLOB, CNN-LSTM, and traditional ML on crypto LOB data. Uses attention mechanisms that capture complex temporal-spatial dependencies in the order book.
- **Why it improves our stack**: Our LSTM is a simple numpy forward pass. LiT provides an architecture proven on crypto LOB data with state-of-the-art accuracy. Even a simplified implementation (attention over L2 depth levels + time) would be a major upgrade.
- **Effort**: **HIGH**. Requires PyTorch, training pipeline, and significant GPU/CPU training. But the architecture is published — we can implement a lightweight version.
- **Key files**: Paper architecture diagrams, attention mechanism over L2 levels
- **Integration path**: Long-term — replace or augment LSTM in `ml_predictor.py` with LOB-aware transformer when we have sustained profitability.

### 4. RAVEN: Regime-Aware Variable-Context Expert Network
- **URL**: https://arxiv.org/abs/2606.24062
- **Paper**: June 2026, fresh from arXiv
- **What it does**: Directly addresses THE core problem you described: "model fires at extremes, misses slow moves and ranging markets." RAVEN dynamically adjusts its context window based on the current market regime. Uses a mixture-of-experts architecture where each expert specializes in a different regime.
- **Why it improves our stack**: Our `unified_predictor.py` uses fixed weights (ML 30%, Exhaustion 25%, Structure 20%, Flow 15%, Macro 10%) regardless of market regime. In ranging markets, structure/flow should dominate. In trending markets, ML/momentum should dominate. RAVEN is the theoretical foundation for regime-adaptive weighting.
- **Effort**: **MEDIUM**. The concept is immediately applicable — we can implement regime-dependent weight adjustment in `unified_predictor.py` without the full neural architecture.
- **Key files**: Paper Section 3 (regime detection), Section 4 (variable-context mechanism)
- **Integration path**: Add `regime_adaptive_weights()` to `unified_predictor.py` that shifts weights based on market regime (already tracked in `market_regime.py`).

### 5. "Better Inputs Matter More Than Stacking Another Hidden Layer" (Paper)
- **URL**: https://arxiv.org/abs/2506.05764
- **Paper**: June 2025, very relevant
- **What it does**: **Critical finding**: LOB feature engineering matters MORE than model architecture for crypto price prediction. Better pre-processing of order book data → more improvement than adding layers. Key insight: stationary LOB features outperform raw LOB data.
- **Why it improves our stack**: Directly validates that we should invest in better microstructure features (#2 above) rather than chasing complex model architectures. Our current setup is directionally correct (XGBoost + simple features).
- **Effort**: **LOW**. The paper provides a feature engineering roadmap — implement the stationary transforms they describe.
- **Key findings to implement**:
  - Log-return normalization of order book levels
  - Multi-level OFI (not just best bid/ask)
  - Time-weighted features (older LOB states decay)
  - Cross-asset features (BTC correlation as input)
- **Integration path**: Enhance `order_book_imbalance.py` with paper's stationary LOB features.

### 6. Liquidation Cascade Prediction (Multiple Repos)
- **URLs**:
  - https://github.com/leionion/liquidation-cluster-signal-scraper (20★) — Detects short/long squeeze events from liquidation heatmaps + OI acceleration
  - https://github.com/aoki-h-jp/py-liquidation-map (133★) — Visualize liquidation maps from Binance/Bybit execution data, real-time WebSocket
  - https://github.com/kukapay/crypto-liquidations-mcp (9★) — MCP server streaming real-time liquidations for AI agents
- **What they do**: Track liquidation clusters as magnetic price levels. When price approaches high-density liquidation zones → high probability of rapid move into those levels.
- **Why it improves our stack**: We have `liquidation_cascade.py` but it's basic. These repos provide real-time liquidation heatmap construction. Known fact: liquidation clusters are the STRONGEST support/resistance levels in crypto because they force mechanical price action.
- **Effort**: **LOW**. The `py-liquidation-map` has clean Python code with WebSocket streaming. We can integrate liquidation cluster levels as inputs to `tick_peak_detector.py` and `peak_exhaustion_detector.py`.
- **Key files**: WebSocket liquidation streaming, heatmap construction, cluster density thresholds
- **Integration path**: Add `liquidation_clusters.py` that tracks real-time liquidation levels → feed into peak detector as "liquidation magnet" signals.

### 7. CryptoFlowEngine (CVD/Delta Divergence Detection)
- **URL**: https://github.com/DevKaranJ/CryptoFlowEngine
- **Stars**: New repo · Python
- **What it does**: Pure Python quantitative crypto trading engine with CVD divergence, stacked imbalance detection, absorption zones, and initiation candle detection. Clean separation of signal computation.
- **Why it improves our stack**: Our `cvd_divergence.py` detects only 4 divergence types (bearish, bullish, hidden_bullish, hidden_bearish) from OHLCV proxy. This adds:
  - **Absorption detection**: Large ask wall gets eaten → bullish absorption (sellers exhausted)
  - **Stacked imbalance**: Multiple consecutive footprint levels with same-direction imbalance
  - **Initiation candles**: First strong candle after absorption zone
- **Effort**: **LOW**. Pure Python, code is readable. Add absorption/stacked-imbalance detection to `cvd_divergence.py`.
- **Key files**: CVD divergence computation, absorption zone detection

### 8. Time Series Library (TSLib) + DLinear — Modern Time Series Models
- **URLs**:
  - https://github.com/thuml/Time-Series-Library (12,615★) — 20+ SOTA time series models in one framework
  - https://github.com/vivva/DLinear — AAAI 2023 paper proving simple linear models beat transformers for time series
- **What it does**: TSLib provides implementations of TimesNet, iTransformer, PatchTST, DLinear, FEDformer, Autoformer, etc. DLinear specifically showed that a simple decomposition + linear layers outperforms complex transformers on long-term forecasting.
- **Why it improves our stack**: 
  - DLinear is trivially implementable (decomposition + 2 linear layers) and outperforms complex models
  - TimesNet transforms 1D time series into 2D tensors to capture intraperiod and interperiod variations — directly relevant to multi-timeframe prediction
  - The crypto-specific paper (CryptoGAT, https://arxiv.org/abs/2606.27670) found that standard time series models fail on crypto — we need crypto-specific architectures
- **Effort**: **MEDIUM-HIGH** for full integration. **LOW** for DLinear proof-of-concept (100 lines of PyTorch).
- **Key files**: `models/DLinear.py`, `models/TimesNet.py` in TSLib
- **Integration path**: Test DLinear against our current XGBoost+LSTM on historical data. If it wins, replace LSTM component in `ml_predictor.py`.

### 9. Time Series Foundation Models (TimesFM / Chronos / Moirai)
- **URLs**:
  - Google TimesFM 2.5: https://github.com/google-research/timesfm
  - Amazon Chronos-2: https://github.com/amazon-science/chronos-forecasting
  - Salesforce Moirai: https://github.com/SalesforceAIResearch/moirai
- **What they do**: Pre-trained on billions of time series points. Zero-shot forecasting without training. TimesFM 2.5 (2026) is the latest.
- **Why it improves our stack**: A pre-trained foundation model could provide a "macro view" of where price is heading without us needing to train our own models. Recent arXiv paper (2606.27100, June 2026) benchmarked all three on financial returns — found they provide useful directional signals even at zero-shot.
- **Caveat from paper**: "Financial return forecasting is a difficult test case...low signal-to-noise ratios, structural breaks, heavy tails." Foundation models help but aren't a silver bullet.
- **Effort**: **LOW** (inference-only). **HIGH** if fine-tuning needed.
- **Integration path**: Add as a "macro opinion" signal to `unified_predictor.py`. Query TimesFM with last 200 candles, get 24h forecast → use as directional bias with low weight (5-10%).
- **Key files**: Inference example scripts in each repo

### 10. FinRL / ElegantRL — Reinforcement Learning for Crypto Trading
- **URL**: https://github.com/AI4Finance-Foundation/FinRL (15,742★) · FinRL-X for production
- **What it does**: Complete RL framework with PPO, SAC, TD3, A2C, DQN agents. Gym environments for crypto trading. Reward function designs (Sharpe-based, differential Sharpe, return-based). Production deployment path via FinRL-X.
- **Why it improves our stack**: Our current decision-making is rule-based weighted ensemble. An RL agent could learn optimal position sizing, dynamic leverage, and when to enter/exit based on ALL our existing signals. The RL agent sees the same features but learns non-linear interaction between them.
- **Key insight**: The 5-layer prediction stack produces excellent FEATURES. An RL agent would learn when to TRUST each layer and how to SIZE positions accordingly — the missing piece.
- **Effort**: **HIGH** (full RL integration). **MEDIUM** for proof-of-concept with our existing features as state space.
- **Integration path**: 
  1. Convert our 5 predictor outputs + market context into a Gym environment state
  2. Train PPO agent on historical Hyperliquid data
  3. The agent outputs: action (long/short/flat) + leverage (continuous 0-1)
  4. Run in paper mode alongside existing system, promote when outperforms
- **Key files**: `finrl/tutorials/`, `finrl/env/env_stocktrading.py`

---

## HONORABLE MENTIONS

### A. "When Does Order Flow Matter? State-Dependent L2 Liquidity-State Transitions" (arXiv: 2607.09230)
- July 2026, extremely fresh. Studies WHEN order flow signals work vs. WHEN they don't — regime-dependent signal reliability. Critical for knowing when to trust our order flow layer.

### B. "The Quarter-Hour Effect" (arXiv: 2607.09426)
- July 2026. Documents periodic volatility bursts at 1/5/15-minute marks in crypto futures. Algorithmic trading activity creates predictable patterns. We can time entries around these known bursts.

### C. "Heads, Not Backbones: Output Heads Dominate Architectures on Fat-Tailed Returns" (arXiv: 2606.30037)
- June 2026. Compared N-BEATS, TimesNet, DLinear, iTransformer heads. Finding: the output head design matters more than backbone for fat-tailed returns (crypto!). We should focus on our ensemble weighting logic more than base models.

### D. "Explainable Patterns in Cryptocurrency Microstructure" (arXiv: 2602.00776)
- Jan 2026. SHAP analysis of crypto LOB features across 12+ coins. Identifies WHICH microstructure features are universally predictive. Use this as a feature selection guide.

### E. Hyperliquid-Specific: Funding Rate Term Structure
- Your `funding_signals.py` and `perfect_predictor.py` already use funding rate. Enhancement: query last 3-5 funding payments from Hyperliquid API, compute slope (accelerating vs decelerating). Accelerating positive funding = FOMO building → peak forming. Already in your RESEARCH_OPTIMIZATION.md as Tier 3 #12.

---

## IMMEDIATE ACTION PLAN (Priority-Ordered)

| Priority | Task | Effort | Files to Modify | Expected Gain |
|----------|------|--------|-----------------|---------------|
| 1 | Add regime-adaptive weights to unified_predictor | 30 min | `unified_predictor.py` | HIGH — fixes "only fires at extremes" |
| 2 | Integrate real liquidation cluster data | 1-2 hr | `liquidation_cascade.py`, `tick_peak_detector.py` | HIGH — liquidation magnets are strongest levels |
| 3 | Port microstructure features from Crypto-Microstructure-Alpha-Lab | 2-3 hr | `order_book_imbalance.py`, new `microstructure_features.py` | MEDIUM-HIGH — better inputs > better models |
| 4 | Add absorption + stacked imbalance to CVD | 1 hr | `cvd_divergence.py` | MEDIUM — catches distribution/accumulation |
| 5 | Add funding rate term structure (acceleration) | 30 min | `funding_signals.py`, `perfect_predictor.py` | MEDIUM — earlier peak/bottom warning |
| 6 | Implement DLinear as ML alternative | 2 hr | `ml_predictor.py` | MEDIUM — AAAI-proven simple model |
| 7 | Add TimesFM zero-shot forecast as macro opinion | 1 hr | `unified_predictor.py` | LOW-MEDIUM — foundation model bias |
| 8 | Extract real tick delta from flowsurface logic | 3-4 hr | `volume_delta.py`, `hyperliquid_ws.py` | HIGH — replaces estimated CVD with real data |

---

## WHAT WE ALREADY HAVE (Don't Reinvent)

Our current stack is actually very strong for the research that's been done:

- ✅ Funding rate crowding signal (`perfect_predictor.py` lines 68-80)
- ✅ CVD divergence detection (`cvd_divergence.py`, 4 types)
- ✅ OI delta tracking (`oi_delta.py`, 4 signal types)
- ✅ Volume profile with POC/VA/HVN/LVN (`volume_profile.py`)
- ✅ Order book imbalance (`order_book_imbalance.py`)
- ✅ Peak exhaustion (RSI div, volume climax, momentum decay, BB/VWAP) (`peak_exhaustion_detector.py`)
- ✅ Tick-level peak detection (OB thinning, flow, CVD, VWAP, liquidation magnet) (`tick_peak_detector.py`)
- ✅ Unified weighted ensemble (`unified_predictor.py`)
- ✅ AI validator layer

**The core problem is not missing signals — it's that the ensemble uses FIXED weights regardless of regime, and the CVD/volume data is OHLCV-estimated rather than tick-level.**

---

## KEY ACADEMIC PAPERS — QUICK REFERENCE

| Paper | Date | Key Finding | Link |
|-------|------|-------------|------|
| RAVEN: Regime-Aware Variable-Context Expert Network | Jun 2026 | Dynamic context windows beat fixed ones for finance | arxiv.org/abs/2606.24062 |
| When Does Order Flow Matter? | Jul 2026 | Order flow signal reliability is state-dependent | arxiv.org/abs/2607.09230 |
| Heads, Not Backbones | Jun 2026 | Output head design > backbone for fat-tailed returns | arxiv.org/abs/2606.30037 |
| CryptoGAT | Jun 2026 | Standard time series models fail on crypto | arxiv.org/abs/2606.27670 |
| Time Series Foundation Models for Financial Returns | Jun 2026 | Benchmark of TimesFM/Chronos/Moirai on financial data | arxiv.org/abs/2606.27100 |
| Better Inputs > More Layers | Jun 2025 | LOB feature engineering beats model complexity | arxiv.org/abs/2506.05764 |
| LiT: Limit Order Book Transformer | Oct 2025 | Transformer for LOB mid-price prediction | frontiersin.org/frai.2025.1616485 |
| Microstructure Alpha: Hierarchical Learning | 2026 | Cross-asset transfer learning for crypto micro | frontiersin.org/fbloc.2026.1811716 |
| Explainable Patterns in Crypto Microstructure | Jan 2026 | SHAP analysis of universal crypto LOB features | arxiv.org/abs/2602.00776 |
| The Quarter-Hour Effect | Jul 2026 | Periodic algorithmic bursts in crypto futures | arxiv.org/abs/2607.09426 |
| Red Queen's Trap: Limits of Deep Evolution in HFT | Dec 2025 | DRL+Evolutionary strategies degrade in non-stationary markets | arxiv.org/abs/2512.15732 |

---

## ARCHITECTURE: PROPOSED REGIME-ADAPTIVE WEIGHTING

Current (fixed):
```python
weights = {"ml": 0.30, "exhaustion": 0.25, "structure": 0.20, "flow": 0.15, "macro": 0.10}
```

Proposed (regime-dependent, informed by RAVEN paper):
```python
def get_regime_weights(regime: str) -> dict:
    if regime == "trending_up" or regime == "trending_down":
        # In trends: ML momentum dominates, exhaustion less relevant
        return {"ml": 0.35, "exhaustion": 0.15, "structure": 0.20, "flow": 0.20, "macro": 0.10}
    elif regime == "ranging":
        # In ranges: structure (S/R) and flow (absorption) dominate
        return {"ml": 0.15, "exhaustion": 0.25, "structure": 0.30, "flow": 0.20, "macro": 0.10}
    elif regime == "volatile":
        # In volatility: exhaustion signals are most predictive
        return {"ml": 0.20, "exhaustion": 0.35, "structure": 0.15, "flow": 0.20, "macro": 0.10}
    else:
        return {"ml": 0.30, "exhaustion": 0.25, "structure": 0.20, "flow": 0.15, "macro": 0.10}
```

This alone could significantly improve detection of slow moves and ranging market turns.

---

*Research compiled 2026-07-16. All URLs verified accessible. Papers sourced from arXiv and Frontiers.*
*See also: `RESEARCH_OPTIMIZATION.md`, `research/latest_trading_research_2024_2026.md` for related prior research.*
