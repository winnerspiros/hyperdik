# Redundant Module Merge Plan — hyperliquid-trader

Generated: 2026-08-09  
Scope: predictors (6 files), liquidation (5 files), peak detectors (3 files)

---

## 1. PREDICTORS — 8 files → 3 files (delete 2 dead, absorb 4 into 1, keep 1 library)

### Files Analyzed

| # | File | Lines | Daemon Import? | Role |
|---|------|-------|----------------|------|
| 1 | `continuous_predictor.py` | 1150 | ✓ L161 — 4 functions imported | **PRIMARY** — 15-layer weighted ensemble, 6 horizons, adaptive learning |
| 2 | `unified_predictor.py` | 485 | ✓ L141 — 3 functions imported | ACTIVE secondary — ML+exhaustion+structure+flow+macro, regime-adaptive |
| 3 | `perfect_predictor.py` | 421 | ✓ L157 — 3 functions, called at L2529 | ACTIVE tertiary — 3 irreducible signals (funding, OB, multi-TF) |
| 4 | `ml_predictor.py` | 511 | ✗ (imported BY unified_predictor.py L59) | LIBRARY — XGBoost DirectionPredictor + LSTM LSTMPredictor |
| 5 | `kronos_predictor.py` | 240 | ✓ L3688 (lazy local import) | OPTIONAL fallback — Kronos-mini transformer trajectory prediction |
| 6 | `cryptogat_predictor.py` | 315 | ✓ L3497 (lazy local import) | OPTIONAL fallback — Graph Attention Network cross-coin prediction |
| 7 | `predictor.py` | 590 | ✗ (zero imports anywhere) | **DEAD** — Pionex/CoinGecko code, hardcoded `os.chdir("/home/ubuntu/revolut-x-trader")` |
| 8 | `market_predictor.py` | 731 | ✗ (shell-only: `run_predictor.sh`) | SHELL-ONLY — LLM strategic forecaster, writes strategy_calendar.json |

### What Each Predicts & Overlaps

| File | What It Predicts | Inputs | Key Output | Overlap With |
|------|-----------------|--------|------------|-------------|
| `continuous_predictor` | Probability of up/down/flat at 6 horizons (1m–1d) | 4 candle TFs, mid, funding, OI, OB, exhaustion signal, taker ratio, cross-exchange, imbalance trend, Hurst, higher TF align | `ContinuousPrediction` with 15 layer scores, horizon drivers | Contains layers that **subsume all others** |
| `unified_predictor` | Single direction + target + confidence | candles, mid, funding, fear_greed, regime | `PredictionResult` with 5 sub-scores | ML layer → continuous.ml_ensemble; exhaustion → continuous.exhaustion; structure/flow → continuous.multi_tf_technical/bollinger/sr_levels |
| `perfect_predictor` | Direction via 3 signals (funding, OB, multi-TF) | 4 candle TFs, mid, coin | `PerfectPrediction` with per-horizon probs | All 3 signals exist as continuous layers: funding_oi, order_book, multi_tf_technical |
| `ml_predictor` | Next-candle direction via XGBoost+LSTM | OHLCV candles | direction + confidence | Used AS IS by unified_predictor; continuous's ml_ensemble layer calls unified_predictor.compute_ml_prediction() which wraps ml_predictor |
| `kronos_predictor` | 12-step OHLCV trajectory (1h) | 150+ candles, coin | `KronosSignal` with target/stop/dd | Redundant with ml_predictor; both are ML-based directional signals |
| `cryptogat_predictor` | Cross-coin directional from correlation graph | candles_dict (multi-coin) | direction + confidence + neighbor_support | Redundant with ml_predictor; both are ML-based directional signals |

### Call Chain in Daemon

```
daemon evaluation loop:
  ├─ unified_predictor.predict_unified()    → ml_sig (entry signal)
  ├─ continuous_predictor.predict_continuous() → cpred (main prediction)
  │    └─ feeds: unified_predictor.compute_ml_prediction() → ml_ensemble layer
  │    └─ feeds: peak_exhaustion_detector.detect_peak_exhaustion() → exhaustion layer
  ├─ perfect_predictor.predict_perfect()     → perfect (supplementary)
  └─ lazy-only: cryptogat_predictor / kronos_predictor (batch prediction fallback)
```

**unified_predictor's `compute_ml_prediction()` is used BY continuous_predictor** (line 59 of unified → imported at continuous line ???). Check: continuous's ml_ensemble layer actually calls unified's API. So unified is both a standalone predictor AND a library for continuous.

### Merge Plan: PREDICTORS

**GOAL: 1 prediction module (`continuous_predictor.py`) that does everything the daemon needs.**

#### Keep (3 files):

| File | Action | Rationale |
|------|--------|-----------|
| `continuous_predictor.py` | **KEEP & EXPAND** — absorb all active predictors | Already the PRIMARY with 15 layers. Extend with a `predict()` method that consolidates unified + perfect + optional kronos/cryptogat |
| `ml_predictor.py` | **KEEP** as library | Pure ML implementation. Used by autoretrain. No daemon logic — just model classes. |
| `market_predictor.py` | **KEEP** standalone | Not imported by daemon. Runs in shell scripts for strategic LLM forecasting. Unrelated scope. |

#### Absorb into `continuous_predictor.py`:

| Source | What Moves | How |
|--------|-----------|-----|
| `unified_predictor.py` | `compute_ml_prediction()`, `compute_structure_score()`, `compute_flow_score()`, regime-adaptive weighting, `PredictionResult` | These ARE continuous layers already (ml_ensemble, multi_tf_technical, bollinger, sr_levels). Move the `consensus()` logic into a `predict_fast()` entry point on continuous_predictor. |
| `perfect_predictor.py` | `_get_funding_signal()`, `_get_order_book_signal()`, `_compute_multi_tf_direction()` | These ARE continuous layers (funding_oi, order_book, multi_tf_technical). Perfect's entire logic is a subset. Move its `predict_perfect()` as a `predict_simple()` mode. |
| `kronos_predictor.py` | Lazy model loading, batch prediction, `KronosSignal` | Make `_layer_kronos()` in continuous_predictor that wraps this. Lazy-load Kronos-mini. Zero-import if not installed. |
| `cryptogat_predictor.py` | Graph building, GAT aggregation, `CryptoGATPredictor` | Make `_layer_cryptogat()` in continuous_predictor. Lazy-load with optional PyTorch. |

#### Delete:

| File | Reason |
|------|--------|
| `predictor.py` | Dead code — references wrong project (`revolut-x-trader`), hardcodes `os.chdir()`, zero imports by any other file |
| `unified_predictor.py` | After absorption → delete |
| `perfect_predictor.py` | After absorption → delete |
| `kronos_predictor.py` | After absorption → delete |
| `cryptogat_predictor.py` | After absorption → delete |

#### Resulting `continuous_predictor.py` structure:

```python
# continuous_predictor.py — Unified Prediction Engine

# PUBLIC API:
def predict(coin, mid, candles_1m, candles_5m, candles_15m, candles_1h, candles_4h, 
            candles_1d=None, funding_rate=0, oi_delta=0, exhaustion_signal=None,
            taker_ratio=None, cross_exchange_divergence=None, imbalance_trend=None,
            mode="full")  # "full" | "fast" | "simple"
    """
    mode='full':   15-layer weighted ensemble, 6 horizons (current behavior)
    mode='fast':   Unified 5-signal consensus (absorbed from unified_predictor)
    mode='simple': 3-signal approach (absorbed from perfect_predictor)
    """
    → returns ContinuousPrediction

def predict_continuous(...):  # ALIAS to predict(mode='full') — backward compat
def predict_unified(...):     # ALIAS to predict(mode='fast') — backward compat
def predict_perfect(...):     # ALIAS to predict(mode='simple') — backward compat

# INTERNAL LAYERS (15 existing + 2 new):
def _layer_ml_ensemble(...):   # calls ml_predictor (existing)
def _layer_kronos(...):        # NEW: lazy-loads Kronos-mini, returns probability dict
def _layer_cryptogat(...):     # NEW: lazy-loads CryptoGAT, returns probability dict
# ... all other existing layers unchanged
```

**Daemon import changes:**
```python
# OLD (6 imports):
from continuous_predictor import predict_continuous, format_prediction_compact, ...
from unified_predictor import predict_unified, format_prediction_for_ai, PredictionResult
from perfect_predictor import predict_perfect, format_perfect_prediction, PerfectPrediction

# NEW (1 import):
from continuous_predictor import (
    predict, predict_continuous, predict_unified, predict_perfect,  # predict_* aliases
    format_prediction_compact, format_prediction_detailed,
    format_prediction_for_ai, format_perfect_prediction,
    ContinuousPrediction, HorizonPrediction, PredictionResult, PerfectPrediction,
    update_layer_accuracy,
)
```

---

## 2. LIQUIDATION — 5 files → 2 files (absorb 2, delete 1)

### Files Analyzed

| # | File | Lines | Daemon Import? | Role |
|---|------|-------|----------------|------|
| 1 | `liquidation_zones.py` | 380 | ✓ L170 — 4 functions | **PRIMARY WRAPPER** — AI context, safe stops, cascade risk via heatmap |
| 2 | `liquidation_heatmap.py` | 436 | ✗ (imported ONLY by zones) | CORE ENGINE — OB depth clustering, LiquidationHeatmap, LiquidationLevel |
| 3 | `liquidation_monitor.py` | 397 | ✓ context_enricher L247 | REAL DATA — reads actual fills from WebSocket userFills stream |
| 4 | `liquidation_cascade.py` | 365 | ✓ context_enricher L230, strategy L1248 | OI PROXY — cascade detection from OI delta when no real fills |
| 5 | `liquidation_data.py` | 131 | ✗ (imported ONLY by market_predictor) | **DEAD** — CoinGecko volatility estimation for Revolut project |

### Overlaps & Relationships

```
liquidation_zones.py  ──imports──▶  liquidation_heatmap.py  (zones IS a wrapper)
liquidation_monitor.py              (independent — reads WS fills)
liquidation_cascade.py              (independent — reads OI delta)
liquidation_data.py                 (dead — wrong project)

context_enricher uses ALL of: zones + monitor + cascade (3 separate calls)
daemon uses ONLY: zones (get_liquidation_context, estimate_safe_stop, detect_cascade_risk)
```

| Overlap | Detail |
|---------|--------|
| **zones ↔ heatmap** | 90% overlap. zones imports and re-exports heatmap. heatmap is NEVER imported directly by anything except zones. This is a classic "thin wrapper" pattern — merge them. |
| **monitor ↔ cascade** | Same goal: detect cascade risk. monitor uses REAL liquidation fills from WS. cascade uses OI-delta heuristic. monitor is superior when WS is connected. cascade is the fallback. |
| **zones ↔ monitor** | Both provide `get_liquidation_context()` for AI enrichment. zones uses OB depth clusters. monitor uses real fill data. Different data, same output format. |

### Merge Plan: LIQUIDATION

**GOAL: 2 files — real data (monitor) with cascade fallback, and zones (with integrated heatmap).**

#### Keep (2 files):

| File | Action |
|------|--------|
| `liquidation_monitor.py` | **KEEP & ABSORB** `liquidation_cascade.py` as fallback |
| `liquidation_zones.py` | **KEEP & ABSORB** `liquidation_heatmap.py` directly |

#### Changes:

**`liquidation_zones.py`** (merged with heatmap):
- Move ALL of `liquidation_heatmap.py`'s classes and functions directly into zones
- `LiquidationHeatmap`, `LiquidationLevel`, `_cluster_levels()`, `_compute_cascade_risk()`, `_compute_depth_score()`, `fetch_heatmap()` → all become internal to zones
- Public API stays identical: `get_liquidation_context()`, `estimate_safe_stop()`, `detect_cascade_risk()`, `get_all_liquidation_context()`
- Delete `liquidation_heatmap.py`

**`liquidation_monitor.py`** (with cascade fallback):
- Move `detect_cascade_from_oi_delta()` and `CascadeSignal` from `liquidation_cascade.py` into monitor
- Add a `_get_cascade_fallback(coin, oi_current, oi_previous, price_change_pct)` that runs when WS fills are unavailable
- Public API: `get_liquidation_activity()`, `get_liquidation_context()`, `get_self_liquidations()`, `has_self_liquidation()`
- Delete `liquidation_cascade.py`

**`context_enricher.py`** update:
```python
# OLD: 3 imports
from liquidation_zones import get_liquidation_context
from liquidation_cascade import detect_cascade_from_candles  
from liquidation_monitor import get_liquidation_context, get_self_liquidation_summary

# NEW: 2 imports
from liquidation_zones import get_liquidation_context  # OB depth clusters
from liquidation_monitor import (
    get_liquidation_context,      # real WS fills
    get_self_liquidation_summary,
    detect_cascade_from_candles,  # now in monitor as fallback
)
```

#### Delete:

| File | Reason |
|------|--------|
| `liquidation_heatmap.py` | Merged into `liquidation_zones.py` |
| `liquidation_cascade.py` | Merged into `liquidation_monitor.py` |
| `liquidation_data.py` | Dead code — CoinGecko/Revolut project, not Hyperliquid. Imported only by `market_predictor.py`. |

---

## 3. PEAK DETECTORS — 3 files → 2 files (merge 1 into 1, keep 1)

### Files Analyzed

| # | File | Lines | Daemon Import? | Role |
|---|------|-------|----------------|------|
| 1 | `peak_exhaustion_detector.py` | 451 | ✓ L136 — 3 functions | **PRIMARY** — candle-level RSI divergence, volume climax, momentum, BB, VWAP, price targets |
| 2 | `tick_peak_detector.py` | 455 | ✓ L145 — 2 functions | SECONDARY — order book thinning, tick flow, CVD divergence, VWAP extension, runs every 10s |
| 3 | `pump_detector.py` | 166 | ✗ (context_enricher L152) | TERTIARY — pump & dump detection for AI risk warnings |

### What Each Detects & Overlaps

| File | Detection Method | Timeframe | Output | Overlap |
|------|-----------------|-----------|--------|---------|
| `peak_exhaustion` | RSI divergence (candle-level), volume climax, momentum decay, BB position, VWAP deviation | Candle-level (1-4h) | `ExhaustionSignal` with price targets, confidence-gated actions | Same goal as tick_peak: detect peaks/bottoms for exits |
| `tick_peak` | Order book thinning, tick flow imbalance, CVD divergence, VWAP extension, multi-TF RSI alignment | Tick-level (10s) | `TickPeakSignal` with profit-aware thresholds | Same goal as peak_exhaustion, different data sources and timeframe |
| `pump_detector` | Volume spike > 2.5x, sharp price move > 5%, reversal detection | Candle-level | Pump confidence dict, wash trading flags | **Different purpose** — risk warning, not exit signal |

### Key Difference: peak_exhaustion vs tick_peak

| Aspect | `peak_exhaustion_detector` | `tick_peak_detector` |
|--------|---------------------------|---------------------|
| **Data source** | Historical candles (close, high, low, volume) | Live WebSocket (order book, tick trades) |
| **Key signal** | RSI divergence (looks back 20 candles) | Order book thinning (real-time bid/ask ratio) |
| **Latency** | Minutes (waits for candle close + divergence confirmation) | Seconds (reacts to order book changes immediately) |
| **Use in daemon** | Fed into `continuous_predictor` exhaustion layer | Called every 10s in position monitor for exit decisions |
| **Strength** | Confirmed signals, less noise, price targets | Speed, catches reversals before they show in candles |
| **Weakness** | Late — signal confirms after the move started | Noisy — false positives from temporary order book changes |

These are **complementary, not redundant**. They should merge because:
1. Same output concept (peak/bottom signal with score + action)
2. Both used together in daemon (tick_peak for rapid exits, peak_exhaustion for confirmed reversals)
3. Combining them enables: tick_peak catches the move early, peak_exhaustion confirms it

### Merge Plan: PEAK DETECTORS

**GOAL: 2 files — `peak_exhaustion_detector.py` with integrated tick detection, `pump_detector.py` kept separate.**

#### Keep & merge:

| File | Action |
|------|--------|
| `peak_exhaustion_detector.py` | **KEEP & ABSORB** `tick_peak_detector.py` |
| `pump_detector.py` | **KEEP AS-IS** — different purpose (scam detection / AI risk context) |

#### Changes to `peak_exhaustion_detector.py`:

```python
# peak_exhaustion_detector.py — Peak & Bottom Exhaustion + Tick Detection

# PUBLIC API (all existing + new):
def detect_peak_exhaustion(coin, candles, current_price) -> ExhaustionSignal
    """Candle-level: RSI divergence, volume climax, momentum, BB, VWAP"""

def detect_tick_peak(coin, current_price, candles_15m=None, candles_1m=None,
                     side="LONG", pnl_pct=0.0) -> TickPeakSignal  # MOVED FROM tick_peak_detector
    """Tick-level: OB thinning, tick flow, CVD, VWAP extension"""

def predict_price_target(signal, candles, current_price) -> ExhaustionSignal

def format_exhaustion_for_ai(signal) -> str

def format_tick_peak(signal) -> str  # MOVED FROM tick_peak_detector

# INTERNAL (shared helpers):
_rsi()           # used by BOTH detectors
_bollinger()     # used by BOTH
_vwap()          # used by BOTH
_ema()           # used by BOTH

# INTERNAL (tick-specific — moved from tick_peak_detector):
_get_ws_data()
_compute_order_book_signal()
_compute_tick_flow_signal()
_compute_cvd_divergence()
_compute_vwap_extension()
```

**Daemon import changes:**
```python
# OLD:
from peak_exhaustion_detector import (
    detect_peak_exhaustion, predict_price_target, format_exhaustion_for_ai, ExhaustionSignal,
)
from tick_peak_detector import (
    detect_tick_peak, format_tick_peak, TickPeakSignal,
)

# NEW:
from peak_exhaustion_detector import (
    detect_peak_exhaustion, detect_tick_peak,
    predict_price_target, format_exhaustion_for_ai, format_tick_peak,
    ExhaustionSignal, TickPeakSignal,
)
```

#### Delete:

| File | Reason |
|------|--------|
| `tick_peak_detector.py` | Merged into `peak_exhaustion_detector.py` |

---

## 4. SUMMARY: Before → After

| Category | Before | After | Files Deleted | Files Absorbed |
|----------|--------|-------|---------------|----------------|
| **PREDICTORS** | 8 files | 3 files | 2 dead + 4 absorbed | `unified_predictor.py`, `perfect_predictor.py`, `kronos_predictor.py`, `cryptogat_predictor.py` → `continuous_predictor.py` |
| **LIQUIDATION** | 5 files | 2 files | 1 dead + 2 absorbed | `liquidation_heatmap.py` → `liquidation_zones.py`; `liquidation_cascade.py` → `liquidation_monitor.py` |
| **PEAK** | 3 files | 2 files | 1 absorbed | `tick_peak_detector.py` → `peak_exhaustion_detector.py` |
| **TOTAL** | 16 files | 7 files | 3 dead + 7 absorbed | |

### Dead Files to Delete Immediately (no absorption needed)

| File | Reason |
|------|--------|
| `predictor.py` | Pionex/CoinGecko code, hardcodes `os.chdir("/home/ubuntu/revolut-x-trader")`, zero imports anywhere |
| `liquidation_data.py` | CoinGecko-based volatility estimation for Revolut project, only imported by `market_predictor.py` |

### Files That Change Imports

| File | Old Imports | New Imports |
|------|-------------|-------------|
| `hyperliquid_daemon.py` | 3 predictor imports + 2 peak imports | 1 predictor import + 1 peak import |
| `context_enricher.py` | 3 liquidation imports + 1 pump import | 2 liquidation imports + 1 pump import |
| `hyperliquid_strategy.py` | `from liquidation_cascade import` | `from liquidation_monitor import` |
| `auto_retrain_ml.py` | `from ml_predictor import` | unchanged (ml_predictor kept) |
| `day_trader.py` | `from liquidation_data import` + `from pump_detector import` | `from pump_detector import` only |
| `unified_direction.py` | References exhaustion indirectly | import from `peak_exhaustion_detector` |

### Execution Order (recommended)

1. **DELETE dead code**: `predictor.py`, `liquidation_data.py`
2. **MERGE heatmap → zones**: absorb `liquidation_heatmap.py` into `liquidation_zones.py`, delete heatmap
3. **MERGE cascade → monitor**: absorb `liquidation_cascade.py` into `liquidation_monitor.py`, delete cascade
4. **MERGE tick_peak → peak_exhaustion**: absorb `tick_peak_detector.py` into `peak_exhaustion_detector.py`, delete tick_peak
5. **MERGE predictors → continuous**: absorb `unified_predictor.py`, `perfect_predictor.py`, `kronos_predictor.py`, `cryptogat_predictor.py` into `continuous_predictor.py`
6. **UPDATE imports** in `hyperliquid_daemon.py`, `context_enricher.py`, `hyperliquid_strategy.py`, `day_trader.py`
7. **TEST**: run daemon dry-run, verify no import errors
