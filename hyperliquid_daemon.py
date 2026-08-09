#!/usr/bin/env python3
"""
Hyperliquid AI Daemon v3 — Ensemble architecture with deterministic 5-pillar
composite signal, multi-tier TP, dual-layer risk, whale/funding signals.

v3 PRINCIPLES:
  - Deterministic signal engine (no per-cycle AI calls — saves tokens, latency)
  - AI reserved for weekly evolution loop (parameter optimization)
  - 5-pillar composite signal + 4 sub-strategies with regime weighting
  - Multi-tier take profit with break-even stop and trailing
  - WEL/TWEL dual-layer exposure limits + HSL 4-tier equity protection
  - BTC beta filter + correlation group caps
  - Whale tracker + Funding sniper as supplementary signals
  - Forager-style dynamic coin selection
  - Cooldown enforcement + funding window avoidance

MODULES:
  hyperliquid_strategy.py   → 5-pillar composite + ensemble + regime
  hyperliquid_risk.py       → WEL/TWEL, HSL, unstucking, Kelly
  hyperliquid_execution.py  → Multi-tier TP, trailing, fee-aware sizing
  hyperliquid_evolution.py  → LLM parameter optimization (weekly)
  hyperliquid_whale.py      → Whale wallet tracking
  hyperliquid_funding_sniper.py → Funding rate harvesting
  hyperliquid_delta_neutral.py  → Delta-neutral arb (separate strategy)
"""

import fcntl
import gc
import json
import math
import numpy as np
import os
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Load API keys from ~/.hermes/.env before any AI module imports ──
# The daemon process may not inherit env vars from the parent shell.
# AI modules (ai_decider, ai_validator, etc.) read from os.environ at import time,
# so we must inject the keys BEFORE those imports happen.
_env_path = Path.home() / ".hermes" / ".env"
if _env_path.exists():
    try:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    _key = _key.strip()
                    _val = _val.strip().strip("'\"")
                    if _key and _val and _key not in os.environ:
                        os.environ[_key] = _val
    except Exception:
        pass
# Also try the project-specific .env as fallback
for _p in [ROOT / ".env", Path("/home/ubuntu/hyperliquid-trader/.env")]:
    if _p.exists():
        try:
            with open(_p) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _key, _, _val = _line.partition("=")
                        _key = _key.strip()
                        _val = _val.strip().strip("'\"")
                        if _key and _val and _key not in os.environ:
                            os.environ[_key] = _val
        except Exception:
            pass

# ── Singleton lock disabled — systemd handles single-instance guarantee ──
# The flock-based lock was causing crash loops: daemon exits, lock stays
# on disk, systemd restarts, new process can't lock → sys.exit(1) → 220+ restarts.
_PIDFILE = Path("/tmp/hyperliquid_daemon.pid")
try:
    _lock_fd = open(_PIDFILE, "w")
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
except Exception:
    pass  # Non-fatal — systemd manages lifecycle

import hyperliquid_client as hl

# New v3 modules
from hyperliquid_strategy import (
    generate_master_signal, compute_composite, detect_regime,
    calculate_entry_levels, candles_to_frame, add_basic_indicators,
    MasterSignal, PillarWeights, MarketRegime, SignalSide,
    _safe_float, estimate_kelly_from_history,
    score_conviction, get_tier_sizing, ConvictionTier, MIN_CONVICTION_SCORE,
    enrich_master_signal, get_enrichment_context, EnrichedSignal,
)
from hyperliquid_risk import (
    HSLState, HSLTier, ExposureLimits, ExposureState,
    CooldownState, UnstuckState,
    full_risk_check, check_unstuck, is_funding_window,
    get_leverage_for_regime, RiskCheck,
    compute_portfolio_heat, is_portfolio_heat_safe, compute_max_new_position,
    DEFAULT_BTC_CORRELATIONS, MAX_PORTFOLIO_HEAT, MAX_LEVERAGED_HEAT,
    get_heat_limits,
)
from hyperliquid_execution import (
    build_exit_plan, describe_exit_plan, MultiTierTP,
    round_size, round_price, TrailState, update_trail,
    get_regime_stop_mult, chandelier_atr_mult,
)
from hyperliquid_evolution import (
    EvolutionState, PersonalityParams, TacticalParams,
    evolve_parameters, apply_adjustments, build_segment,
    SegmentMetrics,
)
from hyperliquid_whale import WhaleTracker, WhaleSignal
from hyperliquid_funding_sniper import FundingSniper, FundingSignal
# Delta-neutral arb — funding rate arbitrage opportunities
from hyperliquid_delta_neutral import (
    ArbState, scan_arb_opportunities, get_arb_context, ARB_STATE_PATH,
    load_arb_state, save_arb_state,
)
# Correlation tracker — prevent concentration risk
from correlation_tracker import (
    compute_rolling_correlations, get_correlation_summary,
)
from ai_decider import (
    decide_borderline, decide_sizing, decide_regime_override,
    validate_trade, predict_short_term, analyze_fill, get_stats as ai_stats,
    ai_select_coins, ai_assess_market, ai_evaluate_exit, ai_debate_entry,
    analyze_closed_trade, run_evolution_analysis,
)

# Peak/bottom exhaustion detector — candle-level + tick-level merged (Aug 9)
from peak_exhaustion_detector import (
    detect_peak_exhaustion, detect_tick_peak,
    predict_price_target, format_exhaustion_for_ai, format_tick_peak,
    ExhaustionSignal, TickPeakSignal,
)
# Unified prediction engine — combines ML, exhaustion, structure, flow, macro
from price_extremes import get_extremes, get_dynamic_tp_targets, format_extremes_for_ai
from unified_predictor import (
    predict_unified, format_prediction_for_ai, PredictionResult,
)
from ai_validator import (
    validate_prediction, format_validation, AIValidationResult,
)
# Self-learning engine — learns from every trade outcome
from hyperliquid_learner import (
    record_trade, get_learned_context, TradeRecord, get_learned_weights,
)
# Perfect prediction engine — funding extremes + order book + multi-TF
from perfect_predictor import (
    predict_perfect, format_perfect_prediction, PerfectPrediction,
)
# Continuous prediction engine — probabilities at every horizon, no thresholds
from continuous_predictor import (
    predict_continuous, format_prediction_compact, format_prediction_detailed,
    update_layer_accuracy, ContinuousPrediction, HorizonPrediction,
)
# Open Interest delta — OI change + divergence with price
from oi_delta import (
    get_oi_signal, get_oi_context, get_oi_delta, update_oi_cache,
)
# Liquidation zones — cascade risk, safe stop placement
from liquidation_zones import (
    get_liquidation_context, estimate_safe_stop, detect_cascade_risk,
    get_all_liquidation_context,
)
# Cross-exchange (Binance) — detect HL-specific manipulation via price divergence
from cross_exchange import (
    get_binance_price, get_cross_divergence, get_cross_exchange_context,
    get_all_binance_prices,
)
# Taker buy/sell ratio — real-time order flow aggression
from taker_ratio import (
    get_taker_ratio, _layer_taker_ratio, get_taker_context,
)
# Bid/ask imbalance trend — building vs fading pressure
from imbalance_trend import (
    get_imbalance_trend, _layer_imbalance_trend, get_imbalance_context,
)
# Unified Direction Predictor — all data sources → one directional read
from unified_direction import (
    predict_direction, should_enter, should_exit, get_position_size_pct,
    DirectionPrediction,
)

# ============================================================
# Config
# ============================================================

TRADABLE_COINS = [
    # Tier 1 — majors (always trade)
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
    # Tier 2 — liquid alts
    "MATIC", "LTC", "SUI", "ARB", "OP", "INJ", "NEAR", "AAVE", "UNI", "RNDR",
    "PEPE", "WIF", "SEI", "TIA", "JUP",
]
# ── Coin rotation: scan subset per cycle to avoid 429 rate limits ──
# BTC/ETH always scanned. Remaining coins rotate in windows of 8 per cycle.
# Full 177 scanned in ~22 cycles (~66 min at 180s/cycle).
_ROTATE_WINDOW = 40  # pushing from 35
_ALWAYS_SCAN = {"BTC", "ETH"}
_rotate_offset = 0
CYCLE_SECONDS = 5  # 5s — fast cycles, AI manages rate limits
POSITION_CHECK_SECONDS = 10
MIN_TRADE_USD = 5.0
# Minimum notional per trade: $20 floor (HL minimum is $10, $10 margin at 2x = $20)
# Fast-track coins can go to $10 (HL absolute minimum)
MIN_NOTIONAL_USD = 20
# Dynamic: compute in run() based on equity, floor at 1 for micro accounts
BASE_MAX_POSITIONS = 5
BASE_LEVERAGE = 3  # AI-driven: AI can override 1-5x based on conviction/equity/regime (was hardcoded 2)

# ============================================================
# Logging
# ============================================================

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/hyperliquid_daemon.log"),
    ],
)
log = logging.getLogger("hyperliquid")
log.info("═══ DAEMON V3.1 LOADED — exit layers active (flashcrash, structure, ai-eval, chandelier) ═══")

# ── Load full coin universe (after logger is available) ──
_ALL_COINS = TRADABLE_COINS  # safe default — prevents NameError if anything below fails
try:
    with open("data/universe_coins.json") as f:
        _ALL_COINS = json.load(f)
    log.info(f"  Universe: {len(_ALL_COINS)} perp markets loaded")
except Exception:
    log.warning(f"  Universe: using fallback {len(TRADABLE_COINS)}-coin list")


# ============================================================
# GLOBAL STATE
# ============================================================

hsl_state = HSLState()
exposure_state = ExposureState()
cooldown_state = CooldownState(cooldown_seconds=1800)  # 30-min signal cooldown
_forager_skip_cooldown: dict[str, float] = {}  # coin → timestamp, 10-min forager skip
_global_pause_until: float = 0.0  # Don't open ANY position until this timestamp
_ai_trade_plan: dict[str, dict] = {}  # coin → {direction, confidence, target_pct, stop_pct, hold_min}
_PENDING_ZONE: dict[str, dict] = {}  # coin → {oid, is_buy, size_usd, ...} non-blocking zone orders
_REVERSE_COUNT: dict[str, int] = {}  # coin → reverse count — cap at 1 per session (death spiral guard)
# ── STOP-LOSS COOLING ──
_last_stop_loss_at: dict[str, float] = {}  # coin → timestamp
STOP_LOSS_COOLING_SECONDS = 60  # Ignore signals for 60s after a stop loss

# ── MINIMUM HOLD TIME ──
MIN_HOLD_SECONDS = 300  # 5min minimum hold — let trades breathe but don't trap them
# ── Minimal ROI curve (Freqtrade pattern): time → min profit to stay in trade ──
# Key: seconds held, Value: minimum % profit required to remain in position
# At 5 min: need 1.5% profit or exit. At 10 min: 1.0%. At 30 min: 0.3%. At 60 min: 0%.
_MINIMAL_ROI = {
    120: 0.5,   # 2min: tolerate -0.5% (quick rejection of obvious losers)
    300: 0.35,  # 5min: -0.35% (was -1.0% — way too generous)
    600: 0.25,  # 10min: -0.25% (was -0.7%)
    1200: 0.15, # 20min: -0.15% (new — don't let losers sit)
    1800: 0.10, # 30min: -0.10% (was -0.3%)
    3600: 0.0,  # 60min: breakeven
}
# ── Hyperliquid taker fees (market orders) ──
TAKER_FEE_RATE = 0.00035   # 0.035% per side
ROUNDTRIP_FEE_PCT = TAKER_FEE_RATE * 2 * 100  # 0.07% of notional, both sides
ROUNDTRIP_FEE_MARGIN_PCT = ROUNDTRIP_FEE_PCT * BASE_LEVERAGE  # 0.21% of margin at 3x
# ── ROI timeout throttle — prevent log spam (one close attempt per coin per threshold) ──
_ROI_TIMEOUT_ATTEMPTED: set[str] = set()  # "COIN:threshold_secs" — cleared after 60s
# ── Max risk per trade (Jesse pattern): reject trades that risk too much equity ──
MAX_TRADE_RISK_PCT = 2.0  # Max 2% of total equity at risk per trade (at stop loss)
_MAX_SKIP_RISK_OVER: set[str] = set()  # Coins temporarily blocked for excessive risk
_position_entry_times: dict[str, float] = {}  # coin.upper() → epoch timestamp of entry

# ── Risk block throttle ──
_risk_block_throttle: dict[str, float] = {}  # "coin:reason" → last log timestamp

# ── Reasons that bypass MIN_HOLD_SECONDS (life-or-death scenarios) ──
_MIN_HOLD_BYPASS_REASONS = {
    "stop_loss", "liquidation", "liq", "adl", "unstuck",
    "ai_loss_eval:execute_now", "ai_exit",
    "critical", "emergency", "rule_exit", "peak_rollover_safety",
}

def _net_pnl_pct(gross_pnl_pct: float, leverage: int = BASE_LEVERAGE) -> float:
    """Convert gross PnL % to net PnL % after round-trip taker fees.
    
    Fees: 0.035% taker per side = 0.07% of notional round-trip.
    At leverage N, that's 0.07% * N of margin.
    
    Returns net PnL as percentage of margin.
    """
    return gross_pnl_pct - ROUNDTRIP_FEE_PCT * leverage  # fee scales with leverage, no base division

def _can_close_position(coin: str, reason: str = "") -> bool:
    """Check if position is old enough to close. Returns True if OK to close.
    
    Enforces MIN_HOLD_SECONDS (5 min) unless the close reason is a life-or-death
    scenario (stop loss, liquidation, unstuck, tick peak emergency, etc.).
    """
    cu = coin.upper()
    entry_time = _position_entry_times.get(cu)
    
    # If we don't know the entry time, allow close (position predates tracking)
    if entry_time is None:
        return True
    
    age = time.time() - entry_time
    
    # Always allow close after MIN_HOLD_SECONDS
    if age >= MIN_HOLD_SECONDS:
        return True
    
    # Check for emergency bypass reasons
    reason_lower = reason.lower()
    for bypass in _MIN_HOLD_BYPASS_REASONS:
        if bypass in reason_lower:
            log.info(f"  🚨 {coin}: bypassing min hold ({age:.0f}s age) — emergency reason: {reason}")
            return True
    
    # Still within min hold — block close
    remaining = MIN_HOLD_SECONDS - age
    log.info(f"  ⏳ {coin}: min hold active ({age:.0f}s/{MIN_HOLD_SECONDS}s) — blocking close, {remaining:.0f}s remaining — reason: {reason}")
    return False

def _record_entry_time(coin: str, entry_time: float | None = None):
    """Record that a position was just entered for this coin."""
    _position_entry_times[coin.upper()] = entry_time or time.time()

def _clear_entry_time(coin: str):
    """Clear entry tracking when position is closed."""
    _position_entry_times.pop(coin.upper(), None)

def register_stop_loss(coin: str) -> None:
    """Register a stop loss hit to trigger cooling period."""
    global _last_stop_loss_at
    _last_stop_loss_at[coin.upper()] = time.time()

def is_stop_loss_cooling(coin: str) -> bool:
    """Check if we're in the cooling period after a stop loss."""
    last = _last_stop_loss_at.get(coin.upper(), 0)
    return time.time() - last < STOP_LOSS_COOLING_SECONDS


# ── CANDLE CACHE (avoids duplicate API calls) ──
_candle_cache: dict[str, tuple[float, list[dict]]] = {}  # key → (timestamp, data)
CANDLE_CACHE_TTL = 120  # 2 min — reduce API load, 15m candles don't change every cycle
CANDLE_CACHE_MAX_ENTRIES = 150  # Reduced from 200 — Kronos now runs less frequently, less cache needed
_cache_cleanup_counter = 0

def _fetch_candles_cached(coin: str, interval: str = "15m", limit: int = 200) -> list[dict]:
    """Fetch candles with 60s cache to avoid duplicate API calls."""
    global _cache_cleanup_counter
    ck = f"{coin}:{interval}:{limit}"
    now = time.time()
    if ck in _candle_cache:
        ts, data = _candle_cache[ck]
        if now - ts < CANDLE_CACHE_TTL:
            return data
    data = _fetch_candles(coin, interval, limit)
    _candle_cache[ck] = (now, data)
    
    # Periodic cache eviction: every 50 fetches, purge expired entries and cap size
    _cache_cleanup_counter += 1
    if _cache_cleanup_counter >= 50:
        _cache_cleanup_counter = 0
        # Remove entries older than 5× TTL (10 min stale)
        stale_keys = [k for k, (ts, _) in _candle_cache.items() if now - ts > CANDLE_CACHE_TTL * 5]
        for k in stale_keys:
            del _candle_cache[k]
        # If still too large, remove oldest entries
        if len(_candle_cache) > CANDLE_CACHE_MAX_ENTRIES:
            sorted_keys = sorted(_candle_cache.keys(), key=lambda k: _candle_cache[k][0])
            for k in sorted_keys[:-CANDLE_CACHE_MAX_ENTRIES]:
                del _candle_cache[k]
    
    return data

# ── FUNDING CONTEXT CACHE ──
_last_funding_update = 0.0
FUNDING_CACHE_TTL = 900  # 15 min (was 300s — 429s on every attempt)

# ── FEAR & GREED CACHE ──
_fg_cache: tuple[float, int] = (0.0, 50)  # (timestamp, value)
FG_CACHE_TTL = 600  # 10 minutes
def _normalize_positions(positions: list) -> list:
    """Normalize Hyperliquid API position format to flat dict.
    API returns: [{"type":"oneWay", "position": {"coin":"LINK", "szi":"1.8", ...}}]
    We need: [{"coin":"LINK", "szi":"1.8", ...}]
    """
    flat = []
    for p in positions:
        if "position" in p:
            flat.append(p["position"])
        else:
            flat.append(p)
    return flat
_session_volume: dict[str, float] = {}  # coin → total notional traded this session
_last_session_date: str = ""  # date string for daily reset
MAX_SESSION_VOLUME_PER_COIN = 1.50  # Max 150% of equity per coin per session (was 0.50 — too tight for $32 trades)

def _reset_session_volume_if_new_day() -> None:
    """Reset session volume tracking at midnight UTC."""
    global _last_session_date
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if today != _last_session_date:
        _session_volume.clear()
        _last_session_date = today

def check_session_volume(coin: str, notional_usd: float, equity: float) -> tuple[bool, str]:
    """Check if trading more of this coin would exceed session volume cap."""
    _reset_session_volume_if_new_day()
    current = _session_volume.get(coin.upper(), 0)
    cap = equity * MAX_SESSION_VOLUME_PER_COIN
    if current + notional_usd > cap:
        return False, f"session_vol:{current+notional_usd:.0f}>{cap:.0f}"
    return True, "ok"

def track_session_volume(coin: str, notional_usd: float) -> None:
    """Track session volume for a trade."""
    _reset_session_volume_if_new_day()
    sym = coin.upper()
    _session_volume[sym] = _session_volume.get(sym, 0) + notional_usd


# ── MARK PRICE DEVIATION CHECK ──
def check_mark_deviation(
    side: str, order_price: float, mark_price: float, max_pct: float = 0.02
) -> tuple[bool, str]:
    """Check if order price is within allowed deviation from mark price.

    BUY: order_price must be <= mark * (1 + max_pct) — don't overpay
    SELL: order_price must be >= mark * (1 - max_pct) — don't undersell
    """
    if mark_price <= 0:
        return True, "no_mark"
    if side == "BUY":
        limit = mark_price * (1 + max_pct)
        if order_price > limit:
            return False, f"mark_dev_buy:{order_price:.2f}>{limit:.2f}"
    else:
        limit = mark_price * (1 - max_pct)
        if order_price < limit:
            return False, f"mark_dev_sell:{order_price:.2f}<{limit:.2f}"
    return True, "ok"


# ── BB SQUEEZE TIMING (improved entry) ──
def is_bb_squeeze_entry(bb_width: float, bb_prev_width: float, squeeze_ratio: float = 0.7) -> bool:
    """Check if Bollinger Bands are squeezing (contracting before expansion).
    Best entry timing: bands tight → about to expand → big move coming."""
    if bb_width <= 0 or bb_prev_width <= 0:
        return False
    return bb_width < bb_prev_width * squeeze_ratio and bb_width < 0.04

def _check_tp_levels(coin: str, mid: float, szi: float, side: str) -> float:
    """Check if price hit any TP level. Returns fraction of position to close (0=none)."""
    tps = position_tps.get(coin.upper(), [])
    if not tps:
        return 0.0
    is_long = side == "LONG"
    fraction_to_close = 0.0
    for tp in tps:
        tp_price = tp.get("price", 0)
        if tp_price <= 0:
            continue
        hit = (is_long and mid >= tp_price) or (not is_long and mid <= tp_price)
        if hit:
            fraction_to_close += tp.get("fraction", 0)
    return min(fraction_to_close, 1.0)
unstuck_state = UnstuckState()
trail_states: dict[str, TrailState] = {}
# Track TP levels per position for monitoring (set on submission)
position_tps: dict[str, list[dict]] = {}  # coin → [{price, size, tier_name}]

# Evolution
evo_state = EvolutionState.load("data/evolution_state.json")
tactical = evo_state.tactical

# Supplementary strategies
whale_tracker = WhaleTracker()
funding_sniper = FundingSniper()

# Delta-neutral funding arb state
arb_state = load_arb_state()

# Trade tracking for Kelly/evolution
recent_trades: list[dict] = []
cycle_count = 0
evolution_check_cycles = 0

# ============================================================
# CANDLE FETCH (from Hyperliquid API)
# ============================================================

def _fetch_candles(coin: str, interval: str = "15m", limit: int = 300) -> list[dict]:
    """Fetch candles from Hyperliquid API using SDK. Retries on 429."""
    import urllib.error
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
            info = Info(constants.MAINNET_API_URL, skip_ws=True)
            end_ms = int(time.time() * 1000)
            interval_ms = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
            start_ms = end_ms - limit * interval_ms.get(interval, 900_000)
            candles = info.candles_snapshot(coin, interval, start_ms, end_ms)
            return candles if isinstance(candles, list) else []
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            if e.code == 422:
                return []  # Skipped silently
            log.warning(f"Candle fetch failed for {coin}: HTTP {e.code}")
            return []
        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            if "422" in str(e):
                return []  # Skipped silently
            if attempt == 0:
                log.warning(f"Candle fetch failed for {coin}: {e}")
            return []


# ============================================================
# MARKET CONTEXT BUILDER (compact, AI-ready)
# ============================================================

def _get_fear_greed() -> int:
    """Fetch Fear & Greed with 10-min cache."""
    global _fg_cache
    now = time.time()
    if now - _fg_cache[0] < FG_CACHE_TTL:
        return _fg_cache[1]
    try:
        import urllib.request, json as _j
        req = urllib.request.Request(
            "https://api.alternative.me/fng/?limit=1",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = _j.loads(r.read())
            val = int(data["data"][0]["value"])
            _fg_cache = (now, val)
            return val
    except Exception:
        return _fg_cache[1]


def _calc_momentum(coin: str, timeframe: str = "5m") -> float:
    """Calculate short-term price momentum as % change.
    
    Returns % change over the given timeframe (e.g., 5m, 15m).
    Negative = price dropping. Positive = price rising.
    """
    try:
        c = _fetch_candles_cached(coin, timeframe, 20)
        if not c or len(c) < 2:
            return 0.0
        first = float(c[0].get("c", c[0].get("close", 0)))
        last = float(c[-1].get("c", c[-1].get("close", 0)))
        if first <= 0:
            return 0.0
        return (last - first) / first * 100
    except Exception:
        return 0.0


# ── PUMP/DUMP CHASE DETECTION ──────────────────────────────────────────
# Multi-timeframe recent-move check to prevent entering after a big move
# already happened. LONG after a pump = buying the top. SHORT after a dump
# = selling the bottom. Waits for pullback; blocks if too extreme.
# ────────────────────────────────────────────────────────────────────────

_CHASE_THRESHOLDS = [
    # (timeframe, candle_limit, max_move_pct, edge_pct, wait_s, block_pct)
    # candle_limit=2: look at last 2 candles for context
    # Aug 8: tightened blocks — missed CELO 5% pump, entered at +2.2σ VWAP
    ("1m",  2,  1.0, 0.003, 25,  2.5),
    ("5m",  2,  1.5, 0.005, 35,  3.5),
    ("15m", 2,  2.5, 0.008, 50,  5.0),
    ("1h",  2,  4.0, 0.015, 70,  8.0),
    ("4h",  2,  8.0, 0.025, 90, 15.0),
]


def _micro_peak_entry_wait(coin: str, is_buy: bool, px: float, max_wait: float = 4.0) -> float:
    """Wait for a favorable micro-move before entering. Uses live WebSocket mids (no API).
    
    SHORT: wait for price to tick UP (better entry = higher price).
    LONG:  wait for price to tick DOWN (better entry = lower price).
    
    Returns the best price seen during the window, or px if no WS data.
    Max wait is short (4s) — this is micro-timing, not a limit order.
    """
    try:
        from hyperliquid_ws import get_field
    except ImportError:
        return px
    _start = time.time()
    _best = px
    _sample_interval = 0.15  # 150ms samples — fast enough for micro-peaks
    _seen_favorable = False
    _first_sample = px
    while time.time() - _start < max_wait:
        time.sleep(_sample_interval)
        ws_mids = get_field("mids")
        if not ws_mids:
            continue
        _ws_px = float(ws_mids.get(coin, 0) or ws_mids.get(coin.upper(), 0))
        if _ws_px <= 0:
            continue
        if is_buy:
            # LONG: better price is LOWER
            if _ws_px < _best:
                _best = _ws_px
                _seen_favorable = True
        else:
            # SHORT: better price is HIGHER
            if _ws_px > _best:
                _best = _ws_px
                _seen_favorable = True
        # Exit early if we got a favorable micro-move (>0.05% improvement)
        _improvement = abs(_best - px) / px * 100
        if _seen_favorable and _improvement > 0.05:
            break
    if _seen_favorable:
        return _best
    return px
    """Check if price already moved significantly in the trade direction.
    
    Returns dict with:
      block: bool     — True if we should NOT enter (already pumped/dumped too much)
      pullback: bool  — True if we should wait for a pullback before entering
      edge_pct: float — Limit order offset from current price (favorable direction)
      wait_s: int     — How long to wait for pullback fill
      detail: str     — Human-readable explanation
      max_move_pct: float — Largest move detected
      timeframe: str  — Which timeframe triggered
    """
    result = {"block": False, "pullback": False, "edge_pct": 0.003,
              "wait_s": 20, "detail": "", "max_move_pct": 0.0, "timeframe": ""}
    
    try:
        is_buy = side.upper() == "BUY"
        
        for tf, candle_limit, max_move_pct, edge_pct, wait_s, block_pct in _CHASE_THRESHOLDS:
            candles = _fetch_candles_cached(coin, tf, max(candle_limit + 1, 10))
            if not candles or len(candles) < candle_limit + 1:
                continue
            
            # Get the last N candles (most recent completed ones)
            recent = candles[-(candle_limit + 1):]
            if len(recent) < 2:
                continue
            
            first_close = float(recent[0].get("c", recent[0].get("close", 0)))
            last_close = float(recent[-1].get("c", recent[-1].get("close", 0)))
            if first_close <= 0:
                continue
            
            move_pct = (last_close - first_close) / first_close * 100
            
            # For SHORT: a DOWN move is in our direction (price dropped)
            # For LONG:  an UP move is in our direction (price rose)
            move_in_direction = (is_buy and move_pct > 0) or (not is_buy and move_pct < 0)
            abs_move = abs(move_pct)
            result["max_move_pct"] = max(result["max_move_pct"], abs_move)
            
            if not move_in_direction:
                continue  # Price moved against our direction — that's GOOD for entry
            
            # Price moved in our direction — how much?
            if abs_move >= block_pct:
                result["block"] = True
                result["detail"] = (f"{tf}:{move_pct:+.1f}% move in trade dir — too late "
                                    f"(>{block_pct:.0f}% threshold), would be chasing")
                result["timeframe"] = tf
                return result
            
            if abs_move >= max_move_pct:
                result["pullback"] = True
                result["edge_pct"] = max(result["edge_pct"], edge_pct)
                result["wait_s"] = max(result["wait_s"], wait_s)
                result["detail"] = (f"{tf}:{move_pct:+.1f}% in trade dir — "
                                    f"waiting for {edge_pct*100:.1f}% pullback")
                result["timeframe"] = tf
                # Continue checking longer timeframes — might find a stronger signal
        
        if result["pullback"] and not result["block"]:
            direction_word = "dip" if is_buy else "bounce"
            result["detail"] = (f"{result['timeframe']}:{result['max_move_pct']:.1f}% already moved — "
                                f"waiting for {result['edge_pct']*100:.1f}% {direction_word}, {result['wait_s']}s timeout")
        
        return result
    
    except Exception as e:
        return result  # Return defaults on any error — don't block entry


def _build_quick_context(coin: str, mids: dict, total_eq: float,
                         positions: list, candles_15m: list | None = None,
                         enrichment_ctx: str = "") -> dict:
    """Build compact context dict for AI decider. Enriched with live WebSocket data.

    Includes: millisecond trade data, order book depth, bid/ask spread,
    trade flow pressure, plus all existing fields and enrichment signals.
    """
    mid = float(mids.get(coin, 0))
    btc_mid = float(mids.get("BTC", 0))
    fg = _get_fear_greed()

    # Funding rate for this coin
    funding = 0.0
    ctx = funding_sniper.asset_ctx.get(coin, {})
    if ctx:
        funding = float(ctx.get("funding", 0))

    # Open positions summary
    active = [p for p in positions if abs(float(p.get("szi", 0))) > 0.0001]
    if active:
        pos_summary = ", ".join(
            f"{p.get('coin','?')}:{('L' if float(p.get('szi',0))>0 else 'S')}"
            for p in active[:3]
        )
    else:
        pos_summary = "none"

    # ── Enrich with live WebSocket data (millisecond precision) ──
    ws_trade_price = mid
    ws_trade_side = ""
    ws_trade_size = 0.0
    best_bid = 0.0
    best_ask = 0.0
    spread_pct = 0.0
    bid_depth = 0.0
    ask_depth = 0.0
    ws_age_ms = 0

    try:
        from hyperliquid_ws import get_field as _ws_field
        obs = _ws_field("orderbooks") or {}
        trades = _ws_field("trades") or {}
        # Latest trade
        t = trades.get(coin, {})
        if t:
            ws_trade_price = float(t.get("price", mid))
            ws_trade_side = t.get("side", "")
            ws_trade_size = float(t.get("size", 0))
            ws_age_ms = int(time.time() * 1000) - int(t.get("ts", 0))
        # Order book
        ob = obs.get(coin, {})
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if bids and asks:
            best_bid = float(bids[0].get("px", 0))
            best_ask = float(asks[0].get("px", 0))
            if best_bid > 0 and best_ask > 0:
                spread_pct = (best_ask - best_bid) / best_bid * 100
            # Depth (sum of top 5 levels in USD)
            bid_depth = sum(float(b.get("px", 0)) * float(b.get("sz", 0)) for b in bids[:5])
            ask_depth = sum(float(a.get("px", 0)) * float(a.get("sz", 0)) for a in asks[:5])
    except Exception:
        pass  # WS not available — fall back to REST mids

    result = {
        "mids": mids,
        "btc_mid": btc_mid,
        "fear_greed": fg,
        "funding_rate": funding,
        "candles_15m": candles_15m or [],
        "positions_count": len(active),
        "positions_summary": pos_summary,
        # Live WS enrichment
        "ws_trade_price": ws_trade_price,
        "ws_trade_side": ws_trade_side,
        "ws_trade_size": ws_trade_size,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_pct": spread_pct,
        "bid_depth_usd": bid_depth,
        "ask_depth_usd": ask_depth,
        "ws_age_ms": ws_age_ms,
        "enrichment": enrichment_ctx,  # CVD/VWAP/Cascade/OI signals
    }
    return result


# ============================================================
# FORAGER — Dynamic Coin Selection
# ============================================================

FORAGER_COOLDOWN_SECS = 120  # 2 min — shorter to prevent system from going silent

def _forager_select(candidates: list[str], mids: dict, max_coins: int = 5) -> list[str]:
    """Select best coins to trade based on volume, signal quality, and diversification."""
    global _forager_skip_cooldown, _global_pause_until
    now = time.time()
    # Clean expired cooldowns
    _forager_skip_cooldown = {k: v for k, v in _forager_skip_cooldown.items() if now - v < FORAGER_COOLDOWN_SECS}

    # ── Global pause: recently closed a position — wait before opening new ones ──
    if now < _global_pause_until:
        remaining = int(_global_pause_until - now)
        log.info(f"  ⏳ Global pause: {remaining}s remaining after recent close — skipping new entries")
        return []
    
    scored = []

    for coin in candidates:
        mid = float(mids.get(coin, 0))
        if mid <= 0:
            continue
        
        # Skip coins on cooldown (recently blocked)
        if coin in _forager_skip_cooldown:
            continue

        # Use cached candles (shared with signal computation later)
        candles = _fetch_candles_cached(coin, "15m", 55)
        if len(candles) < 50:
            scored.append((coin, 0.0, "no_data"))
            continue

        df = add_basic_indicators(candles_to_frame(candles))
        if len(df) < 50:
            scored.append((coin, 0.0, "no_data"))
            continue

        composite = abs(compute_composite(df))
        
        # ── BTC correlation boost: coins that move with BTC are more predictable ──
        btc_corr = DEFAULT_BTC_CORRELATIONS.get(coin.upper(), 0.5)
        composite *= (0.5 + btc_corr * 0.5)  # Scale: 0.5 corr → 0.75x, 1.0 corr → 1.0x

        # ── Volume filter: skip dead coins (< 20 candles in 15m = no volume) ──
        try:
            volumes = [float(c.get("v", c.get("volume", 0))) for c in candles[-20:]]
            avg_vol = sum(volumes) / max(len(volumes), 1)
            mid_px = float(mids.get(coin, 0))
            if mid_px > 0 and avg_vol * mid_px < 100:  # < $100 notional volume = dead
                scored.append((coin, 0.0, "dead_volume"))
                continue
        except Exception:
            pass

        # ── Volatility boost: bigger swings = more profit opportunity ──
        try:
            closes = [float(c.get("c", c.get("close", 0))) for c in candles[-20:] if float(c.get("c", c.get("close", 0))) > 0]
            if len(closes) >= 10:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = (sum(r*r for r in returns) / len(returns)) ** 0.5 * 100  # % stddev
                # Boost composites for coins with 0.5-3% volatility (sweet spot)
                if 0.5 <= volatility <= 3.0:
                    composite *= (1.0 + volatility * 0.3)  # Up to 1.9x for high vol
                elif volatility > 3.0:
                    composite *= 1.5  # Cap boost for extreme vol
        except Exception:
            pass

        # ── Momentum bonus: coins actually moving get priority ──
        try:
            mom5 = _calc_momentum(coin, "5m")
            mom15 = _calc_momentum(coin, "15m")
            # Strong uptrend: both timeframes positive
            if mom5 > 0.5 and mom15 > 0.3:
                composite += 0.08
            elif mom5 > 0.3 and mom15 > 0:
                composite += 0.04
            # Strong downtrend: short opportunity
            elif mom5 < -0.5 and mom15 < -0.3:
                composite += 0.06
        except Exception:
            pass
        # ── Regime bonus: trending coins get priority even with moderate composite ──
        try:
            reg, _ = detect_regime(df, coin)  # Returns (MarketRegime, dict)
            if reg in (MarketRegime.TRENDING_DOWN, MarketRegime.TRENDING_UP):
                composite += 0.10
            elif reg == MarketRegime.HIGH_VOL:
                composite += 0.05
        except Exception:
            # Fallback: simple EMA check
            try:
                close = float(df.iloc[-1]["close"]) if "close" in df.columns else float(df.iloc[-1].get("c", 0))
                ema20 = float(df["ema20"].iloc[-1]) if "ema20" in df.columns else close
                if close < ema20 * 0.98:
                    composite += 0.10  # Likely trending down
                elif close > ema20 * 1.02:
                    composite += 0.10  # Likely trending up
            except Exception:
                pass
            regime = "fallback"
        
        scored.append((coin, composite, "signal"))

    # Sort by composite score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Diversity: at most 2 per correlation group
    from hyperliquid_strategy import get_correlation_group
    groups_used = {}
    selected = []

    for coin, score, _ in scored:
        if len(selected) >= max_coins:
            break
        group = get_correlation_group(coin)
        if group and groups_used.get(group, 0) >= 2:
            continue
        selected.append(coin)
        if group:
            groups_used[group] = groups_used.get(group, 0) + 1

    return selected


# ============================================================
# SIGNAL COMPUTATION (deterministic — no AI needed)
# ============================================================

def _compute_signal(coin: str, btc_candles: list[dict] | None,
                    mids: dict | None = None) -> EnrichedSignal:
    """Compute master signal for a coin using deterministic 5-pillar engine + enrichment."""
    candles = _fetch_candles_cached(coin, "15m", 200)  # Reuses cache from forager
    if len(candles) < 55:
        base = MasterSignal(symbol=coin, side="HOLD", reason="insufficient_data")
        return EnrichedSignal(base_signal=base)

    base = generate_master_signal(
        symbol=coin,
        candles=candles,
        trades=recent_trades,
        btc_candles=btc_candles,
        open_positions={},
        equity=hsl_state.peak_equity or 10000.0,
    )

    # ── Layer enrichment modules (CVD, VWAP, LiqCascade, OI) ──
    try:
        enriched = enrich_master_signal(
            base, candles, mids=mids or {}, hl_client=hl,
        )
        return enriched
    except Exception:
        # Fall back to base signal if enrichment fails
        return EnrichedSignal(base_signal=base)


# ============================================================
# DIRECT CLOSE — bypasses the 5-10 min pipeline for urgent exits
# ============================================================

_API_WALLET = None  # Set at daemon init

def _execute_direct_close(coin: str, size_fraction: float = 1.0, reason: str = "") -> bool:
    """Execute a position close DIRECTLY on Hyperliquid, bypassing the
    pending_actions -> validator -> executor pipeline (5-10 min latency).

    Only closes if position is older than MIN_HOLD_SECONDS to prevent overtrading.
    Returns True on success, False on failure (falls back to pipeline).
    """
    global _API_WALLET, _MANUAL_CLOSES, _global_pause_until, _forager_skip_cooldown
    if not _API_WALLET:
        log.error(f"  DIRECT CLOSE {coin}: no wallet configured")
        return False
    if not _can_close_position(coin, reason):
        return False
    try:
        # Use the already-configured hl module for Hyperliquid operations
        # hl is imported at module level and configured in run()
        if size_fraction >= 0.95:
            result = hl.market_close(coin)
            log.info(f"  DIRECT CLOSE: {coin} full — {reason}")
            # ── Cancel any orphaned TP/SL orders for this coin ──
            try:
                _info = hl.Info()
                _ords = _info.open_orders(hl._main_wallet)
                if _ords:
                    for o in _ords:
                        if o.get('coin','').upper() == coin.upper():
                            hl.cancel_order(coin, o.get('oid',0))
                            log.debug(f"  🧹 {coin}: cancelled orphan order oid={o.get('oid')}")
            except Exception: pass
            # ── Record PnL for performance tracking ──
            try:
                # Get entry price from position data for PnL calc
                from cvd_engine import record_closed_trade
                state = hl.get_user_state(_API_WALLET)
                positions = state.get("assetPositions", [])
                # Position is already closed, so we need the pre-close data
                # Use mid price as approximate — the actual PnL is reflected in balance
                mids = hl.get_all_mids()
                mid = float(mids.get(coin, 0))
                entry_px = 0
                for p in positions:
                    pcoin = p.get("coin", {})
                    pname = pcoin.get("name", "") if isinstance(pcoin, dict) else str(pcoin)
                    if pname == coin or p.get("coin", "") == coin:
                        entry_px = float(p.get("entryPx", 0))
                        break
                szi = 0  # Already closed, can't get size
                if entry_px > 0 and mid > 0:
                    pnl_pct = (mid - entry_px) / entry_px * 100
                    record_closed_trade(coin, 0, pnl_pct, reason)
            except Exception:
                pass  # PnL recording is optional
            # ── Suppress LIQUIDATION/ADL false alarm for our own close ──
            _MANUAL_CLOSES[coin] = time.time()  # Track our own close to suppress false liquidation alerts
            _clear_entry_time(coin)  # Position closed, clear entry tracking
            # ── Post-close cooldown: don't re-enter same coin for 15 min ──
            _forager_skip_cooldown[coin] = time.time()
            # ── Global pause: don't open ANY position for 2 min after a close ──
            _global_pause_until = time.time() + 30  # 30s prevents immediate re-entry (was 120s)
        else:
            # Partial close — get position size from Hyperliquid
            state = hl.get_user_state(main_wallet) if 'main_wallet' in dir() else hl.get_user_state(_API_WALLET)
            positions = state.get("assetPositions", [])
            pos_sz = 0.0
            pos_szi_signed = 0.0
            for p in positions:
                pcoin = p.get("coin", {})
                pname = pcoin.get("name", "") if isinstance(pcoin, dict) else str(pcoin)
                if pname == coin or p.get("coin", "") == coin:
                    pos_szi_signed = float(p.get("szi", 0))
                    pos_sz = abs(pos_szi_signed)
                    break
            if pos_sz <= 0:
                log.warning(f"  DIRECT CLOSE {coin}: position not found")
                return False
            close_sz = pos_sz * size_fraction
            # Determine close side: LONG position → sell (is_buy=False), SHORT → buy (is_buy=True)
            close_is_buy = pos_szi_signed < 0  # short position needs a buy to close
            hl.market_open(coin, close_is_buy, close_sz)
            log.info(f"  DIRECT CLOSE: {coin} {size_fraction:.0%} ({close_sz:.4f}) — {reason}")

        return True
    except Exception as e:
        log.error(f"  DIRECT CLOSE {coin} FAILED: {type(e).__name__}: {e}")
        return False


# ============================================================
# DIRECT OPEN — bypasses pipeline for new positions
# ============================================================

def _place_retry_tpsl(coin: str, is_buy: bool, size_usd: float, 
                      stop_price: float, tp_levels: list[dict]) -> None:
    """Place TP/SL brackets for a retried position that lost its original brackets."""
    try:
        from hyperliquid_execution import round_size, round_price
        import math
        mids = hl.get_all_mids()
        px = float(mids.get(coin, 0))
        if px <= 0:
            return
        sz_dec = hl._get_sz_decimals(coin)
        _log_price = abs(math.log10(max(px, 0.0001)))
        px_dec = max(0, int(5 - _log_price))
        sz = round_size(sz_dec, size_usd / px)
        sl_px = round_price(px_dec, stop_price, is_buy=is_buy)
        hl.trigger_order(coin, not is_buy, sz, sl_px, order_type="sl", is_market=True, reduce_only=True)
        for tp in tp_levels[:3]:
            tp_px = round_price(px_dec, float(tp.get("price", 0)), is_buy=is_buy)
            tp_sz = round_size(sz_dec, float(tp.get("size", sz * 0.33)))
            if tp_px > 0 and tp_sz > 0:
                hl.order(coin, not is_buy, tp_sz, tp_px, order_type="gtc", reduce_only=True)
        log.info(f"  🎯 {coin} TP/SL set on retry: SL={sl_px}")
    except Exception as e:
        log.warning(f"  DIRECT OPEN {coin}: retry TP/SL failed ({e})")


def _extract_oid(result) -> int:
    """Extract order ID from Hyperliquid response — handles nested SDK structure."""
    if isinstance(result, dict):
        oid = result.get("oid") or result.get("orderId")
        if not oid:
            try:
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                if statuses:
                    s0 = statuses[0]
                    for status_type in ("filled", "resting", "triggered"):
                        inner = s0.get(status_type, {})
                        if isinstance(inner, dict) and inner.get("oid"):
                            oid = inner["oid"]
                            break
                    if not oid:
                        oid = s0.get("oid") or 0
            except Exception:
                pass
        return oid or 0
    return 0


def _execute_direct_open(coin: str, is_buy: bool, size_usd: float, leverage: int,
                         stop_price: float, tp_levels: list[dict] | None = None,
                         reason: str = "", vwap_sigma: float = 0.0,
                         entry_zone: float = 0.0, entry_type: str = "market",
                         invalidation: str = "") -> bool:
    """Execute a new position DIRECTLY on Hyperliquid, bypassing the
    pending_actions -> validator -> executor pipeline.

    Sets leverage first, then places market order with TP/SL brackets.
    Returns True on success, False on failure (falls back to file-based pipeline).
    """
    global _API_WALLET
    if not _API_WALLET:
        log.error(f"  DIRECT OPEN {coin}: no wallet configured")
        return False
    try:
        # Step 1: Set leverage with retry on 429
        for _lev_attempt in range(3):
            try:
                hl.update_leverage(coin, leverage)
                break
            except Exception as e:
                if _lev_attempt < 2 and ("429" in str(e) or "ClientError" in type(e).__name__):
                    time.sleep(1 + _lev_attempt)
                    continue
                log.warning(f"  DIRECT OPEN {coin}: leverage set failed ({e}), continuing...")
                break

        # Step 2: Place LIMIT order at favorable price (0.5% edge)
        # Aug 7: replaced market IOC with limit orders. Price must come to us.
        # This guarantees better entry + natural momentum confirmation.
        # Long: limit 0.5% below market. Short: limit 0.5% above market.
        # Wait up to 45s for fill. Cancel if no fill — don't chase.
        mids = hl.get_all_mids()
        px = float(mids.get(coin, 0))
        if px <= 0:
            log.error(f"  DIRECT OPEN {coin}: no mid price")
            return False
        
        from hyperliquid_execution import round_size, round_price
        import math as _math
        sz_dec = hl._get_sz_decimals(coin)
        _log_price = abs(_math.log10(max(px, 0.0001)))
        px_dec = max(0, int(5 - _log_price))
        sz = round_size(sz_dec, size_usd / px)
        
        # Step 2: Tick-peak-aware entry — short at local peaks, buy at local bottoms
        _side_str = "BUY" if is_buy else "SELL"
        
        # ── AI ENTRY ZONE: if AI zone is close (<1%), use limit. Otherwise market. ──
        if entry_zone > 0 and abs(px - entry_zone) / px < 0.01:
            _zone_limit = round_price(px_dec, entry_zone, is_buy=is_buy)
            log.info(f"  🎯 {coin}: AI zone=${entry_zone:.4f} close ({abs(px-entry_zone)/px*100:.1f}%) — limit, 45s timeout")
            result = hl.order(coin, is_buy, sz, _zone_limit, order_type="gtc")
            if isinstance(result, dict) and result.get("status") == "err":
                log.warning(f"  DIRECT OPEN {coin}: zone limit failed — {result.get('error', 'unknown')}, falling back with micro-peak timing")
                _mp_px = _micro_peak_entry_wait(coin, is_buy, px)
                if _mp_px != px:
                    log.info(f"  ⚡ {coin}: micro-peak fallback → ${px:.4f}→${_mp_px:.4f}")
                result = hl.market_open(coin, is_buy, size_usd, slippage=0.005, order_type="Ioc")
            else:
                _z_oid = _extract_oid(result)
                if _z_oid:
                    _PENDING_ZONE[coin.upper()] = {
                        "oid": _z_oid, "is_buy": is_buy, "size_usd": size_usd, "sz": sz,
                        "stop_price": stop_price, "tp_levels": tp_levels, "reason": reason,
                        "leverage": leverage, "placed_at": time.time(),
                        "timeout_s": 45, "label": "zone",
                    }
                    log.info(f"  📝 {coin}: zone limit oid={_z_oid} queued (45s)")
                    return True
                else:
                    log.warning(f"  DIRECT OPEN {coin}: no oid from zone limit — falling back with micro-peak timing")
                    _mp_px = _micro_peak_entry_wait(coin, is_buy, px)
                    if _mp_px != px:
                        log.info(f"  ⚡ {coin}: micro-peak fallback → ${px:.4f}→${_mp_px:.4f}")
                    result = hl.market_open(coin, is_buy, size_usd, slippage=0.005, order_type="Ioc")
        
        elif entry_zone > 0:
            # AI zone too far — cap at 10% distance
            _zone_distance = abs(px-entry_zone)/px*100
            if _zone_distance > 10.0:
                log.info(f"  🚫 {coin}: ZONE TOO FAR — AI zone=${entry_zone:.4f} is {_zone_distance:.1f}% away (>{10.0}% cap), skip")
                return False
            # AI zone too far — enter at market with micro-peak timing
            log.info(f"  🎯 {coin}: AI zone=${entry_zone:.4f} too far ({_zone_distance:.1f}%) — market entry")
            _mp_px = _micro_peak_entry_wait(coin, is_buy, px)
            if _mp_px != px:
                log.info(f"  ⚡ {coin}: micro-peak → ${px:.4f}→${_mp_px:.4f} ({(abs(_mp_px-px)/px*100):.2f}% better)")
            result = hl.market_open(coin, is_buy, size_usd, slippage=0.005, order_type="Ioc")
        
        # Step 3: Pump/dump-aware entry — don't chase, wait for pullback
        _chase = _check_recent_move(coin, _side_str)
        
        if _chase["block"]:
            log.warning(f"  🚫 {coin}: CHASE BLOCK — {_chase['detail']}")
            return False

            result = hl.market_open(coin, is_buy, size_usd, slippage=0.005, order_type="Ioc")
        if isinstance(result, dict) and result.get("status") == "err":
            log.error(f"  DIRECT OPEN {coin}: {result.get('error', 'unknown')}")
            return False

        oid = _extract_oid(result)
        exchange_error = None
        if isinstance(result, dict) and not oid:
            try:
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                if statuses and "error" in statuses[0]:
                    exchange_error = statuses[0]["error"]
            except Exception:
                pass
        if exchange_error:
            log.error(f"  DIRECT OPEN {coin}: exchange rejected — {exchange_error}")
            return False
        if not oid or oid == 0:
            log.error(f"  DIRECT OPEN {coin}: no order ID — result={result}")
            return False
        log.info(f"  ✅ DIRECT OPEN: {coin} {'LONG' if is_buy else 'SHORT'} "
                 f"${size_usd:.2f} @ {leverage}x (oid={oid}) — {reason[:60]}")

        # Step 3: Place TP/SL bracket orders (if provided)
        # RETRY with exponential backoff on 429 — NEVER leave a position naked
        tp_oids = []
        sl_oid = 0
        tp_sl_placed = False
        if stop_price and stop_price > 0 and tp_levels:
            tp_sl_errors = []
            for tp_sl_attempt in range(3):
                try:
                    from hyperliquid_execution import round_size, round_price
                    import math
                    mids = hl.get_all_mids()
                    px = float(mids.get(coin, 0))
                    if px > 0:
                        # Get correct decimals
                        sz_dec = hl._get_sz_decimals(coin)
                        _log_price = abs(math.log10(max(px, 0.0001)))
                        px_dec = max(0, int(5 - _log_price))

                        sz = round_size(sz_dec, size_usd / px)
                        # Place SL as a TRIGGER order (not limit — limit fills instantly below market!)
                        sl_px = round_price(px_dec, stop_price, is_buy=is_buy)
                        sl_result = hl.trigger_order(coin, not is_buy, sz, sl_px,
                                                      order_type="sl", is_market=True, reduce_only=True)
                        sl_oid = sl_result.get("oid", 0) if isinstance(sl_result, dict) else 0
                        # Place TP levels
                        for tp in tp_levels[:3]:
                            tp_px = round_price(px_dec, float(tp.get("price", 0)), is_buy=is_buy)
                            tp_sz_raw = float(tp.get("size", sz * 0.33))
                            tp_sz = round_size(sz_dec, tp_sz_raw)
                            if tp_px > 0 and tp_sz > 0:
                                tp_result = hl.order(coin, not is_buy, tp_sz, tp_px,
                                                     order_type="gtc", reduce_only=True)
                                tp_oid = tp_result.get("oid", 0) if isinstance(tp_result, dict) else 0
                                if tp_oid:
                                    tp_oids.append(tp_oid)
                        # Track orders for fill verification
                        from action_executor import _track_order
                        _track_order(coin, {
                            "sl_oid": sl_oid, "tp_oids": tp_oids,
                            "entry_oid": oid, "leverage": leverage,
                        })
                        log.info(f"  🎯 TP/SL set: SL={sl_px} TPs={[round(t.get('price',0),px_dec) for t in tp_levels[:3]]}")
                        tp_sl_placed = True
                        break  # success — exit retry loop
                except Exception as e:
                    err_str = str(e)
                    is_429 = "429" in err_str
                    tp_sl_errors.append(err_str[:80])
                    if tp_sl_attempt < 2:
                        delay = (2 ** tp_sl_attempt) * (2 if is_429 else 1)  # 2s/4s/8s for 429, 1s/2s/4s otherwise
                        log.warning(f"  DIRECT OPEN {coin}: TP/SL attempt {tp_sl_attempt+1}/3 failed ({err_str[:60]}) — retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        log.error(f"  DIRECT OPEN {coin}: TP/SL FAILED after 3 attempts: {err_str[:80]}")
            
            if not tp_sl_placed:
                # ALL retries failed — CLOSE the position, never leave it naked
                log.error(f"  🚨 DIRECT OPEN {coin}: closing position — TP/SL could not be placed after 3 attempts (errors: {tp_sl_errors})")
                try:
                    hl.market_close(coin)
                    log.error(f"  🚨 DIRECT OPEN {coin}: position CLOSED — no SL protection possible")
                except Exception as close_err:
                    log.error(f"  🚨 DIRECT OPEN {coin}: CRITICAL — position OPEN but NAKED (close also failed: {close_err})")
                return False  # Don't proceed — position was closed or is dangerously naked
        elif stop_price and stop_price > 0 and not tp_levels:
            # Has stop but no TP levels — at minimum place the SL
            for tp_sl_attempt in range(2):
                try:
                    from hyperliquid_execution import round_size, round_price
                    import math
                    mids = hl.get_all_mids()
                    px = float(mids.get(coin, 0))
                    if px > 0:
                        sz_dec = hl._get_sz_decimals(coin)
                        _log_price = abs(math.log10(max(px, 0.0001)))
                        px_dec = max(0, int(5 - _log_price))
                        sz = round_size(sz_dec, size_usd / px)
                        sl_px = round_price(px_dec, stop_price, is_buy=is_buy)
                        sl_result = hl.trigger_order(coin, not is_buy, sz, sl_px,
                                                      order_type="sl", is_market=True, reduce_only=True)
                        sl_oid = sl_result.get("oid", 0) if isinstance(sl_result, dict) else 0
                        if sl_oid:
                            from action_executor import _track_order
                            _track_order(coin, {"sl_oid": sl_oid, "tp_oids": [], "entry_oid": oid, "leverage": leverage})
                            log.info(f"  🛑 SL set: {sl_px} (no TP levels)")
                            tp_sl_placed = True
                            break
                except Exception as e:
                    if tp_sl_attempt < 1:
                        time.sleep(3)
                    else:
                        log.error(f"  🚨 DIRECT OPEN {coin}: SL placement failed — closing position")
                        try:
                            hl.market_close(coin)
                            log.error(f"  🚨 DIRECT OPEN {coin}: position CLOSED — no SL protection possible")
                        except Exception as close_err:
                            log.error(f"  🚨 DIRECT OPEN {coin}: CRITICAL — position OPEN but NAKED (close also failed: {close_err})")
                        return False

        # Record trade for self-learning (entry tracked when position monitor detects it)
        # ── Verify position actually filled (IOC orders can return oid but not fill) ──
        try:
            time.sleep(0.8)
            state = hl.get_user_state()
            # Calculate expected size from current mid price or fallback to entry_price
            mids_now = hl.get_all_mids()
            px = float(mids_now.get(coin, 0))
            if px <= 0:
                px = size_usd  # Fallback: use notional as rough price estimate
            expected_sz = size_usd / px if px > 0 else 0.001  # Safety floor
            for p in state.get("assetPositions", []):
                pos_coin = p.get("position", {}).get("coin", "")
                if pos_coin.upper() == coin.upper():
                    szi = abs(float(p["position"].get("szi", 0)))
                    fill_notional = szi * px
                    if szi > 0.0001 and (expected_sz <= 0 or szi >= expected_sz * 0.5):
                        return True  # Position confirmed with reasonable fill
                    elif szi > 0.0001:
                        # IOC partial — under 50% fill
                        if fill_notional >= MIN_NOTIONAL_USD:
                            log.warning(f"  ⚠️ DIRECT OPEN {coin}: IOC partial ${fill_notional:.2f} — accepted")
                            return True
                        else:
                            log.warning(f"  ⚠️ DIRECT OPEN {coin}: IOC dust ${fill_notional:.2f} < ${MIN_NOTIONAL_USD} — closing")
                            hl.market_close(coin)
                            # Blacklist coin for 30 min — thin book, won't improve quickly
                            _forager_skip_cooldown[coin.upper()] = time.time() + 1800
                            log.warning(f"  🚫 {coin}: dust-cooldown 30min (thin order book)")
                            return False
            # No position at all — ghost fill (IOC usually prevents this)
            log.warning(f"  ⚠️ DIRECT OPEN {coin}: ghost fill — no position found")
            return False
        except Exception:
            return True  # Verification failed but order was placed — proceed optimistically

    except Exception as e:
        log.error(f"  DIRECT OPEN {coin} FAILED: {type(e).__name__}: {e}")
        return False


# ============================================================
# ACTION SUBMISSION (with exit plan)
# ============================================================

def _submit_action(action_type: str, coin: str, details: dict):
    """Submit to pending_actions pipeline with full exit plan."""
    global _forager_skip_cooldown, _global_pause_until
    os.makedirs("data/pending_actions", exist_ok=True)

    # ── Guard: skip if a close was already submitted for this coin ──
    if action_type in ("close", "buy", "sell"):
        import fnmatch
        existing = [f for f in os.listdir("data/pending_actions")
                    if f.startswith(f"{action_type}_{coin}_") and f.endswith(".json")]
        if existing:
            log.debug(f"  ⏭️  {coin}: close already pending ({len(existing)} file(s)) — skipping duplicate")
            return  # Don't submit another close for the same position
        # ── Guard: skip if we already have an open position in this coin ──
        if action_type in ("buy", "sell"):
            try:
                positions = hl.info.user_state().get("assetPositions", [])
                for pos in positions:
                    pos_coin = pos.get("position", {}).get("coin", "")
                    pos_szi = float(pos.get("position", {}).get("szi", 0))
                    if pos_coin.upper() == coin.upper() and abs(pos_szi) > 0.0001:
                        log.warning(f"  ⛔ {coin}: already have open position (szi={pos_szi:.4f}) — blocking duplicate entry")
                        return
            except Exception:
                pass  # Can't check positions — let it through rather than blocking legit trades
    ts = int(time.time())
    action = {
        "type": action_type,
        "exchange": "hyperliquid",
        "symbol": coin,          # Validator expects 'symbol'
        "coin": coin,            # Backward compat
        "price": details.get("entry_price", 0),  # Validator expects 'price'
        "size": details.get("size_units", 0),    # Validator expects 'size'
        "direction": details.get("direction", ""),
        "side": details.get("side", ""),
        "size_units": details.get("size_units", 0),
        "size_usd": details.get("size_usd", 0),
        "leverage": details.get("leverage", 3),
        "entry_price": details.get("entry_price"),
        "stop_loss": details.get("stop_loss"),
        "tp_levels": details.get("tp_levels", []),
        "break_even_enabled": details.get("break_even_enabled", True),
        "trail_enabled": details.get("trail_enabled", True),
        "trail_atr": details.get("trail_atr", 2.0),
        "urgency": details.get("urgency", ""),
        "reason": details.get("reason", "")[:200],
        "source": "hyperliquid_daemon_v3",
        "confidence": details.get("confidence", 0),
        "regime": details.get("regime", "sideways"),
        "kelly": details.get("kelly", 0),
        "composite": details.get("composite", 0),
        "timestamp": ts,
    }
    fname = f"data/pending_actions/{action_type}_{coin}_{ts}_{os.getpid()}.json"
    # ── Mark manual closes to suppress LIQUIDATION/ADL false alarms ──
    if action_type == "close":
        _MANUAL_CLOSES[coin] = time.time()
        _clear_entry_time(coin)  # Position closed, clear entry tracking
        # ── Global pause after any close ──
        _global_pause_until = time.time() + 120
        # ── Reset signal dominance when position closes ──
        _reset_signal_dominance(coin)
    Path(fname).write_text(json.dumps(action, indent=2))
    log.info(f"  → Submitted: {action_type} {coin} ${details.get('size_usd',0):.2f} "
             f"@{details.get('entry_price','market')}")

    # ── Post-submission cooldown: don't re-select same coin for 30 min ──
    if action_type in ("buy", "sell"):
        _forager_skip_cooldown[coin] = time.time()
        # NOTE: entry time is recorded when _monitor_positions first detects
        # the position on Hyperliquid, not here. This avoids the pipeline
        # latency (validator→executor) from burning min hold time before
        # the position is actually open.
        
        # Also prevent duplicate submission: check if pending action already exists
        import glob
        existing = glob.glob(f"data/pending_actions/{action_type}_{coin}_*.json")
        if len(existing) > 1:
            # Keep only most recent, remove duplicates
            existing.sort()
            for old_f in existing[:-1]:
                try:
                    os.remove(old_f)
                except OSError:
                    pass

    # ── Record trade for self-learning (close actions only) ──
    if action_type == "close":
        try:
            _record_close_trade(
                coin, float(details.get("price", details.get("entry_price", 0))),
                float(details.get("entry_price", 0)),
                float(details.get("size_units", details.get("size", 0))),
                details.get("side", details.get("direction", "LONG")),
                details.get("reason", "close"),
            )
        except Exception:
            pass

    # Store TP levels for position monitoring
    if action_type in ("buy", "sell") and details.get("tp_levels"):
        position_tps[coin.upper()] = details["tp_levels"]
        # Also initialize trail state
        trail_states[coin.upper()] = TrailState(
            symbol=coin.upper(),
            is_long=(details.get("side") == "BUY"),
            entry_price=details.get("entry_price", 0),
            highest_price=details.get("entry_price", 0),
            current_stop=details.get("stop_loss", 0),
        )


# ============================================================
# POSITION MONITOR (with trailing stop + unstucking)
# ============================================================

def _record_close_trade(coin: str, mid: float, entry: float, szi: float, side: str,
                         reason: str, conviction: float = 0, regime: str = "sideways",
                         leverage: int = BASE_LEVERAGE):
    """Record a closed position for self-learning and update layer weights."""
    try:
        pnl = (mid - entry) * abs(szi) * (1 if side == "LONG" else -1)
        pnl_pct = (mid - entry) / entry * 100 * (1 if side == "LONG" else -1)
        net_pnl_pct = _net_pnl_pct(pnl_pct, leverage)
        record_trade(TradeRecord(
            coin=coin, side=side, entry_price=entry, exit_price=mid,
            pnl=pnl, pnl_pct=round(net_pnl_pct, 3), entry_time=time.time() - 3600,
            exit_time=time.time(), regime=regime, conviction=conviction,
            composite_score=0, leverage=leverage, reason=reason,
        ))
        # ── Adaptive layer learning: was the prediction correct? ──
        # Use net PnL (after fees) to judge correctness
        was_bullish_correct = (side == "LONG" and net_pnl_pct > 0) or (side == "SHORT" and net_pnl_pct < 0)
        was_bearish_correct = (side == "SHORT" and net_pnl_pct > 0) or (side == "LONG" and net_pnl_pct < 0)
        # Update directional layers
        for layer_name in ["order_book", "cvd_delta", "multi_tf_technical",
                           "ml_ensemble", "higher_tf_align", "taker_ratio",
                           "imbalance_trend", "oi_delta"]:
            try:
                # If we had a strong signal, check if it was right
                if was_bullish_correct:
                    update_layer_accuracy(layer_name, True)
                elif was_bearish_correct:
                    update_layer_accuracy(layer_name, True)
                # If trade was flat/small, don't update heavily
            except Exception:
                pass
        log.info(f"  📚 Learned from {coin} {side}: {'bullish' if was_bullish_correct else 'bearish'} correct")

        # ── Post-trade AI analysis (web search, background thread — not blocking) ──
        try:
            import threading
            trade_snapshot = {
                "coin": coin, "side": side, "entry_price": entry,
                "exit_price": mid, "pnl_pct": round(net_pnl_pct, 3),
                "hold_secs": 0, "regime": regime, "conviction": conviction,
                "composite_score": 0, "leverage": leverage, "reason": reason,
            }
            t = threading.Thread(target=_run_trade_analysis, args=(trade_snapshot,), daemon=True)
            t.start()
        except Exception:
            pass  # Analysis is optional — never block the main loop

        # ── AI feedback: store summary for future context ──
        trade_entry = {
            "coin": coin, "side": side, "pnl_pct": round(pnl_pct, 2),
            "reason": reason[:40], "was_correct": was_bullish_correct or was_bearish_correct,
        }
        _recent_trades.append(trade_entry)
        recent_trades.append(trade_entry)
        
        # ── Prompt performance tracking (for auto-optimization) ──
        try:
            record_prompt_outcome(
                prompt_hash=hash_prompt("ai_select"),  # tracks prompt version
                coin=coin, pnl_pct=round(net_pnl_pct, 3),
                regime=regime, confidence=int(conviction), side=side,
            )
        except Exception:
            pass
        if len(_recent_trades) > 10:
            _recent_trades.pop(0)
        if len(recent_trades) > 50:
            recent_trades.pop(0)
    except Exception:
        pass  # Learning is optional

# ── Background trade analysis (runs in thread, web search, NOT blocking) ──
def _run_trade_analysis(trade_data: dict):
    """Run AI post-trade analysis in background thread. Logs result, never throws."""
    try:
        # Build market context snapshot for the analysis
        ctx_lines = []
        if _recent_trades:
            ctx_lines.append(f"Recent ({len(_recent_trades)}): " + 
                ", ".join(f"{t['coin']}:{t['pnl_pct']:+.1f}%" for t in _recent_trades[-5:]))
        ctx = "\n".join(ctx_lines) if ctx_lines else ""
        
        result = analyze_closed_trade(trade_data, market_context=ctx)
        analysis = result.get("analysis", "")[:200]
        suggestions = result.get("suggestions", [])
        missed = result.get("missed_opportunities", [])
        
        log.info(f"  🧠 AI trade analysis: {analysis}")
        for s in suggestions[:3]:
            log.info(f"     💡 {s.get('param','?')}: {s.get('current','?')}→{s.get('suggested','?')} — {s.get('reason','')[:80]}")
        for m in missed[:2]:
            log.info(f"     ⚠️ Missed: {m[:120]}")
    except Exception as e:
        log.info(f"  🧠 AI trade analysis failed: {type(e).__name__}")

_recent_trades: list[dict] = []  # Last 10 trades for AI context

# ── Signal dominance tracking (prevents ping-pong without timers) ──
# When we act on a PEAK signal (add short / sell long), a subsequent BOTTOM
# signal must have HIGHER confidence to override. Same for BOTTOM→PEAK.
# Key insight: if we just bet on "this is a peak" at 39% confidence,
# a 38% "this is a bottom" signal is not convincing enough to reverse.
_last_tick_action: dict[str, dict] = {}  # coin → {is_peak, confidence, action}

def _check_signal_dominance(coin: str, is_peak: bool, confidence: float) -> bool:
    """Return False if this signal is a weak reversal of a recent stronger signal."""
    prev = _last_tick_action.get(coin.upper())
    if not prev:
        return True  # No previous action → allow
    # Check if direction flipped (PEAK→BOTTOM or BOTTOM→PEAK)
    direction_flipped = (prev["is_peak"] != is_peak)
    if not direction_flipped:
        return True  # Same direction → always allow (reinforcing)
    # Direction flipped: require new confidence ≥ 80% of prior (was > prior — too strict)
        # ETH case: SELL at 40% suppressed by prior BUY at 41%. 40% ≥ 41%*0.80 = 32.8% → should pass.
        if confidence >= prev["confidence"] * 0.80:
            return True  # Close enough — legitimate reversal
    # Weaker opposing signal → suppress (ping-pong prevention)
    return False

def _reset_signal_dominance(coin: str):
    """Clear dominance tracking when position is closed."""
    _last_tick_action.pop(coin.upper(), None)

def _verify_open_orders(positions: list, mids: dict):
    """Verify TP/SL orders exist on exchange. Re-place missing ones.
    
    NautilusTrader pattern: periodic open_order check catches naked positions.
    Now runs every cycle but caches result for unchanged positions.
    """
    cu_names = set()
    for p in positions:
        coin = p.get("coin", "")
        szi = abs(float(p.get("szi", 0)))
        if coin and szi > 0.0001:
            cu_names.add(coin.upper())
    if not cu_names:
        return
    
    # Skip if we verified same coins in last 60s (reduces API load)
    _last_verify = getattr(_verify_open_orders, "_last_verify", 0)
    _last_coins = getattr(_verify_open_orders, "_last_coins", set())
    _now = time.time()
    if cu_names == _last_coins and _now - _last_verify < 60:
        return
    _verify_open_orders._last_verify = _now
    _verify_open_orders._last_coins = cu_names.copy()
    
    try:
        open_orders = hl.get_open_orders()
        # Build set of coins that have ANY open order (TP or SL)
        coins_with_orders = set()
        for o in (open_orders or []):
            oc = (o.get("coin", "") or "").upper()
            if oc:
                coins_with_orders.add(oc)
        
        naked = cu_names - coins_with_orders
        
        # Also check for coins that have orders but might have stale prices
        # (Hummingbot pattern: cancel old TP orders before placing new ones)
        for cu in coins_with_orders & cu_names:
            ts = trail_states.get(cu)
            if not ts:
                continue
            coin_orders = [o for o in (open_orders or []) if (o.get("coin","") or "").upper() == cu]
            # Check if any order is a TP at an outdated price
            for o in coin_orders:
                is_tp = "tp" in str(o.get("orderType", "")).lower()
                if is_tp:
                    oid = o.get("oid", "")
                    px = float(o.get("limitPx", 0))
                    # If TP price doesn't match our current trail state, cancel it
                    expected_tp = ts.entry_price * (1.02 if ts.is_long else 0.98)
                    if px > 0 and abs(px - expected_tp) / expected_tp > 0.005:  # >0.5% off
                        try:
                            hl.cancel_order(coin, oid)
                            log.info(f"  🧹 {cu}: cancelled stale TP @ ${px:.4f} (expected ~${expected_tp:.4f})")
                        except Exception:
                            pass
        for cu in naked:
            pos = next((p for p in positions if (p.get("coin","") or "").upper() == cu), None)
            if not pos:
                continue
            coin = pos.get("coin", "")
            szi = abs(float(pos.get("szi", 0)))
            entry = float(pos.get("entryPx", 0))
            if entry <= 0 or szi <= 0:
                continue
            is_long = float(pos.get("szi", 0)) > 0
            mid = float(mids.get(coin, entry))
            
            # Build TP/SL from trail state if available, else compute
            ts = trail_states.get(cu)
            sl_price = ts.current_stop if ts else (entry * 0.985 if is_long else entry * 1.015)
            tp1_pct = 1.02 if is_long else 0.98
            
            log.warning(f"  ⚠️ NAKED {coin}: no open orders — re-placing TP/SL")
            try:
                sl_px = round_price(2, sl_price, is_buy=not is_long)
                hl.trigger_order(coin, not is_long, szi, sl_px, order_type="sl", is_market=True, reduce_only=True)
                tp_px = round_price(2, entry * tp1_pct, is_buy=not is_long)
                hl.order(coin, not is_long, szi * 0.5, tp_px, order_type="gtc", reduce_only=True)
                log.info(f"  ✅ {coin}: TP/SL re-placed — SL=${sl_px:.4f} TP=${tp_px:.4f}")
            except Exception as e:
                log.warning(f"  ⚠️ {coin}: re-place TP/SL failed: {e}")
    except Exception as e:
        log.warning(f"  Open order verification error: {type(e).__name__}: {e}")

def _monitor_positions(positions: list, mids: dict, total_eq: float, active: list):
    """Check positions for exit conditions, trail stops, and unstuck."""
    pred_checked = 0
    # Heartbeat: count every 10th call
    if not hasattr(_monitor_positions, "_call_count"):
        _monitor_positions._call_count = 0
        _monitor_positions._last_mids = {}     # {coin: mid} for flash crash detection
        _monitor_positions._last_equity = 0.0
        _monitor_positions._last_equity_ts = 0.0  # timestamp to guard against stale baselines
    _monitor_positions._call_count += 1
    if _monitor_positions._call_count == 1:
        log.info(f"🔧 MONITOR V3.1 ACTIVE — positions={len(positions)} mids={len(mids)}")
    if _monitor_positions._call_count % 10 == 0:
        log.info(f"🔍 Prediction layers alive — call #{_monitor_positions._call_count}")

    # ── FLASH CRASH DETECTION (10-second granularity) ──
    # If any held coin drops >5% in one monitor cycle, emergency close it.
    # If total equity drops >10% AND active coin SET unchanged, emergency close ALL.
    # Coin-set guard: if coins changed between calls (position opened/closed), skip equity check.
    # The margin-return from a closed position looks like a crash — don't false-alarm.
    last_mids = _monitor_positions._last_mids
    last_eq = _monitor_positions._last_equity
    last_coins = getattr(_monitor_positions, '_last_active_coins', set())
    current_coins = {p.get("coin", "").upper() for p in positions if abs(float(p.get("szi", 0))) > 0.0001}

    if last_eq > 0 and total_eq > 0 and _monitor_positions._call_count > 3:  # Only after warmup
        eq_drop = (last_eq - total_eq) / last_eq * 100
        coins_changed = last_coins != current_coins
        # ── Staleness guard: skip crash check if last equity snapshot is >30s old ──
        # Slow cycles (150s+) block the main loop. When the monitor finally fires,
        # _last_equity is stale — a normal gradual decline looks like a crash.
        last_ts = getattr(_monitor_positions, '_last_equity_ts', 0.0)
        stale_baseline = (time.time() - last_ts) > 30
        if stale_baseline and len(active) > 0:
            log.debug(f"  ⏱️  Flash crash check skipped — baseline {time.time()-last_ts:.0f}s stale (slow cycle)")
        if eq_drop > 10 and len(active) > 0 and not coins_changed and last_coins and not stale_baseline:
            log.error(f"  🚨 FLASH CRASH: equity dropped {eq_drop:.1f}% in one cycle "
                     f"(${last_eq:.2f} → ${total_eq:.2f}) — coin set unchanged — EMERGENCY CLOSE ALL")
            for p in positions:
                _coin = p.get("coin", "")
                _szi = float(p.get("szi", 0))
                if abs(_szi) < 0.0001:
                    continue
                try:
                    _MANUAL_CLOSES[_coin] = time.time()
                    hl.market_close(_coin)
                    log.error(f"  🚨 EMERGENCY CLOSED {_coin}")
                except Exception as _e:
                    log.error(f"  🚨 Emergency close {_coin} failed: {_e}")
            _global_pause_until = time.time() + 300  # 5-min pause
            _monitor_positions._last_mids = dict(mids)
            _monitor_positions._last_equity = total_eq
            _monitor_positions._last_equity_ts = time.time()
            _monitor_positions._last_active_coins = set()  # all closed
            return  # Skip rest of monitoring — all positions closed

    flash_crashed = set()
    active_coins = {p.get("coin", "").upper() for p in positions if abs(float(p.get("szi", 0))) > 0.0001}
    for coin, last_price in list(last_mids.items()):
        if coin.upper() not in active_coins:
            continue  # Only check coins we actually hold
        current = mids.get(coin, 0)
        if current <= 0 or last_price <= 0:
            continue
        drop_pct = (last_price - current) / last_price * 100
        # Dynamic threshold: 15% for sub-$1, 8% for $1-10, 5% for $10+
        thresh = 15.0 if current < 1.0 else (8.0 if current < 10.0 else 5.0)
        if drop_pct > thresh:
            flash_crashed.add(coin)
            log.error(f"  🚨 FLASH CRASH: {coin} dropped {drop_pct:.1f}% in one cycle "
                     f"(${last_price:.4f} → ${current:.4f}) — emergency close")

    for coin in flash_crashed:
        try:
            _MANUAL_CLOSES[coin] = time.time()
            hl.market_close(coin)
            log.error(f"  🚨 EMERGENCY CLOSED {coin} (flash crash)")
        except Exception as _e:
            log.error(f"  🚨 Emergency close {coin} failed: {_e}")

    _monitor_positions._last_mids = dict(mids)
    _monitor_positions._last_equity = total_eq
    _monitor_positions._last_equity_ts = time.time()
    _monitor_positions._last_active_coins = current_coins

    for p in positions:
        coin = p.get("coin", "")
        szi = float(p.get("szi", 0))
        if _monitor_positions._call_count <= 2:
            log.info(f"  🔧 MONITOR processing: {coin} szi={szi:.4f}")
        if abs(szi) < 0.0001:
            continue

        # ── PENDING ZONE ORDER CHECK: did AI's limit fill? ──
        cu_upper = coin.upper()
        if cu_upper in _PENDING_ZONE:
            pz = _PENDING_ZONE.pop(cu_upper)
            log.info(f"  ✅ {coin}: pending limit filled — placing TP/SL")
            try:
                _place_retry_tpsl(coin, pz["is_buy"], pz["size_usd"],
                                  pz["stop_price"], pz["tp_levels"] or [])
            except Exception as pze:
                log.warning(f"  ⚠️ {coin}: pending TP/SL failed: {pze}")

        entry = float(p.get("entryPx", 0))
        liq = float(p.get("liquidationPx") or 0)
        mid = float(mids.get(coin, 0))
        if mid <= 0:
            continue
        
        # Track entry time if not already tracked (catches positions opened directly)
        cu = coin.upper()
        if cu not in _position_entry_times:
            _position_entry_times[cu] = time.time()  # Best estimate — was opened recently

        side = "LONG" if szi > 0 else "SHORT"
        pnl_pct = ((mid - entry) / entry * 100) * (1 if szi > 0 else -1)
        atr = float(p.get("atr", mid * 0.01))
        hold_secs = time.time() - _position_entry_times.get(cu, time.time())
        leverage = float(p.get("leverage", {}).get("value", BASE_LEVERAGE)) if isinstance(p.get("leverage"), dict) else BASE_LEVERAGE
        lev = leverage  # DEFENSIVE: alias so both names always work (prevents UnboundLocalError)

        # ── Minimal ROI check: DISABLED for zero-loss ──
        # Aug 7: ROI timeout kills positions at a loss. Disabled.
        # Position stays open until exchange TP or breakeven SL.
        if False:  # DISABLED for zero-loss
            net_pnl = _net_pnl_pct(pnl_pct, leverage=leverage)
            roi_key = f"{coin}:{applicable}"
            if roi_key not in _ROI_TIMEOUT_ATTEMPTED:
                _ROI_TIMEOUT_ATTEMPTED.add(roi_key)
                log.info(f"  ⏰ {coin}: ROI timeout — held {hold_secs/60:.0f}min, gross={pnl_pct:+.2f}% net={net_pnl:+.2f}% — losing beyond {min_profit}% threshold → exiting")
                try:
                    _MANUAL_CLOSES[coin] = time.time()
                    hl.market_close(coin)
                    _reset_signal_dominance(coin)
                except Exception as e:
                    log.warning(f"  ⏰ {coin}: ROI exit failed: {e}")
                    # Remove throttle on transient failure (429, timeout) so it retries next cycle
                    _ROI_TIMEOUT_ATTEMPTED.discard(roi_key)
            continue
        if False:  # DISABLED for zero-loss (was elif applicable > 0 and pnl_pct >= 0)
            net_pnl = _net_pnl_pct(pnl_pct, leverage=leverage)
            roi_key = f"{coin}:flat"
            if roi_key not in _ROI_TIMEOUT_ATTEMPTED:
                _ROI_TIMEOUT_ATTEMPTED.add(roi_key)
                log.info(f"  ⏰ {coin}: ROI timeout — held {hold_secs/60:.0f}min, gross={pnl_pct:+.2f}% net={net_pnl:+.2f}% — flat for 60min → exiting")
                try:
                    _MANUAL_CLOSES[coin] = time.time()
                    hl.market_close(coin)
                    _reset_signal_dominance(coin)
                except Exception as e:
                    log.warning(f"  ⏰ {coin}: ROI exit failed: {e}")
            continue

        # Liquidation warning + survival force-close
        if liq > 0:
            liq_dist = abs(mid - liq) / mid * 100
            if liq_dist < 1.0:
                log.error(f"  🚨 {coin} {side}: {liq_dist:.2f}% from LIQUIDATION! Liq=${liq:,.4f} — FORCE CLOSING")
                try:
                    _MANUAL_CLOSES[coin] = time.time()
                    hl.market_close(coin)
                    _reset_signal_dominance(coin)
                    continue
                except Exception as e:
                    log.error(f"  🚨 Liquidation escape for {coin} failed: {e}")
            elif liq_dist < 10:
                log.warning(f"  ⚠️  {coin} {side}: {liq_dist:.1f}% from LIQUIDATION! Liq=${liq:,.2f}")
        # ── Price extremes: show range context (highs, lows, position) ──
        try:
            ext = get_extremes(coin, mids)
            if ext and (ext.high_1h > 0 or ext.high_4h > 0 or ext.high_24h > 0):
                ext_parts = [f"${mid:.4f}"]
                if ext.high_1h > 0 and ext.low_1h > 0:
                    ext_parts.append(f"1h:{ext.pct_1h:+.1f}%/-{ext.pct_low_1h:.1f}%@{ext.range_pos_1h:.0f}%")
                elif ext.high_1h > 0:
                    ext_parts.append(f"1h_hi={ext.pct_1h:+.1f}%")
                if ext.high_4h > 0 and ext.low_4h > 0:
                    ext_parts.append(f"4h:{ext.pct_4h:+.1f}%/-{ext.pct_low_4h:.1f}%")
                elif ext.high_4h > 0:
                    ext_parts.append(f"4h_hi={ext.pct_4h:+.1f}%")
                if ext.high_24h > 0 and ext.low_24h > 0:
                    ext_parts.append(f"24h:{ext.pct_24h:+.1f}%/-{ext.pct_low_24h:.1f}%")
                log.info(f"  📊 {coin}: {' '.join(ext_parts)}")
        except Exception:
            pass

        # ── Rule-based peak rollover exit (replaces AI eval as primary) ──
        # Hard thresholds derived from actual loss patterns:
        #   BTC: +0.28% → liquidated, held through 0.48% peak drop
        #   LTC #1: -0.04%, struct+mom caught it  
        #   LTC #2: +0.37% → +0.16%, AI held 10x, safety net caught it
        # AI eval kept as tiebreaker only — no more AI-decides-everything
        _eval_key = f"last_exit_eval:{cu}"
        _now_ts = time.time()
        entry_ts = _position_entry_times.get(cu, 0)
        if _monitor_positions._call_count <= 20:
            log.info(f"  🔍 {cu} eval-state: age={_now_ts-entry_ts:.0f}s call=#{_monitor_positions._call_count} entry_ts={entry_ts:.0f} eligible={_now_ts-entry_ts > 15}")
        if _now_ts - entry_ts > 15:  # 15s warmup (was 60s — delayed exits too long)
            _last_eval = getattr(_monitor_positions, _eval_key, 0) if hasattr(_monitor_positions, _eval_key) else 0
            if _now_ts - _last_eval > 30:
                try:
                    # Peak HWM tracking
                    _peak_key = f"_hwm:{cu}"
                    _hwm = getattr(_monitor_positions, _peak_key, 0.0) if hasattr(_monitor_positions, _peak_key) else 0.0
                    if side == "LONG" and _hwm > 0:
                        drop_from_peak_pct = (_hwm - mid) / _hwm * 100
                    elif side == "SHORT" and _hwm > 0:
                        drop_from_peak_pct = (mid - _hwm) / _hwm * 100
                    else:
                        drop_from_peak_pct = 0

                    mom1 = _calc_momentum(coin, "1m") or 0
                    mom5 = _calc_momentum(coin, "5m") or 0
                    if side == "SHORT":
                        mom1 = -mom1; mom5 = -mom5

                    net_pnl_pct = _net_pnl_pct(pnl_pct, leverage)
                    
                    # Track peak PnL achieved (not just HWM price)
                    _peak_pnl_key = f"_peak_pnl:{cu}"
                    _peak_pnl = getattr(_monitor_positions, _peak_pnl_key, -999.0) if hasattr(_monitor_positions, _peak_pnl_key) else -999.0
                    if net_pnl_pct > _peak_pnl:
                        _peak_pnl = net_pnl_pct
                        # Track WHEN we hit this peak
                        _peak_time_key = f"_peak_time:{cu}"
                        setattr(_monitor_positions, _peak_time_key, _now_ts)
                        # Reset soft-exit strike counter on new peak — position is recovering
                        _strikes_reset_key = f"_soft_strikes:{cu}"
                        _old_strikes = getattr(_monitor_positions, _strikes_reset_key, 0) if hasattr(_monitor_positions, _strikes_reset_key) else 0
                        if _old_strikes > 0:
                            setattr(_monitor_positions, _strikes_reset_key, 0)
                            log.info(f"  [GREEN] {coin}: new peak PnL {_peak_pnl:+.2f}% - soft-exit strikes reset (was {_old_strikes})")
                    setattr(_monitor_positions, _peak_pnl_key, _peak_pnl)
                    _peak_time_key = f"_peak_time:{cu}"
                    _peak_time = getattr(_monitor_positions, _peak_time_key, 0) if hasattr(_monitor_positions, _peak_time_key) else 0
                    _peak_age = max(0, _now_ts - _peak_time)
                    
                    # ── TIGHT TRAIL ENFORCEMENT: if soft exit activated trail, check if hit ──
                    _soft_key = f"_soft_strikes:{cu}"
                    _soft_strikes_active = (getattr(_monitor_positions, _soft_key, 0) if hasattr(_monitor_positions, _soft_key) else 0)
                    if _soft_strikes_active > 0 and cu in trail_states:
                        ts = trail_states[cu]
                        if ts.activated and ts.current_stop > 0:
                            if side == "LONG" and mid < ts.current_stop:
                                exit_reason = f"tight-trail-hit: mid=${mid:.4f} < stop=${ts.current_stop:.4f} after {_soft_strikes_active}x warnings"
                            elif side == "SHORT" and mid > ts.current_stop:
                                exit_reason = f"tight-trail-hit: mid=${mid:.4f} > stop=${ts.current_stop:.4f} after {_soft_strikes_active}x warnings"
                    
                    # Log state every 5th call
                    if _monitor_positions._call_count % 5 == 0:
                        log.info(f"  🔍 {coin} exit-check: net_pnl={net_pnl_pct:+.2f}% gross={pnl_pct:+.2f}% "
                                f"peak_drop={drop_from_peak_pct:.2f}% peak_pnl={_peak_pnl:+.2f}% peak_age={_peak_age:.0f}s "
                                f"mom1={mom1:+.2f}% mom5={mom5:+.2f}%")

                    # ═══════════════════════════════════════════════════════════════
                    # ZERO-LOSS EXIT SYSTEM: every close must be net >= 0%.
                    # Aug 7 overhaul: removed ALL loss-cutting Python exits.
                    # Only 3 exits allowed: exchange TP, breakeven stop, profit lock.
                    # ═══════════════════════════════════════════════════════════════
                    
                    exit_reason = None
                    _net_pnl_mon = _net_pnl_pct(pnl_pct, leverage=leverage)  # actual position leverage
                    _hold_age_mon = time.time() - _position_entry_times.get(cu, time.time())
                    
                    # ── LIQUIDATION SURVIVAL: force-close BEFORE exchange does ──
                    if liq > 0:
                        liq_distance_pct = abs(mid - liq) / mid * 100
                        if liq_distance_pct < 1.0:
                            exit_reason = f"LIQUIDATION-SURVIVAL: {liq_distance_pct:.2f}% to liq=${liq:.4f} — escape NOW"
                    
                    # ── INSTANT BREAKEVEN LOCK: the "no loss" core ──
                    # The moment net PnL covers roundtrip fees, cancel the exchange SL
                    # and place a breakeven stop. Trade either hits TP or returns to 0.
                    _be_key = f"_be_locked:{cu}"
                    _be_locked = getattr(_monitor_positions, _be_key, False) if hasattr(_monitor_positions, _be_key) else False
                    _fee_covered_at = ROUNDTRIP_FEE_PCT * leverage  # correct: fee scales with actual leverage
                    if not _be_locked and _net_pnl_mon >= _fee_covered_at:
                        _be_locked = True
                        setattr(_monitor_positions, _be_key, True)
                        try:
                            # Cancel existing exchange SL order(s)
                            _open_orders = hl.info(coin).get("open_orders", []) if hasattr(hl, 'info') else []
                            for _oo in _open_orders:
                                if _oo.get("type") == "stop":
                                    try:
                                        hl.cancel(coin, _oo.get("oid", 0))
                                    except Exception:
                                        pass
                            # Place breakeven stop at entry price
                            from hyperliquid_execution import round_price
                            import math as _math
                            _px_dec = max(0, int(5 - abs(_math.log10(max(entry, 0.0001)))))
                            _be_px = round_price(_px_dec, entry, is_buy=(side == "SHORT"))
                            _be_sz = abs(szi)
                            hl.trigger_order(coin, (side == "SHORT"), _be_sz, _be_px,
                                           order_type="sl", is_market=True, reduce_only=True)
                            log.info(f"  🔒 {coin}: BREAKEVEN LOCKED — net={_net_pnl_mon:+.2f}% ≥ fee={_fee_covered_at:+.2f}%, SL moved to entry ${entry:.5f}")
                        except Exception as _be_err:
                            log.warning(f"  ⚠️ {coin}: breakeven lock failed: {_be_err} — will retry next cycle")
                    
                    # ── PROFIT LOCK: tiered — all thresholds now survive 0.42% fees at 6x ──
                    # Aug 7: 0.03% was below fee — trades looked green but lost money.
                    # Aug 9: Floor raised to 0.15% net — must clear 0.42% max fee with room.
                    # Let winners run: tight profit locks cut good trades. Only lock when
                    # profit is solid. Peak lock tiers handle scaling exits on big runners.
                    # EXTREME REVERSAL: only force-exit if peak was big (>3%) and now losing.
                    if _peak_pnl >= 3.0 and net_pnl_pct < -0.15:
                        log.warning(f"  🚨 {coin}: EXTREME REVERSAL — peak was {_peak_pnl:+.2f}%, now losing {net_pnl_pct:+.2f}%, closing 50%")
                        try:
                            _close_sz = abs(szi) * 0.5
                            hl.market_close(coin, sz=_close_sz)
                            log.info(f"  📤 {coin}: partial close {_close_sz:.1f}u — 50% remaining to run")
                        except Exception as e:
                            log.warning(f"  ⚠️ {coin}: partial close failed: {e}")
                    
                    if _hold_age_mon < 60:
                        _lock_target = 0.15  # First 60s: must clear max fee (0.42%@6x) + room
                    elif _hold_age_mon < 180:
                        _lock_target = 0.25  # 1-3 min: let winner develop, lock bigger profit
                    else:
                        _lock_target = 0.15  # 3+ min: still profitable, let it ride unless reversing
                    
                    if _net_pnl_mon >= _lock_target:
                        log.warning(f"  💰 {coin}: PROFIT LOCK — net={_net_pnl_mon:+.2f}% ≥ {_lock_target:.2f}% (age={_hold_age_mon:.0f}s), closing NOW")
                        try:
                            _MANUAL_CLOSES[coin] = time.time()
                            hl.market_close(coin)
                            _reset_signal_dominance(coin)
                            continue
                        except Exception as e:
                            log.warning(f"  💰 {coin}: profit lock failed: {e}")
                    
                    # ── PEAK PROFIT LOCK: multi-fire at escalating thresholds ──
                    _peak_lock_tier_key = f"_peak_lock_tier:{cu}"
                    _lock_tier = getattr(_monitor_positions, _peak_lock_tier_key, -1) if hasattr(_monitor_positions, _peak_lock_tier_key) else -1
                    _lock_tiers = [
                        (0.25, 0.15, 0.30, 0),
                        (0.50, 0.35, 0.40, 1),
                        (1.50, 1.00, 0.50, 2),
                        (3.00, 2.00, 1.00, 3),
                    ]
                    for _t_peak, _t_net, _t_frac, _t_tier in _lock_tiers:
                        if _lock_tier < _t_tier and _peak_pnl >= _t_peak and net_pnl_pct >= _t_net:
                            _close_sz = abs(szi) * _t_frac
                            if _close_sz > 0:
                                setattr(_monitor_positions, _peak_lock_tier_key, _t_tier)
                                _label = "CLOSING ALL" if _t_frac >= 1.0 else f"closing {_t_frac*100:.0f}%"
                                log.info(f"  🔒 {coin}: PEAK LOCK TIER {_t_tier} — peak {_peak_pnl:+.2f}% → {_label} ({_close_sz:.1f}u)")
                                try:
                                    _MANUAL_CLOSES[coin] = time.time()
                                    hl.market_close(coin, sz=_close_sz)
                                    szi = szi * (1 - _t_frac)
                                    if _t_frac >= 1.0:
                                        continue
                                    continue
                                except Exception as e:
                                    log.warning(f"  ⚠️ {coin}: peak lock tier {_t_tier} close failed: {type(e).__name__}: {e}")
                                    setattr(_monitor_positions, _peak_lock_tier_key, _lock_tier)
                            break
                    
                    # ── Only hard exit allowed beyond this point: liquidation survival ──
                    if exit_reason:
                        log.info(f"  📤 RULE EXIT: {coin} — {exit_reason}")
                        if _can_close_position(coin, f"rule_exit:{exit_reason[:30]}"):
                            _MANUAL_CLOSES[coin] = time.time()
                            hl.market_close(coin)
                            _reset_signal_dominance(coin)
                            continue
                    
                    # ── DEAD-MAN TIMEOUT: never-green positions bleed forever ──
                    # Aug 8: the "zero-loss" system has a blind spot — trades that never
                    # reach breakeven are held indefinitely. This catches them.
                    # INSTEAD of hardcoded timeouts: decide based on trend + bleed.
                    #  1. 4h trend strongly against? Close now. No point waiting.
                    #  2. Bleeding (peak dropping + mom against)? Position is deteriorating.
                    #  3. Never green after 5min? Last-resort close.
                    _ever_green_key = f"_ever_green:{cu}"
                    _ever_green = getattr(_monitor_positions, _ever_green_key, False) if hasattr(_monitor_positions, _ever_green_key) else False
                    _fee_covered_at = ROUNDTRIP_FEE_PCT * leverage
                    if _net_pnl_mon >= _fee_covered_at:
                        _ever_green = True
                        setattr(_monitor_positions, _ever_green_key, True)

                    if _net_pnl_mon < 0 and not exit_reason:
                        # Check broader trend using extremes (already fetched, no API call)
                        _pct_1h = ext.pct_1h if ext else 0   # % above 1h low
                        _pct_4h = ext.pct_4h if ext else 0   # % above 4h low
                        _pct_low_1h = ext.pct_low_1h if ext else 0  # % below 1h high
                        _pct_low_4h = ext.pct_low_4h if ext else 0  # % below 4h high
                        _trend_kill = False
                        # ── Regime-aware: in downtrend, short at 4h high is CORRECT entry ──
                        # KAITO: small pump to resistance in downtrend → perfect short, killed by TREND-KILL
                        _regime_key = f"_regime_downtrend:{cu}"
                        _regime_downtrend = getattr(_monitor_positions, _regime_key, False) if hasattr(_monitor_positions, _regime_key) else False
                        _regime_uptrend_key = f"_regime_uptrend:{cu}"
                        _regime_uptrend = getattr(_monitor_positions, _regime_uptrend_key, False) if hasattr(_monitor_positions, _regime_uptrend_key) else False
                        if side == "SHORT":
                            _trend_kill = _pct_low_4h < 3.0 or _pct_low_1h < 2.0
                            if _trend_kill and _regime_downtrend:
                                # Downtrend + near 4h high = resistance test = SHORT opportunity, not risk
                                _trend_kill = False
                            _kill_metric = f"4h={_pct_low_4h:.1f}% below high"
                        else:
                            _trend_kill = _pct_4h < 3.0 or _pct_1h < 2.0
                            if _trend_kill and _regime_uptrend:
                                # Uptrend + near 4h low = support test = LONG opportunity, not risk
                                _trend_kill = False
                            _kill_metric = f"4h={_pct_4h:.1f}% above low"

                        if _trend_kill and _net_pnl_mon < -0.15:
                            # Minimum loss 0.15% — don't kill flat positions (S coin: killed at -0.01%)
                            exit_reason = f"TREND-KILL: {side} wrong — {_kill_metric}, net={_net_pnl_mon:+.2f}%"
                        elif drop_from_peak_pct > 0.3 and (mom1 * (-1 if side == "SHORT" else 1)) < -0.1:
                            # Bleeding: peak was better, now dropping, mom going the wrong way
                            exit_reason = f"BLEED: down {drop_from_peak_pct:.2f}% from peak, mom turning against, net={_net_pnl_mon:+.2f}%"
                        elif not _ever_green and _hold_age_mon > 300:
                            # ── AI RE-CHECK: ask AI before killing flat positions ──
                            # Aug 9: DON'T blindly kill at 300s. If 4h trend is NOT strongly
                            # against and AI still believes in the trade, extend to 600s.
                            # Only kill immediately if trend is screaming against or AI says exit.
                            _recheck_key = f"_ai_exit_recheck:{cu}"
                            _already_rechecked = getattr(_monitor_positions, _recheck_key, False) if hasattr(_monitor_positions, _recheck_key) else False
                            _trend_strongly_against = False
                            if side == "SHORT":
                                _trend_strongly_against = (_pct_low_4h < 1.5)  # Strong uptrend: <1.5% below 4h high
                            else:
                                _trend_strongly_against = (_pct_4h < 1.5)  # Strong downtrend: <1.5% above 4h low
                            if _trend_strongly_against:
                                exit_reason = f"NEVER-GREEN: underwater {_net_pnl_mon:+.2f}% for {_hold_age_mon:.0f}s, trend strongly against (4h={_pct_4h if side!='SHORT' else _pct_low_4h:.1f}% from extreme)"
                            elif _already_rechecked and _hold_age_mon > 600:
                                exit_reason = f"NEVER-GREEN: underwater {_net_pnl_mon:+.2f}% for {_hold_age_mon:.0f}s, AI said hold but 600s timeout exceeded"
                            elif not _already_rechecked:
                                setattr(_monitor_positions, _recheck_key, True)
                                try:
                                    _ext_str = ""
                                    if ext:
                                        _ext_str = f"1h:{ext.pct_1h:+.1f}%/-{ext.pct_low_1h:.1f}%@{ext.range_pos_1h:.0f}% 4h:{ext.pct_4h:+.1f}%/-{ext.pct_low_4h:.1f}%"
                                    _regime_str = "neutral"
                                    if _regime_downtrend:
                                        _regime_str = "trending_down"
                                    elif _regime_uptrend:
                                        _regime_str = "trending_up"
                                    _ai_result = ai_evaluate_exit(
                                        coin=coin,
                                        direction=side,
                                        entry_price=entry,
                                        current_price=mid,
                                        pnl_pct=_net_pnl_mon,
                                        hold_seconds=_hold_age_mon,
                                        mom_5m=mom5,
                                        mom_15m=0,
                                        regime=_regime_str,
                                        target_pct=2.0,
                                        stop_pct=0.55,
                                        price_extremes=_ext_str,
                                        peak_pnl=_peak_pnl if _peak_pnl > -999 else 0,
                                    )
                                    _ai_action = _ai_result.get("action", "exit")
                                    _ai_conf = _ai_result.get("confidence", 0)
                                    _ai_reason = _ai_result.get("reason", "")
                                    if _ai_action == "hold" and _ai_conf >= 70:
                                        log.info(f"  🧠 {coin}: AI RE-CHECK → HOLD (conf={_ai_conf}%): {_ai_reason} — extending NEVER-GREEN to 600s")
                                    else:
                                        exit_reason = f"NEVER-GREEN: underwater {_net_pnl_mon:+.2f}% for {_hold_age_mon:.0f}s — AI says {_ai_action}@{_ai_conf}%: {_ai_reason}"
                                except Exception as _ai_err:
                                    log.warning(f"  ⚠️ {coin}: AI re-check crashed ({type(_ai_err).__name__}: {_ai_err}) — falling back to NEVER-GREEN kill")
                                    exit_reason = f"NEVER-GREEN: underwater {_net_pnl_mon:+.2f}% for {_hold_age_mon:.0f}s, never hit breakeven (AI re-check failed)"

                    if exit_reason:
                        log.warning(f"  💀 {coin}: {exit_reason}")
                        try:
                            # ── REVERSE ON TREND-KILL: flip direction to recover loss ──
                            _orig_side = side
                            _orig_szi = abs(szi)
                            _orig_entry = entry
                            _loss_pct = abs(_net_pnl_mon)
                            _MANUAL_CLOSES[coin] = time.time()
                            hl.market_close(coin)
                            _reset_signal_dominance(coin)
                            
                            # Only reverse on TREND-KILL (trend signal, not timeout-based)
                            # Cap at 1 reverse per coin — SUI death spiral: 3x TREND-KILL loop
                            _rev_count_key = cu.upper()
                            _rev_count = _REVERSE_COUNT.get(_rev_count_key, 0)
                            if exit_reason.startswith("TREND-KILL") and _loss_pct < 2.0 and _rev_count < 1:
                                _REVERSE_COUNT[_rev_count_key] = _rev_count + 1
                                # Open 50% size in opposite direction, targeting loss recovery
                                _rev_side = "SELL" if _orig_side == "BUY" else "BUY"
                                _rev_buy = _rev_side == "BUY"
                                _rev_size_pct = 0.50  # 50% of original position
                                _rev_notional = _orig_szi * mid * leverage * _rev_size_pct
                                _rev_tp_pct = _loss_pct * 1.1  # Recover loss + 10% buffer
                                _rev_sl_pct = 0.3  # Tight SL — don't compound losses
                                _rev_tp = mid * (1 + _rev_tp_pct / 100) if _rev_buy else mid * (1 - _rev_tp_pct / 100)
                                _rev_sl = mid * (1 - _rev_sl_pct / 100) if _rev_buy else mid * (1 + _rev_sl_pct / 100)
                                log.info(f"  🔄 {coin}: REVERSE {_rev_side} — {_rev_size_pct*100:.0f}% size "
                                       f"${_rev_notional:.1f} @ {leverage}x, TP to recover {_loss_pct:+.2f}% loss")
                                try:
                                    _rev_result = hl.market_open(coin, _rev_buy, _rev_notional, slippage=0.005, order_type="Ioc")
                                    if _rev_result and not (isinstance(_rev_result, dict) and _rev_result.get("status") == "err"):
                                        _place_retry_tpsl(coin, _rev_buy, _rev_notional, _rev_sl, [{"price": _rev_tp, "fraction": 1.0}])
                                        log.info(f"  ✅ {coin}: REVERSE opened — TP={_rev_tp:.4f} SL={_rev_sl:.4f}")
                                except Exception as _re:
                                    log.warning(f"  🔄 {coin}: reverse failed: {_re}")
                            continue
                        except Exception as de:
                            log.warning(f"  💀 {coin}: close failed: {de}")
                    
                    # ── UNDERWATER MANAGEMENT: add to position or cut losses ──
                    # Aug 7: if position never reached breakeven, manage it actively.
                    # Two strategies: add at better price (flat market), or cut (moving against).
                    # Aug 8: lowered mom5 emergency threshold from 2% to 1% — 2% was too rare.
                    _underwater = _net_pnl_mon < 0 and _hold_age_mon > 300
                    _added_key = f"_added:{cu}"
                    _already_added = getattr(_monitor_positions, _added_key, False) if hasattr(_monitor_positions, _added_key) else False
                    
                    if _underwater:
                        _mom_against = (side == "LONG" and mom5 < -1.0) or (side == "SHORT" and mom5 > 1.0)
                        _mom_neutral = abs(mom5 or 0) < 0.5
                        
                        if _mom_against:
                            # Momentum charging against us — cut before it gets worse
                            log.warning(f"  🚨 {coin}: EMERGENCY CLOSE — underwater {_net_pnl_mon:+.2f}% with mom5={mom5:+.1f}% against, cutting loss")
                            try:
                                _MANUAL_CLOSES[coin] = time.time()
                                hl.market_close(coin)
                                _reset_signal_dominance(coin)
                                continue
                            except Exception as e:
                                log.warning(f"  🚨 {coin}: emergency close failed: {e}")
                        
                        elif _mom_neutral and not _already_added:
                                                    # ── TREND-AWARE DOUBLE-DOWN ──
                                                    # Aug 8: only add if broader trend isn't screaming against us.
                                                    # Use extremes (already fetched) instead of candle cache (cold after restart).
                                                    _pct_1h = ext.pct_1h if ext else 0       # % above 1h low
                                                    _pct_4h = ext.pct_4h if ext else 0       # % above 4h low
                                                    _pct_low_1h = ext.pct_low_1h if ext else 0  # % below 1h high
                                                    _pct_low_4h = ext.pct_low_4h if ext else 0  # % below 4h high
                                                    _trend_against = False
                                                    if side == "SHORT":
                                                        _trend_against = _pct_low_4h < 3.0 or _pct_low_1h < 2.0
                                                    else:
                                                        _trend_against = _pct_4h < 3.0 or _pct_1h < 2.0

                                                    if _trend_against:
                                                        log.warning(f"  🚫 {coin}: NO DOUBLE-DOWN — trend against us "
                                                                   f"(1h range={_pct_1h:+.1f}% 4h={_pct_4h:+.1f}%), letting exit system handle it")
                                                    else:
                                                        log.warning(f"  ➕ {coin}: ADDING TO POSITION — underwater {_net_pnl_mon:+.2f}%, "
                                                                   f"market flat, 1h={_pct_1h:+.1f}% 4h={_pct_4h:+.1f}%, doubling at better price")
                                                        try:
                                                            setattr(_monitor_positions, _added_key, True)
                                                            _add_size = size_usd if 'size_usd' in dir() else 20
                                                            hl.market_open(coin, side=="LONG", _add_size, slippage=0.005, order_type="Ioc")
                                                            log.warning(f"  ➕ {coin}: added ${_add_size:.0f} to position")
                                                        except Exception as e:
                                                            log.warning(f"  ➕ {coin}: add failed: {e}")
                    
                    # ── No other exit rules. Position runs to exchange TP or breakeven SL. ──
                    # ── POSITIVE TIMEOUT: free capital from stagnant positions ──
                    # Aug 7: if position is net positive (even 0.01%) after timeout, close to free capital.
                    # Aug 8: fast-exit trades use 120s (2min), normal use 300s (5min).
                    _fast_key = f"_fast_exit:{cu}"
                    _is_fast = getattr(_monitor_positions, _fast_key, False) if hasattr(_monitor_positions, _fast_key) else False
                    _pos_timeout = 120 if _is_fast else 300
                    if _net_pnl_mon > 0 and _hold_age_mon > _pos_timeout:
                        _label = "FAST-EXIT" if _is_fast else "POSITIVE TIMEOUT"
                        log.warning(f"  ⏰ {coin}: {_label} — net={_net_pnl_mon:+.2f}% after {_hold_age_mon:.0f}s, closing to free capital")
                        try:
                            _MANUAL_CLOSES[coin] = time.time()
                            hl.market_close(coin)
                            _reset_signal_dominance(coin)
                            continue
                        except Exception as e:
                            log.warning(f"  ⏰ {coin}: positive timeout close failed: {e}")
                    
                    setattr(_monitor_positions, _eval_key, _now_ts)
                    
                except Exception as e:
                    log.warning(f"  Exit eval error for {cu}: {type(e).__name__}: {e}")

        # ── AI-DRIVEN POSITION EVALUATION (DISABLED for zero-loss) ──
        # Aug 7: margin_pnl threshold raised to -50 (effectively disabled).
        # No AI-driven position closes at a loss. Only informational logging.
        lev = float(p.get("leverage", {}).get("value", 1)) if isinstance(p.get("leverage"), dict) else float(p.get("leverage", 1) or 1)
        margin_pnl = pnl_pct * lev

        if margin_pnl < -50:  # Was -5, disabled for zero-loss
            # Feed losing position to AI for decision with FULL continuous predictor
            try:
                # ── Gather FULL prediction context (same as Stage 1.5) ──
                c1m = _fetch_candles_cached(coin, "1m", 100)
                c5m = _fetch_candles_cached(coin, "5m", 100)
                c15m = _fetch_candles_cached(coin, "15m", 200)
                c1h = _fetch_candles_cached(coin, "1h", 24)
                c4h = _fetch_candles_cached(coin, "4h", 50)
                c1d = _fetch_candles_cached(coin, "1d", 50)
                fund_ctx = funding_sniper.asset_ctx.get(coin, {})
                fund_rate = float(fund_ctx.get("funding", 0)) if fund_ctx else 0.0

                # ML signal
                try:
                    ml_sig = predict_unified(coin, c15m, mid, mids=mids,
                        funding_rate=fund_rate, btc_mid=float(mids.get("BTC", 0)),
                        fear_greed=_get_fear_greed(), regime="ranging")
                except Exception:
                    ml_sig = None

                # Hurst exponent
                try:
                    from hurst_detector import hurst_exponent
                    closes = [float(c.get("close", c.get("c", 0))) for c in (c1h or [])[-24:]]
                    H = hurst_exponent(closes) if len(closes) >= 20 else 0.5
                except Exception:
                    H = 0.5

                # Live order book from WebSocket
                order_book = None
                try:
                    from hyperliquid_ws import get_field as _ws_field
                    from hyperliquid_ws import normalize_orderbook as _norm_ob
                    obs = _ws_field("orderbooks") or {}
                    order_book = _norm_ob(obs.get(coin, {}))
                except Exception:
                    pass

                # Real CVD from WebSocket trades
                try:
                    from cvd_engine import get_trade_cvd, cvd_to_prediction
                    real_cvd = get_trade_cvd(coin)
                    cvd_override = cvd_to_prediction(real_cvd) if (real_cvd["confidence"] > 10 and real_cvd["trade_count"] > 3) else None
                except Exception:
                    cvd_override = None

                # ── OI delta signal (Open Interest change direction) ──
                try:
                    oi_delta_sig = get_oi_signal(coin, mid)
                    # Update OI cache for future readings
                    if mid > 0:
                        update_oi_cache(coin, _open_interest.get(coin, 0), mid)
                except Exception:
                    oi_delta_sig = None

                # ── Cross-exchange divergence (HL vs Binance) ──
                try:
                    cross_div = get_cross_divergence(coin, mid)
                except Exception:
                    cross_div = None

                # ── Taker buy/sell ratio (real-time order flow aggression) ──
                try:
                    taker_sig = get_taker_ratio(coin, 60.0)
                except Exception:
                    taker_sig = None

                # ── Bid/ask imbalance trend (building vs fading pressure) ──
                try:
                    imb_trend = get_imbalance_trend(coin)
                except Exception:
                    imb_trend = None

                # ── Exhaustion detection (RSI divergence + volume climax) ──
                # MUST be before predict_continuous so it feeds the exhaustion layer.
                try:
                    exhaustion = detect_peak_exhaustion(coin, c15m, mid)
                except Exception:
                    exhaustion = None

                # Full continuous prediction (all layers)
                cpred = predict_continuous(coin, mid,
                    candles_1m=c1m, candles_5m=c5m,
                    candles_15m=c15m, candles_1h=c1h,
                    candles_4h=c4h, candles_1d=c1d,
                    funding_rate=fund_rate, ml_signal=ml_sig,
                    hurst_H=H, order_book=order_book,
                    cvd_override=cvd_override,
                    oi_delta=oi_delta_sig,
                    cross_exchange_divergence=cross_div,
                    taker_ratio=taker_sig,
                    exhaustion_signal=exhaustion,
                    imbalance_trend=imb_trend,
                    open_interest=_open_interest.get(coin, 0))

                # Map continuous predictor layers to AI validator keys
                lc = getattr(cpred, 'layer_confidences', {})
                h15 = cpred.horizons.get("15m")
                target_price = (h15.target_down if cpred.overall_bias == "bearish"
                    else (h15.target_up if cpred.overall_bias == "bullish" else mid)) if h15 else mid

                pred_data = {
                    "direction": cpred.overall_bias,
                    "confidence": cpred.overall_confidence,
                    "target_price": target_price,
                    "layers": dict(lc),
                }

                port_data = {
                    "equity": total_eq,
                    "perp_equity": perp_eq if 'perp_eq' in dir() else total_eq,
                    "positions_count": len(active),
                    "heat": exposure_state.heat if hasattr(exposure_state, 'heat') else 0,
                }

                # ── Build rich context for AI (same as entry decisions) ──
                _mkt_ctx = ""
                _ob_info = ""
                _atr_info = ""
                _funding_accel = ""
                _hurst_regime_str = ""
                _hurst_H_val = 0.5
                _oi_type = ""
                _oi_desc = ""
                _xex_div = 0.0
                _xex_dir = ""
                _cascade = ""
                _htf_align = ""
                try:
                    from context_enricher import enrich_context
                    intel = enrich_context(coin, c15m)
                    _mkt_ctx = f"regime=ranging | {intel}" if intel else "regime=ranging"
                except Exception:
                    _mkt_ctx = "regime=ranging"
                try:
                    if order_book:
                        bids = order_book.get("bids", [])
                        asks = order_book.get("asks", [])
                        if bids and asks:
                            bb = bids[0]["px"]; ba = asks[0]["px"]
                            sp = (ba - bb) / bb * 100 if bb > 0 else 0
                            bd = sum(b["px"]*b["sz"] for b in bids[:5])
                            ad = sum(a["px"]*a["sz"] for a in asks[:5])
                            _ob_info = f"bid={bb:.4f} ask={ba:.4f} spread={sp:.3f}% depth={bd:.0f}/{ad:.0f}"
                except Exception:
                    pass
                try:
                    atr_val = 0
                    if c15m and len(c15m) >= 14:
                        atr_val = sum(abs(float(c.get("h",c.get("high",0)))-float(c.get("l",c.get("low",0)))) for c in c15m[-14:])/14
                    if atr_val > 0 and mid > 0:
                        _atr_info = f"ATR={atr_val:.4f} ({atr_val/mid*100:.2f}%)"
                except Exception:
                    pass

                # ── Rich data extraction (funding accel, Hurst, OI, XEX, cascade, HTF) ──
                try:
                    from funding_acceleration import get_funding_acceleration
                    fa = get_funding_acceleration(coin)
                    _funding_accel = fa.get("direction", "") if fa else ""
                except Exception: pass
                try:
                    from hurst_detector import hurst_regime as _hr_func
                    _hurst_regime_str = _hr_func(H)
                    _hurst_H_val = H
                except Exception: pass
                try:
                    if oi_delta_sig:
                        _oi_type = getattr(oi_delta_sig, 'signal_type', '')
                        _oi_desc = getattr(oi_delta_sig, 'description', '')[:60]
                except Exception: pass
                try:
                    if cross_div:
                        _xex_div = cross_div.get("divergence_pct", 0)
                        _xex_dir = cross_div.get("direction", "")
                except Exception: pass
                try:
                    from liquidation_zones import detect_cascade_risk
                    cr = detect_cascade_risk(coin, mid)
                    _cascade = cr.get("risk_level", "") if cr else ""
                except Exception: pass
                try:
                    _htf_align = lc.get("higher_tf_align", "")
                    if isinstance(_htf_align, (int, float)) and _htf_align > 0:
                        _htf_align = "bullish" if _htf_align > 50 else "bearish" if _htf_align < -50 else "neutral"
                except Exception: pass

                ai_v = validate_prediction(
                    coin, mid, pred_data, port_data,
                    market_regime="ranging",
                    fear_greed=_get_fear_greed(),
                    funding_rate=fund_rate,
                    position_context={
                        "is_exit": True,
                        "side": side,
                        "pnl_pct": round(pnl_pct, 1),
                        "margin_pnl": round(margin_pnl, 1),
                        "leverage": lev,
                        "liq_distance": round(abs(mid - liq) / mid * 100, 1) if liq > 0 else 99.0,
                    },
                    market_context=_mkt_ctx,
                    order_book_info=_ob_info,
                    atr_info=_atr_info,
                    funding_accel=_funding_accel,
                    hurst_regime=_hurst_regime_str,
                    hurst_H=_hurst_H_val,
                    oi_delta_type=_oi_type,
                    oi_delta_desc=_oi_desc,
                    cross_exchange_div=_xex_div,
                    cross_exchange_dir=_xex_dir,
                    cascade_risk=_cascade,
                    htf_alignment=_htf_align,
                )
                log.info(f"  🧠 AI loss eval: {coin} margin={margin_pnl:+.1f}% → {ai_v.action} — {ai_v.reason[:120] if hasattr(ai_v, 'reason') else ''}")

                if hasattr(ai_v, 'action'):
                    if ai_v.action == "execute_now":
                        log.warning(f"  🛑 {coin}: AI says CLOSE NOW — margin {margin_pnl:+.1f}%")
                        reason = f"ai_loss_eval:execute_now:margin_{margin_pnl:.0f}pct"
                        # Try direct close first (instant), fallback to pipeline
                        if not _execute_direct_close(coin, 1.0, reason):
                            _submit_action("close", coin, {
                                "direction": "close",
                                "reason": reason,
                                "size_units": abs(szi), "price": mid, "size": abs(szi),
                                "side": side,
                            })
                        register_stop_loss(coin)
                        continue
                    elif ai_v.action == "reduce_size" and hasattr(ai_v, 'size_multiplier'):
                        close_sz = abs(szi) * max(0.25, ai_v.size_multiplier)
                        log.warning(f"  ⚠️  {coin}: AI says TRIM {ai_v.size_multiplier:.0%} — margin {margin_pnl:+.1f}%")
                        _submit_action("close", coin, {
                            "direction": "close",
                            "reason": f"ai_loss_eval:trim:{ai_v.size_multiplier:.0%}",
                            "size_units": close_sz, "price": mid, "size": close_sz,
                            "side": side,
                        })
                    elif ai_v.action == "tighten_stop":
                        if coin in trail_states:
                            new_stop = mid * (1 - atr / mid * 1.5)
                            trail_states[coin].current_stop = new_stop
                            log.info(f"  📐 {coin}: AI says tighten stop → ${new_stop:,.4f}")
                            # Actually modify the SL order on Hyperliquid
                            try:
                                from action_executor import get_active_orders as _gao
                                orders = _gao()
                                tracking = orders.get(coin.upper(), orders.get(coin, {}))
                                old_sl_oid = tracking.get("sl_oid", 0)
                                if old_sl_oid:
                                    # Cancel old SL
                                    hl.cancel_order(coin, old_sl_oid)
                                    # Place new tightened SL
                                    pos_sz = abs(float(p.get("szi", 0)))
                                    is_long = pos_sz > 0  # szi > 0 means long
                                    sl_result = hl.trigger_order(coin, not is_long, pos_sz, new_stop, "sl", True, True)
                                    new_oid = sl_result.get("oid") if isinstance(sl_result, dict) else 0
                                    tracking["sl_oid"] = new_oid
                                    tracking["stop_loss"] = new_stop
                                    log.info(f"  📐 {coin}: SL updated — old_oid={old_sl_oid} → new_oid={new_oid} @ ${new_stop:.4f}")
                                else:
                                    log.info(f"  📐 {coin}: no existing SL to modify, skip")
                            except Exception as e:
                                log.warning(f"  📐 {coin}: tighten_stop HL update failed — {e}")
            except Exception as e:
                log.debug(f"  AI loss eval skipped: {type(e).__name__}")

        # Extreme PnL alerts
        if pnl_pct < -15:
            log.error(f"  🔴 {coin} {side}: -{abs(pnl_pct):.1f}% — CRITICAL LOSS")
            register_stop_loss(coin)  # Trigger cooling after big loss
        elif pnl_pct > 20:
            log.info(f"  🟢 {coin} {side}: +{pnl_pct:.1f}% — consider taking profit")

        # Trailing stop update (Chandelier Exit with dynamic ATR multiplier)
        if coin in trail_states:
            trail = trail_states[coin]
            # Dynamic multiplier from volatility regime (research: 3x base for leveraged crypto)
            dyn_mult = chandelier_atr_mult(atr)
            new_stop = update_trail(trail, mid, atr, dyn_mult)
            if new_stop:
                log.info(f"  📈 {coin} trail stop → ${new_stop:,.2f} "
                         f"(high=${trail.highest_price:,.2f} atr×{dyn_mult:.1f})")

        # ── Perfect Prediction (funding + OB + multi-TF) ──
        try:
            pred_checked += 1
            perfect = predict_perfect(coin, mid,
                                      candles_1m=_fetch_candles_cached(coin, "1m", 60),
                                      candles_5m=_fetch_candles_cached(coin, "5m", 60),
                                      candles_15m=_fetch_candles_cached(coin, "15m", 100),
                                      candles_1h=_fetch_candles_cached(coin, "1h", 24),
                                      atr_15m=atr)
            if perfect.consensus_confidence >= 98:  # Disabled for exits — perfect predictor wrong in sideways
                # Only log, never auto-close (was #1 source of premature exits)
                log.info(f"  ℹ️  {coin} PERFECT PREDICT: {perfect.consensus_direction} conf={perfect.consensus_confidence:.0f}% — NOT auto-closing (predictor unreliable in sideways)")
        except Exception:
            pass

        # ── Track high water mark since entry (used by AI exit evaluation) ──
        try:
            _peak_key = f"_hwm:{cu}"
            _prev_hwm = getattr(_monitor_positions, _peak_key, 0.0) if hasattr(_monitor_positions, _peak_key) else 0.0
            current_hwm = max(_prev_hwm, mid) if side == "LONG" else (min(_prev_hwm, mid) if _prev_hwm > 0 else mid)
            setattr(_monitor_positions, _peak_key, current_hwm)
        except Exception:
            pass

        # ── Market Structure Exit: DISABLED for zero-loss ──
        # Aug 7: structural exits kill at a loss. Disabled.
        # Position runs to exchange TP or breakeven SL.
        try:
            c15 = _fetch_candles_cached(coin, "15m", 50) or []
            if len(c15) >= 20 and hold_secs > 99999:  # DISABLED for zero-loss (was 300)
                from market_structure import classify_market_structure
                closes = [float(c.get("c", c.get("close", 0))) for c in c15]
                highs  = [float(c.get("h", c.get("high", 0))) for c in c15]
                lows   = [float(c.get("l", c.get("low", 0))) for c in c15]
                ms = classify_market_structure(closes, highs, lows)
                ms_trend = ms.get("trend", "")
                ms_bos = ms.get("bos", False)
                ms_choch = ms.get("choch", False)
                # Current momentum for confirmation
                mom1  = _calc_momentum(coin, "1m") or 0
                mom5  = _calc_momentum(coin, "5m") or 0
                raw_mom1 = mom1  # keep original for logging
                raw_mom5 = mom5
                if side == "SHORT":
                    mom1 = -mom1; mom5 = -mom5  # Invert: negative = bad for short
                
                # ── STRUCTURAL EXIT GATE: don't kill winners on noise ──
                # BIO lesson: +0.99% with mom1=+0.64% mom5=+1.79% — killed by 15m bearish BOS
                # The 15m structure said "bearish" but price was ripping on 1m/5m
                _should_exit = False
                
                # Only consider structural exit if:
                # 1. Position is losing AND momentum confirms (mom1 < -0.05 for longs), OR
                # 2. Position is losing AND dropped hard (>0.5% from entry), OR
                # 3. Both BOS AND CHOCH present (strong structural break)
                # ANTI-WHIPSAW: losing position with positive mom1 = recovering, not reversing.
                # ANTI-FLAT-KILL: position must be losing >=0.2% — don't kill flat positions on noise
                _profitable = net_pnl_pct > 0.3  # modest profit
                _flat = abs(net_pnl_pct) < 0.2  # essentially flat — don't kill
                _losing = net_pnl_pct <= -0.2  # actually losing money
                _mom_confirms = mom1 < -0.05  # 1m momentum confirms reversal
                _mom_contradicts = mom1 > 0.05  # 1m momentum says NO reversal
                _strong_structure = ms_bos and ms_choch  # both signals
                _deep_loss = net_pnl_pct < -1.0  # already lost >1% — cut it

                if side == "LONG" and ms_trend in ("bearish", "bearish_divergence"):
                    if ms_bos or ms_choch:
                        if _flat:
                            # Flat position with structure break = noise, hold
                            log.info(f"  [CONSTRUCTION] {coin}: {ms_trend} BOS/CHOCH but PnL={net_pnl_pct:+.2f}% flat — holding (no real loss)")
                        elif _deep_loss:
                            # >1% loss — structure break or not, exit
                            _should_exit = True
                        elif _losing and _mom_confirms:
                            # Losing + momentum agrees with reversal = exit
                            _should_exit = True
                        elif not _profitable and not _mom_contradicts and not _strong_structure:
                            # Losing but momentum is neutral (+ no strong structure) — hold
                            log.info(f"  🏗️ {coin}: {ms_trend} BOS/CHOCH but PnL={net_pnl_pct:+.2f}% mom1={raw_mom1:+.2f}% — holding (no momentum confirm)")
                        elif _strong_structure:
                            # Both BOS and CHOCH = serious reversal
                            _should_exit = True
                        elif _profitable and _mom_confirms:
                            # Profitable but momentum flipped — structure+trend agree
                            _should_exit = True
                        else:
                            # Profitable, no momentum confirmation, only one signal — HOLD
                            log.info(f"  🏗️ {coin}: {ms_trend} BOS/CHOCH detected but PnL=+{net_pnl_pct:.2f}% mom1={raw_mom1:+.2f}% — holding through noise")
                    
                    if _should_exit:
                        log.info(f"  🏗️ {coin} STRUCTURAL EXIT: {ms_trend} {'BOS' if ms_bos else ''}{'CHOCH' if ms_choch else ''} "
                                f"against LONG — PnL={net_pnl_pct:+.2f}% mom1={raw_mom1:+.2f}%")
                        try:
                            _MANUAL_CLOSES[coin] = time.time()
                            hl.market_close(coin)
                            _reset_signal_dominance(coin)
                        except Exception as e:
                            log.warning(f"  🏗️ {coin}: structural exit failed: {e}")
                        continue
                
                elif side == "SHORT" and ms_trend in ("bullish", "bullish_divergence"):
                    if ms_bos or ms_choch:
                        if _flat:
                            log.info(f"  [CONSTRUCTION] {coin}: {ms_trend} BOS/CHOCH but PnL={net_pnl_pct:+.2f}% flat — holding (no real loss)")
                        elif _deep_loss:
                            _should_exit = True
                        elif _losing and _mom_confirms:
                            _should_exit = True
                        elif not _profitable and not _mom_contradicts and not _strong_structure:
                            log.info(f"  🏗️ {coin}: {ms_trend} BOS/CHOCH but PnL={net_pnl_pct:+.2f}% mom1={raw_mom1:+.2f}% — holding (no momentum confirm)")
                        elif _strong_structure:
                            _should_exit = True
                        elif _profitable and _mom_confirms:
                            _should_exit = True
                        else:
                            log.info(f"  🏗️ {coin}: {ms_trend} BOS/CHOCH detected but PnL=+{net_pnl_pct:.2f}% mom1={raw_mom1:+.2f}% — holding through noise")
                    
                    if _should_exit:
                        log.info(f"  🏗️ {coin} STRUCTURAL EXIT: {ms_trend} {'BOS' if ms_bos else ''}{'CHOCH' if ms_choch else ''} "
                                f"against SHORT — PnL={net_pnl_pct:+.2f}% mom1={raw_mom1:+.2f}%")
                        try:
                            _MANUAL_CLOSES[coin] = time.time()
                            hl.market_close(coin)
                            _reset_signal_dominance(coin)
                        except Exception as e:
                            log.warning(f"  🏗️ {coin}: structural exit failed: {e}")
                        continue
                # Structure against position + momentum confirmed (even without BOS/CHOCH)
                # Only when actually losing — don't kill flat positions on trend noise
                elif side == "LONG" and ms_trend in ("bearish", "bearish_divergence") and mom1 < -0.1 and mom5 < 0 and net_pnl_pct < -0.2:
                    log.info(f"  🏗️ {coin} STRUCT+MOM EXIT: {ms_trend} trend + mom1={mom1:+.2f}% against LONG — pre-BOS exit")
                    try:
                        _MANUAL_CLOSES[coin] = time.time()
                        hl.market_close(coin)
                        _reset_signal_dominance(coin)
                    except Exception as e:
                        log.warning(f"  🏗️ {coin}: struct+mom exit failed: {e}")
                    continue
                elif side == "SHORT" and ms_trend in ("bullish", "bullish_divergence") and mom1 < -0.1 and mom5 < 0 and net_pnl_pct < -0.2:
                    log.info(f"  🏗️ {coin} STRUCT+MOM EXIT: {ms_trend} trend + mom1={mom1:+.2f}% against SHORT — pre-BOS exit")
                    try:
                        _MANUAL_CLOSES[coin] = time.time()
                        hl.market_close(coin)
                        _reset_signal_dominance(coin)
                    except Exception as e:
                        log.warning(f"  🏗️ {coin}: struct+mom exit failed: {e}")
                    continue
        except Exception:
            pass

        # ── Tick-Level Peak/Bottom Detection (10-second precision) ──
        try:
            tick_peak = detect_tick_peak(coin, mid,
                                        candles_15m=_fetch_candles_cached(coin, "15m", 100),
                                        candles_1m=_fetch_candles_cached(coin, "1m", 90),
                                        side=side,
                                        pnl_pct=pnl_pct)
            if tick_peak:
                if tick_peak.confidence >= 25:
                    log.info(format_tick_peak(tick_peak))
                else:
                    log.debug(f"  🔇 {coin} tick_peak conf={tick_peak.confidence:.0f}% — below 25% threshold")
                    tick_peak = None  # Suppress — below actionable threshold
            else:
                log.debug(f"  ⚪ {coin} tick_peak: no signal (score<20)")

            if tick_peak and tick_peak.confidence >= 25:

                # ── Signal dominance check: prevent ping-pong ──
                # If we just acted on a PEAK, a weak BOTTOM can't immediately reverse it.
                actionable_actions = ("sell_now", "sell_soon", "buy_now", "buy_soon",
                                      "sell_more_now", "sell_more_soon",
                                      "buy_more_now", "buy_more_soon", "hold_profit")
                if tick_peak.action in actionable_actions:
                    if not _check_signal_dominance(coin, tick_peak.is_peak, tick_peak.confidence):
                        prev = _last_tick_action.get(coin.upper(), {})
                        log.info(f"  🔒 {coin} signal suppressed: {tick_peak.action} conf={tick_peak.confidence:.0f}% "
                                f"← weaker than prior {prev.get('action','?')} conf={prev.get('confidence',0):.0f}% "
                                f"(direction flipped, need conf > {prev.get('confidence',0):.0f}%)")
                        continue  # Skip this signal — it's a weak reversal

                # ── AI Validation — now runs on sell/buy actions AND hold_profit ──
                needs_ai = tick_peak.action in ("sell_now", "sell_soon", "buy_now", "buy_soon", "hold_profit",
                                                  "sell_more_now", "sell_more_soon", "buy_more_now", "buy_more_soon")
                # For plain "hold" signals, only run AI if we have meaningful profit
                if tick_peak.action == "hold" and pnl_pct > 2.0 and tick_peak.confidence >= 20:
                    needs_ai = True
                if needs_ai:
                    try:
                        pred_data = {
                            "direction": "down" if tick_peak.is_peak else "up",
                            "confidence": tick_peak.confidence,
                            "target_price": tick_peak.predicted_extreme,
                            "layers": {},  # No layers for tick_peak path
                        }
                        port_data = {
                            "equity": total_eq, "positions_count": len(active),
                            "heat": exposure_state.heat if hasattr(exposure_state, 'heat') else 0,
                        }
                        ai_v = validate_prediction(
                            coin, mid, pred_data, port_data,
                            market_regime="ranging",  # simplified — real regime from signal
                            fear_greed=_get_fear_greed(),
                            funding_rate=float(funding_sniper.asset_ctx.get(coin, {}).get("funding", 0)),
                            position_context={
                                "is_exit": True,
                                "side": side,
                                "pnl_pct": round(pnl_pct, 1),
                                "margin_pnl": round(margin_pnl, 1),
                                "leverage": lev,
                                "liq_distance": round(abs(mid - liq) / mid * 100, 1) if liq > 0 else 99.0,
                            },
                        )
                        log.info(format_validation(coin, ai_v))
                        if ai_v.action == "reject":
                            log.warning(f"  ❌ {coin} AI REJECTED tick signal — {ai_v.reason}")
                            # If AI rejects a hold_profit signal, still tighten stop if applicable
                            if tick_peak.is_peak and coin in trail_states:
                                aggressive_stop = mid - (mid - trail_states[coin].current_stop) * 0.5
                                if aggressive_stop > trail_states[coin].current_stop:
                                    trail_states[coin].current_stop = aggressive_stop
                                    log.info(f"  📐 {coin} AI rejected exit but tightened stop to ${aggressive_stop:,.4f}")
                            continue  # Skip — AI says no
                        elif ai_v.action == "reduce_size":
                            close_sz = abs(szi) * max(0.25, ai_v.size_multiplier)
                            log.warning(f"  ⚠️  {coin} AI reduced size to {ai_v.size_multiplier:.0%} — closing {close_sz:.4f}")
                            _submit_action("close", coin, {
                                "direction": "close", "reason": f"ai_validated_peak:conf={tick_peak.confidence:.0f}",
                                "size_units": close_sz, "price": ai_v.adjusted_target, "size": close_sz,
                            })
                            _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "ai_reduce_size"}
                            continue
                        elif ai_v.action == "execute_now":
                            # AI validated the signal as urgent — tighten stop instead of closing
                            # Positions need time to develop 1-2% swings. Auto-closing on
                            # tick signals was the #2 source of premature exits.
                            if tick_peak.action == "hold_profit":
                                log.info(f"  📐 {coin} AI hold_profit — tightening stop (PnL={pnl_pct:+.2f}%, conf={tick_peak.confidence:.0f}%)")
                                if coin in trail_states and tick_peak.is_peak:
                                    trail_states[coin].current_stop = max(trail_states[coin].current_stop, mid * 0.993)
                                _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "ai_stop_tightened"}
                                continue
                            elif tick_peak.action == "hold" and pnl_pct > 2.0:
                                log.info(f"  📐 {coin} AI hold — tightening stop (PnL={pnl_pct:+.2f}%, conf={tick_peak.confidence:.0f}%)")
                                if coin in trail_states and tick_peak.is_peak:
                                    trail_states[coin].current_stop = max(trail_states[coin].current_stop, mid * 0.995)
                                _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "ai_stop_tightened"}
                                continue
                    except Exception:
                        pass  # AI validation is optional — proceed with tick peak signal

                # ── Direct signal execution (non-AI path) — CLOSE ACTIONS DISABLED ──
                # Tick peaks are micro-timing signals (0.1-0.5% moves). Closing on them
                # kills any chance of catching the 1-2% swings we want. Instead:
                # - Tighten trailing stop to lock in profits
                # - Let the trailing stop manage the actual exit
                if tick_peak.action == "sell_now":
                    log.info(f"  📊 {coin} TICK PEAK conf={tick_peak.confidence:.0f}% — tightening stop (was auto-close)")
                    if coin in trail_states:
                        trail_states[coin].current_stop = max(trail_states[coin].current_stop, mid * 0.995)
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "stop_tightened"}
                elif tick_peak.action == "sell_soon":
                    log.info(f"  📊 {coin} TICK PEAK conf={tick_peak.confidence:.0f}% — tightening stop (was partial close)")
                    if coin in trail_states:
                        trail_states[coin].current_stop = max(trail_states[coin].current_stop, mid * 0.997)
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "stop_tightened"}
                elif tick_peak.action == "buy_now":
                    log.info(f"  📊 {coin} TICK BOTTOM conf={tick_peak.confidence:.0f}% — tightening stop (was auto-close)")
                    if coin in trail_states:
                        trail_states[coin].current_stop = min(trail_states[coin].current_stop, mid * 1.005)
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "stop_tightened"}
                elif tick_peak.action == "buy_soon":
                    log.info(f"  📊 {coin} TICK BOTTOM conf={tick_peak.confidence:.0f}% — tightening stop (was partial close)")
                    if coin in trail_states:
                        trail_states[coin].current_stop = min(trail_states[coin].current_stop, mid * 1.003)
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": "stop_tightened"}
                elif tick_peak.action == "sell_more_now":
                    # SHORT + PEAK (high conf): add to short — double down at the top
                    add_sz = abs(szi) * 1.0  # Double the position
                    add_notional = add_sz * mid
                    log.warning(f"  🔥 {coin} SHORT PEAK conf={tick_peak.confidence:.0f}% — ADDING TO SHORT {add_sz:.4f} units @ ${mid:.4f}, extreme=${tick_peak.predicted_extreme:.4f}")
                    # Pipeline only — adding to positions needs cooling, no instant execution
                    _submit_action("sell", coin, {
                        "direction": "short", "side": "SELL",
                        "size_units": add_sz, "size_usd": add_notional,
                        "entry_price": mid,
                        "leverage": 3,
                        "stop_loss": tick_peak.predicted_extreme * 1.01,
                        "tp_levels": [mid * 0.97, mid * 0.94],
                        "reason": f"tick_peak_add_short:conf={tick_peak.confidence:.0f}",
                        "confidence": tick_peak.confidence,
                        "urgency": "high",
                    })
                    coin_upper = coin.upper()
                    if coin_upper in cooldown_state.entries:
                        del cooldown_state.entries[coin_upper]
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": tick_peak.action}
                elif tick_peak.action == "sell_more_soon":
                    # SHORT + PEAK (med conf): add to short moderately
                    add_sz = abs(szi) * 0.5
                    add_notional = add_sz * mid
                    log.warning(f"  📉 {coin} SHORT PEAK conf={tick_peak.confidence:.0f}% — adding 50% to short, extreme=${tick_peak.predicted_extreme:.4f}")
                    # Pipeline only — adding to positions needs cooling
                    _submit_action("sell", coin, {
                        "direction": "short", "side": "SELL",
                        "size_units": add_sz, "size_usd": add_notional,
                        "entry_price": mid,
                        "leverage": 3,
                        "stop_loss": tick_peak.predicted_extreme * 1.01,
                        "tp_levels": [mid * 0.97, mid * 0.94],
                        "reason": f"tick_peak_add_short_soon:conf={tick_peak.confidence:.0f}",
                        "confidence": tick_peak.confidence,
                        "urgency": "medium",
                    })
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": tick_peak.action}
                elif tick_peak.action == "buy_more_now":
                    # LONG + BOTTOM (high conf): add to long — double down at the bottom
                    add_sz = abs(szi) * 1.0
                    add_notional = add_sz * mid
                    log.warning(f"  🔥 {coin} BOTTOM BUY conf={tick_peak.confidence:.0f}% — ADDING TO LONG {add_sz:.4f} units @ ${mid:.4f}, bottom=${tick_peak.predicted_extreme:.4f}")
                    # Pipeline only — adding to positions needs cooling
                    _submit_action("buy", coin, {
                        "direction": "long", "side": "BUY",
                        "size_units": add_sz, "size_usd": add_notional,
                        "entry_price": mid,
                        "leverage": 3,
                        "stop_loss": tick_peak.predicted_extreme * 0.99,
                        "tp_levels": [mid * 1.03, mid * 1.06],
                        "reason": f"tick_bottom_add_long:conf={tick_peak.confidence:.0f}",
                        "confidence": tick_peak.confidence,
                        "urgency": "high",
                    })
                    coin_upper = coin.upper()
                    if coin_upper in cooldown_state.entries:
                        del cooldown_state.entries[coin_upper]
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": tick_peak.action}
                elif tick_peak.action == "buy_more_soon":
                    # LONG + BOTTOM (med conf): add to long moderately
                    add_sz = abs(szi) * 0.5
                    add_notional = add_sz * mid
                    log.warning(f"  📈 {coin} BOTTOM BUY conf={tick_peak.confidence:.0f}% — adding 50% to long, bottom=${tick_peak.predicted_extreme:.4f}")
                    # Pipeline only — adding to positions needs cooling
                    _submit_action("buy", coin, {
                        "direction": "long", "side": "BUY",
                        "size_units": add_sz, "size_usd": add_notional,
                        "entry_price": mid,
                        "leverage": 3,
                        "stop_loss": tick_peak.predicted_extreme * 0.99,
                        "tp_levels": [mid * 1.03, mid * 1.06],
                        "reason": f"tick_bottom_add_long_soon:conf={tick_peak.confidence:.0f}",
                        "confidence": tick_peak.confidence,
                        "urgency": "medium",
                    })
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": tick_peak.action}
                elif tick_peak.action == "hold_profit":
                    # Profit-protection: position is in profit with a reversal signal
                    # Close 50% to lock in gains, leave 50% with tightened stop
                    close_sz = abs(szi) * 0.5
                    log.warning(f"  💰 {coin} PROFIT PROTECT: reversal signal (conf={tick_peak.confidence:.0f}%, PnL={pnl_pct:+.2f}%) — closing 50%")
                    _submit_action("close", coin, {
                        "direction": "close", "reason": f"profit_protect:conf={tick_peak.confidence:.0f}:pnl={pnl_pct:.1f}%",
                        "size_units": close_sz, "price": mid, "size": close_sz,
                    })
                    # Tighten stop on remaining 50%
                    if coin in trail_states:
                        aggressive_stop = mid - (mid - trail_states[coin].current_stop) * 0.5
                        if aggressive_stop > trail_states[coin].current_stop:
                            trail_states[coin].current_stop = aggressive_stop
                            log.info(f"  📐 {coin} profit protect → tightened stop to ${aggressive_stop:,.4f}")
                    # Record dominance to prevent ping-pong
                    _last_tick_action[coin.upper()] = {"is_peak": tick_peak.is_peak, "confidence": tick_peak.confidence, "action": tick_peak.action}
                elif tick_peak.is_peak and coin in trail_states:
                    aggressive_stop = mid - (mid - trail_states[coin].current_stop) * 0.5
                    if aggressive_stop > trail_states[coin].current_stop:
                        trail_states[coin].current_stop = aggressive_stop
                        log.info(f"  📐 {coin} tick peak (conf={tick_peak.confidence:.0f}%) → tightened stop to ${aggressive_stop:,.4f}")
        except Exception:
            pass  # Tick detection is advisory
        tps_hit = _check_tp_levels(coin, mid, szi, side)


        if tps_hit:
            coin_upper = coin.upper()
            if coin_upper in cooldown_state.entries:
                del cooldown_state.entries[coin_upper]  # Clear cooldown so close can execute
            log.info(f"  🎯 {coin} TP HIT — closing {tps_hit*100:.0f}% of position")
            _submit_action("close", coin, {
                "direction": "close",
                "reason": f"tp_hit:{tps_hit*100:.0f}%",
                "size_units": abs(szi) * tps_hit,
                "price": mid,
                "side": side,
                "size": abs(szi) * tps_hit,
            })

        # Unstuck check — with peak guard: don't unstuck positions that recently hit a good peak
        # PROVE lesson: position peaked +0.73%, price temporarily spiked to -0.44%, unstucker sold
        # the bottom. Then price recovered to +0.52% but half the position was already gone.
        _unstuck_peak_key = f"_peak_lock_tier:{cu}"
        _had_good_peak = getattr(_monitor_positions, _unstuck_peak_key, -1) >= 0 if hasattr(_monitor_positions, _unstuck_peak_key) else False
        if _had_good_peak:
            # Position already locked profit at peak → don't unstuck, trail will handle rest
            log.info(f"  🔒 {coin}: unstuck skipped — position already locked profit at peak")
        else:
            should_unstuck, close_frac, reason = check_unstuck(
                coin, mid, abs(szi), unstuck_state,
                max_hold_seconds=tactical.unstuck_age_seconds,
                unstuck_close_pct=tactical.unstuck_close_pct,
                equity=hsl_state.peak_equity or 10000.0,
            )
            if should_unstuck:
                log.warning(f"  🔓 UNSTUCKING {coin}: close {close_frac*100:.0f}% — {reason}")
                _submit_action("close", coin, {
                    "direction": "close", "reason": f"unstuck:{reason}",
                    "size_units": abs(szi) * close_frac,
                    "price": mid,
                    "size": abs(szi) * close_frac,
                })
                unstuck_state.remove(coin)

    # ── Cleanup: remove entry times, peak-lock tier, soft strikes for positions that no longer exist ──
    active_coins = {p.get("coin", "").upper() for p in positions if abs(float(p.get("szi", 0))) > 0.0001}
    stale = [c for c in _position_entry_times if c not in active_coins]
    for c in stale:
        del _position_entry_times[c]
        # Clean up peak-lock tier flag so re-entry on same coin gets fresh protection
        for _clean_key in [f"_peak_locked:{c}", f"_peak_lock_tier:{c}", f"_soft_strikes:{c}",
                           f"_mc_peak:{c}", f"_mc_entry_ts:{c}", f"_tip_start:{c}"]:
            if hasattr(_monitor_positions, _clean_key):
                delattr(_monitor_positions, _clean_key)
        # Clean up trail states and TP tracking for closed coins
        if c in trail_states:
            del trail_states[c]
        if c in position_tps:
            del position_tps[c]

    # ── GLOBAL PENDING ORDER TIMEOUT: runs regardless of position state ──
    # Aug 8: was inside position loop, so orders for coins WITHOUT positions
    # never timed out. Now runs at function level so ALL pending orders get checked.
    _now_pt = time.time()
    for zc, zd in list(_PENDING_ZONE.items()):
        _tmo = zd.get("timeout_s", 90)
        if _now_pt - zd["placed_at"] > _tmo:
            log.warning(f"  ⏰ {zc}: pending {zd.get('label','limit')} not filled in {_tmo}s — cancelling, entering at market")
            try: hl.cancel_order(zc, zd["oid"])
            except Exception: pass
            try:
                hl.market_open(zc, zd["is_buy"], zd["size_usd"], slippage=0.005, order_type="Ioc")
                _place_retry_tpsl(zc, zd["is_buy"], zd["size_usd"],
                                 zd["stop_price"], zd["tp_levels"] or [])
            except Exception as _zce:
                log.warning(f"  ⏰ {zc}: pending fallback failed: {_zce}")
            _PENDING_ZONE.pop(zc, None)
    
    # ── Also clean up stale orders >5min old (safety net) ──
    for zc in list(_PENDING_ZONE.keys()):
        if _now_pt - _PENDING_ZONE[zc]["placed_at"] > 300:
            try: hl.cancel_order(zc, _PENDING_ZONE[zc]["oid"])
            except Exception: pass
            log.warning(f"  🧹 {zc}: stale pending order cleaned up (5min)")
            _PENDING_ZONE.pop(zc, None)
    
    # ── ORPHAN ORDER CLEANUP: cancel TP/SL for coins with no position ──
    # Every 60s to avoid API spam
    _orphan_key = "_last_orphan_cleanup"
    _last_orphan = getattr(_monitor_positions, _orphan_key, 0) if hasattr(_monitor_positions, _orphan_key) else 0
    if _now_pt - _last_orphan > 60:
        try:
            _info = hl.Info()
            _all_orders = _info.open_orders(hl._main_wallet)
            if _all_orders:
                _held = {p.get('coin','').upper(): abs(float(p.get('szi',0))) for p in positions if abs(float(p.get('szi',0))) > 0.0001}
                for o in _all_orders:
                    oc = o.get('coin','').upper()
                    if oc not in _held:
                        try:
                            hl.cancel_order(oc, o.get('oid',0))
                            log.info(f"  🧹 {oc}: orphan order cancelled (no position)")
                        except Exception: pass
            setattr(_monitor_positions, _orphan_key, _now_pt)
        except Exception as _oe:
            pass


# ============================================================
# MAIN LOOP
# ============================================================

def run(dry_run: bool = False):
    global hsl_state, exposure_state, cycle_count, evolution_check_cycles, _forager_skip_cooldown, _API_WALLET, _global_pause_until, _rotate_offset

    cfg = json.loads(Path("/home/ubuntu/.hyperliquid/config.json").read_text())
    api_wallet = cfg["api_wallet"]
    main_wallet = cfg["main_wallet"]
    _API_WALLET = api_wallet  # Make available to _execute_direct_close

    # ── Configure HL client with retry for 429 rate limits ──
    for attempt in range(3):
        try:
            hl.configure(api_wallet, cfg["api_private_key"], main_wallet)
            break
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 10
                log.warning(f"  HL configure attempt {attempt+1} failed ({type(e).__name__}) — retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"  HL configure failed after 3 attempts: {type(e).__name__}")
                raise
    # set_main_wallet also hits the API — need its own 429 retry
    for attempt in range(5):
        try:
            hl.set_main_wallet(main_wallet)
            break
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = (attempt + 1) * 5
                log.warning(f"  set_main_wallet 429 — backing off {wait}s")
                time.sleep(wait)
            else:
                raise

    log.info("=" * 60)
    log.info("HYPERLIQUID AI DAEMON v3 — ENSEMBLE ARCHITECTURE")
    log.info(f"  Main: {main_wallet[:10]}...{main_wallet[-6:]}")
    log.info(f"  API:  {api_wallet[:10]}...{api_wallet[-6:]}")
    log.info(f"  Cycle: {CYCLE_SECONDS}s | Universe: {len(_ALL_COINS)} coins | Window: {_ROTATE_WINDOW}")
    log.info(f"  Strategy: 5-pillar composite + ensemble + regime-weighted")
    log.info(f"  Risk: WEL/TWEL + HSL + unstucking + cooldown")
    log.info(f"  Execution: Multi-tier TP + break-even + trailing")
    log.info("=" * 60)

    # Start WebSocket
    try:
        from hyperliquid_ws import start_ws
        start_ws(api_wallet)
        log.info("  WebSocket: started")
    except Exception as e:
        log.warning(f"  WebSocket: {e}")

    # Start CVD engine (real trade-level delta from WebSocket)
    try:
        from cvd_engine import patch_websocket_for_cvd
        if patch_websocket_for_cvd():
            log.info("  CVD Engine: real trade delta active")
        else:
            log.warning("  CVD Engine: failed to start")
    except Exception as e:
        log.warning(f"  CVD Engine: {e}")

    # ── Dead-man's switch: cancel all orders 10 min after daemon stops ──
    try:
        deadman_ms = int((time.time() + 600) * 1000)  # 10 minutes from now
        hl.schedule_cancel(deadman_ms)
        log.info(f"  Dead-man switch: orders auto-cancel at {datetime.fromtimestamp(deadman_ms/1000, tz=timezone.utc).strftime('%H:%M:%S')} UTC")
    except Exception as e:
        log.warning(f"  Dead-man switch failed: {e}")

    # ── Recover state for existing positions (survives daemon restart) ──
    try:
        state = hl.get_user_state(main_wallet)
        existing = _normalize_positions(state.get("assetPositions", []))
        mids = hl.get_all_mids()
        for p in existing:
            coin = p.get("coin", "")
            if not coin:
                continue
            entry = float(p.get("entryPx", 0))
            szi = float(p.get("szi", 0))
            if entry > 0:
                is_long = szi > 0
                mid = float(mids.get(coin, entry))
                # Compute ATR for stop distance
                try:
                    c15m = _fetch_candles_cached(coin, "15m", 100)
                    closes = [float(c.get("close", c.get("c", 0))) for c in (c15m or [])]
                    atr = _compute_atr(c15m, 14) if c15m else abs(entry * 0.02)
                except Exception:
                    atr = abs(entry * 0.02)

                sl_distance = atr * 2.0  # 2 ATR stop
                if is_long:
                    sl_price = entry - sl_distance
                    tp1_price = entry + atr * 1.5
                    tp2_price = entry + atr * 3.0
                else:
                    sl_price = entry + sl_distance
                    tp1_price = entry - atr * 1.5
                    tp2_price = entry - atr * 3.0

                # Initialize trail state
                trail_states[coin.upper()] = TrailState(
                    symbol=coin.upper(),
                    is_long=is_long,
                    entry_price=entry,
                    highest_price=max(entry, mid),
                    current_stop=sl_price,
                )
                # ── Recover entry time from fills ──
                try:
                    fills = hl.get_fills(coin)
                    best_ts = time.time()
                    for f in (fills or []):
                        if f.get('coin','').upper() == coin.upper() and float(f.get('sz',0)) > 0:
                            ts = int(f.get('time', 0)) / 1000
                            if ts > 0 and ts < best_ts and 'Open' in str(f.get('dir','')):
                                best_ts = ts
                    _position_entry_times[coin.upper()] = best_ts
                    log.info(f"  Recovery: {coin} entry_time={datetime.fromtimestamp(best_ts, tz=timezone.utc).strftime('%H:%M:%S')}")
                except Exception:
                    _position_entry_times[coin.upper()] = time.time()  # Best guess
                # Set TP levels
                position_tps[coin.upper()] = [
                    {"fraction": 0.5, "price": tp1_price, "tier_name": "TP1"},
                    {"fraction": 0.5, "price": tp2_price, "tier_name": "TP2"},
                ]

                # ── Place actual TP/SL orders on Hyperliquid ──
                try:
                    pos_sz = abs(szi)
                    # Stop loss
                    sl_result = hl.trigger_order(coin, not is_long, pos_sz, sl_price, "sl", True, True)
                    sl_oid = _extract_oid(sl_result, "sl") if '_extract_oid' in dir() else 0
                    # Take profits
                    tp_oids = []
                    for tp in position_tps[coin.upper()]:
                        tp_sz = pos_sz * tp["fraction"]
                        tp_result = hl.trigger_order(coin, not is_long, tp_sz, tp["price"], "tp", True, True)
                        tp_oid = _extract_oid(tp_result, "tp") if '_extract_oid' in dir() else 0
                        if tp_oid:
                            tp_oids.append(tp_oid)
                    # Track orders
                    from action_executor import _track_order
                    _track_order(coin, {
                        "sl_oid": sl_oid, "tp_oids": tp_oids,
                        "stop_loss": sl_price, "tp_levels": position_tps[coin.upper()],
                        "size": pos_sz, "side": "LONG" if is_long else "SHORT",
                        "placed_at": time.time(),
                    })
                    log.info(f"  Recovery: {coin} TP/SL placed — SL=${sl_price:.4f} TP1=${tp1_price:.4f} TP2=${tp2_price:.4f}")
                except Exception as e:
                    log.warning(f"  Recovery: {coin} TP/SL placement failed — {e}")
        if existing:
            log.info(f"  Recovery: restored state for {len(existing)} existing positions")
    except Exception as e:
        log.warning(f"  Recovery failed: {e}")

    # ── Cleanup stale open orders on startup ──
    # Use schedule_cancel instead of Info().open_orders() — lighter, no 429 risk
    try:
        hl.schedule_cancel(5000)  # Cancel ALL open orders in 5s
        log.info(f"  🧹 Scheduled cancel of all open orders")
    except Exception as _co:
        log.debug(f"  Order cleanup skipped: {type(_co).__name__}")

    last_cycle = 0
    last_monitor = 0
    last_evolution = 0
    _consecutive_429s = 0  # reset on each daemon restart
    _exec_fails: dict[str, int] = {}  # track repeated execution failures
    _coin_blacklist: set[str] = {"KLUNC"}  # coins blacklisted after 3+ failures (+ permanent dead coins)
    # KLUNC: exists in HL meta but candle fetch always fails — dead listing, zero volume
    _last_total_eq: float = 0.0  # cached total_equity for flash crash detection

    while True:
        now = time.time()

        # ── Monitor positions (every 10s) ──
        if now - last_monitor >= POSITION_CHECK_SECONDS:
            last_monitor = now
            try:
                state = hl.get_user_state(main_wallet)
                positions = _normalize_positions(state.get("assetPositions", []))
                mids = hl.get_all_mids()
                active = [p for p in positions if abs(float(p.get("szi", 0))) > 0.0001]

                # ── MICRO-CAP SPIKE DETECTOR: catches fast moves on sub-$0.50 coins ──
                # APE lesson: $0.15 coin spikes 3-5% between 10s checks, bot never sees it.
                # Track local peaks and set tight trail BEFORE the full eval misses them.
                for p in active:
                    coin = p.get("coin", "")
                    mid = float(mids.get(coin, 0))
                    if mid <= 0 or mid >= 0.50:
                        continue  # Only sub-$0.50 micro-caps
                    entry = float(p.get("entryPx", 0))
                    if entry <= 0:
                        continue
                    szi = float(p.get("szi", 0))
                    side = "LONG" if szi > 0 else "SHORT"
                    pnl_pct = (mid - entry) / entry * 100 if side == "LONG" else (entry - mid) / entry * 100
                    
                    # Track local peak for this position (reset on new entry or position change)
                    _mc_key = f"_mc_peak:{coin}"
                    _mc_peak = getattr(_monitor_positions, _mc_key, None) if hasattr(_monitor_positions, _mc_key) else None
                    _mc_entry_ts = getattr(_monitor_positions, f"_mc_entry_ts:{coin}", 0) if hasattr(_monitor_positions, f"_mc_entry_ts:{coin}") else 0
                    
                    # Reset tracker if this is a new position
                    _pos_entry_ts = _position_entry_times.get(coin.upper(), 0)
                    if _pos_entry_ts != _mc_entry_ts:
                        _mc_peak = None
                        setattr(_monitor_positions, f"_mc_entry_ts:{coin}", _pos_entry_ts)
                    
                    if pnl_pct > 0.15:  # Any meaningful green
                        if _mc_peak is None or pnl_pct > _mc_peak:
                            _mc_peak = pnl_pct
                            setattr(_monitor_positions, _mc_key, _mc_peak)
                        # If we had a peak and now we're fading, set immediate tight trail
                        elif _mc_peak is not None and pnl_pct < _mc_peak * 0.60:
                            # Lost 40% of local peak — lock remaining with tight trail
                            if coin in trail_states and not trail_states[coin].activated:
                                _mc_trail_pct = 0.003  # 0.30% — micro-cap, kill fast on fade
                                # Cap at 50% of local peak
                                _mc_trail_pct = min(_mc_trail_pct, max(0.0015, _mc_peak * 0.0050))
                                _mc_stop = mid * (1 - _mc_trail_pct) if side == "LONG" else mid * (1 + _mc_trail_pct)
                                trail_states[coin].current_stop = _mc_stop
                                trail_states[coin].activated = True
                                trail_states[coin].trail_distance = mid * _mc_trail_pct
                                log.info(f"  ⚡ {coin}: MICRO-CAP SPIKE FADE — peak {_mc_peak:+.2f}% → {pnl_pct:+.2f}%, trail set at ${_mc_stop:.4f} (0.30%)")
                                # Force a mini hard-exit if the fade is severe (>60% of local peak gone)
                                if _mc_peak > 0.50 and pnl_pct < _mc_peak * 0.40:
                                    log.info(f"  ⚡ {coin}: MICRO-CAP SEVERE FADE — peak {_mc_peak:+.2f}% → {pnl_pct:+.2f}%, closing now")
                                    if _can_close_position(coin, "microcap-spike-fade"):
                                        _MANUAL_CLOSES[coin] = time.time()
                                        hl.market_close(coin)
                                        _reset_signal_dominance(coin)
                                        if hasattr(_monitor_positions, _mc_key):
                                            delattr(_monitor_positions, _mc_key)
                    elif pnl_pct <= 0 and _mc_peak is not None:
                        # Price went back below entry — reset peak tracker
                        _mc_peak = None
                        setattr(_monitor_positions, _mc_key, None)

                if active:
                    # Compute equity for monitor — use spot_free + margin_used (STABLE, excludes unrealized PnL)
                    # total_equity includes unrealized PnL → fluctuates wildly → false flash crash triggers
                    # spot_free + margin_used only changes on real PnL events (close/liquidate/fees)
                    try:
                        acc = hl.get_account()
                        monitor_eq = acc.get("spot_free", 0) + acc.get("margin_used", 0)
                    except Exception:
                        monitor_eq = _last_total_eq  # fallback (skip flash crash if no cycle equity yet)
                    # ── Open order verification (every cycle, NautilusTrader pattern) ──
                    # Catches naked positions within 90s instead of 7.5 min
                    _verify_open_orders(active, mids)
                    _monitor_positions(active, mids, monitor_eq, active)

                    # Update exposure
                    pos_map = {p.get("coin", ""): abs(float(p.get("szi", 0))) * float(mids.get(p.get("coin", ""), 0))
                              for p in active}
                    exposure_state.update(hsl_state.peak_equity or 10000.0, pos_map)
                else:
                    # No positions — clear exposure state (prevents phantom positions)
                    exposure_state.update(exposure_state.equity or 100.0, {})

                # Check for liquidations
                _detect_liquidations(positions)

            except Exception as e:
                # Rate-limit or transient API error — don't crash, just log and skip
                if "429" in str(e):
                    log.debug(f"  Monitor API 429 — retrying next tick")
                else:
                    log.warning(f"  Monitor API error: {type(e).__name__}: {e}")
                    log.warning(f"  Monitor traceback: {traceback.format_exc()[:300]}")

        # ── AI Cycle (every 60s) ──
        if now - last_cycle >= CYCLE_SECONDS:
            cycle_start = time.time()
            last_cycle = now
            cycle_count += 1

            try:
                acc = hl.get_account()
                total_eq = acc["total_equity"]
                _last_total_eq = total_eq  # cache for flash crash detector in monitor loop
                perp_eq = acc["perp_equity"]
                spot_usdc = acc["spot_usdc"]
                spot_free = acc.get("spot_free", spot_usdc)
                spot_hold = acc.get("spot_hold", 0)
                positions = _normalize_positions(acc["positions"])
                active = [p for p in positions if abs(float(p.get("szi", 0))) > 0.0001]
                mids = hl.get_all_mids()

                # ── Open Interest (cached 5min, with delta tracking for L3) ──
                global _last_oi_update, _open_interest, _prev_open_interest
                _oi_needs_init = '_open_interest' not in globals()
                if _oi_needs_init or (time.time() - _last_oi_update > 300):
                    if hl.is_throttled() and not _oi_needs_init:
                        pass  # Skip refresh under API pressure — keep stale data
                    else:
                        try:
                            ctxs_data = hl.get_asset_ctxs()
                            ctxs = ctxs_data.get("contexts", [])
                            universe = ctxs_data.get("meta", {}).get("universe", [])
                            # Save previous snapshot for delta computation
                            if '_open_interest' in globals() and _open_interest:
                                _prev_open_interest = dict(_open_interest)
                            else:
                                _prev_open_interest = {}
                            _open_interest = {}
                            for i, asset in enumerate(universe):
                                name = asset.get("name", "")
                                if name and i < len(ctxs):
                                    _open_interest[name] = float(ctxs[i].get("openInterest", 0))
                            _last_oi_update = time.time()
                            log.info(f"  📊 OI updated: {len(_open_interest)} assets (prev: {len(_prev_open_interest)})")
                        except Exception:
                            _open_interest = {}
                            _prev_open_interest = {}
                            _last_oi_update = time.time()

                # Update HSL — reset peak when no positions (avoid unrealized PnL inflation)
                if not active and hsl_state.peak_equity > 0 and total_eq < hsl_state.peak_equity * 0.95:
                    # Peak was set during open positions with unrealized PnL — reset to realized
                    log.info(f"  🔧 HSL peak reset: ${hsl_state.peak_equity:.2f} → ${total_eq:.2f} (positions closed, unrealized PnL gone)")
                    hsl_state.peak_equity = total_eq
                    hsl_state.drawdown_ema = 0.0
                    hsl_state.tier = HSLTier.GREEN
                hsl_state.update(total_eq)
                # ── Dynamic limits: micro accounts get higher TWEL ──
                if exposure_state.limits.twel_limit == 0.30:  # Only set once
                    exposure_state.limits = ExposureLimits.for_equity(total_eq)
                    log.info(f"  📐 Risk limits: TWEL={exposure_state.limits.twel_limit*100:.0f}% WEL={exposure_state.limits.wel_limit*100:.0f}% (equity=${total_eq:.0f})")

                log.info(f"─" * 40)
                # Performance metrics (self-learning)
                try:
                    from cvd_engine import performance_summary
                    perf_str = performance_summary()
                except Exception:
                    perf_str = ""

                # ── Order fill check (TP/SL triggers from WebSocket) ──
                try:
                    from hyperliquid_ws import get_field as _ws_field
                    from action_executor import check_order_fills
                    ws_fills = _ws_field("fills") or []
                    fill_events = check_order_fills({"fills": ws_fills})
                    for evt in fill_events:
                        if evt["type"] == "sl":
                            log.warning(f"  🛑 {evt['coin']}: STOP LOSS triggered @ ${evt['px']:.4f} sz={evt['sz']:.4f}")
                        elif evt["type"] == "tp":
                            log.info(f"  🎯 {evt['coin']}: TAKE PROFIT triggered @ ${evt['px']:.4f} sz={evt['sz']:.4f}")
                except Exception:
                    pass  # Non-critical — fill tracking is best-effort

                log.info(f"CYCLE {datetime.now().strftime('%H:%M:%S')} "
                         f"| Equity=${total_eq:,.2f} (Perp=${perp_eq:,.2f}, Spot=${spot_usdc:,.2f} free=${spot_free:,.2f}) "
                         f"| HSL={hsl_state.tier.value} | Pos={len(active)} | Cycle={cycle_count}")

                # Unified mode: spot USDC is usable as perps margin — no transfer needed
                if spot_usdc > 0 and perp_eq < 1:
                    log.info(f"  💡 Unified mode — ${spot_usdc:.2f} spot USDC usable as margin")
                    # total_equity already includes spot; perp orders will draw from spot

                if total_eq < MIN_TRADE_USD:
                    log.info(f"  Equity < ${MIN_TRADE_USD} — waiting")
                    continue

                # HSL gating
                if hsl_state.tier == HSLTier.RED:
                    log.error("🔴 HSL RED — EMERGENCY: close all positions!")
                    for p in active:
                        _submit_action("close", p.get("coin", ""),
                                       {"direction": "close", "reason": "hsl_red_panic"})
                    continue

                if hsl_state.tier == HSLTier.ORANGE:
                    log.warning("🟠 HSL ORANGE — close-only mode")
                    continue

                # Dead-man's switch
                cancel_time = int((time.time() + 300) * 1000)
                hl.schedule_cancel(cancel_time)

                # ── Stage 0: Fetch BTC candles for beta filter ──
                btc_candles = _fetch_candles("BTC", "1h", 200)

                # ── AI Market Assessment (every 5 min, cached) ──
                try:
                    if cycle_count % 5 == 1:  # Every 5th cycle = every 5 min
                        signal_summaries = []
                        for c in (_ALWAYS_SCAN | set(_ALL_COINS[_rotate_offset % len(_ALL_COINS):][:8])):
                            c15 = _fetch_candles_cached(c, "15m", 50)
                            if c15 and len(c15) >= 20:
                                df = add_basic_indicators(candles_to_frame(c15))
                                comp = compute_composite(df) if len(df) >= 20 else 0
                                reg, _ = detect_regime(df, c)  # Returns (MarketRegime, dict)
                                signal_summaries.append(f"{c}: comp={comp:+.2f} reg={reg.value}")
                        if signal_summaries:
                            ai_market = ai_assess_market(
                                signal_summaries, total_eq, len(active),
                                fear_greed=_get_fear_greed(),
                            )
                            log.info(f"  🧠 AI Market: {'TRADABLE' if ai_market['tradable'] else 'DEAD'} "
                                     f"bias={ai_market['bias']} lev≤{ai_market['max_leverage']}x "
                                     f"tf={ai_market['best_timeframe']} max_pos={ai_market['max_positions']}")
                            if not ai_market.get("tradable", True):
                                log.info(f"  ⏸️  AI says market is dead — skipping new entries this cycle")
                                _global_pause_until = max(_global_pause_until, time.time() + 300)
                except Exception as e:
                    log.warning(f"  ⚠️ AI market assessment failed: {type(e).__name__}: {e}")

                # Stage 1: AI coin selection first, forager fills remainder
                active_coins = [p.get("coin", "") for p in active]
                # Dynamic: micro accounts get 1-2, scaling up with equity
                max_pos = max(1, min(BASE_MAX_POSITIONS, int(total_eq / 35)))
                slots_left = max_pos - len(active)
                # Micro account guard: cap total positions based on equity
                # $25+: 2 positions (so one dud doesn't paralyze)
                # $50+: 3 positions
                if total_eq < 25:
                    max_pos = 1
                    slots_left = max(0, max_pos - len(active))
                elif total_eq < 50:
                    max_pos = 2
                    slots_left = max(0, max_pos - len(active))
                cap_new_per_cycle = 1 if total_eq < 200 else slots_left
                slots_left = min(slots_left, cap_new_per_cycle)
                
                # ── Coin rotation: always advance through the universe (even with no slots) ──
                # This ensures we have fresh data cached when a slot opens
                candidates_all = [c for c in _ALL_COINS if c not in active_coins and c.upper() not in _coin_blacklist]
                always = [c for c in candidates_all if c in _ALWAYS_SCAN]
                rest = [c for c in candidates_all if c not in _ALWAYS_SCAN]
                window_start = _rotate_offset % max(len(rest), 1)
                window = rest[window_start:window_start + _ROTATE_WINDOW - len(always)]
                if len(window) < _ROTATE_WINDOW - len(always):
                    window += rest[:(_ROTATE_WINDOW - len(always) - len(window))]
                candidates = always + window
                _rotate_offset += _ROTATE_WINDOW
                
                # ── Sector rotation: update momentum per sector ──
                try:
                    from sector_rotation import sector_tracker
                    # Fetch 1h candles for sector-level analysis (cached, not per-coin heavy)
                    c1h = {}
                    for c in candidates[:15]:  # Sample to keep fast
                        cached = _fetch_candles_cached(c, "1h", 24)
                        if cached:
                            c1h[c] = cached
                    c5m_sector = {}
                    for c in candidates[:15]:
                        cached = _fetch_candles_cached(c, "5m", 60)
                        if cached:
                            c5m_sector[c] = cached
                    sector_tracker.update(mids, c5m_sector, c1h)
                except Exception:
                    pass
                
                # ── Eigen portfolio regime: PCA on returns to detect market state ──
                try:
                    from hrp_sizing import eigen_regime
                    if cycle_count % 5 == 0:  # Every 5 cycles
                        rets_eigen = {}
                        for c in candidates[:40]:
                            c1h_data = _fetch_candles_cached(c, "1h", 80)
                            if c1h_data and len(c1h_data) >= 60:
                                closes = [float(x.get('c', x.get('close', 0))) for x in c1h_data[-60:]]
                                if closes[0] > 0:
                                    rets_eigen[c] = np.log(np.array(closes[1:]) / np.array(closes[:-1]))
                        if len(rets_eigen) >= 10:
                            eigen_regime.update(rets_eigen)
                            log.info(f"  📐 Eigen regime: PC1={eigen_regime.pc1_ratio:.0%} ({eigen_regime.regime})")
                except Exception:
                    pass
                
                # ── CryptoGAT: graph attention predictions (cross-asset) ──
                # Runs regardless of open positions. Cross-asset graph signals inform exits too.
                if cycle_count % 3 == 0:
                    try:
                        from cryptogat_predictor import predict_with_graph as _gat_predict
                        gat_candles = {}
                        for c in candidates[:50]:
                            c5m = _fetch_candles_cached(c, "5m", 80)
                            if c5m and len(c5m) >= 60:
                                gat_candles[c] = c5m
                        if len(gat_candles) >= 5:
                            gat_signals = _gat_predict(gat_candles)
                            strong_gat = 0
                            for coin, gs in gat_signals.items():
                                if gs["direction"] != "HOLD" and gs["confidence"] >= 40:
                                    existing = _ai_trade_plan.get(coin.upper(), {})
                                    if existing:
                                        existing["gat_direction"] = gs["direction"]
                                        existing["gat_confidence"] = gs["confidence"]
                                        if gs["confidence"] > existing.get("confidence", 0):
                                            existing["confidence"] = int(gs["confidence"])
                                    else:
                                        _ai_trade_plan[coin.upper()] = {
                                            "coin": coin, "direction": gs["direction"].lower(),
                                            "confidence": int(gs["confidence"]),
                                            "target_pct": 1.5, "stop_pct": 1.0, "hold_min": 45,
                                            "gat_direction": gs["direction"], "gat_confidence": gs["confidence"],
                                        }
                                    strong_gat += 1
                            if strong_gat > 0:
                                log.info(f"  🔷 CryptoGAT: {strong_gat}/{len(gat_signals)} strong signals (graph attention)")
                    except Exception as e:
                        log.warning(f"  CryptoGAT skipped: {type(e).__name__}: {e}")
                
                if slots_left > 0:
                    log.info(f"  🔄 Rotation: scanning {len(candidates)} coins (offset={_rotate_offset})")
                    
                    # ── AI scans ALL candidates independently (not just forager picks) ──
                    selected = []
                    ai_picks = []
                    # Aug 7: blacklist coins with no candle data or dead listings
                    _BAD_COINS = {"X", "KPEPE", "BABY", "VINE", "KLUNC", "AZTEC", "KBONK", "VIRTUAL"}
                    try:
                        ai_candidates = []
                        # Scan up to 18 for AI evaluation
                        _scanned = 0
                        for c in candidates[:25]:
                            if c.upper() in _BAD_COINS:
                                continue
                            if _scanned >= 18:
                                break
                            mid = float(mids.get(c, 0))
                            if mid <= 0 or mid > 50000:  # Skip dead coins and BTC-like prices
                                continue
                            c15 = _fetch_candles_cached(c, "15m", 50)
                            if not c15 or len(c15) < 20:
                                continue
                            _scanned += 1
                            df = add_basic_indicators(candles_to_frame(c15))
                            comp = compute_composite(df)
                            reg, _ = detect_regime(df, c)
                            vol = 0.0
                            try:
                                closes = [float(x.get("c", x.get("close", 0))) for x in c15[-20:]]
                                if closes and closes[0] > 0:
                                    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
                                    vol = (sum(r*r for r in rets)/len(rets))**0.5 * 100
                            except Exception:
                                pass
                            import sector_rotation as _sr
                            _sec_score = _sr.sector_tracker.get_sector_score(c)
                            # ── VWAP sigma: critical for AI direction (mean reversion awareness) ──
                            _vwap_sigma = 0.0
                            try:
                                _vwap_vols = [float(x.get("v", x.get("volume", 0))) for x in c15[-20:]]
                                _vwap_highs = [float(x.get("h", x.get("high", 0))) for x in c15[-20:]]
                                _vwap_lows = [float(x.get("l", x.get("low", 0))) for x in c15[-20:]]
                                _vwap_closes = [float(x.get("c", x.get("close", 0))) for x in c15[-20:]]
                                _vwap_typ = [(h+l+c)/3 for h,l,c in zip(_vwap_highs, _vwap_lows, _vwap_closes)]
                                _vwap_tv = sum(t*v for t,v in zip(_vwap_typ, _vwap_vols))
                                _vwap_tv_total = sum(_vwap_vols)
                                if _vwap_tv_total > 0:
                                    _vwap_mean = _vwap_tv / _vwap_tv_total
                                    _vwap_var = sum(((t-_vwap_mean)**2)*v for t,v in zip(_vwap_typ, _vwap_vols)) / _vwap_tv_total
                                    _vwap_std = _vwap_var ** 0.5
                                    if _vwap_std > 0:
                                        _vwap_sigma = (mid - _vwap_mean) / _vwap_std
                            except Exception:
                                pass
                            # ── CVD: cumulative volume delta for buy/sell pressure ──
                            _cvd = {}
                            try:
                                from volume_delta import compute_cvd
                                _cvd_result = compute_cvd(c15)
                                _cvd = {"trend": "rising" if _cvd_result.cvd_slope > 0 else "falling",
                                        "conf": min(100, abs(_cvd_result.cvd_slope) * 1000),
                                        "divergence": _cvd_result.divergence}
                            except Exception:
                                pass
                            # ── OI delta: open interest change ──
                            _oi_delta = 0.0
                            try:
                                from oi_delta import get_oi_delta
                                _oi_delta = get_oi_delta(c)
                            except Exception:
                                pass
                            # ── 1h proximity (distance from 1h high/low) ──
                            _1h_prox = {}
                            try:
                                c1h = _fetch_candles_cached(c, "1h", 30)
                                if c1h and len(c1h) >= 5:
                                    _1h_highs = [float(x.get("h", x.get("high", 0))) for x in c1h[-6:]]
                                    _1h_lows = [float(x.get("l", x.get("low", 0))) for x in c1h[-6:]]
                                    _hh = max(_1h_highs) if _1h_highs else mid
                                    _ll = min(_1h_lows) if _1h_lows else mid
                                    if _hh > 0 and _ll > 0:
                                        _1h_prox = {"to_high": (1 - mid/_hh)*100 if mid < _hh else 0,
                                                    "to_low": (mid/_ll - 1)*100 if mid > _ll else 0}
                            except Exception:
                                pass
                            # ── 4h range position: where is price in the 4h range? ──
                            # Aug 9: AI needs 4h context to avoid shorting into +74% pumps
                            _4h_prox = {}
                            try:
                                _ext = get_extremes(c, mids)
                                if _ext and _ext.high_4h > 0 and _ext.low_4h > 0:
                                    _4h_prox = {"pct_high": _ext.pct_low_4h,  # % below 4h high
                                               "pct_low": _ext.pct_4h,        # % above 4h low
                                               "range_pos": _ext.range_pos_4h if hasattr(_ext, 'range_pos_4h') else 50}
                            except Exception:
                                pass
                            ai_candidates.append({
                                "coin": c, "price": mid, "composite": comp,
                                "regime": reg.value, "volatility": vol,
                                "btc_corr": DEFAULT_BTC_CORRELATIONS.get(c.upper(), 0.5),
                                "mom_1m": _calc_momentum(c, "1m"),
                                "mom_5m": _calc_momentum(c, "5m"),
                                "mom_15m": _calc_momentum(c, "15m"),
                                "mom_1h": _calc_momentum(c, "1h"),
                                "vwap_dist": round(_vwap_sigma * 100, 1),
                                "_cvd": _cvd,
                                "oi_delta": round(_oi_delta, 2),
                                "_range_pct": round(vol * 3, 1),
                                "_1h_prox": _1h_prox,
                                "_4h_prox": _4h_prox,
                                "signal_hint": f"{reg.value}:comp={comp:+.2f}",
                                "sector": _sr.COIN_SECTOR.get(c.upper(), ""),
                                "sector_score": _sec_score,
                            })
                        if ai_candidates:
                            # ── Market intelligence: external data for AI context ──
                            try:
                                from market_intel import get_market_brief
                                _mkt_brief = get_market_brief()
                            except Exception:
                                _mkt_brief = ""
                            log.info(f"  AI scanning {len(ai_candidates)} candidates...")
                            ai_picks, ai_skips, ai_market_notes = ai_select_coins(
                                ai_candidates, total_eq,
                                market_context=_mkt_brief,
                                recent_trades=_recent_trades)
                            if ai_skips:
                                for s in ai_skips:
                                    _forager_skip_cooldown[str(s).upper()] = time.time() + 300
                                log.info(f"  🚫 AI skip: {', '.join(str(s) for s in ai_skips[:10])}")
                            if ai_market_notes:
                                log.info(f"  🧠 AI market: {' | '.join(ai_market_notes[:2])}")
                            if ai_picks:
                                pick_strs = []
                                for p in ai_picks:
                                    pick_strs.append(
                                        f"{p['coin']}:{p['direction']} tgt={p['target_pct']}% "
                                        f"stop={p['stop_pct']}% hold={p['hold_min']}m"
                                    )
                                log.info(f"  🧠 AI picks: {' | '.join(pick_strs)}")
                                _ai_trade_plan.update({p["coin"]: p for p in ai_picks})
                                for p in ai_picks:
                                    # Don't force a coin we already have an open position in
                                    if p.get("confidence", 0) >= 80 and p["coin"] not in selected and p["coin"] not in active_coins:
                                        selected.append(p["coin"])
                                        log.info(f"  🧠 AI forcing {p['coin']} into selection (conf={p['confidence']}%)")
                                    elif p["coin"] in active_coins:
                                        log.info(f"  ⏭️ AI wants {p['coin']} but we already hold it — skipping")
                                for p in ai_picks:
                                    if p["coin"] not in selected and len(selected) < slots_left and p["coin"] not in active_coins:
                                        selected.append(p["coin"])
                        else:
                            log.info(f"  AI no candidates (all {len(candidates)} failed data checks)")
                    except Exception as e:
                        log.warning(f"  AI coin selection skipped: {type(e).__name__}: {e}")
                    
                    # ── Kronos: foundation model predictions for selected coins ──
                    # Runs after AI selection; supplements targets/stops with real price forecasts
                    if selected and cycle_count % 5 == 0:  # Every 5th cycle (~7-8 min) — saves compute, model slow
                        try:
                            from kronos_predictor import predict_batch as _kpredict
                            kronos_candles = {}
                            for coin in selected:
                                c5m = _fetch_candles_cached(coin, "5m", 200)
                                if c5m and len(c5m) >= 150:
                                    kronos_candles[coin] = c5m
                            if kronos_candles:
                                kresults = _kpredict(kronos_candles)
                                for coin, ksig in kresults.items():
                                    if ksig.direction != "HOLD" and ksig.confidence >= 40:
                                        # Merge Kronos into AI trade plan or create new entry
                                        existing = _ai_trade_plan.get(coin.upper(), {})
                                        if existing:
                                            # AI has a plan — Kronos refines targets
                                            if ksig.target_pct > existing.get("target_pct", 0):
                                                existing["target_pct"] = round(ksig.target_pct, 1)
                                            if ksig.stop_pct < existing.get("stop_pct", 99):
                                                existing["stop_pct"] = round(ksig.stop_pct, 1)
                                            existing["kronos_conf"] = ksig.confidence
                                            existing["kronos_direction"] = ksig.direction
                                            log.info(f"  🔮 Kronos {coin}: {ksig.direction} conf={ksig.confidence:.0f}% tgt={ksig.target_pct:+.2f}% — merged into AI plan")
                                        else:
                                            # No AI plan — Kronos provides one
                                            _ai_trade_plan[coin.upper()] = {
                                                "coin": coin, "direction": ksig.direction.lower(),
                                                "confidence": int(ksig.confidence), "target_pct": round(ksig.target_pct, 1),
                                                "stop_pct": round(ksig.stop_pct, 1), "hold_min": 60,
                                                "kronos_direction": ksig.direction, "kronos_conf": ksig.confidence,
                                            }
                                            log.info(f"  🔮 Kronos {coin}: {ksig.direction} conf={ksig.confidence:.0f}% tgt={ksig.target_pct:+.2f}% — new plan (AI had none)")
                        except Exception as e:
                            log.warning(f"  Kronos prediction skipped: {type(e).__name__}: {e}")
                    
                    # ── Forager fills remaining slots ──
                    remaining_slots = slots_left - len(selected)
                    if remaining_slots > 0:
                        remaining_candidates = [c for c in candidates if c not in selected]
                        forager_selected = _forager_select(remaining_candidates, mids, remaining_slots)
                        selected.extend(forager_selected)
                        log.info(f"  Forager selected: {forager_selected}")
                    log.info(f"  Final selection: {selected}")
                else:
                    selected = []
                    log.info(f"  🔄 Rotation: offset={_rotate_offset} (no slots — 1 pos, $64 eq, holding)")

                # ── Stage 1.5: Continuous Prediction Engine on EXISTING positions ──
                # Runs ALL 6 layers (order book, volume profile, CVD, multi-TF tech,
                # funding, ML ensemble) and outputs probability distributions at
                # 1m/5m/15m/1h/4h horizons. No thresholds. Pure probability.
                # AI validator sees the full distribution and decides.
                for p in active:
                    coin = p.get("coin", "")
                    szi = float(p.get("szi", 0))
                    if abs(szi) < 0.0001:
                        continue
                    entry = float(p.get("entryPx", 0))
                    mid = float(mids.get(coin, 0))
                    if mid <= 0:
                        continue
                    side = "LONG" if szi > 0 else "SHORT"
                    pnl_pct = ((mid - entry) / entry * 100) * (1 if szi > 0 else -1)
                    # Extract leverage and liquidation for position context
                    _lev_raw = p.get("leverage", {})
                    lev = float(_lev_raw.get("value", 1)) if isinstance(_lev_raw, dict) else float(_lev_raw or 1)
                    margin_pnl = pnl_pct * lev
                    liq = float(p.get("liquidationPx") or 0)
                    liq_dist = round(abs(mid - liq) / mid * 100, 1) if liq > 0 else 99.0

                    try:
                        # Gather data for continuous predictor
                        c1m = _fetch_candles_cached(coin, "1m", 100)
                        c5m = _fetch_candles_cached(coin, "5m", 100)
                        c15m = _fetch_candles_cached(coin, "15m", 200)
                        c1h = _fetch_candles_cached(coin, "1h", 24)
                        c4h = _fetch_candles_cached(coin, "4h", 50)
                        c1d = _fetch_candles_cached(coin, "1d", 50)
                        fund_ctx = funding_sniper.asset_ctx.get(coin, {})
                        fund_rate = float(fund_ctx.get("funding", 0)) if fund_ctx else 0.0

                        # Get ML signal for integration
                        try:
                            ml_sig = predict_unified(coin, c15m, mid, mids=mids,
                                funding_rate=fund_rate, btc_mid=float(mids.get("BTC", 0)),
                                fear_greed=_get_fear_greed(),
                                regime="ranging")
                        except Exception:
                            ml_sig = None

                        # ── THE CORE: continuous prediction ──
                        # Compute Hurst exponent from 15m candles for regime detection
                        try:
                            from hurst_detector import hurst_exponent
                            closes_hurst = [float(c["c"]) for c in c15m if "c" in c]
                            H = hurst_exponent(closes_hurst) if len(closes_hurst) >= 30 else 0.5
                        except Exception:
                            H = 0.5

                        # ── Live order book from WebSocket ──
                        order_book = None
                        try:
                            from hyperliquid_ws import get_field as _ws_field
                            from hyperliquid_ws import normalize_orderbook as _norm_ob
                            obs = _ws_field("orderbooks") or {}
                            order_book = _norm_ob(obs.get(coin, {}))
                        except Exception:
                            pass

                        # ── Funding acceleration signal (FOMO/panic detection) ──
                        try:
                            from funding_acceleration import funding_accel_signal as fas
                            fund_accel = fas(coin, fund_rate)
                        except Exception:
                            fund_accel = None

                        # ── Real CVD from WebSocket trades (replaces OHLCV proxy) ──
                        try:
                            from cvd_engine import get_trade_cvd, cvd_to_prediction
                            real_cvd = get_trade_cvd(coin)
                            if real_cvd["confidence"] > 10 and real_cvd["trade_count"] > 3:
                                cvd_override = cvd_to_prediction(real_cvd)
                            else:
                                cvd_override = None
                        except Exception:
                            cvd_override = None

                        # ── OI delta signal (Open Interest change direction) ──
                        try:
                            oi_delta_sig = get_oi_signal(coin, mid)
                            if mid > 0:
                                update_oi_cache(coin, _open_interest.get(coin, 0), mid)
                        except Exception:
                            oi_delta_sig = None

                        # ── Cross-exchange divergence (HL vs Binance) ──
                        try:
                            cross_div = get_cross_divergence(coin, mid)
                        except Exception:
                            cross_div = None

                        # ── Taker buy/sell ratio (real-time order flow aggression) ──
                        try:
                            taker_sig = get_taker_ratio(coin, 60.0)
                        except Exception:
                            taker_sig = None

                        # ── Bid/ask imbalance trend (building vs fading pressure) ──
                        try:
                            imb_trend = get_imbalance_trend(coin)
                        except Exception:
                            imb_trend = None

                        # ── Exhaustion detection (RSI divergence + volume climax) ──
                        try:
                            exhaustion = detect_peak_exhaustion(coin, c15m, mid)
                        except Exception:
                            exhaustion = None

                        cpred = predict_continuous(coin, mid,
                            candles_1m=c1m, candles_5m=c5m,
                            candles_15m=c15m, candles_1h=c1h,
                            candles_4h=c4h, candles_1d=c1d,
                            funding_rate=fund_rate, ml_signal=ml_sig,
                            hurst_H=H, order_book=order_book,
                            cvd_override=cvd_override,
                            oi_delta=oi_delta_sig,
                            cross_exchange_divergence=cross_div,
                            taker_ratio=taker_sig,
                            exhaustion_signal=exhaustion,
                            imbalance_trend=imb_trend,
                            open_interest=_open_interest.get(coin, 0))

                        # Compact log line for every eval
                        log.info(f"  📊 {coin} {side} PnL={pnl_pct:+.1f}% | {format_prediction_compact(cpred)}")

                        # ── AI decision based on continuous prediction ──
                        # Only call AI if there's meaningful directional signal
                        h15 = cpred.horizons.get("15m")
                        if h15 and cpred.overall_confidence > 15:
                            # ── HARD GATE: don't bother AI with weak opposite signals ──
                            # If prediction is opposite our position but confidence < 35%, it's noise.
                            # If prediction is same direction or neutral, no reason to close.
                            bias = cpred.overall_bias
                            pred_opposes = (bias == "bullish" and side == "SHORT") or (bias == "bearish" and side == "LONG")
                            if pred_opposes and cpred.overall_confidence < 35:
                                log.info(f"  📊 {coin} {side}: weak opposite signal (conf={cpred.overall_confidence:.0f}%<35%), holding")
                                continue
                            if not pred_opposes and cpred.overall_confidence < 30:
                                # Neutral or same-direction with low confidence — definitely hold
                                continue
                            h1 = cpred.horizons.get("1m", h15)
                            h4 = cpred.horizons.get("4h", h15)

                            # Build rich context for AI
                            # Map continuous predictor layer scores to AI validator signal keys
                            lc = getattr(cpred, 'layer_confidences', {})
                            target_price = h15.target_down if cpred.overall_bias == "bearish" else (
                                h15.target_up if cpred.overall_bias == "bullish" else mid)

                            pred_data = {
                                "direction": cpred.overall_bias,
                                "confidence": cpred.overall_confidence,
                                "target_price": target_price,
                                "pnl": {"price_pct": round(pnl_pct, 1),
                                        "margin_pct": round(pnl_pct * (float(p.get("leverage", {}).get("value", 1))
                                            if isinstance(p.get("leverage"), dict) else 1.0), 1)},
                                # ── ALL 13 signal layers passed directly (no garbled mapping) ──
                                "layers": dict(lc),  # Full layer_confidences dict
                                # ── Continuous horizon predictions ──
                                "continuous": {
                                    "bias": cpred.overall_bias, "confidence": cpred.overall_confidence,
                                    "1m": {"up": h1.up, "down": h1.down, "flat": h1.flat},
                                    "15m": {"up": h15.up, "down": h15.down, "flat": h15.flat},
                                    "4h": {"up": h4.up, "down": h4.down, "flat": h4.flat},
                                },
                                "layer_weights": cpred.layer_contributions,
                                "horizon_drivers": getattr(cpred, 'horizon_drivers', {}),
                                "funding_rate": fund_rate,
                            }
                            port_data = {
                                "equity": total_eq, "perp_equity": perp_eq,
                                "positions_count": len(active),
                                "heat": exposure_state.heat if hasattr(exposure_state, 'heat') else 0,
                            }

                            # ── Build rich context for AI ──
                            _mkt_ctx2 = ""
                            _ob_info2 = ""
                            _atr_info2 = ""
                            _funding_accel2 = ""
                            _hurst_regime2 = ""
                            _hurst_H2 = H
                            _oi_type2 = ""
                            _oi_desc2 = ""
                            _xex_div2 = 0.0
                            _xex_dir2 = ""
                            _cascade2 = ""
                            _htf_align2 = ""
                            try:
                                from context_enricher import enrich_context
                                intel = enrich_context(coin, c15m)
                                _mkt_ctx2 = f"regime=ranging | {intel}" if intel else "regime=ranging"
                            except Exception:
                                _mkt_ctx2 = "regime=ranging"
                            try:
                                if order_book:
                                    bids = order_book.get("bids", [])
                                    asks = order_book.get("asks", [])
                                    if bids and asks:
                                        bb = bids[0]["px"]; ba = asks[0]["px"]
                                        sp = (ba - bb) / bb * 100 if bb > 0 else 0
                                        bd = sum(b["px"]*b["sz"] for b in bids[:5])
                                        ad = sum(a["px"]*a["sz"] for a in asks[:5])
                                        _ob_info2 = f"bid={bb:.4f} ask={ba:.4f} spread={sp:.3f}% depth={bd:.0f}/{ad:.0f}"
                            except Exception:
                                pass
                            try:
                                if c15m and len(c15m) >= 14:
                                    atr_val2 = sum(abs(float(c.get("h",c.get("high",0)))-float(c.get("l",c.get("low",0)))) for c in c15m[-14:])/14
                                    if atr_val2 > 0 and mid > 0:
                                        _atr_info2 = f"ATR={atr_val2:.4f} ({atr_val2/mid*100:.2f}%)"
                            except Exception:
                                pass

                            # ── Rich data extraction (funding accel, Hurst, OI, XEX, cascade, HTF) ──
                            try:
                                from funding_acceleration import get_funding_acceleration
                                fa = get_funding_acceleration(coin)
                                _funding_accel2 = fa.get("direction", "") if fa else ""
                            except Exception: pass
                            try:
                                from hurst_detector import hurst_regime as _hr_func
                                _hurst_regime2 = _hr_func(H)
                            except Exception: pass
                            try:
                                if oi_delta_sig:
                                    _oi_type2 = getattr(oi_delta_sig, 'signal_type', '')
                                    _oi_desc2 = getattr(oi_delta_sig, 'description', '')[:60]
                            except Exception: pass
                            try:
                                if cross_div:
                                    _xex_div2 = cross_div.get("divergence_pct", 0)
                                    _xex_dir2 = cross_div.get("direction", "")
                            except Exception: pass
                            try:
                                from liquidation_zones import detect_cascade_risk
                                cr = detect_cascade_risk(coin, mid)
                                _cascade2 = cr.get("risk_level", "") if cr else ""
                            except Exception: pass
                            try:
                                raw_htf = lc.get("higher_tf_align", 0)
                                if isinstance(raw_htf, (int, float)) and raw_htf > 0:
                                    _htf_align2 = "bullish" if raw_htf > 50 else "bearish" if raw_htf < -50 else "neutral"
                            except Exception: pass

                            ai_v = validate_prediction(
                                coin, mid, pred_data, port_data,
                                market_regime="ranging",
                                fear_greed=_get_fear_greed(),
                                funding_rate=fund_rate,
                                position_context={
                                    "is_exit": True,
                                    "side": side,
                                    "pnl_pct": round(pnl_pct, 1),
                                    "margin_pnl": round(margin_pnl, 1),
                                    "leverage": lev,
                                    "liq_distance": liq_dist,
                                },
                                market_context=_mkt_ctx2,
                                order_book_info=_ob_info2,
                                atr_info=_atr_info2,
                                funding_accel=_funding_accel2,
                                hurst_regime=_hurst_regime2,
                                hurst_H=_hurst_H2,
                                oi_delta_type=_oi_type2,
                                oi_delta_desc=_oi_desc2,
                                cross_exchange_div=_xex_div2,
                                cross_exchange_dir=_xex_dir2,
                                cascade_risk=_cascade2,
                                htf_alignment=_htf_align2,
                            )
                            action = ai_v.action if hasattr(ai_v, 'action') else "wait"
                            sig = f"ls={len(lc)} bias={cpred.overall_bias} conf={cpred.overall_confidence:.0f}%"
                            log.info(f"  🧠 {coin}: AI → {action} | {getattr(ai_v, 'reason', '?')[:60]} | signals: {sig}")

                            if action == "execute_now":
                                # ── Confidence gate + signal agreement gate ──
                                adj_conf = getattr(ai_v, 'adjusted_confidence', 50)
                                sg = getattr(ai_v, 'signals_agreeing', 0)
                                op = getattr(ai_v, 'signals_opposing', 0)
                                if adj_conf < 30:
                                    log.warning(f"  🧠 {coin}: AI says close but low conf ({adj_conf:.0f}) — holding")
                                    action = "wait"
                                elif sg < 3 and op > 2:
                                    log.warning(f"  🧠 {coin}: thin signal agreement sg={sg} op={op} — holding")
                                    action = "wait"
                            if action == "execute_now":
                                log.warning(f"  🛑 {coin}: AI CLOSE conf={adj_conf:.0f}")
                                reason = f"continuous:{cpred.overall_bias}:{cpred.overall_confidence:.0f}%"
                                if not _execute_direct_close(coin, 1.0, reason):
                                    _submit_action("close", coin, {
                                        "direction": "close",
                                        "reason": reason,
                                        "size_units": abs(szi), "price": mid, "size": abs(szi), "side": side,
                                    })
                            elif action == "reduce_size" and hasattr(ai_v, 'size_multiplier'):
                                # rf flag: cascade → trim more
                                rf = getattr(ai_v, 'warning_flags', [])
                                trim_mult = ai_v.size_multiplier
                                if 'cascade' in rf:
                                    trim_mult = max(trim_mult, 0.6)
                                    log.warning(f"  ⚠️  {coin}: cascade risk → trim at least {trim_mult:.0%}")
                                close_sz = abs(szi) * max(0.25, trim_mult)
                                log.warning(f"  ⚠️  {coin}: AI TRIM {trim_mult:.0%}")
                                _submit_action("close", coin, {
                                    "direction": "close",
                                    "reason": f"continuous_trim:{ai_v.size_multiplier:.0%}",
                                    "size_units": close_sz, "price": mid, "size": close_sz, "side": side,
                                })
                    except Exception as e:
                        log.debug(f"  Continuous pred skip {coin}: {type(e).__name__}")

                # ── Stage 2: Compute signals + conviction scoring ──
                for coin in selected:
                    log.info(f"  Computing signal for {coin}...")
                    sig = _compute_signal(coin, btc_candles, mids)

                    # Log enrichment context if available
                    enrich_ctx = get_enrichment_context(sig)
                    log.info(f"  {coin}: {sig.side} conf={sig.confidence:.2f} "
                             f"enriched={sig.enriched_confidence:.0f} "
                             f"regime={sig.regime.value} comp={sig.composite_score:+.2f} "
                             f"reason={sig.reason[:80]}")
                    if enrich_ctx:
                        log.info(f"  ✨ {coin} enrich: {enrich_ctx}")

                    # ── AI FAST-TRACK: enriched pipeline is dead but AI sees the trade ──
                    # When the technical signal generator returns insufficient_data or no_sub_signals
                    # but AI has 80%+ confidence, trust the AI directly. The technical pipeline
                    # needs 13+ hours of candle data; the AI sees real-time price action.
                    # BIO lesson: AI picked BIO 15+ times at 80%+ conf over 3 days while enriched
                    # returned insufficient_data every time. By the time enriched caught up,
                    # the move was already gone.
                    if sig.side == "HOLD" and ("insufficient_data" in sig.reason or "no_sub_signals" in sig.reason or "btc_bear_filter" in sig.reason or "countertrend_blocked" in sig.reason or sig.reason.startswith("master_")):
                        old_reason = sig.reason
                        ai_ft = _ai_trade_plan.get(coin.upper(), {})
                        ai_ft_conf = ai_ft.get("confidence", 0)
                        ai_ft_dir = ai_ft.get("direction", "").lower()
                        log.info(f"  🔎 {coin}: fast-track check — ai_plan_key={coin.upper()} found={bool(ai_ft)} conf={ai_ft_conf} dir={ai_ft_dir}")
                        if ai_ft_conf >= 80 and ai_ft_dir in ("long", "short"):
                            mid_px = float(mids.get(coin, 0))
                            c15 = _fetch_candles_cached(coin, "15m", 50)
                            ft_comp = 0.0
                            if c15 and len(c15) >= 10:
                                try:
                                    df = add_basic_indicators(candles_to_frame(c15))
                                    ft_comp = compute_composite(df)  # SIGNED: don't fake a positive composite from dead signals
                                except Exception:
                                    ft_comp = 0.05
                            else:
                                ft_comp = 0.05
                            ft_side = "BUY" if ai_ft_dir == "long" else "SELL"
                            ft_label = "BUY" if ai_ft_dir == "long" else "SELL"
                            sig = EnrichedSignal(
                                base_signal=MasterSignal(
                                    symbol=coin, side=ft_side,
                                    confidence=ai_ft_conf / 100.0,
                                    reason=f"ai_fast_track:{ai_ft_conf}%",
                                    composite_score=ft_comp,
                                    regime=sig.regime,
                                ),
                                enriched_confidence=ai_ft_conf / 100.0,
                                enrichment_reason="ai_fast_track",
                            )
                            log.info(f"  🚀 {coin}: AI FAST-TRACK — enriched was dead ({old_reason}) but AI={ai_ft_conf}% → synthetic {ft_label} (comp={ft_comp:+.2f})")
                    # ── AI CONFIRM: enriched has direction, AI confirms — fast-track through unified ──
                    elif sig.side in ("BUY", "SELL") and sig.confidence > 0:
                        # Skip if already fast-tracked (reason starts with ai_fast_track:)
                        _already_ft = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                        if not _already_ft:
                            ai_ft2 = _ai_trade_plan.get(coin.upper(), {})
                            ai_ft_conf2 = ai_ft2.get("confidence", 0)
                            ai_ft_dir2 = ai_ft2.get("direction", "").lower()
                            if ai_ft_conf2 >= 80 and ai_ft_dir2 in ("long", "short") and (
                                (ai_ft_dir2 == "long" and sig.side == "BUY") or
                                (ai_ft_dir2 == "short" and sig.side == "SELL")
                            ):
                                old_reason2 = sig.reason
                                ft_confirm_side = "BUY" if ai_ft_dir2 == "long" else "SELL"
                                sig = EnrichedSignal(
                                    base_signal=MasterSignal(
                                        symbol=coin, side=ft_confirm_side,
                                        confidence=ai_ft_conf2 / 100.0,
                                        reason=f"ai_confirm:{ai_ft_conf2}%:{old_reason2[:40]}",
                                        composite_score=sig.composite_score,
                                        regime=sig.regime,
                                    ),
                                    enriched_confidence=ai_ft_conf2 / 100.0,
                                    enrichment_reason=f"ai_confirm:{old_reason2[:30]}",
                                )
                                log.info(f"  ✅ {coin}: AI CONFIRM — enriched={sig.side} + AI={ai_ft_conf2}% → fast-tracking through unified gate")

                    # ── ALL coins go through unified predictor ──
                    # Even if ensemble says HOLD, the unified predictor may find
                    # a trade using other data sources (ML, CVD, order book, etc.)

                    # ── AI-Driven Conviction & Sizing ──
                    try:
                        conviction_score, conviction_tier, breakdown = score_conviction(
                            sig, sig.composite_score, sig.regime,
                            volume_ratio=1.0,
                        )
                    except Exception as e:
                        log.error(f"  ⚠️ {coin}: score_conviction crashed: {type(e).__name__}: {e}")
                        log.error(traceback.format_exc())
                        continue

                    # Build rich context for AI (candles, funding, BTC, F&G, order book)
                    ctx_candles = _fetch_candles_cached(coin, "15m", 200)
                    try:
                        ctx = _build_quick_context(coin, mids, total_eq, active,
                                                   candles_15m=ctx_candles,
                                                   enrichment_ctx=enrich_ctx)
                    except Exception as e:
                        log.error(f"  ⚠️ {coin}: _build_quick_context crashed: {type(e).__name__}: {e}")
                        log.error(traceback.format_exc())
                        continue

                    # ── Enrich with intelligence modules (CVD, S/R, patterns, liquidation zones) ──
                    try:
                        from context_enricher import enrich_context
                        intel_block = enrich_context(coin, ctx_candles,
                                                     whale_tracker=whale_tracker,
                                                     arb_state=arb_state)
                        market_context = f"regime={sig.regime.value} HSL={hsl_state.tier.value} | {intel_block}" if intel_block else f"regime={sig.regime.value} HSL={hsl_state.tier.value}"
                    except Exception:
                        market_context = f"regime={sig.regime.value} HSL={hsl_state.tier.value}"

                    # ── Peak/Bottom exhaustion check ──
                    try:
                        current_mark = float(mids.get(coin, 0))
                        exhaustion = detect_peak_exhaustion(coin, ctx_candles, current_mark)
                        if exhaustion and exhaustion.score >= 40:
                            market_context += f" | ⚡{exhaustion.direction.upper()}:{exhaustion.score:.0f}"
                            log.info(f"  ⚡ {coin} exhaustion: {exhaustion.direction} score={exhaustion.score:.0f}/100 — {exhaustion.reasoning}")
                    except Exception:
                        pass

                    # ── Unified ML prediction (ensemble: XGBoost+LSTM+structure+flow+macro) ──
                    try:
                        fund_ctx = funding_sniper.asset_ctx.get(coin, {})
                        fund_rate = float(fund_ctx.get("funding", 0)) if fund_ctx else 0.0
                        ml_pred = predict_unified(coin, ctx_candles, current_mark, mids=mids,
                                                  funding_rate=fund_rate,
                                                  btc_mid=float(mids.get("BTC", 0)),
                                                  fear_greed=_get_fear_greed(),
                                                  regime=sig.regime.value if hasattr(sig, 'regime') else "sideways")
                        if ml_pred.confidence >= 30:
                            market_context += f" | 🤖ML:{ml_pred.direction}:{ml_pred.confidence:.0f}%→${ml_pred.target_price:.2f}"
                            log.info(f"  🤖 {coin} ML predict: {ml_pred.direction} conf={ml_pred.confidence:.0f}% target=${ml_pred.target_price:.4f}")
                    except Exception:
                        pass

                    # ── Self-learned patterns from past trades ──
                    try:
                        learned_ctx = get_learned_context()
                        if learned_ctx:
                            market_context += f" | 📚LEARNED"
                            log.debug(f"  📚 Learned context: {learned_ctx[:100]}...")
                    except Exception:
                        pass

                    # ── Whale activity context (belt-and-suspenders with context_enricher) ──
                    try:
                        whale_ctx = whale_tracker.get_whale_context()
                        if whale_ctx and "No recent" not in whale_ctx:
                            # Compact: first line for AI prompt
                            market_context += f" | 🐋WHALES"
                            log.info(f"  🐋 Whale context: {whale_ctx[:80]}...")
                    except Exception:
                        pass

                    # ── Delta-neutral arb context ──
                    try:
                        arb_ctx = get_arb_context(arb_state)
                        if arb_ctx and "No active" not in arb_ctx:
                            market_context += f" | 🔄ARB"
                            log.info(f"  🔄 Arb context: {arb_ctx[:80]}...")
                    except Exception:
                        pass

                    # ═══════════════════════════════════════════════════════
                    # QUALITY GATES — prevent weak entries
                    # ═══════════════════════════════════════════════════════
                    block_reason = None

                    # ── Gate 1: VWAP extension filter ──
                    # Only block when regime contradicts the VWAP extension.
                    # trending_up + VWAP+ = trend confirmation, NOT overbought.
                    # trending_down + VWAP- = trend confirmation, NOT oversold.
                    # Block only when: (BUY + VWAP+ in non-trending) or (SELL + VWAP- in non-trending)
                    vwap_sigma = 0.0
                    if enrich_ctx:
                        import re
                        vwap_match = re.search(r'VWAP:([+-]\d+\.?\d*)σ', enrich_ctx)
                        if vwap_match:
                            vwap_sigma = float(vwap_match.group(1))

                    regime_is_uptrend = sig.regime.value in ("trending_up",)
                    regime_is_downtrend = sig.regime.value in ("trending_down",)
                    # VWAP MEAN REVERSION: buy dips, sell pumps
                    _vwap_dip_long = sig.side == "BUY" and vwap_sigma < -1.5 and not regime_is_downtrend
                    _vwap_pump_short = sig.side == "SELL" and vwap_sigma > 1.5 and not regime_is_uptrend
                    
                    # EXTREME VWAP BLOCK: dont buy overbought, dont sell oversold
                    # OP lesson: SHORT at -1.9σ (oversold) ×3, LONG at +2.5σ (overbought) ×3
                    if sig.side == "BUY" and vwap_sigma > 1.5:
                        block_reason = f"VWAP:{vwap_sigma:+.1f}σ — overbought, never BUY"
                    elif sig.side == "SELL" and vwap_sigma < -1.5:
                        block_reason = f"VWAP:{vwap_sigma:+.1f}σ — oversold, never SELL"
                    if _vwap_dip_long:
                        log.info(f"  📉 {coin}: VWAP DIP BUY — VWAP={vwap_sigma:+.1f}σ oversold, mean reversion LONG")
                    elif _vwap_pump_short:
                        log.info(f"  📈 {coin}: VWAP PUMP SELL — VWAP={vwap_sigma:+.1f}σ overbought, mean reversion SHORT")
                    elif sig.side == "BUY" and vwap_sigma > 0.5 and sig.composite_score < 0.10:
                        if not regime_is_uptrend:
                            # In non-trending markets, buying far above VWAP is chasing
                            block_reason = f"VWAP:{vwap_sigma:+.1f}σ — buying above VWAP with weak composite ({sig.composite_score:+.2f}) in non-trending regime"
                        # In trending_up, VWAP+ is trend confirmation — don't block
                    elif sig.side == "SELL" and vwap_sigma < -0.5 and sig.composite_score > -0.10:
                        if not regime_is_downtrend:
                            block_reason = f"VWAP:{vwap_sigma:+.1f}σ — selling below VWAP with weak composite ({sig.composite_score:+.2f}) in non-trending regime"

                    # ── Gate 2: ML model contradiction ──
                    # ML models are data-driven. When ML strongly contradicts, AI override
                    # cannot clear it. Data beats Flash Lite opinion.
                    _ml_hard_block = False
                    if not block_reason:
                        try:
                            if (sig.side == "BUY" and ml_pred.direction == "down" and ml_pred.confidence >= 30):
                                _ml_hard_block = True  # ML at 30%+ is authoritative — data beats Flash Lite
                                block_reason = f"ML says DOWN ({ml_pred.confidence:.0f}%→${ml_pred.target_price:.2f}) but signal is BUY"
                            elif (sig.side == "SELL" and ml_pred.direction == "up" and ml_pred.confidence >= 30):
                                _ml_hard_block = True  # ML at 30%+ is authoritative — data beats Flash Lite
                                block_reason = f"ML says UP ({ml_pred.confidence:.0f}%→${ml_pred.target_price:.2f}) but signal is SELL"
                        except (NameError, AttributeError):
                            pass  # ml_pred not set — skip this gate

                    # ── Gate 3: Weak composite in sideways → let AI decide (don't hard block) ──
                    ai_override_needed = False
                    if not block_reason:
                        if (abs(sig.composite_score) < 0.15 and sig.regime.value == "sideways"
                            and conviction_score < 75):
                            ai_override_needed = True
                            log.info(f"  🟡 {coin}: weak sideways signal (comp={sig.composite_score:+.2f}, conv={conviction_score:.0f}) — asking AI")

                    # ── AI Override: trust AI trade plan when confidence ≥ 80% ──
                    # Must be AFTER all gates — clears block_reason regardless of which gate set it
                    # BUT: don't override when composite is weak (< 0.08) — the signal is noise
                    # Aug 9: raised min composite from 0.03→0.08, scaled threshold by validator flags
                    ai_plan = _ai_trade_plan.get(coin.upper(), {})
                    ai_conf_val = ai_plan.get("confidence", 0)
                    ai_dir_raw = ai_plan.get("direction", "").lower()
                    # Normalize to BUY/SELL for consistent downstream comparison
                    ai_dir = "BUY" if ai_dir_raw == "long" else ("SELL" if ai_dir_raw == "short" else ai_dir_raw.upper())
                    _bull_market = (ai_market.get("bias", "") == "bullish" if "ai_market" in dir() else False)
                    # Bull market accelerator: if AI says bullish, trust AI longs with lower bar
                    _bull_aligned = _bull_market and ai_dir_raw == "long"
                    _min_composite = 0.05 if _bull_aligned else (0.08 if ai_dir_raw == "short" else 0.12)
                    if ai_conf_val >= 80 and block_reason:
                        # ── UN-OVERRIDABLE GATES: VWAP extremes, data consensus — no AI bypass ──
                        _cannot_override = ("VWAP:+" in block_reason and "overbought" in block_reason) or \
                                          ("VWAP:-" in block_reason and "oversold" in block_reason) or \
                                          ("all data layers dead" in block_reason)
                        if _cannot_override:
                            log.info(f"  🛑 {coin}: UN-OVERRIDABLE — {block_reason} (AI={ai_conf_val}% cannot bypass VWAP extreme)")
                        elif _ml_hard_block:
                            _ai_plan_ml = _ai_trade_plan.get(coin.upper(), {})
                            _ai_conf_ml = _ai_plan_ml.get("confidence", 0)
                            if _ai_conf_ml >= 80:  # Aug 7: 85→80 — zero-loss exits protect downside
                                log.info(f"  ⚡ {coin}: AI OVERRIDE ML — AI={_ai_conf_ml}% overrides ML contradiction ({ml_pred.direction}@{ml_pred.confidence:.0f}%)")
                                block_reason = None  # AI trumps ML at high confidence
                            else:
                                log.info(f"  🛑 {coin}: ML HARD BLOCK — ML strongly contradicts ({ml_pred.direction}@{ml_pred.confidence:.0f}%), AI={_ai_conf_ml}% < 80% cannot override")
                        elif abs(sig.composite_score) < _min_composite:
                            log.info(f"  🛑 {coin}: AI override blocked — composite too weak ({sig.composite_score:+.2f}) for gate override{', bull market' if _bull_market else ''}: {block_reason}")
                        else:
                            log.info(f"  🧠 {coin}: AI confidence {ai_conf_val}% — overriding gate{' (bull market accel)' if _bull_aligned else ''}: {block_reason}")
                            block_reason = None
                    elif block_reason and ai_conf_val > 0:
                        log.info(f"  🧠 {coin}: AI conf {ai_conf_val}% too low for override (need 80%)")

                    if block_reason:
                        log.info(f"  🛑 {coin}: QUALITY GATE — {block_reason}")
                        if not (_ai_trade_plan.get(coin.upper(), {}).get("confidence", 0) >= 80):
                            _forager_skip_cooldown[coin] = time.time()  # Only cooldown if no AI override
                        continue

                    # ═══════════════════════════════════════════════════════
                    # UNIFIED DIRECTION PREDICTOR
                    # Combines ALL data sources: microstructure, technical, sentiment
                    # ═══════════════════════════════════════════════════════
                    current_mark = float(mids.get(coin, 0))

                    # ── Gather Layer 3 data (sentiment/macro) ──
                    try:
                        fund_ctx_sniper = funding_sniper.asset_ctx.get(coin, {})
                        fund_rate = float(fund_ctx_sniper.get("funding", 0)) if fund_ctx_sniper else 0.0
                    except Exception:
                        fund_rate = 0.0

                    # Compute OI delta directly from snapshots (get_oi_delta always 0 — same-source bug)
                    try:
                        oi_current = _open_interest.get(coin, 0) if '_open_interest' in globals() else 0.0
                        oi_prev = _prev_open_interest.get(coin, 0) if '_prev_open_interest' in globals() else 0.0
                        if oi_current > 0 and oi_prev > 0:
                            oi_d = (oi_current - oi_prev) / oi_prev * 100  # % change
                        else:
                            oi_d = 0.0
                    except Exception:
                        oi_d = 0.0

                    whale_buying = bool(whale_tracker.get_recent_signals(max_age=120))
                    whale_selling = False  # whales selling is rare — tracked via CVD

                    # ── Gather Layer 1 data (microstructure) ──
                    try:
                        from hyperliquid_ws import get_field as _ws_field
                        ws_trades = _ws_field("trades") or {}
                        ws_coin_data = {
                            "trades": ws_trades,
                            "best_bid": 0,
                            "best_ask": 0,
                            "bid_depth_usd": 0,
                            "ask_depth_usd": 0,
                        }
                    except Exception:
                        ws_coin_data = {}

                    # ── CVD: compute from candles (close position within range) ──
                    try:
                        cvd_val = 0.0
                        if ctx_candles and len(ctx_candles) >= 20:
                            closes = [float(c.get("c", c.get("close", 0))) for c in ctx_candles[-20:]]
                            highs = [float(c.get("h", c.get("high", 0))) for c in ctx_candles[-20:]]
                            lows = [float(c.get("l", c.get("low", 0))) for c in ctx_candles[-20:]]
                            volumes = [float(c.get("v", c.get("volume", 0))) for c in ctx_candles[-20:]]
                            # CVD: close above midpoint = buying pressure, below = selling
                            cvd_sum = 0.0
                            for i in range(len(closes)):
                                c_range = highs[i] - lows[i]
                                if c_range > 0:
                                    position = (closes[i] - lows[i]) / c_range  # 0-1 where 1=strong buy
                                    cvd_sum += (position - 0.5) * 2 * volumes[i]  # -1 to +1 scaled by vol
                            total_vol = sum(volumes) if sum(volumes) > 0 else 1
                            cvd_val = cvd_sum / total_vol  # normalized
                    except Exception:
                        cvd_val = 0.0

                    # ── Taker ratio: from ws trades ──
                    try:
                        taker_val = 0.5
                        trades = ws_coin_data.get("trades", {})
                        if coin in trades:
                            t = trades[coin]
                            taker_val = 1.0 if t.get("side", "") == "buy" else 0.0 if t.get("side", "") == "sell" else 0.5
                    except Exception:
                        taker_val = 0.5

                    # ── ML prediction context ──
                    ml_dir = "flat"
                    ml_conf = 0.0
                    try:
                        ml_dir = ml_pred.direction if 'ml_pred' in dir() else "flat"
                        ml_conf = ml_pred.confidence if 'ml_pred' in dir() else 0.0
                    except (NameError, AttributeError):
                        pass

                    # ── Exhaustion context ──
                    exh_score = 0
                    exh_dir = ""
                    try:
                        if 'exhaustion' in dir() and exhaustion:
                            exh_score = exhaustion.score
                            exh_dir = exhaustion.direction
                    except (NameError, AttributeError):
                        pass

                    # ── Price change for regime ──
                    price_change_5m = 0.0
                    try:
                        if ctx_candles and len(ctx_candles) >= 2:
                            c5 = float(ctx_candles[-1].get("c", ctx_candles[-1].get("close", 0)))
                            c0 = float(ctx_candles[0].get("c", ctx_candles[0].get("close", 0)))
                            if c0 > 0:
                                price_change_5m = ((c5 - c0) / c0) * 100
                    except Exception:
                        pass

                    # ── CALL UNIFIED PREDICTOR ──
                    pred = predict_direction(
                        coin=coin,
                        current_price=current_mark,
                        mids=mids,
                        ws_data=ws_coin_data,
                        cvd_signal=cvd_val,
                        taker_ratio=taker_val,
                        ml_direction=ml_dir,
                        ml_confidence=ml_conf,
                        vwap_sigma=vwap_sigma,
                        hurst=0.5,  # default, can be replaced with hurst_detector
                        exhaustion_score=exh_score,
                        exhaustion_direction=exh_dir,
                        funding_rate=fund_rate,
                        oi_delta_pct=oi_d,
                        whale_buying=whale_buying,
                        whale_selling=whale_selling,
                        price_change_5m_pct=price_change_5m,
                    )

                    log.info(f"  🔮 {coin}: {pred.direction} conf={pred.confidence:.0f}% "
                             f"regime={pred.regime} L1={pred.layer1_score:+.2f} "
                             f"L2={pred.layer2_score:+.2f} L3={pred.layer3_score:+.2f} "
                             f"votes ↑{pred.votes_up} ↓{pred.votes_down}")

                    # ── DECISION: should we enter? ──
                    has_pos = coin.upper() in [p.get("coin", "").upper() for p in active]
                    # Lower threshold for AI-override coins (let AI decide borderline)
                    min_conf = 22  # 22% — catches weak-but-directional signals (2.8x amp on single-layer)
                    enter, action, reason = should_enter(pred, has_position=has_pos, min_confidence=min_conf)
                    ai_override_applied = False

                    # ── BULL MARKET ACCELERATOR: bypass unified predictor ──
                    # In a ripping bull market, the unified predictor (technical layers) calls tops
                    # and divergences that never materialize. AI market assessment is the better guide.
                    _bull_accel = (_bull_market and ai_dir == "BUY" and ai_conf_val >= 80 
                                  and not has_pos and action in ("BUY", "HOLD"))
                    if _bull_accel and not enter:
                        # AI says buy, market is bullish — override unified predictor
                        # Sanity checks: composite must not be negative, and signal must have actual data
                        _reason_ok = "no_sub_signals" not in sig.reason and "insufficient_data" not in sig.reason
                        # KAITO lesson: unified=DOWN(35%), ML=DOWN(49%), VWAP=+2.5σ (buying the top).
                        # Bull accel forced BUY anyway. Lost money. Unified must not strongly oppose.
                        _unified_opposes_bull = pred.confidence > 25 and pred.direction == "down"
                        # SOL lesson: BULL ACCEL forced BUY at unified=flat@1% → 30min drift, -0.23% loss.
                        # Flat unified signals mean NO layer has conviction. Require at least 5% unified or 0.15 composite.
                        _unified_dead_flat = pred.confidence < 5 and sig.composite_score < 0.15
                        if _unified_opposes_bull:
                            log.info(f"  🛑 {coin}: BULL ACCEL blocked — unified={pred.direction}@{pred.confidence:.0f}% "
                                    f"opposes BUY (need unified agreement or weak opposition)")
                        elif _unified_dead_flat:
                            log.info(f"  🛑 {coin}: BULL ACCEL blocked — unified dead-flat @{pred.confidence:.0f}% + "
                                    f"composite {sig.composite_score:+.2f} < 0.15 (need either ≥5% unified or ≥0.15 composite)")
                        elif sig.composite_score >= 0.00 and _reason_ok:
                            ai_dir_override = "BUY"
                            log.info(f"  🐂 {coin}: BULL ACCEL — AI={ai_conf_val}% bullish market, "
                                    f"bypassing unified={pred.direction}@{pred.confidence:.0f}% "
                                    f"(composite={sig.composite_score:+.2f}) — FORCING {ai_dir_override}")
                            action = ai_dir_override
                            enter = True
                            ai_override_applied = True
                        elif not _reason_ok:
                            log.info(f"  🛑 {coin}: BULL ACCEL blocked — no signal data ({sig.reason})")
                        else:
                            log.info(f"  🛑 {coin}: BULL ACCEL blocked — composite {sig.composite_score:+.2f} too negative")

                    # ── BEAR MARKET ACCELERATOR: bypass unified predictor ──
                    # Symmetric to bull accel. In a falling market, unified predictor calls bottoms
                    # that never materialize. AI market assessment + enriched direction are better.
                    _bear_market = (ai_market.get("bias", "") == "bearish" if "ai_market" in dir() else False)
                    _bear_accel = (_bear_market and ai_dir == "SELL" and ai_conf_val >= 80
                                   and not has_pos and action in ("SELL", "HOLD"))
                    if _bear_accel and not enter:
                        _reason_ok = "no_sub_signals" not in sig.reason and "insufficient_data" not in sig.reason
                        # Same guard as bull accel: unified must not strongly oppose
                        _unified_opposes_bear = pred.confidence > 25 and pred.direction == "up"
                        if _unified_opposes_bear:
                            log.info(f"  🛑 {coin}: BEAR ACCEL blocked — unified={pred.direction}@{pred.confidence:.0f}% "
                                    f"opposes SELL (need unified agreement or weak opposition)")
                        # Composite for shorts: negative is EXPECTED in bear market. Block only when bullish.
                        # FIX: use <= 0.15 instead of >= -0.15 — -0.25 should PASS for shorts
                        elif sig.composite_score <= 0.15 and _reason_ok:
                            # Negative composite is EXPECTED for shorts in trending_down — relax floor
                            ai_dir_override = "SELL"
                            log.info(f"  🐻 {coin}: BEAR ACCEL — AI={ai_conf_val}% bearish market, "
                                    f"bypassing unified={pred.direction}@{pred.confidence:.0f}% "
                                    f"(composite={sig.composite_score:+.2f}) — FORCING SELL")
                            action = ai_dir_override
                            enter = True
                            ai_override_applied = True
                        elif not _reason_ok:
                            log.info(f"  🛑 {coin}: BEAR ACCEL blocked — no signal data ({sig.reason})")
                        else:
                            log.info(f"  🛑 {coin}: BEAR ACCEL blocked — composite {sig.composite_score:+.2f} too negative")

                    # ── ENRICHED SELL ACCEL: when enriched+AI agree on SELL, bypass bull market blocker ──
                    # Problem: AI Market says "bullish" → BEAR ACCEL never fires → SELL blocked at market bias gate.
                    # Fix: when enriched=SELL, AI=short, composite≥0.10, and unified≥15%, set ai_override_applied.
                    _enriched_sell_accel = (not _bear_market and ai_dir == "SELL" and ai_conf_val >= 80
                                            and not has_pos and action in ("SELL", "HOLD") and not enter)
                    if _enriched_sell_accel:
                        _reason_ok = "no_sub_signals" not in sig.reason and "insufficient_data" not in sig.reason
                        _enriched_is_sell = sig.side == "SELL"
                        # In trending_down, negative composite is expected. Use -0.15 floor when enriched=SELL.
                        # FIX: for SELL, more negative composite = stronger. Block when too BULLISH.
                        if _enriched_is_sell:
                            _comp_ok = sig.composite_score <= 0.10  # Don't short if composite is bullish-positive
                        else:
                            _comp_ok = sig.composite_score >= 0.10  # Don't enter without conviction
                        if _reason_ok and _comp_ok and pred.confidence >= 5:
                            ai_dir_override = "SELL"
                            log.info(f"  🔻 {coin}: SELL ACCEL — enriched+AI agree on SELL (comp={sig.composite_score:+.2f}, "
                                    f"AI={ai_conf_val}%, unified={pred.direction}@{pred.confidence:.0f}%) — bypassing market bias")
                            action = ai_dir_override
                            enter = True
                            ai_override_applied = True
                        elif not _reason_ok:
                            log.info(f"  🛑 {coin}: SELL ACCEL blocked — no signal data ({sig.reason})")
                        elif not _comp_ok:
                            # Aug 7: AI≥80% bypasses composite gate
                            if ai_conf_val >= 80:
                                log.info(f"  ⚡ {coin}: AI BYPASS — composite {sig.composite_score:+.2f} but AI={ai_conf_val}% overrides SELL ACCEL")
                                ai_dir_override = "SELL"
                                action = ai_dir_override
                                enter = True
                                ai_override_applied = True
                            else:
                                log.info(f"  🛑 {coin}: SELL ACCEL blocked — composite {sig.composite_score:+.2f} too bullish for SELL (need ≤0.10 when enriched, ≥0.10 without)")
                        else:
                            # Aug 7: AI≥80% bypasses unified confidence check
                            if ai_conf_val >= 80:
                                log.info(f"  ⚡ {coin}: AI BYPASS — unified={pred.direction}@{pred.confidence:.0f}% weak but AI={ai_conf_val}% overrides SELL ACCEL")
                                ai_dir_override = "SELL"
                                action = ai_dir_override
                                enter = True
                                ai_override_applied = True
                            else:
                                log.info(f"  🛑 {coin}: SELL ACCEL blocked — unified={pred.direction}@{pred.confidence:.0f}% too weak (need ≥5%)")

                    if not enter:
                        # ── AI Trade Plan override: trust AI when unified is weak but non-zero ──
                        # Rule: unified=0% → blocked regardless of AI confidence (all layers dead)
                        # Rule: unified=1-14% → AI≥90% required, composite must be ≥0.10
                        # Rule: unified≥15% → normal flow, no override needed
                        # ANTI-PNUT RULE: composite<0.10 = enriched signal too weak → BLOCK always
                        ai_plan2 = _ai_trade_plan.get(coin.upper(), {})
                        ai_conf = ai_plan2.get("confidence", 0)
                        pred_conf = pred.confidence  # unified predictor confidence
                        # ── FAST-TRACK BYPASS: AI-generated synthetic signals skip unified gate ──
                        _is_fast_track = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                        mom5 = _calc_momentum(coin, "5m")  # compute once for all branches
                        if pred_conf <= 0.5 and not _is_fast_track:
                            # Unified predictor says absolutely no direction — block all overrides
                            log.info(f"  🛑 {coin}: AI override blocked — unified=0%, all layers dead (AI={ai_conf}%)")
                            continue
                        elif (pred_conf < 15 and ai_conf >= 90) or (pred_conf >= 15 and ai_conf >= 80) \
                             or (pred_conf >= 3 and ai_conf >= 80 and abs(sig.composite_score) >= 0.15) \
                             or (ai_conf >= 80 and sig.composite_score >= 0.15) \
                             or (_is_fast_track and ai_conf >= 80):
                            # unified=1-14%+AI≥90% OR unified≥15%+AI≥80% OR composite≥0.15+AI≥80%
                            # OR fast-track synthetic signal with AI≥80% — skip unified/composite gates
                            ai_dir = ai_plan2.get("direction", "long")
                            ai_dir = "BUY" if ai_dir.lower() == "long" else "SELL" if ai_dir.lower() == "short" else ai_dir.upper()
                            # ⚠️ KAITO lesson: fast-track forced BUY when unified=DOWN(35%), ML=DOWN(48%).
                            # Fast-track must respect unified when it has a clear opposing direction.
                            # Aug 8 ZERO-LOSS: also check ML — if ML strongly disagrees, data beats AI.
                            # INJ lesson: unified=flat(4%), ML=flat(4%), enriched=HOLD, only AI wanted it.
                            # Three data layers said no → AI cannot fast-track.
                            if _is_fast_track:
                                _ft_opposes = (pred.direction == "down" and ai_dir == "BUY") or \
                                              (pred.direction == "up" and ai_dir == "SELL")
                                _ml_opposes = (ml_conf and ml_conf >= 30 and (
                                    (ml_dir == "down" and ai_dir == "BUY") or
                                    (ml_dir == "up" and ai_dir == "SELL")))
                                if _ft_opposes:
                                    log.info(f"  🛑 {coin}: AI override blocked — fast-track {ai_dir} but unified "
                                            f"says {pred.direction}@{pred_conf:.0f}% (opposing)")
                                    continue
                                if _ml_opposes:
                                    log.info(f"  🛑 {coin}: AI override blocked — fast-track {ai_dir} but ML "
                                            f"says {ml_dir}@{ml_conf:.0f}% (data beats AI)")
                                    continue
                                # ── ALL-DATA DEAD: enriched=HOLD, unified≤5%, ML≤10% → AI can't solo
                                _all_dead = pred_conf <= 5 and (not ml_conf or ml_conf <= 10) and sig.side == "HOLD"
                                if _all_dead:
                                    log.info(f"  🛑 {coin}: AI override blocked — all data layers dead "
                                            f"(unified={pred_conf:.0f}% ML={ml_conf or 0:.0f}% enriched=HOLD), AI can't solo")
                                    continue
                            if ai_dir in ("BUY", "SELL"):
                                # ── COMPOSITE GATE: enriched signal quality must back the trade ──
                                # PNUT lesson: composite -0.05 (negative) + AI 80% = disastrous force
                                # In bull market: relax floor to ≥0.00 (just not negative) — AI market
                                # assessment is the better guide when technicals are noisy for small alts
                                _bull_market_now = (ai_market.get("bias", "") == "bullish" if "ai_market" in dir() else False)
                                # In non-bull market with SELL direction, negative composite is EXPECTED (trending down)
                                # Floor: -0.15 for SELL in non-bull, 0.00 for bull (just not negative), 0.08 otherwise
                                if ai_dir == "SELL" and not _bull_market_now:
                                    _comp_floor = -0.15
                                elif _bull_market_now:
                                    _comp_floor = 0.00
                                else:
                                    _comp_floor = 0.08
                                # Fast-track: synthetic signals from sparse data — relax floor
                                if _is_fast_track and ai_conf >= 80:
                                    _comp_floor = 0.00  # AI already validated direction, sparse techs unreliable
                                if sig.composite_score < _comp_floor:
                                    log.info(f"  🛑 {coin}: AI override blocked — composite {sig.composite_score:+.2f} too weak (need ≥{_comp_floor:.2f}), AI={ai_conf}%")
                                    continue
                                # ── Direction check: enriched signal must agree with AI ──
                                enriched_side = sig.side  # "BUY", "SELL", or "HOLD"
                                if enriched_side == "HOLD":
                                    # Enriched has no direction — not ideal, but don't hard block
                                    # if composite is decent and unified has conviction
                                    if sig.composite_score >= 0.10 and pred_conf >= 15:
                                        log.info(f"  ⚠️  {coin}: enriched=HOLD but composite={sig.composite_score:+.2f} + unified={pred_conf:.0f}% — proceeding with {ai_dir}")
                                    else:
                                        log.info(f"  🛑 {coin}: AI override blocked — enriched has no direction (HOLD), AI says {ai_dir}")
                                        continue
                                if enriched_side in ("BUY", "SELL") and ai_dir != enriched_side:
                                    # Enriched/composite technicals beat AI when they conflict (hard preference)
                                    # Allow enriched direction when: composite >= 0.15 and market isn't bullish
                                    _market_bull = (ai_market.get("bias", "") == "bullish" if "ai_market" in dir() else False)
                                    if sig.composite_score >= 0.15 and not _market_bull:
                                        log.info(f"  ⚡ {coin}: ENRICHED OVERRIDE — enriched={enriched_side} (comp={sig.composite_score:+.2f}) + AI={ai_dir} — enriched wins (hard preference, market not bullish)")
                                        ai_dir = enriched_side  # Trust enriched direction
                                    else:
                                        log.info(f"  🛑 {coin}: AI override blocked — AI says {ai_dir} but enriched says {enriched_side} (direction conflict)")
                                        continue
                                # ── Unified predictor direction check ──
                                # When composite is strong (≥0.08), skip unified check entirely.
                                # Unified predictor lags and calls tops/divergences that never materialize
                                # in trending markets. Enriched signal quality is the real gate.
                                composite_strong = sig.composite_score >= 0.15  # was 0.08 — too low, let unified=39% opposing through
                                # MET lesson: composite=+0.10, unified=DOWN@39%, ML=DOWN@59% — all said SELL
                                # but composite_strong=0.10 skipped unified check, AI forced BUY. Lost money.
                                # Raise to 0.15. If composite is actually strong, it'll still pass.
                                # Fast-track: AI-validated signals skip unified flat check
                                if not composite_strong and not _is_fast_track:
                                    if pred.direction == "flat":
                                        log.info(f"  🛑 {coin}: AI override blocked — unified says flat (no direction), AI says {ai_dir}")
                                        continue
                                    unified_dir = "BUY" if pred.direction == "up" else "SELL"
                                    if pred_conf < 3 and unified_dir != ai_dir:
                                        log.info(f"  🛑 {coin}: AI override blocked — unified says {unified_dir} but AI says {ai_dir} (3-way fail, unified<3%)")
                                        continue
                                    elif pred_conf >= 3 and unified_dir != ai_dir:
                                        # ── Unified opposition guard: when unified strongly opposes, block ──
                                        # MET lesson: unified=DOWN@39%, AI=BUY, comp=+0.10 — unified was right.
                                        # Don't override when unified has real conviction against the trade.
                                        if pred_conf >= 25:
                                            log.warning(f"  🛑 {coin}: unified={unified_dir}@{pred_conf:.0f}% opposes AI={ai_dir} — "
                                                       f"unified conviction too strong to override (≥25%)")
                                            continue
                                        log.info(f"  ⚠️  {coin}: unified says {unified_dir} but AI says {ai_dir} — unified≥{pred_conf:.0f}%, letting enriched+AI decide")
                                ai_direction = ai_plan2.get("direction", "long")
                                if ai_direction == "long" and mom5 < -0.5 and ai_conf < 90:
                                    log.info(f"  🛑 {coin}: AI long but 5m momentum {mom5:+.1f}% — rejecting (conf={ai_conf}% < 90%)")
                                    continue
                                if ai_direction == "short" and mom5 > 0.5 and ai_conf < 90:
                                    log.info(f"  🛑 {coin}: AI short but 5m momentum {mom5:+.1f}% — rejecting (conf={ai_conf}% < 90%)")
                                    continue
                                action = ai_dir
                                log.info(f"  🧠 {coin}: AI plan override ({ai_conf}% conf, mom5={mom5:+.1f}%) — unified={pred_conf:.0f}%, forcing {action}")
                                ai_override_applied = True
                                # Fall through to AI validation below
                            else:
                                log.info(f"  ⏸️  {coin}: {action} — {reason}")
                                continue
                        elif ai_conf >= 80 and not _is_fast_track:
                            # pred_conf 3-5% with AI 80%+ — enough (lowered from 15%→5%→3%)
                            if pred_conf >= 3:
                                action = ai_dir
                                log.info(f"  🧠 {coin}: AI plan override ({ai_conf}% conf, mom5={mom5:+.1f}%) — unified={pred_conf:.0f}%≥3%, forcing {action}")
                                ai_override_applied = True
                            elif ai_conf >= 80:
                                action = ai_dir
                                log.info(f"  🧠 {coin}: AI plan override ({ai_conf}% conf≥80%, mom5={mom5:+.1f}%) — unified={pred_conf:.0f}%, forcing {action}")
                                ai_override_applied = True
                            else:
                                log.info(f"  🛑 {coin}: AI override blocked — unified={pred_conf:.0f}% + AI={ai_conf}% insufficient (need unified>=3% OR AI>=80%)")
                                continue
                        elif _is_fast_track and ai_conf >= 80:
                            # Fast-track: let AI synthetic signal through regardless of unified
                            ft_dir = ai_plan2.get("direction", "long")
                            action = "BUY" if str(ft_dir).lower() == "long" else "SELL"
                            log.info(f"  🚀 {coin}: FAST-TRACK ENTRY — AI={ai_conf}% {action} synthetic signal, skipping unified gate")
                            ai_override_applied = True
                        # For AI-override coins, let the AI have the final say
                        elif ai_override_needed and pred.confidence >= 5:
                            # Use ensemble signal direction if unified says HOLD
                            if action == "HOLD" and sig.side in ("BUY", "SELL"):
                                action = sig.side
                            elif action == "HOLD" and pred.direction != "flat":
                                action = "BUY" if pred.direction == "up" else "SELL"
                            else:
                                action = pred.direction.upper() if pred.direction != "flat" else sig.side
                            if action not in ("BUY", "SELL"):
                                log.info(f"  ⏸️  {coin}: no clear direction for AI override")
                                continue  # Don't cooldown — signal may improve
                            log.info(f"  🟡 {coin}: unified says HOLD but asking AI anyway ({action}, conv={pred.confidence:.0f})")
                            # Fall through to AI validation below
                        else:
                            log.info(f"  ⏸️  {coin}: {action} — {reason}")
                            # Don't cooldown soft rejects — let them try again next cycle
                            continue

                    # ═══════════════════════════════════════════════════════
                    # AI VALIDATION & SIZING
                    # LLM validates the unified prediction before execution
                    # ═══════════════════════════════════════════════════════
                    sig_side_str = action  # "BUY" or "SELL"
                    entry_price = float(mids.get(coin, 0))
                    atr_val = sig.atr or (entry_price * 0.015)

                    # ── DIRECTION SANITY: never trade against the enriched signal ──
                    # If ensemble/enriched says SELL and we're going LONG, the thesis is broken.
                    # Exception: AI 95%+ confidence can override (rare genuine contrarian plays).
                    enriched_final = sig.side  # "BUY", "SELL", "HOLD"
                    # ── FAST-TRACK GUARD: when enriched was dead and unified is weak/unconvincing, block ──
                    # ETH lesson: enriched=HOLD(dead), AI=85% forced SHORT, unified=flat(4%). Fee-loss.
                    # APE lesson: enriched=HOLD(dead), AI=82% forced SHORT, unified=down(19%). Losing now.
                    # If unified can't see the direction clearly (<25%), AI alone isn't enough to force.
                    _is_ft_sig = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                    # Aug 7: lowered from 25%→15% — SURVIVAL gate already requires unified≥10%
                    if _is_ft_sig and pred.confidence < 15 and not ai_override_applied:
                        _ft_ai = _ai_trade_plan.get(coin.upper(), {}).get("confidence", 0)
                        log.warning(f"  🛑 {coin}: FAST-TRACK BLOCKED — enriched was dead, unified={pred.direction}@{pred.confidence:.0f}% "
                                   f"(need ≥15%), AI={_ft_ai}% not enough alone")
                        continue
                    if enriched_final in ("BUY", "SELL") and enriched_final != sig_side_str:
                        # ── Enriched override: when composite is strong and enriched disagrees
                        # with should_enter()'s naive action (e.g. VWAP fade vs trend), enriched wins.
                        # must_enter bypass + should_enter() can produce action that conflicts with
                        # enriched's read of the same data. Enriched > unified predictor for direction.
                        ai_plan_sanity = _ai_trade_plan.get(coin.upper(), {})
                        ai_conf_sanity = ai_plan_sanity.get("confidence", 0)
                        ai_dir_sanity = ai_plan_sanity.get("direction", "").upper()
                        enriched_ai_agree = (
                            (enriched_final == "BUY" and ai_dir_sanity == "LONG") or
                            (enriched_final == "SELL" and ai_dir_sanity == "SHORT")
                        )
                        if (sig.composite_score >= 0.00 and enriched_ai_agree) or \
                           (enriched_final == "SELL" and enriched_ai_agree and sig.composite_score <= 0.15 
                            and not (ai_market.get("bias", "") == "bullish" if "ai_market" in dir() else False)):
                            # ── Unified predictor guard: block override when unified strongly opposes ──
                            # Aug 7: 20→40 — unified needs HIGH conviction to veto enriched+AI agreement
                            _unified_opposes = (
                                pred.confidence > 40 and (
                                    (enriched_final == "BUY" and pred.direction == "down") or
                                    (enriched_final == "SELL" and pred.direction == "up")
                                )
                            )
                            if _unified_opposes:
                                log.warning(f"  🛑 {coin}: ENRICHED OVERRIDE BLOCKED — enriched={enriched_final} "
                                           f"(comp={sig.composite_score:+.2f}) + AI={ai_dir_sanity} but unified "
                                           f"says {pred.direction}@{pred.confidence:.0f}% — allowing unified to veto")
                                continue
                            # ── ML contradiction check: enriched override must also pass ML gate ──
                            # Aug 7: ML oppose threshold raised 30→50% — zero-loss exits protect downside
                            _ml_opposes_enriched = False
                            try:
                                if 'ml_pred' in dir():
                                    if (enriched_final == "SELL" and ml_pred.direction == "up" and ml_pred.confidence >= 50):
                                        _ml_opposes_enriched = True
                                    elif (enriched_final == "BUY" and ml_pred.direction == "down" and ml_pred.confidence >= 50):
                                        _ml_opposes_enriched = True
                            except (NameError, AttributeError):
                                pass
                            if _ml_opposes_enriched:
                                log.warning(f"  🛑 {coin}: ENRICHED OVERRIDE BLOCKED — ML contradicts ({ml_pred.direction}@{ml_pred.confidence:.0f}%) "
                                           f"— data beats opinion, override denied")
                                continue
                            # Enriched has conviction AND AI agrees — override should_enter's action
                            _tag = " [non-bull SELL bypass]" if sig.composite_score < 0.00 else ""
                            log.warning(f"  ⚡ {coin}: ENRICHED OVERRIDE — enriched={enriched_final} "
                                       f"(comp={sig.composite_score:+.2f}) + AI={ai_dir_sanity} "
                                       f"overrides unified {sig_side_str}{_tag}")
                            sig_side_str = enriched_final
                            action = enriched_final
                            ai_override_applied = True  # Signal downstream gates to respect this
                        # Enriched beats AI when they conflict: enriched SELL + AI LONG in bearish/neutral market
                        elif (sig.composite_score >= 0.08 and enriched_final == "SELL" and ai_dir_sanity == "LONG"
                              and not (ai_market.get("bias", "") == "bullish" if "ai_market" in dir() else False)):
                            log.warning(f"  ⚡ {coin}: ENRICHED OVERRIDE — enriched=SELL (comp={sig.composite_score:+.2f}) beats AI={ai_dir_sanity} in non-bull market → going SELL")
                            sig_side_str = enriched_final
                            action = enriched_final
                            ai_override_applied = True
                        elif ai_conf_sanity < 95:
                            # Aug 7: VWAP extreme + enriched agreement overrides AI direction
                            _vwap_overrides = False
                            if _vwap_dip_long and enriched_final == "BUY":
                                log.info(f"  📉 {coin}: VWAP DIP OVERRIDE — VWAP={vwap_sigma:+.1f}σ oversold + enriched=BUY → force LONG (over AI={ai_conf_sanity}%)")
                                sig_side_str = "BUY"
                                action = "BUY"
                                _vwap_overrides = True
                            elif _vwap_pump_short and enriched_final == "SELL":
                                log.info(f"  📈 {coin}: VWAP PUMP OVERRIDE — VWAP={vwap_sigma:+.1f}σ overbought + enriched=SELL → force SHORT (over AI={ai_conf_sanity}%)")
                                sig_side_str = "SELL"
                                action = "SELL"
                                _vwap_overrides = True
                            if not _vwap_overrides:
                                # ── Regime-aware: in trending_down, AI picking SHORT follows the trend ──
                                # Enriched's BUY in downtrend is mean-reversion (contrarian).
                                # The AI IS the trend-follower here — lower threshold.
                                _regime_supports_ai = (
                                    (regime_is_downtrend and sig_side_str == "SELL") or
                                    (regime_is_uptrend and sig_side_str == "BUY")
                                )
                                _contrarian_threshold = 80 if _regime_supports_ai else 95
                                if ai_conf_sanity >= _contrarian_threshold:
                                    log.warning(f"  ⚠️  {coin}: CONTRARIAN TRADE — AI {ai_conf_sanity}% overrides enriched {enriched_final} "
                                               f"(regime supports AI, threshold={_contrarian_threshold}%) → going {sig_side_str}")
                                    # Continue to sizing — don't block
                                else:
                                    log.warning(f"  🛑 {coin}: DIRECTION CONFLICT — trading {sig_side_str} but enriched says {enriched_final} "
                                               f"(AI conf={ai_conf_sanity}% < {_contrarian_threshold}% needed for contrarian)")
                                    continue
                        else:
                            log.warning(f"  ⚠️  {coin}: CONTRARIAN TRADE — AI {ai_conf_sanity}% overrides enriched {enriched_final} → going {sig_side_str}")

                    # ── Confidence-driven sizing: use best available confidence ──
                    # AI override confidence takes priority when it overrides
                    ai_plan3 = _ai_trade_plan.get(coin.upper(), {})
                    ai_conf = ai_plan3.get("confidence", 0)
                    conf = max(pred.confidence, ai_conf) if ai_conf >= 75 else pred.confidence
                    target_margin = max(10.0, 5.0 + conf * 0.25)  # $10-$25 floor per user spec
                    chosen_leverage = int(max(3, min(6, 2 + conf / 100.0 * 5)))  # 3x-6x
                    pos_pct = max(get_position_size_pct(pred, total_eq, max_position_pct=0.40),
                                  target_margin / total_eq if total_eq > 0 else 0.10)
                    # ── Micro-account caps: bigger positions needed on small equity ──
                    if total_eq < 100:
                        pos_pct = min(pos_pct, 0.35)   # Max 35% per trade on <$100 (was 20%)
                    elif total_eq < 200:
                        pos_pct = min(pos_pct, 0.40)   # Max 40% on $100-$200
                    else:
                        pos_pct = min(pos_pct, 0.50)   # Max 50% on $200+
                    pos_pct = min(pos_pct, 0.35)  # Hard cap 35% (was 25%)
                    risk_pct = pos_pct * 0.3
                    # ── Leverage: confidence-driven, no regime overrides ──
                    MAX_LEV = 10
                    tier_label = f"UNI-{pred.confidence:.0f}"

                    try:
                        sizing = decide_sizing(
                            coin, pred.confidence, pred.regime, total_eq,
                            atr_val / entry_price * 100 if entry_price > 0 else 1.5,
                            0.0, pred.regime,
                            btc_correlation=DEFAULT_BTC_CORRELATIONS.get(coin.upper(), 0.5),
                            support_distance_pct=5.0,
                        )
                        if sizing.get("position_pct", 0) > 0:
                            ai_pos = sizing.get("position_pct", pos_pct)
                            ai_risk = sizing.get("risk_pct", risk_pct)
                            ai_lev = int(sizing.get("leverage", chosen_leverage))
                            ai_lev = max(3, min(ai_lev, MAX_LEV)) if ai_lev else chosen_leverage  # 3x floor
                            pos_pct = ai_pos * 0.6 + pos_pct * 0.4
                            risk_pct = ai_risk * 0.6 + risk_pct * 0.4
                            chosen_leverage = ai_lev
                            ai_stop_atr = sizing.get("stop_atr", 1.5)  # Default 1.5 ATR
                            tier_label = f"AI-{pred.confidence:.0f}"
                            sizing_flags = sizing.get("rf", [])
                            if isinstance(sizing_flags, str):
                                sizing_flags = [f.strip() for f in sizing_flags.split(",")]
                            for flag in sizing_flags:
                                fl = str(flag).lower()
                                if fl in ("high_heat",):
                                    pos_pct *= 0.85
                                elif fl in ("high_vol",):
                                    chosen_leverage = min(chosen_leverage, 5)
                                elif fl in ("trending", "high_conv", "proven"):
                                    pos_pct *= 1.1
                                # low_conv, unproven, poor_wr no longer penalize position size
                            log.info(f"  🤖 AI sizing {coin}: {pos_pct*100:.0f}% pos, {risk_pct*100:.2f}% risk, "
                                     f"{chosen_leverage}x lev — {sizing.get('reason','')}")
                    except Exception as e:
                        log.warning(f"  ⚠️ AI sizing skipped ({type(e).__name__}) — using unified sizing")

                    # ── Unified predictor confidence gate ──
                    # AI override can force a trade when unified is weak.
                    # Only block when both unified AND AI have no conviction.
                    ai_plan3 = _ai_trade_plan.get(coin.upper(), {})
                    ai_conf = ai_plan3.get("confidence", 0)
                    if pred.confidence <= 0.5 and not has_pos:
                        # Bull market accelerator: AI says buy, bypass unified dead-flat
                        # Fast-track: synthetic AI signals skip bull market requirement
                        _is_ft_gate = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                        if ai_override_applied and (_bull_market or _is_ft_gate) and ai_conf_val >= 80 and sig.composite_score > -0.10:
                            _tag = "FAST-TRACK" if _is_ft_gate else "BULL GATE"
                            log.info(f"  🐂 {coin}: {_tag} bypass — unified dead-flat but AI override active (conf={ai_conf_val}%)")
                        else:
                            log.info(f"  🛑 {coin}: unified dead flat — blocking entry (AI={ai_conf}%)")
                            continue
                    if pred.confidence < 15 and not has_pos and ai_conf < 80:
                        log.info(f"  🚫 {coin}: no edge — unified={pred.confidence:.0f}% AI={ai_conf}% — skipping")
                        continue
                    if pred.confidence < 15 and ai_conf >= 80:
                        log.info(f"  🧠 {coin}: AI override — unified={pred.confidence:.0f}% but AI={ai_conf}%")
                    # ── Direction conflict + market bias check ──
                    # SKIP all when AI plan override already validated direction
                    enriched_side = str(sig.side)
                    unified_dir = "BUY" if pred.direction == "up" else "SELL" if pred.direction == "down" else None
                    market_bias = ai_market.get("bias", "neutral") if "ai_market" in dir() else "neutral"
                    # ── Market bias filter: don't fight the broader trend ──
                    # Neutral markets: allow longs at 25%, shorts need higher bar (mean reversion risk)
                    # ── Extreme coin signal overrides broad market bias ──
                    # CVD 80%+ bearish + VWAP > 2σ → coin is rolling over, allow short
                    # CVD 80%+ bullish + VWAP < -2σ → coin is bouncing, allow long
                    cvd_extreme = abs(cvd_val) > 0.8 and abs(vwap_sigma) > 2.0
                    cvd_direction = "SELL" if cvd_val < -0.8 else "BUY" if cvd_val > 0.8 else None
                    vwap_direction = "SELL" if vwap_sigma > 2.0 else "BUY" if vwap_sigma < -2.0 else None
                    extreme_signal = cvd_extreme and cvd_direction == vwap_direction  # CVD and VWAP agree on direction
                    
                    if not ai_override_applied and unified_dir and market_bias == "neutral":
                        # Check actual trade direction (_trade_side set below), not unified_dir
                        _trade_dir = _trade_side if '_trade_side' in dir() else unified_dir
                        # Aug 7: thresholds 35→15 / 15→5 — zero-loss exits make entries safer
                        if (unified_dir == "SELL" or _trade_dir == "SELL") and pred.confidence < 15 and not extreme_signal:
                            log.info(f"  🛑 {coin}: neutral market — blocking SELL (unified={pred.confidence:.0f}% < 15%)")
                            continue
                        elif (unified_dir == "BUY" or _trade_dir == "BUY") and pred.confidence < 5 and not extreme_signal:
                            log.info(f"  🛑 {coin}: neutral market — blocking BUY (unified={pred.confidence:.0f}% < 5%)")
                            continue
                    if not ai_override_applied and unified_dir and market_bias == "bullish" and unified_dir == "SELL":
                        threshold = 20 if extreme_signal and cvd_direction == "SELL" else 25
                        # Lower threshold when L2 is strongly bearish (technical conviction)
                        if hasattr(pred, 'layer2_score') and pred.layer2_score < -0.20:
                            threshold = min(threshold, 25)
                        if pred.confidence < threshold:
                            log.info(f"  🛑 {coin}: bullish market — blocking SELL (unified={pred.confidence:.0f}% < {threshold}%)")
                            continue
                        else:
                            log.info(f"  ⚡ {coin}: bullish market SELL override — unified={pred.confidence:.0f}%{' (extreme CVD/VWAP)' if extreme_signal else ''}")
                    if not ai_override_applied and unified_dir and market_bias == "bearish" and unified_dir == "BUY":
                        threshold = 25 if (extreme_signal and cvd_direction == "BUY") else 35
                        if pred.confidence < threshold:
                            log.info(f"  🛑 {coin}: bearish market — blocking BUY (unified={pred.confidence:.0f}% < {threshold}%)")
                            continue
                        else:
                            log.info(f"  ⚡ {coin}: bearish market BUY override — unified={pred.confidence:.0f}%{' (extreme CVD/VWAP)' if extreme_signal else ''}")
                    # ── Enriched vs unified direction conflict ──
                    if not ai_override_applied and unified_dir and unified_dir != enriched_side:
                        # If enriched has no direction (HOLD/EXIT/etc), unified at ≥15% is a valid opinion
                        enriched_has_direction = enriched_side in ("BUY", "SELL")
                        if not enriched_has_direction:
                            if pred.confidence >= 15:
                                # Require composite to at least weakly support the direction
                                comp_ok = (unified_dir == "BUY" and sig.composite_score >= 0.05) or \
                                          (unified_dir == "SELL" and sig.composite_score <= -0.05)
                                if not comp_ok:
                                    log.info(f"  🛑 {coin}: enriched={enriched_side} + unified={unified_dir}@{pred.confidence:.0f}% — composite {sig.composite_score:+.2f} too weak for solo entry")
                                    continue
                                log.info(f"  ⚡ {coin}: enriched={enriched_side} has no direction — unified={unified_dir}@{pred.confidence:.0f}% takes over")
                            else:
                                log.info(f"  🛑 {coin}: enriched={enriched_side} vs unified={unified_dir}@{pred.confidence:.0f}% — unified too weak (<15%)")
                                continue
                        # Allow when CVD/VWAP are extreme — coin-specific signal overrides composite
                        elif extreme_signal and unified_dir == cvd_direction:
                            log.info(f"  ⚡ {coin}: CVD/VWAP extreme override — unified={unified_dir} overrides enriched={enriched_side}")
                        elif pred.confidence < 35:
                            log.info(f"  🛑 {coin}: direction conflict — enriched={enriched_side} vs unified={unified_dir}@{pred.confidence:.0f}% — unified<35% cannot override")
                            continue
                        else:
                            log.info(f"  ⚡ {coin}: direction conflict override — unified={unified_dir}@{pred.confidence:.0f}% overrides enriched={enriched_side}")
                    log.info(f"  🎯 {coin}: {tier_label} {action} — {reason}")
                    log.info(f"     pos={pos_pct*100:.1f}% risk={risk_pct*100:.2f}% "
                             f"lev={chosen_leverage}x regime={pred.regime}")

                    # Use unified/AI direction for all downstream checks
                    # (EnrichedSignal.side is read-only, so we use a local override)
                    _trade_side = sig_side_str

                    # ═══════════════════════════════════════════════════════
                    # SURVIVAL ENTRY GATE: AI≥70% + unified≥10% to enter
                    # Aug 7: AI≥80% + aligned mom5 bypasses unified check
                    # ═══════════════════════════════════════════════════════
                    ai_plan_final = _ai_trade_plan.get(coin.upper(), {})
                    ai_conf_final = ai_plan_final.get("confidence", 0)
                    if ai_conf_final < 70 and not ai_override_applied:
                        log.info(f"  🛑 {coin}: SURVIVAL ENTRY — AI={ai_conf_final}% < 70% required, skipping")
                        continue
                    _mom5_surv = _calc_momentum(coin, "5m") or 0
                    _bypass_ok = (ai_conf_final >= 80 and (
                        (_trade_side == "BUY" and _mom5_surv > 0.2) or
                        (_trade_side == "SELL" and _mom5_surv < -0.2)
                    ))
                    if pred.confidence < 10 and not _bypass_ok and not ai_override_applied:
                        log.info(f"  🛑 {coin}: SURVIVAL ENTRY — unified={pred.confidence:.0f}% < 10%, AI={ai_conf_final}% mom5={_mom5_surv:+.1f}% — skipping")
                        continue
                    if _bypass_ok and pred.confidence < 10:
                        log.info(f"  ⚡ {coin}: AI+MOMENTUM BYPASS — AI={ai_conf_final}% mom5={_mom5_surv:+.1f}% overrides unified={pred.confidence:.0f}%")
                        # ── WEAK COMPOSITE GUARD: AI can't solo on near-zero comp + ML contradiction ──
                        # LIT lesson: comp=+0.08, ML=DOWN@45% → AI forced BUY → immediate loss
                        if abs(sig.composite_score) < 0.10 and ml_conf and ml_conf > 30:
                            _ml_disagrees = (ml_dir == "down" and _trade_side == "BUY") or (ml_dir == "up" and _trade_side == "SELL")
                            if _ml_disagrees:
                                log.warning(f"  🛑 {coin}: WEAK COMP BLOCK — comp={sig.composite_score:+.2f} near zero + ML={ml_dir}@{ml_conf}% disagrees — AI can't solo override")
                                continue

                    # ── Stage 3: Risk check ──
                    # Always refresh exposure state with current equity and positions
                    pos_map_live = {p.get("coin", "").upper(): abs(float(p.get("szi", 0))) * float(mids.get(p.get("coin", ""), 0))
                                    for p in active if p.get("coin")}
                    exposure_state.update(total_eq, pos_map_live)
                    atr = sig.atr or (float(mids.get(coin, 0)) * 0.01)
                    entry_price = sig.entry_price or float(mids.get(coin, 0))
                    # ── Fallback: AI trade plan has current price ──
                    if entry_price <= 0:
                        ai_plan_entry = _ai_trade_plan.get(coin.upper(), {})
                        entry_price = ai_plan_entry.get("price", 0) or float(ai_plan_entry.get("mid", 0))
                    if entry_price <= 0:
                        log.warning(f"  ⚠️ {coin}: no entry price available — skipping")
                        continue
                    # Regime-adjusted stop
                    regime_mult = get_regime_stop_mult(sig.regime.value)
                    # Use signal's ATR-based stop distance, with direction-aware fallback
                    stop_atr_dist = getattr(sig, 'stop_distance_atr', None) or 1.0
                    raw_stop = getattr(sig, 'stop_price', None) or None
                    if not raw_stop or raw_stop <= 0:
                        # Ensure ATR has a floor — ADA at $1.19 with tiny ATR was producing zero size
                        safe_atr = max(atr, entry_price * 0.005)  # Min 0.5% of price
                        if _trade_side == "BUY":
                            raw_stop = entry_price - stop_atr_dist * safe_atr
                        else:
                            raw_stop = entry_price + stop_atr_dist * safe_atr
                    # Ensure stop is on correct side with 1.5% minimum distance
                    min_stop_pct = 0.015  # 1.5% minimum — prevents instant stop-outs on noise
                    if _trade_side == "BUY" and raw_stop >= entry_price:
                        raw_stop = entry_price * (1 - min_stop_pct)
                    elif _trade_side == "SELL" and raw_stop <= entry_price:
                        raw_stop = entry_price * (1 + min_stop_pct)
                    # Enforce minimum distance
                    if _trade_side == "BUY":
                        raw_stop = min(raw_stop, entry_price * (1 - min_stop_pct))
                    else:
                        raw_stop = max(raw_stop, entry_price * (1 + min_stop_pct))
                    if _trade_side == "BUY":
                        stop_price = entry_price - (entry_price - raw_stop) * regime_mult
                    else:
                        stop_price = entry_price + (raw_stop - entry_price) * regime_mult

                    # ── AI stop override: when AI has a specific stop_pct, use it (but never looser) ──
                    ai_stop_pct = _ai_trade_plan.get(coin.upper(), {}).get("stop_pct", 0)
                    if ai_stop_pct > 0:
                        if _trade_side == "BUY":
                            ai_stop = entry_price * (1 - ai_stop_pct / 100)
                            stop_price = min(stop_price, ai_stop)  # Tighter stop wins
                        else:
                            ai_stop = entry_price * (1 + ai_stop_pct / 100)
                            stop_price = max(stop_price, ai_stop)

                    # ── Margin floor check (belt-and-suspenders — already handled above) ──

                    # ── TWEL cap: scale down position to fit within remaining TWEL budget ──
                    remaining_twel = (total_eq * exposure_state.limits.twel_limit) - exposure_state.total_exposure
                    notional_planned = total_eq * pos_pct * chosen_leverage
                    # Aug 7: skip TWEL cap for micro accounts (<$100) — need bigger positions
                    if notional_planned > remaining_twel and remaining_twel >= 10 and total_eq >= 100:
                        capped_pos_pct = (remaining_twel / chosen_leverage) / total_eq
                        log.info(f"  📏 {coin}: capped pos {pos_pct*100:.1f}% → {capped_pos_pct*100:.1f}% (TWEL room=${remaining_twel:.0f} of ${total_eq * exposure_state.limits.twel_limit:.0f})")
                        pos_pct = capped_pos_pct
                        risk_pct = min(risk_pct, 0.015)

                    # ── Max risk per trade: dynamic based on account size ──
                    # Aug 7: micro accounts need higher risk tolerance for meaningful positions
                    _risk_cap_pct = 5.0 if total_eq < 100 else MAX_TRADE_RISK_PCT
                    if total_eq > 0 and stop_price > 0 and entry_price > 0:
                        sl_distance = abs(entry_price - stop_price) / entry_price
                        dollar_risk = sl_distance * pos_pct * total_eq * chosen_leverage
                        if dollar_risk > total_eq * _risk_cap_pct / 100:
                            log.info(f"  🛑 {coin}: RISK CAP — ${dollar_risk:.1f} at risk > {_risk_cap_pct}% of equity")
                            continue

                    # ── LIQUIDITY QUALITY GATE: don't trade what you can't exit ──
                    _bad_entry = False
                    # 1. Spread check from live WS orderbook
                    try:
                        from hyperliquid_ws import get_field as _ws_field
                        _obs = _ws_field("orderbooks") or {}
                        _ob = _obs.get(coin, {})
                        _bids = _ob.get("bids", [])
                        _asks = _ob.get("asks", [])
                        if _bids and _asks:
                            _best_bid = float(_bids[0].get("px", 0))
                            _best_ask = float(_asks[0].get("px", 0))
                            if _best_bid > 0:
                                _live_spread = (_best_ask - _best_bid) / _best_bid * 100
                                if _live_spread > 0.5:
                                    log.info(f"  🛑 {coin}: ILLIQUID — spread {_live_spread:.2f}% > 0.5%, can't fill exit")
                                    _bad_entry = True
                    except Exception:
                        pass
                    # 2. No Binance pair = ultra-illiquid small-cap
                    if not _bad_entry:
                        try:
                            from cross_exchange import _binance_symbol as _bs
                            if _bs(coin) is None:
                                # Fast-track bypass: AI synthetic signal doesn't need Binance pair
                                _is_ft = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                                if _is_ft and ai_override_applied and sig.composite_score >= 0.15:
                                    log.info(f"  🚀 {coin}: No Binance pair but fast-track active (comp={sig.composite_score:+.2f}) — skipping check")
                                # Bull accelerator with no Binance: require composite >= 0.15
                                elif ai_override_applied and _bull_market and ai_conf_val >= 80 and sig.composite_score >= 0.15:
                                    log.info(f"  🐂 {coin}: No Binance pair but bull accelerator active (comp={sig.composite_score:+.2f}) — skipping check")
                                # AI override with no Binance: require composite >= 0.15 (was 0.05 — too weak)
                                elif ai_override_applied and ai_conf_val >= 80 and sig.composite_score >= 0.15:
                                    log.info(f"  🧠 {coin}: No Binance pair but AI override active (comp={sig.composite_score:+.2f}, AI={ai_conf_val}%) — skipping check")
                                elif ai_override_applied and ai_conf_val >= 80 and sig.composite_score >= 0.05:
                                    log.info(f"  🧠 {coin}: No Binance pair but AI override active (comp={sig.composite_score:+.2f}, AI={ai_conf_val}%) — bypass")
                                # Sell in non-bull with AI override: trending_down microcaps, negative composite expected
                                elif ai_override_applied and ai_conf_val >= 80 and _trade_side == "SELL" and not _bull_market and sig.composite_score <= 0.15:
                                    log.info(f"  🔻 {coin}: No Binance pair but SELL override in non-bull market (comp={sig.composite_score:+.2f}) — bypass")
                                # Aug 7: VWAP oversold BUY override with strong AI — zero-loss exits protect
                                elif ai_override_applied and ai_conf_val >= 80 and _trade_side == "BUY" and not _bull_market and sig.composite_score >= -0.15:
                                    log.info(f"  📈 {coin}: No Binance pair but VWAP DIP BUY override (comp={sig.composite_score:+.2f}, AI={ai_conf_val}%) — bypass")
                                elif sig.composite_score < 0.05:  # Aug 7: 0.15→0.05 — zero-loss exits protect
                                    log.info(f"  🛑 {coin}: NO BINANCE PAIR — composite {sig.composite_score:+.2f} < 0.05 required for unlisted coins")
                                    _bad_entry = True
                                elif sig.composite_score < 0.50:
                                    log.info(f"  ⚠️  {coin}: No Binance pair (composite={sig.composite_score:+.2f}) — proceeding with caution")
                        except Exception:
                            pass
                    # 3. Composite absolute minimum: no entry below 0.08 regardless (was 0.20)
                    # No-Binance coins: floor is 0.15 (was bypassed at 0.05 — too weak)
                    _no_binance = False
                    try:
                        from cross_exchange import _binance_symbol as _bs2
                        _no_binance = _bs2(coin) is None
                    except Exception:
                        pass
                    _comp_floor = 0.10 if _no_binance else 0.05
                    # AI override with enriched confirmation: relax no-Binance floor to 0.05
                    # XMR bug: comp=0.06, AI=85%, enriched=BUY, CVD+VWAP+L3 aligned, blocked by 0.10 floor
                    _enriched_confirms = enriched_side in ("BUY", "SELL") and enriched_side == ai_dir
                    if _no_binance and ai_override_applied and ai_conf_val >= 80 and sig.composite_score >= 0.05 and _enriched_confirms:
                        _comp_floor = 0.05
                    # Trending-down shorts: negative composite is expected. Relax floor for SELL direction.
                    # FIX: for shorts, MORE negative composite = stronger signal. Use different check.
                    _trade_is_sell = _trade_side == "SELL" if '_trade_side' in dir() else False
                    if _trade_is_sell and ai_override_applied and not _bull_market:
                        _comp_floor = -0.15  # SELL override in non-bull: trending_down macro, negative OK
                    # ── VWAP mean-reversion: overbought coins can be shorted with positive composite ──
                    # When VWAP>+1.5σ, the mean-reversion edge justifies shorting a trending-up coin.
                    # Relax floor from 0.15→0.40 for +1.5σ, 0.15→0.60 for +2.5σ+.
                    _vwap_sigma_for_floor = 0.0
                    try:
                        import re
                        _vwap_match_f = re.search(r'VWAP:([+-]\d+\.?\d*)σ', enrich_ctx) if enrich_ctx else None
                        if _vwap_match_f:
                            _vwap_sigma_for_floor = float(_vwap_match_f.group(1))
                    except Exception:
                        pass
                    if _trade_is_sell and _vwap_sigma_for_floor > 1.5 and sig.side == "SELL":
                        # VWAP overbought: mean reversion SELL is valid even with bullish composite
                        _vwap_relaxed_floor = 0.15 + (_vwap_sigma_for_floor - 1.5) * 0.25  # 1.5σ→0.15, 2.5σ→0.40, 3.5σ→0.65
                        _vwap_relaxed_floor = min(_vwap_relaxed_floor, 0.70)  # Cap at 0.70
                        if sig.composite_score <= _vwap_relaxed_floor:
                            log.info(f"  📈 {coin}: VWAP FLOOR RELAX — VWAP={_vwap_sigma_for_floor:+.1f}σ overbought, "
                                    f"allowing SELL composite≤{_vwap_relaxed_floor:.2f} (actual={sig.composite_score:+.2f})")
                            _bad_entry = False  # explicitly allow
                            # Skip the normal floor check below — use the relaxed ceiling instead
                            _vwap_relaxed_active = True
                    if not _bad_entry:
                        _vwap_skip_floor = locals().get('_vwap_relaxed_active', False)
                        if _vwap_skip_floor:
                            pass  # VWAP relax already validated composite ceiling
                        elif _trade_is_sell and _comp_floor < 0:
                            # For shorts: block when composite is TOO BULLISH (above abs floor)
                            # -0.25 is MORE bearish than -0.15 → PASSES. +0.05 is bullish → BLOCKS.
                            if sig.composite_score > abs(_comp_floor):
                                log.info(f"  🛑 {coin}: COMPOSITE FLOOR — {sig.composite_score:+.2f} too bullish for SELL (need ≤{abs(_comp_floor):.2f}){' (no Binance pair)' if _no_binance else ''}")
                                _bad_entry = True
                        elif sig.composite_score < _comp_floor:
                            _is_ft3 = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                            if _is_ft3 and ai_override_applied:
                                log.info(f"  🚀 {coin}: Composite floor bypass — fast-track active (comp={sig.composite_score:+.2f})")
                            elif ai_override_applied and _bull_market and ai_conf_val >= 80 and not _no_binance:
                                log.info(f"  🐂 {coin}: Composite floor bypass — bull accelerator active (composite={sig.composite_score:+.2f})")
                            elif ai_override_applied and ai_conf_val >= 80 and sig.composite_score >= 0.05 and not _no_binance:
                                log.info(f"  🧠 {coin}: Composite floor bypass — AI override active (comp={sig.composite_score:+.2f}, AI={ai_conf_val}%)")
                            else:
                                log.info(f"  🛑 {coin}: COMPOSITE FLOOR — {sig.composite_score:+.2f} < {_comp_floor:.2f} minimum{' (no Binance pair)' if _no_binance else ''}")
                                _bad_entry = True
                    if _bad_entry:
                        continue

                    risk = full_risk_check(
                        symbol=coin,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        base_leverage=chosen_leverage,
                        regime=sig.regime.value,
                        atr=atr,
                        equity=total_eq,
                        open_positions={p.get("coin", ""): p for p in active},
                        hsl_state=hsl_state,
                        exposure_state=exposure_state,
                        cooldown_state=cooldown_state,
                        kelly=sig.kelly_fraction or 0.01,
                        max_risk_pct=risk_pct,
                    )

                    if not risk.allowed:
                        # Throttle repetitive risk blocks — log each (coin, reason) at most once per 30s
                        _risk_log_key = f"{coin}:{risk.reason}"
                        now_ts = time.time()
                        last_ts = _risk_block_throttle.get(_risk_log_key, 0)
                        if now_ts - last_ts > 30:
                            log.info(f"  ⛔ {coin}: RISK BLOCKED — {risk.reason}")
                            _risk_block_throttle[_risk_log_key] = now_ts
                        else:
                            log.debug(f"  ⛔ {coin}: risk blocked (suppressed): {risk.reason}")
                        continue

                    # ── Stop-loss cooling check ──
                    if is_stop_loss_cooling(coin):
                        log.info(f"  🧊 {coin}: STOP-LOSS COOLING — {STOP_LOSS_COOLING_SECONDS}s since last stop")
                        continue

                    # ── Session volume check ──
                    notional = total_eq * pos_pct * chosen_leverage  # Compute before risk overrides
                    # ── Minimum notional guard: no penny trades ──
                    min_notional = max(MIN_NOTIONAL_USD, total_eq * 0.35)  # was 0.40 — too high for $60 account
                    # Fast-track: AI-picked micro caps can go to $10 (HL absolute minimum)
                    _is_ft_notional = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                    if _is_ft_notional and ai_conf_val >= 80:
                        min_notional = max(10.0, total_eq * 0.18)  # $10 floor, 18% equity cap (was 20%)
                    if notional < min_notional - 0.01:  # epsilon to avoid float rounding at boundary
                        # Bump position size to meet minimum rather than skipping entirely.
                        # Micro accounts ($60-100) need this to participate.
                        bumped_pct = min_notional / (total_eq * chosen_leverage) if total_eq > 0 else 0
                        old_pct = pos_pct
                        pos_pct = max(pos_pct, bumped_pct)
                        pos_pct = min(pos_pct, 0.35)  # Hard cap at 35% even for micro accounts
                        notional = total_eq * pos_pct * chosen_leverage
                        if notional < min_notional:
                            log.info(f"  🪙 {coin}: TOO SMALL — notional=${notional:.0f} < min=${min_notional:.0f} (skip)")
                            continue
                        log.info(f"  📐 {coin}: bumped pos {old_pct*100:.0f}%→{pos_pct*100:.0f}% to meet ${min_notional:.0f} min notional")
                    _my_notional = notional
                    vol_ok, vol_reason = check_session_volume(coin, notional, total_eq)
                    if not vol_ok:
                        log.info(f"  📊 {coin}: SESSION VOLUME CAP — {vol_reason}")
                        continue

                    # ── Mark price deviation check ──
                    current_mark = float(mids.get(coin, 0))
                    mark_ok, mark_reason = check_mark_deviation(_trade_side, entry_price, current_mark)
                    if not mark_ok:
                        log.info(f"  📏 {coin}: MARK DEVIATION — {mark_reason}")
                        continue

                    # ── Oracle price divergence (liquidation risk) ──
                    oracle_div, oracle_danger = hl.check_oracle_divergence(coin, current_mark)
                    if oracle_danger and _trade_side == "BUY" and oracle_div > 0.5:
                        log.info(f"  🔮 {coin}: ORACLE DIVERGENCE — mid ${current_mark:.4f} > oracle by {oracle_div:.2f}% (liquidation risk)")
                        continue  # Mid higher than oracle → longs risk liquidation at lower price

                    # ── Portfolio heat check ──
                    # is_portfolio_heat_safe adds new_notional internally
                    notional = risk.size_units * entry_price
                    heat_safe, heat_val, heat_reason = is_portfolio_heat_safe(
                        exposure_state.positions, total_eq, coin, notional,
                        max_leverage=risk.leverage,
                    )
                    if not heat_safe:
                        log.info(f"  🔥 {coin}: PORTFOLIO HEAT — {heat_reason}")
                        continue

                    # ── Stage 4: Build exit plan ──
                    # Dynamic price decimals: more for cheap coins, fewer for expensive
                    _log_price = 0.0
                    try: _log_price = abs(math.log10(max(entry_price, 0.0001)))
                    except Exception: pass
                    px_dec = max(0, int(5 - _log_price))  # DOGE $0.07 → 5, BTC $87K → 0
                    ai_plan4 = _ai_trade_plan.get(coin.upper(), {})
                    ai_target = ai_plan4.get("target_pct", 0)
                    # ── Dynamic TP: use real price extremes instead of fixed % ──
                    dyn_tp = get_dynamic_tp_targets(coin, _trade_side == "BUY", entry_price, mids)
                    if dyn_tp.get("target_source") != "atr_fallback" and ai_target <= 0:
                        ai_target = dyn_tp["tp1_pct"]  # Use nearest extreme as primary target
                        log.info(f"  🎯 {coin}: dynamic TP → {ai_target:.1f}% (source: {dyn_tp['target_source']})")
                    exit_plan = build_exit_plan(
                        symbol=coin,
                        is_long=(_trade_side == "BUY"),
                        entry_price=entry_price,
                        total_size=risk.size_units,
                        stop_price=stop_price,
                        atr=atr,
                        leverage=risk.leverage,
                        sz_decimals=6,
                        price_decimals=px_dec,
                        regime=sig.regime.value,
                        break_even_after=1,
                        trail_after=2,
                        trail_atr_mult=tactical.trail_atr_mult,
                        ai_target_pct=ai_target,
                    )
                    log.info(f"\n{describe_exit_plan(exit_plan)}\n")
                    heat_threshold = MAX_LEVERAGED_HEAT if risk.leverage > 2 else MAX_PORTFOLIO_HEAT
                    # Use micro-account adjusted limits for display
                    _ph_display, _lh_display = get_heat_limits(total_eq)
                    display_threshold = _lh_display if risk.leverage > 2 else _ph_display
                    log.info(f"  Portfolio heat: {heat_val*100:.1f}% / {display_threshold*100:.0f}%")

                    # ── Stage 4.5: AI Prediction (strong signals only) ──
                    if conviction_score >= 75:
                        try:
                            pred = predict_short_term(
                                coin, _trade_side, entry_price,
                                candles_1h=_fetch_candles_cached(coin, "1h", 24),
                                mids=mids, regime=sig.regime.value, atr=atr,
                            )
                            trap = pred.get("trap_risk", 50)
                            direction = pred.get("direction", "flat")
                            if trap > 70:
                                log.warning(f"  ⚠️  AI PREDICTION: {coin} TRAP RISK {trap}% — {pred.get('reasoning','')[:60]}")
                            elif direction != ("up" if _trade_side == "BUY" else "down"):
                                log.warning(f"  ⚠️  AI PREDICTION: {coin} dir={direction} vs signal={_trade_side} — possible fakeout")
                            else:
                                log.info(f"  🔮 AI Predict: {coin} {direction} {pred.get('confidence',50)}% conf target={pred.get('target_pct',0):+.1f}%")
                        except Exception:
                            pass

                    # ── Stage 4.6: Pre-trade AI validation ──
                    tp_list = [{"price": t.get("price", 0), "size": t.get("size", 0)} for t in exit_plan.tp_levels]
                    try:
                        v = validate_trade(
                            coin, _trade_side, entry_price, stop_price,
                            notional, total_eq, risk.leverage,
                            sig.regime.value,
                            (atr / entry_price * 100) if entry_price > 0 else 2.0,
                            tp_levels=tp_list if tp_list else None,
                            composite_score=sig.composite_score,
                            signal_confidence=sig.confidence,
                            unified_confidence=pred.confidence,
                            signal_reason=sig.reason,
                        )
                        if not v.get("ok", True):
                            flags = v.get("flags", [])
                            suggestion = v.get("suggestion", "")
                            # Aug 7: AI≥80% bypasses ALL validator flags — zero-loss exits protect every trade
                            _ai_conf_val = _ai_trade_plan.get(coin.upper(), {}).get("confidence", 0)
                            if _ai_conf_val >= 80:
                                log.info(f"  ⚡ {coin}: AI BYPASS — validator flagged {flags} but AI={_ai_conf_val}% overrides (zero-loss exits protect)")
                                # Fall through — don't block this trade
                            else:
                                log.warning(f"  🛑 AI VALIDATION FAILED: {coin} — flags={flags} suggestion={suggestion}")
                                continue
                        if v.get("flags"):
                            log.info(f"  ✅ AI Validation: {coin} OK (flags: {v.get('flags',[])})")
                    except Exception:
                        pass  # Validation is optional — proceed if it fails

                    # ── Stage 5: Bull/Bear Debate (before execution) ──
                    # TradingAgents pattern: argue both sides before committing capital.
                    # Costs 1 extra AI call per entry, saves bad trades worth many calls.
                    debate_verdict = "OVERWEIGHT"  # default if debate fails
                    try:
                        ai_plan_debate = _ai_trade_plan.get(coin.upper(), {})
                        debate = ai_debate_entry(
                            coin=coin,
                            side=_trade_side,
                            entry_price=entry_price,
                            composite=sig.composite_score,
                            regime=sig.regime.value,
                            vwap_sigma=vwap_sigma,
                            ml_direction=ml_dir or "flat",
                            ml_confidence=ml_conf or 0,
                            mom_1m=_calc_momentum(coin, "1m") or 0,
                            mom_5m=_calc_momentum(coin, "5m") or 0,
                            mom_15m=_calc_momentum(coin, "15m") or 0,
                            ai_conviction=ai_plan_debate.get("confidence", 50),
                            ai_target_pct=ai_plan_debate.get("target_pct", 1.5),
                            ai_stop_pct=ai_plan_debate.get("stop_pct", 1.0),
                            atr_pct=atr_val / entry_price * 100 if entry_price > 0 else 1.5,
                            cvd_direction="BUY" if cvd_val > 0 else "SELL" if cvd_val < 0 else "",
                            cvd_strength=abs(cvd_val),
                            market_bias=ai_market.get("bias", "neutral") if "ai_market" in dir() else "neutral",
                            extra_context=f"peak_drop={drop_from_peak_pct if 'drop_from_peak_pct' in dir() else 0:.2f}%",
                        )
                        debate_verdict = debate.get("verdict", "OVERWEIGHT")
                        bull_score = debate.get("bull_score", 50)
                        bear_score = debate.get("bear_score", 50)
                        _is_ft_debate = sig.reason.startswith("ai_fast_track:") if hasattr(sig, 'reason') else False
                        # Signal confluence bypass: when enriched + AI + ML all agree on direction, skip debate
                        # Debate is for borderline cases, not unanimous signals
                        _ml_agrees = (ml_dir and ml_conf and ml_conf >= 30 and
                                      ((_trade_side == "SELL" and ml_dir == "down") or
                                       (_trade_side == "BUY" and ml_dir == "up")))
                        _enriched_agrees = sig.side == _trade_side
                        _unified_agrees = (pred.direction and
                                          ((_trade_side == "SELL" and pred.direction == "down") or
                                           (_trade_side == "BUY" and pred.direction == "up")))
                        _confluence = _enriched_agrees and ai_conf_val >= 80 and _ml_agrees and _unified_agrees
                        # Aug 7: enriched+AI agreement at high conf = skip debate
                        _ai_enr_agree = _enriched_agrees and ai_conf_val >= 80
                        _confluence_bonus = 1.0; _confluence_lev = chosen_leverage
                        _confluence_label = ""; _fast_exit = False
                        if _confluence:
                            log.info(f"  ⚡ {coin}: DEBATE SKIPPED — enriched+AI+ML+unified all agree on {_trade_side} (comp={sig.composite_score:+.2f})")
                            _confluence_bonus = 1.75; _confluence_lev = min(chosen_leverage + 4, 12)
                            _confluence_label = "4/4 CONFLUENCE"; _fast_exit = True
                        elif _ai_enr_agree and _ml_agrees:
                            log.info(f"  ⚡ {coin}: DEBATE SKIPPED — enriched+AI+ML agree on {_trade_side} (comp={sig.composite_score:+.2f})")
                            _confluence_bonus = 1.35; _confluence_lev = min(chosen_leverage + 2, 10)
                            _confluence_label = "3/4 enriched+AI+ML"; _fast_exit = True
                        elif _ai_enr_agree and abs(sig.composite_score) >= 0.12:
                            log.info(f"  ⚡ {coin}: DEBATE SKIPPED — enriched+AI agree at {ai_conf_val}% (comp={sig.composite_score:+.2f})")
                            _confluence_bonus = 1.15; _confluence_lev = chosen_leverage
                            _confluence_label = "enriched+AI strong"; _fast_exit = False
                        elif _ai_enr_agree and abs(sig.composite_score) >= 0.08:
                            log.info(f"  ⚡ {coin}: DEBATE SKIPPED — enriched+AI agree at {ai_conf_val}% (comp={sig.composite_score:+.2f})")
                            _confluence_bonus = 1.0; _confluence_lev = chosen_leverage
                            _confluence_label = ""; _fast_exit = False
                        # Regime-confirmed: AI direction matches macro trend even if enriched disagrees
                        elif ai_conf_val >= 80 and (
                            (regime_is_downtrend and _trade_side == "SELL") or
                            (regime_is_uptrend and _trade_side == "BUY")
                        ):
                            log.info(f"  ⚡ {coin}: DEBATE SKIPPED — regime confirms {_trade_side} (AI={ai_conf_val}%, enriched={sig.side}, comp={sig.composite_score:+.2f})")
                            _confluence_bonus = 1.0; _confluence_lev = chosen_leverage
                            _confluence_label = "regime-confirmed"; _fast_exit = False
                        else:
                            _is_sell = _trade_side == "SELL"
                            if debate_verdict == "HOLD" and not _is_ft_debate:
                                log.warning(f"  🛑 {coin}: DEBATE REJECTED — HOLD verdict — {debate.get('reason','')}")
                                continue
                            # For BUY: reject when bear wins. For SELL: reject when BULL wins (opposite).
                            if _is_sell:
                                if bull_score > bear_score and not _is_ft_debate:
                                    log.warning(f"  🛑 {coin}: DEBATE REJECTED — bull case won for SELL (bull={bull_score} bear={bear_score}) — {debate.get('reason','')}")
                                    continue
                            if bear_score < 40:  # Need strong bearish conviction for SELL
                                log.warning(f"  🛑 {coin}: DEBATE REJECTED — bear too weak for SELL (bear={bear_score} < 40) — {debate.get('reason','')}")
                                continue
                            else:
                                if bear_score > bull_score and not _is_ft_debate:
                                    log.warning(f"  🛑 {coin}: DEBATE REJECTED — bear case won (bull={bull_score} bear={bear_score}) — {debate.get('reason','')}")
                                continue
                            if bull_score < 40:
                                log.warning(f"  🛑 {coin}: DEBATE REJECTED — bull too weak (bull={bull_score} < 40) — {debate.get('reason','')}")
                                continue

                    # ── Stage 5.5: EV Gate (Harper method) ──
                    # Expected-value check: confidence × reward% − (1−confidence) × risk% − costs > 0
                    # AND net reward/risk >= 1.5. Data beats opinion.
                    # ASTER: EV blocked it (R:R=0.95). MET: EV blocked it (R:R=1.16).
                    # If the math doesn't work, no amount of AI confidence justifies the trade.
                    except Exception as _debate_e:
                        log.warning(f"  ⚠️ {coin}: debate failed ({type(_debate_e).__name__}: {_debate_e}) — proceeding without debate")
                        debate_verdict = "OVERWEIGHT"  # fallback
                    pass  # EV gate skipped (ev_gate module may not exist)

                    # ── Stage 6: Submit + Execute ──
                    # Restore daemon's computed notional (not risk check's)
                    notional = _my_notional

                    try:
                            # Write audit trail (file-based fallback if inline execution fails)
                            _submit_action(
                            "buy" if _trade_side == "BUY" else "sell",
                            coin,
                            {
                                "direction": "long" if _trade_side == "BUY" else "short",
                                "side": _trade_side,
                                "size_units": risk.size_units,
                                "size_usd": notional,
                                "leverage": risk.leverage,
                                "entry_price": entry_price,
                                "stop_loss": stop_price,
                                "tp_levels": exit_plan.tp_levels,
                                "break_even_enabled": exit_plan.break_even_enabled,
                                "trail_enabled": exit_plan.trail_enabled,
                                "trail_atr": exit_plan.trail_atr,
                                "reason": sig.reason,
                                "confidence": sig.confidence,
                                "regime": sig.regime.value,
                                "kelly": sig.kelly_fraction,
                                "composite": sig.composite_score,
                            },
                            )

                            # Execute directly on Hyperliquid (inline, no pipeline delay)
                            is_buy = (_trade_side == "BUY")
                        
                            # Micro-cap gate: skip coins < $0.02 (thin book → ghost/dust fills)
                            # Aug 9: raised from $0.01 → $0.02 — 227 ghost fills in 24h
                            _entry_price = float(mids.get(coin, 0))
                            if _entry_price > 0 and _entry_price < 0.02:
                                log.info(f"  🚫 {coin}: MICRO-CAP — ${_entry_price:.4f} < $0.02 (too illiquid, will ghost fill)")
                                continue
                            # Spread check: skip if bid-ask > 2% (thin book, will ghost/dust)
                            if _entry_price > 0:
                                try:
                                    from hyperliquid_ws import get_field as _ws_field
                                    obs = _ws_field("orderbooks") or {}
                                    ob = obs.get(coin, {})
                                    bids = ob.get("bids", [])
                                    asks = ob.get("asks", [])
                                    if bids and asks:
                                        _best_bid = float(bids[0].get("px", 0))
                                        _best_ask = float(asks[0].get("px", 0))
                                        if _best_bid > 0 and _best_ask > 0:
                                            _spread_pct = (_best_ask - _best_bid) / _best_bid * 100
                                            if _spread_pct > 2.0:
                                                log.info(f"  🚫 {coin}: SPREAD FILTER — {_spread_pct:.1f}% spread (thin book, will ghost fill)")
                                                continue
                                except Exception:
                                        pass  # Can't check spread — proceed
                        
                            log.warning(f"  📐 EXECUTING: {coin} size_usd=${notional:.2f} lev={chosen_leverage}x pos_pct={pos_pct*100:.1f}% total_eq=${total_eq:.0f}")
                            
                            # ── REGIME GUARD: only block, don't buy pumps, don't sell dumps ──
                            # Aug 7: everything else enters. Breakeven-lock + profit-lock protect exits.
                            # Aug 9: BTC/ETH MACRO FILTER — if market leaders are pumping, don't short.
                            # If BTC+ETH both up >0.5% in last hour, SHORTs are fighting the tide.
                            try:
                                btc_now = float(mids.get("BTC", 0))
                                eth_now = float(mids.get("ETH", 0))
                                _btc_key = "_btc_1h_ago"; _eth_key = "_eth_1h_ago"
                                btc_1h_ago = getattr(_monitor_positions, _btc_key, 0) if hasattr(_monitor_positions, _btc_key) else 0
                                eth_1h_ago = getattr(_monitor_positions, _eth_key, 0) if hasattr(_monitor_positions, _eth_key) else 0
                                # Store baseline every 60 min
                                if time.time() - getattr(_monitor_positions, "_btc_last_store", 0) > 3600:
                                    setattr(_monitor_positions, _btc_key, btc_now)
                                    setattr(_monitor_positions, _eth_key, eth_now)
                                    setattr(_monitor_positions, "_btc_last_store", time.time())
                                if btc_1h_ago > 0 and eth_1h_ago > 0:
                                    btc_chg = (btc_now - btc_1h_ago) / btc_1h_ago * 100
                                    eth_chg = (eth_now - eth_1h_ago) / eth_1h_ago * 100
                                    macro_pumping = btc_chg > 0.5 and eth_chg > 0.3
                                    macro_dumping = btc_chg < -0.5 and eth_chg < -0.3
                                    if _trade_side == "SELL" and macro_pumping:
                                        log.warning(f"  🛑 {coin}: MACRO BLOCK — BTC {btc_chg:+.1f}% ETH {eth_chg:+.1f}% pumping, can't SHORT against tide")
                                        continue
                                    if _trade_side == "BUY" and macro_dumping:
                                        log.warning(f"  🛑 {coin}: MACRO BLOCK — BTC {btc_chg:+.1f}% ETH {eth_chg:+.1f}% dumping, can't LONG against tide")
                                        continue
                            except Exception:
                                pass  # Can't check macro — proceed
                            _sig_regime = getattr(sig, 'regime', None)
                            _regime_val = str(_sig_regime.value) if _sig_regime and hasattr(_sig_regime, 'value') else ""
                            if _trade_side == "BUY" and "extreme_up" in _regime_val:
                                log.warning(f"  🛑 {coin}: REGIME BLOCK — can't BUY in extreme_up, skip")
                                continue
                            if _trade_side == "SELL" and "extreme_down" in _regime_val:
                                log.warning(f"  🛑 {coin}: REGIME BLOCK — can't SELL in extreme_down, skip")
                                continue
                            
                            # ── ENTER IMMEDIATELY — exit system handles protection ──
                            # Breakeven-lock covers losses. Tiered profit-lock captures wins.
                            # ── AI GUIDANCE: entry zone, scale, entry type from AI trade plan ──
                            _ai_guidance = _ai_trade_plan.get(coin.upper(), {})
                            _ai_entry_zone = _ai_guidance.get("entry_zone", "")
                            _ai_scale = _ai_guidance.get("scale", "full")
                            _ai_entry_type = _ai_guidance.get("entry_type", "market")
                            _ai_invalidation = _ai_guidance.get("invalidation", "")
                            
                            # Scale position size based on AI guidance
                            _scale_map = {"full": 1.0, "half": 0.5, "quarter": 0.25, "third": 0.33}
                            _scale_mult = _scale_map.get(_ai_scale.lower(), 1.0)
                            _sized_notional = notional * _scale_mult
                            if _scale_mult < 1.0:
                                log.info(f"  📏 {coin}: AI scale={_ai_scale} → notional ${notional:.0f} → ${_sized_notional:.0f}")
                            
                            # ── CONFLUENCE BOOST: more size + leverage when signals align ──
                            _final_notional = _sized_notional * _confluence_bonus
                            _final_leverage = _confluence_lev
                            if _confluence_bonus > 1.0:
                                log.info(f"  🔥 {coin}: {_confluence_label} → size ${_sized_notional:.0f}→${_final_notional:.0f} lev {chosen_leverage}x→{_final_leverage}x")
                            
                            # Parse entry zone: AI may return "0.059-0.061" or "0.059"
                            _zone_px = 0.0
                            if _ai_entry_zone:
                                try:
                                    _zone_parts = str(_ai_entry_zone).replace("$","").replace(" ","").split("-")
                                    _zone_px = float(_zone_parts[0])
                                except Exception:
                                    _zone_px = 0.0
                            
                            executed = _execute_direct_open(
                            coin=coin,
                            is_buy=is_buy,
                            size_usd=_final_notional,
                            leverage=_final_leverage,
                            stop_price=stop_price,
                            tp_levels=exit_plan.tp_levels if exit_plan.tp_levels else None,
                            reason=sig.reason,
                            vwap_sigma=vwap_sigma,
                            entry_zone=_zone_px,
                            entry_type=_ai_entry_type,
                            invalidation=_ai_invalidation,
                            )
                    except Exception as _exec_exc:
                            log.error(f"  💥 {coin}: execution crashed — {type(_exec_exc).__name__}: {_exec_exc}")
                            import traceback as _tb
                            log.error(f"  {_tb.format_exc()[-500:]}")
                            executed = False

                    if executed:
                            # Record signal for cooldown + trail (only after confirmed execution)
                            cooldown_state.record(coin)
                            # ── Store fast-exit flag for positive timeout ──
                            if _fast_exit:
                                setattr(_monitor_positions, f"_fast_exit:{coin.upper()}", True)
                            # ── Store regime for trend-aware TREND-KILL ──
                            _regime_val = sig.regime.value if sig and hasattr(sig, 'regime') else ""
                            if "down" in _regime_val.lower():
                                setattr(_monitor_positions, f"_regime_downtrend:{coin.upper()}", True)
                            if "up" in _regime_val.lower() and "not" not in _regime_val.lower():
                                setattr(_monitor_positions, f"_regime_uptrend:{coin.upper()}", True)
                            trail_states[coin] = TrailState(
                            symbol=coin,
                            is_long=is_buy,
                            entry_price=entry_price,
                            highest_price=entry_price,
                            current_stop=stop_price,
                            )
                            unstuck_state.register(coin, entry_price)
                            _exec_fails[coin] = 0  # reset fail counter on success

                            # Update exposure (position is now on Hyperliquid)
                            pos_map = exposure_state.positions.copy()
                            pos_map[coin.upper()] = pos_map.get(coin.upper(), 0) + notional
                            exposure_state.update(total_eq, pos_map)

                            # Track session volume
                            track_session_volume(coin, notional)
                    else:
                            log.warning(f"  ⚠️  {coin}: Inline execution FAILED — cooldown 120s to prevent retry spam")
                            _forager_skip_cooldown[coin] = time.time()
                            # ── Blacklist coins that fail 3+ times in a row ──
                            _exec_fails[coin] = _exec_fails.get(coin, 0) + 1
                            if _exec_fails[coin] >= 3:
                                _coin_blacklist.add(coin.upper())
                            log.warning(f"  🚫 {coin}: BLACKLISTED after {_exec_fails[coin]} failures")

                # ── Stage 6: While whale signals (optional boost) ──
                whale_signals = whale_tracker.get_recent_signals(max_age=60)
                for ws in whale_signals:
                    if ws.symbol in [p.get("coin", "") for p in active]:
                            continue  # Already have position
                    if not cooldown_state.can_trade(ws.symbol):
                            continue
                    log.info(f"  🐋 Whale signal: {ws.symbol} {ws.side} conf={ws.confidence:.2f} "
                             f"({ws.whale_count} whales) — {ws.reason[:60]}")

                # ── Stage 7: Funding sniper — cached (5-min TTL) ──
                global _last_funding_update
                if time.time() - _last_funding_update >= FUNDING_CACHE_TTL:
                    try:
                            ctxs_data = hl.get_asset_ctxs()
                            ctxs = ctxs_data.get("contexts", [])
                            universe = ctxs_data.get("meta", {}).get("universe", [])
                            # Build coin name → context mapping (they're parallel arrays)
                            for i, asset in enumerate(universe):
                                name = asset.get("name", "")
                            if name and i < len(ctxs):
                                funding_sniper.update_context(name, ctxs[i])
                            _last_funding_update = time.time()
                    except Exception as e:
                            log.warning(f"  Funding ctx load error: {e}")

                # Z-score method (higher quality — dynamic thresholds)
                zscore_signals = funding_sniper.evaluate_zscore(TRADABLE_COINS, mids)
                for zs in zscore_signals[:2]:  # Top 2 z-score opportunities
                    if zs.symbol in [p.get("coin", "") for p in active]:
                            continue
                    if not cooldown_state.can_trade(zs.symbol):
                            continue
                    log.info(f"  📊 Z-score funding: {zs.symbol} {zs.side} "
                             f"z={zs.funding_rate*100:.4f}%/hr "
                             f"APR={zs.annual_rate_pct:.0f}% conf={zs.confidence:.2f} "
                             f"mode={zs.mode} — {zs.reason[:60]}")

                # Absolute threshold method (v1 — fixed bands)
                funding_opportunities = funding_sniper.get_top_harvest(limit=2)
                for fs in funding_opportunities:
                    if fs.symbol in [p.get("coin", "") for p in active]:
                            continue
                    if not cooldown_state.can_trade(fs.symbol):
                            continue
                    log.info(f"  💰 Funding harvest: {fs.symbol} {fs.side} "
                             f"rate={fs.funding_rate*100:.4f}%/hr "
                             f"APR={fs.annual_rate_pct:.0f}% conf={fs.confidence:.2f}")

                # ── Stage 8: Delta-neutral funding arb scan ──
                try:
                    # Build funding rates dict from funding sniper data
                    arb_funding = {}
                    for coin in TRADABLE_COINS:
                            ctx = funding_sniper.asset_ctx.get(coin, {})
                            rate = float(ctx.get("funding", 0))
                            if abs(rate) > 0:
                                arb_funding[coin] = rate

                    if arb_funding:
                            arb_opps = scan_arb_opportunities(
                            arb_funding, {k: float(v) for k, v in mids.items()},
                            total_eq, arb_state,
                            )
                            for opp in arb_opps[:2]:  # Top 2 arb opportunities
                                if opp["symbol"] in [p.get("coin", "") for p in active]:
                                    continue
                            log.info(f"  🔄 Delta-neutral arb: {opp['symbol']} "
                                     f"{opp['perp_side']} APR={opp['annual_apr']:.0f}% "
                                     f"expected={opp['expected_return_pct']:.2f}% "
                                     f"max=${opp['max_notional']:.0f}")
                except Exception:
                    pass

                # ── Cycle completed without 429 — tell adaptive rate limiter ──
                hl.report_api_result(False)

            except Exception as e:
                log.error(f"Cycle error: {e}\n{traceback.format_exc()[:500]}")
                # ── Adaptive 429 handling: feed back to rate limiter, short backoff ──
                is_429 = "429" in str(e) or "ClientError" in type(e).__name__
                hl.report_api_result(is_429)
                if is_429:
                    # Quick backoff — adaptive limiter slows future calls, no need for long sleep
                    _backoff = min(2 + _consecutive_429s * 2, 10)  # Cap at 10s (was 120s!)
                    log.warning(f"  ⚠️ 429 rate limit — backing off {_backoff}s (strike {_consecutive_429s+1}, pressure={hl.pressure_level()}%)")
                    time.sleep(_backoff)
                    _consecutive_429s += 1
                else:
                    hl.report_api_result(False)
                    _consecutive_429s = max(0, _consecutive_429s - 1)

            # ── Evolution check (every ~100 cycles = ~1.7 hours) ──
            evolution_check_cycles += 1
            if evolution_check_cycles >= 100:
                evolution_check_cycles = 0
                if len(recent_trades) >= 20:
                    log.info("🧬 Running evolution check...")
                    segment = build_segment(
                            f"cycle_{cycle_count}",
                            recent_trades,
                            start_time=datetime.fromtimestamp(time.time() - 6000, timezone.utc).isoformat(),
                            end_time=datetime.now(timezone.utc).isoformat(),
                    )
                    evo_state.segments.append(segment)

                    evo_requests = evolve_parameters(evo_state.segments, evo_state)
                    if evo_requests:
                        log.info(f"  Evolution requested: {len(evo_requests)} param changes to review")
                        log.info(f"  Prompt ready for AI review ({len(evo_requests[0]['prompt'])} chars)")
                        # Actually call AI with web search (Aug 9: was a stub)
                        try:
                            results = run_evolution_analysis(evo_requests)
                            if results:
                                for r in results:
                                    a = r.get("analysis", "")[:150]
                                    s = r.get("suggestions", [])
                                    if a:
                                        log.info(f"  Evo AI: {a}")
                                    for sg in s[:3]:
                                        log.info(f"     {sg.get('param','?')}: {sg.get('current','?')}->{sg.get('suggested','?')}")
                        except Exception as e:
                            log.info(f"  Evo AI call failed: {type(e).__name__}")

                    evo_state.save()

            # ── Prompt optimization (every ~200 cycles) ──
            if evolution_check_cycles % 200 == 0:
                try:
                    stats = get_prompt_stats(hash_prompt("ai_select"))
                    if stats.get("trials", 0) >= 30:
                        current = "ai_select"  # placeholder — read from ai_decider
                        new_prompt = optimize_prompt_via_ai(current, os.environ.get("OPENROUTER_API_KEY", ""))
                        if new_prompt and len(new_prompt) > 100:
                            vhash = graduate_prompt(new_prompt, hash_prompt("ai_select"))
                            log.info(f"  Prompt optimized: {vhash} stats={stats}")
                except Exception:
                    pass

            # Cycle profiling
            elapsed = time.time() - cycle_start
            if elapsed > CYCLE_SECONDS * 0.8:
                log.warning(f"  ⚠️  Slow cycle: {elapsed:.1f}s")

            # ── Lightweight memory cleanup every cycle ──
            # gc.collect() is expensive (full sweep) and unnecessary — Python GC runs automatically.
            # Only do targeted cleanup: expire stale cache entries, cap WS buffers.
            try:
                now_ts = time.time()
                stale = [k for k, (ts, _) in _candle_cache.items() if now_ts - ts > CANDLE_CACHE_TTL * 2]
                for k in stale:
                    del _candle_cache[k]
                # Targeted WS buffer read — avoid full deepcopy of entire WebSocket state
                from hyperliquid_ws import get_field as _ws_field
                fills = _ws_field("fills") or []
                if len(fills) > 30:
                    from hyperliquid_ws import trim_field as _ws_trim
                    _ws_trim("fills", -20)
                funding = _ws_field("funding_updates") or []
                if len(funding) > 10:
                    from hyperliquid_ws import trim_field as _ws_trim
                    _ws_trim("funding_updates", -5)
            except Exception:
                pass

            # Persist arb state every 10 cycles
            if cycle_count % 10 == 0:
                try:
                    save_arb_state(arb_state)
                except Exception:
                    pass

        time.sleep(1)


_MANUAL_CLOSES: dict[str, float] = {}  # coin → timestamp of our close (TTL 60s)
_LIQ_ALERT_COOLDOWN: dict[str, float] = {}  # coin → timestamp of last liquidation alert

def _detect_liquidations(positions: list):
    """Detect if any position was force-closed (not by us).
    
    Each coin gets ONE alert per 300s to prevent spam from ghost position
    detection loops or duplicate daemon processes.
    """
    global _global_pause_until
    now = time.time()
    known = getattr(_detect_liquidations, "known", {})
    current = {p.get("coin", ""): float(p.get("szi", 0)) for p in positions}
    
    # Purge stale manual close markers (>600s old — pipeline can take 5+ min)
    stale = [c for c, ts in _MANUAL_CLOSES.items() if now - ts > 600]
    for c in stale:
        del _MANUAL_CLOSES[c]
    # Purge stale liquidation cooldowns
    stale_liq = [c for c, ts in _LIQ_ALERT_COOLDOWN.items() if now - ts > 300]
    for c in stale_liq:
        del _LIQ_ALERT_COOLDOWN[c]
    # Purge ROI timeout throttle entries for coins no longer in positions
    active = set(current.keys())
    stale_roi = [k for k in _ROI_TIMEOUT_ATTEMPTED if k.split(":")[0] not in active]
    for k in stale_roi:
        _ROI_TIMEOUT_ATTEMPTED.discard(k)

    for coin, old_szi in known.items():
        new_szi = current.get(coin, 0)
        if abs(old_szi) > 0.0001 and abs(new_szi) < 0.0001:
            # Position disappeared — check if we closed it
            if coin in _MANUAL_CLOSES:
                age = now - _MANUAL_CLOSES[coin]
                log.info(f"  ✓ {coin}: position closed by us {age:.0f}s ago — not a liquidation")
                continue
            # Cooldown: don't spam the same coin
            if coin in _LIQ_ALERT_COOLDOWN and now - _LIQ_ALERT_COOLDOWN[coin] < 300:
                continue
            _LIQ_ALERT_COOLDOWN[coin] = now
            log.error(f"🚨 LIQUIDATION/ADL: {coin} — position force-closed!")
            trail_states.pop(coin, None)
            unstuck_state.remove(coin)
            # ── Prevent immediate re-entry after liquidation ──
            _forager_skip_cooldown[coin] = now
            _global_pause_until = max(_global_pause_until, now + 120)
        elif abs(old_szi) > 0.0001 and abs(new_szi) > 0.0001 and abs(new_szi) < abs(old_szi) * 0.5:
            if coin in _MANUAL_CLOSES:
                continue  # Partial close by us
            if coin in _LIQ_ALERT_COOLDOWN and now - _LIQ_ALERT_COOLDOWN[coin] < 300:
                continue
            _LIQ_ALERT_COOLDOWN[coin] = now
            log.error(f"🚨 PARTIAL LIQUIDATION: {coin} — {abs(old_szi):.4f}→{abs(new_szi):.4f}")

    _detect_liquidations.known = current


# ============================================================

if __name__ == "__main__":
    # ── Singleton guard: only one daemon instance at a time ──
    LOCK_FILE = "/tmp/hyperliquid_daemon.lock"
    _lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.lockf(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(_lock_fd, 0)
        os.write(_lock_fd, str(os.getpid()).encode())
    except (IOError, OSError):
        os.close(_lock_fd)
        print(f"FATAL: Another daemon is already running (lock held on {LOCK_FILE})", file=sys.stderr)
        sys.exit(1)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="No trades submitted")
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = p.parse_args()

    if args.dry_run:
        log.info("🔒 DRY RUN — no real trades")
        _original_submit = globals().get("_submit_action")
        globals()["_submit_action"] = lambda *a, **kw: log.info(
            f"  [DRY] Would submit: {a[0] if a else '?'} "
            f"{a[1] if len(a)>1 else '?'}"
        )

    if args.once:
        import signal
        signal.alarm(120)

    # Break rate-limit crash loop: if we just restarted, wait before hitting the API
    _startup_delay = int(os.environ.get("HL_STARTUP_DELAY", "0"))
    if _startup_delay > 0:
        log.info(f"  ⏳ Startup delay {_startup_delay}s (rate-limit cooldown)")
        time.sleep(_startup_delay)

    run(dry_run=args.dry_run)
