#!/usr/bin/env python3
"""
AI Decider v4 — AI-driven leverage, market-aware risk, liquidation-aware validation.
Every byte of prompt is signal. No filler.

Changes from v1:
  - Richer market data in prompts (price action, supports, funding, BTC, F&G)
  - Persistent SHA256 cache via llm_optimizer (survives restarts)
  - Pre-trade safety validation (AI double-checks stops/leverage/sizing)
  - Post-trade fill analysis (was entry good? stop too tight?)
  - 4h price prediction for strong signals (identifies traps)
  - Smarter model routing: flash-lite for simple, deepseek for complex
  - Extracts warning flags from AI responses (manipulation, fakeout, illiquidity)
  - Token budget: ~400 input, ~150 output (was ~350/100) — richer but minimal overhead
"""

import json
import os
import time
import hashlib
import logging
from typing import Any

log = logging.getLogger(__name__)

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_KEY:
    # Fallback: read from ~/.hermes/.env (belt-and-suspenders)
    _env_path = os.path.join(os.path.expanduser("~"), ".hermes", ".env")
    try:
        with open(_env_path) as _f:
            for _line in _f:
                if "OPENROUTER_API_KEY" in _line and "=" in _line and not _line.strip().startswith("#"):
                    _val = _line.split("=", 1)[1].strip().strip("'\"")
                    if _val:
                        OPENROUTER_KEY = _val
                        os.environ["OPENROUTER_API_KEY"] = _val
                        break
    except Exception:
        pass
# ── Model routing: ALL bot AI calls use Gemini 2.5 Flash (non-lite, smarter) ──
# Flash Lite was unreliable for directional calls. Flash has better reasoning.
# DeepSeek V4 Pro is BANNED from all bot components — chat session only.
# Jul 30: upgraded from flash-lite → flash after INJ, SYRUP, ORDI losses from bad AI picks.
# Aug 9 v2: Switched to Qwen 235B — best decision quality in benchmark (tight sizing,
# VWAP-aware reasoning, specific picks). $0.64/M, 2.0s avg. Scout is fallback.
MODEL_CHEAP = "qwen/qwen3-235b-a22b-2507"    # Qwen 235B MoE $0.64/M — best decisions
MODEL_SMART = "qwen/qwen3-235b-a22b-2507"
MODEL_PREMIUM = "openai/gpt-4.1"              # GPT-4.1 $2/M — evolution, code gen, 1M ctx
MODEL_PICKS = "meta-llama/llama-4-maverick"   # Llama 4 Maverick $1.00/M — proven picks
MODEL_DEEP = MODEL_CHEAP
MODEL_FALLBACK = "meta-llama/llama-4-scout"    # $0.40/M — fast fallback if Qwen rate-limited
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Rate limiting (Qwen 235B is ~2s avg — slower than Scout) ──
_MAX_CALLS_PER_MINUTE = 20
_MIN_CALL_SPACING = 0.5
_call_timestamps: list[float] = []
_last_call_time: float = 0.0

def _rate_limit_check() -> bool:
    """Check if we're within rate limits. Returns True if allowed.
    Also enforces minimum spacing between calls."""
    global _call_timestamps, _last_call_time
    now = time.time()
    # Enforce minimum spacing
    if now - _last_call_time < _MIN_CALL_SPACING:
        time.sleep(_MIN_CALL_SPACING - (now - _last_call_time) + 0.1)
    now = time.time()
    _call_timestamps = [t for t in _call_timestamps if now - t < 60]
    if len(_call_timestamps) >= _MAX_CALLS_PER_MINUTE:
        return False
    _call_timestamps.append(now)
    _last_call_time = now
    return True

# ── Persistent cache (survives restarts) ──
CACHE_DIR = "data"
CACHE_FILE = f"{CACHE_DIR}/ai_decider_cache.json"
_cache: dict[str, dict] = {}
_cache_hits = 0
_cache_misses = 0
_cache_total_cost = 0.0

# Cost model
# Cost model (per 1K tokens input, output) — updated for Qwen 235B
COST_PER_1K = {
    "qwen/qwen3-235b-a22b-2507": (0.00032, 0.00032),    # $0.32/$0.32 per 1M = $0.64/M total
    "meta-llama/llama-4-scout": (0.00020, 0.00020),      # $0.40/M — fallback
    "meta-llama/llama-4-maverick": (0.00050, 0.00050),   # $1.00/M — picks
    "qwen/qwen3-30b-a3b-instruct-2507": (0.000048, 0.00019),  # legacy
}


def _load_cache():
    global _cache
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                data = json.load(f)
                _cache = data.get("entries", {})
                # Purge entries older than 6 hours
                now = time.time()
                _cache = {k: v for k, v in _cache.items() if now - v.get("ts", 0) < 21600}
    except Exception:
        _cache = {}


def _save_cache():
    try:
        # Keep max 200 entries
        if len(_cache) > 200:
            oldest = sorted(_cache.items(), key=lambda x: x[1].get("ts", 0))[:len(_cache) - 200]
            for k, _ in oldest:
                del _cache[k]
        with open(CACHE_FILE, "w") as f:
            json.dump({"entries": _cache, "saved": time.time()}, f)
    except Exception:
        pass


_load_cache()


def _cache_key(prefix: str, *args) -> str:
    """Stable cache key: prefix + SHA256 of args. 5-min window per coin."""
    parts = [prefix]
    for a in args:
        if isinstance(a, float):
            # Round floats that change every cycle (prices, ATR)
            parts.append(f"{a:.4f}")
        else:
            parts.append(str(a))
    raw = "|".join(parts)
    # 5-minute time window
    window = int(time.time() / 300)
    return f"{window}:{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _call_llm(prompt: str, model: str = MODEL_CHEAP,
              max_tokens: int = 200, temperature: float = 0.3,
              cache_prefix: str = "") -> dict:
    """Call LLM with persistent caching and cost tracking."""
    global _cache_hits, _cache_misses, _cache_total_cost

    # Check cache
    ck = _cache_key(cache_prefix or "llm", prompt, model, max_tokens)
    if ck in _cache:
        entry = _cache[ck]
        if time.time() - entry.get("ts", 0) < 900:  # 15-min TTL (was 10 — lite is cheaper, but fewer calls = less risk of bad output)
            _cache_hits += 1
            return json.loads(entry["response"]) if isinstance(entry["response"], str) else entry["response"]

    _cache_misses += 1

    # ── Rate limit check (prevent 429) ──
    if not _rate_limit_check():
        return {"decision": "HOLD", "reason": "rate_limited"}

    if not OPENROUTER_KEY:
        return {"decision": "HOLD", "reason": "no_api_key"}

    # ── Build system prompt (minimized for token efficiency) ──
    system = (
        "Crypto trading AI. Output ONLY valid JSON. No markdown, no text outside JSON. "
        "Picks keys: picks,skip,market_note. Trade keys: decision,reason. Be terse."
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "data_collection": "allow",
        "provider": {"order": ["Groq", "DeepInfra", "Together"], "allow_fallbacks": True},
        "transforms": ["prompt-caching-v1"],   # OpenRouter caching: 90% off repeated system prompts
    }).encode()

    import urllib.request
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "X-Title": "hyperliquid-trader",       # OpenRouter analytics tag
            })

    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            resp = json.loads(r.read())
            choices = resp.get("choices", [])
            if not choices:
                return {"decision": "HOLD", "reason": "empty_choices"}
            msg = choices[0].get("message") or {}
            content = msg.get("content", "") or ""

            # ── Parse response ──
            raw_content = content  # save for debug
            content = content.strip()
            # Strip markdown fences
            if content.startswith("```"):
                nl = content.find("\n")
                content = content[nl+1:] if nl > 0 else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            # Find JSON object
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                content = content[start:end+1]
            else:
                # No JSON braces — likely just ```json with no body
                return {"picks": [], "skip": [], "market_note": ""}

            if not content.strip():
                return {"picks": [], "skip": [], "market_note": ""}

            result = json.loads(content)

            # Track cost
            input_tokens = resp.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = resp.get("usage", {}).get("completion_tokens", 0)
            icost, ocost = COST_PER_1K.get(model, (0.00014, 0.00028))
            call_cost = (input_tokens / 1000 * icost) + (output_tokens / 1000 * ocost)
            _cache_total_cost += call_cost

            # Cache response (only successful ones — don't cache errors)
            _cache[ck] = {"response": json.dumps(result), "ts": time.time()}
            if _cache_misses % 10 == 0:
                _save_cache()

            return result

    except json.JSONDecodeError:
        _raw = raw_content[:300] if 'raw_content' in dir() else 'N/A'
        log.warning(f"  ⚠️ JSON parse error — raw={_raw}")
        # Clear this key from cache so next cycle retries (don't cache errors)
        if ck in _cache:
            del _cache[ck]
        return {"decision": "HOLD", "reason": "parse_error", "picks": [], "skip": []}
    except Exception as e:
        _raw = raw_content[:300] if 'raw_content' in dir() else 'N/A'
        log.warning(f"  ⚠️ API error: {type(e).__name__}: {e} — raw={_raw}")
        if ck in _cache:
            del _cache[ck]
        return {"decision": "HOLD", "reason": f"api:{str(e)[:60]}", "picks": [], "skip": []}

# ── OUTER CATCH-ALL: never let _call_llm return None ──
def _call_llm_safe(prompt, model=MODEL_CHEAP, max_tokens=200, temperature=0.3, cache_prefix=""):
    """Wrapper that guarantees a dict return."""
    try:
        return _call_llm(prompt, model, max_tokens, temperature, cache_prefix)
    except Exception as e:
        log.warning(f"  ⚠️ _call_llm CRASH: {type(e).__name__}: {e}")
        return {"decision": "HOLD", "reason": f"crash:{str(e)[:40]}", "picks": [], "skip": []}

# Patch: make _call_llm self-safe with fallback
_call_llm_original = _call_llm
def _call_llm(prompt, model=MODEL_CHEAP, max_tokens=200, temperature=0.3, cache_prefix=""):
    try:
        return _call_llm_original(prompt, model, max_tokens, temperature, cache_prefix)
    except:
        # If primary model fails, try fallback (Scout) once
        if model != MODEL_FALLBACK:
            try:
                return _call_llm_original(prompt, MODEL_FALLBACK, max_tokens, temperature, cache_prefix)
            except:
                pass
        return {"decision": "HOLD", "reason": "crash", "picks": [], "skip": []}


# ============================================================
# PRICE / MARKET CONTEXT BUILDER (shared across prompts)
# ============================================================

def _build_market_snapshot(candles_15m: list, mids: dict, coin: str,
                           btc_mid: float = 0, fear_greed: int = 50) -> str:
    """Build compact market snapshot from candle data. ~150 chars."""
    if not candles_15m or len(candles_15m) < 5:
        return "no_recent_data"

    mid = float(mids.get(coin, 0))

    # Last 5 candles summary
    closes = [float(c.get("c", c.get("close", 0))) for c in candles_15m[-5:]]
    highs = [float(c.get("h", c.get("high", 0))) for c in candles_15m[-5:]]
    lows = [float(c.get("l", c.get("low", 0))) for c in candles_15m[-5:]]
    volumes = [float(c.get("v", c.get("volume", 0))) for c in candles_15m[-5:]]

    price_change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0
    vol_trend = "rising" if sum(volumes[-2:]) > sum(volumes[:2]) * 1.3 else "falling" if sum(volumes[-2:]) < sum(volumes[:2]) * 0.7 else "stable"
    high_5 = max(highs)
    low_5 = min(lows)
    range_pct = (high_5 - low_5) / low_5 * 100 if low_5 > 0 else 1

    # Support/resistance from last 20
    if len(candles_15m) >= 20:
        all_highs = [float(c.get("h", c.get("high", 0))) for c in candles_15m[-20:]]
        all_lows = [float(c.get("l", c.get("low", 0))) for c in candles_15m[-20:]]
        top20 = sorted(all_highs, reverse=True)[:3]
        bot20 = sorted(all_lows)[:3]
        # Closest support below mid, closest resistance above mid
        supports = [l for l in bot20 if l < mid]
        resistances = [h for h in top20 if h > mid]
        nearest_s = f"${supports[0]:.2f}" if supports else "none"
        nearest_r = f"${resistances[0]:.2f}" if resistances else "none"
    else:
        nearest_s, nearest_r = "unknown", "unknown"

    parts = [
        f"{coin}:${mid:.2f} 5m:{price_change:+.1f}% vol:{vol_trend} 5c_range:{range_pct:.1f}%",
        f"supp:{nearest_s} res:{nearest_r}",
    ]
    if btc_mid > 0:
        btc_delta = (mid / btc_mid - 1) * 100 if btc_mid > 0 else 0
        parts.append(f"vsBTC:{btc_delta:+.1f}%")
    parts.append(f"F&G:{fear_greed}")

    return " | ".join(parts)


# ============================================================
# DECIDE BORDERLINE (conviction 50-69)
# ============================================================

def decide_borderline(
    coin: str,
    conviction_score: float,
    composite: float,
    regime: str,
    signal_side: str,
    signal_reason: str,
    entry_price: float,
    atr: float,
    equity: float,
    positions_count: int,
    market_context: str = "",
    # ── Rich market data ──
    candles_15m: list | None = None,
    mids: dict | None = None,
    btc_mid: float = 0.0,
    fear_greed: int = 50,
    funding_rate: float = 0.0,
    open_positions_summary: str = "",
    # ── Live WS order book data (millisecond precision) ──
    best_bid: float = 0.0,
    best_ask: float = 0.0,
    spread_pct: float = 0.0,
    ws_trade_price: float = 0.0,
    ws_trade_side: str = "",
    bid_depth_usd: float = 0.0,
    ask_depth_usd: float = 0.0,
) -> dict:
    """
    AI decides for borderline signals (conviction 50-69).
    v2: Richer context — price action, supports, funding, BTC, F&G.

    Returns:
      decision, conviction_boost, position_pct, risk_pct, stop_atr, reason,
      warnings (list of risk flags: manipulation, fakeout, illiquid, etc.)
    """
    mids = mids or {}
    candles_list = candles_15m or []

    # Build rich market snapshot
    market_snap = _build_market_snapshot(candles_list, mids, coin, btc_mid, fear_greed)
    atr_pct = (atr / entry_price * 100) if entry_price > 0 else 2.0
    max_pos = equity * 0.15  # 15% max (was 20%)
    risk_baseline = equity * 0.0015  # 0.15% (was 0.2%)

    pos_info = f"open_positions:{positions_count}" if positions_count == 0 else f"open_positions:{positions_count} {open_positions_summary}"

    # Build order book context
    ob_info = ""
    if best_bid > 0 and best_ask > 0:
        ob_info = f"Book:bid=${best_bid:.4f} ask=${best_ask:.4f} spread={spread_pct:.3f}% depth=${bid_depth_usd:.0f}/${ask_depth_usd:.0f}"
        if ws_trade_price > 0:
            ob_info += f" last={ws_trade_side}@${ws_trade_price:.4f}"

    # Liquidation risk calc -- v4: AI-driven leverage with liquidation awareness
    # Default safe values (actual leverage/sizing decided by decide_sizing)
    _leverage = 2
    _stop_pct = 5.0
    _stop_price = entry_price * 0.95 if signal_side == "BUY" else entry_price * 1.05
    if signal_side == "BUY":
        liq_dist_pct = (entry_price - _stop_price) / entry_price * _leverage * 100
        liq_price = entry_price * (1 - (_stop_pct / 100) * _leverage / 2)
    else:
        liq_dist_pct = (_stop_price - entry_price) / entry_price * _leverage * 100
        liq_price = entry_price * (1 + (_stop_pct / 100) * _leverage / 2)
    lev_risk = (
        f"LEV:{_leverage}x liq_dist:{liq_dist_pct:.1f}%_move "
        f"(liq~${liq_price:.2f}) eq:${equity:.0f}"
    )

    prompt = (
        f"ENTRY:{coin} {signal_side} conv={conviction_score:.0f} comp={composite:+.2f} regime={regime}\n"
        f"SIG:{signal_reason[:60]}\n"
        f"MKT:{market_snap}\n"
        f"PRC:entry={entry_price:.2f} ATR={atr:.2f}({atr_pct:.1f}%) fund={funding_rate*100:+.4f}%/hr\n"
        f"ACCT:eq={equity:.0f} max_pos={max_pos:.0f} risk={risk_baseline:.2f} {pos_info}\n"
        + (f"OB:{ob_info}\n" if ob_info else "")
        + f"CTX:{market_context or 'none'}\n"
        f"RULES:ranging=mean_revert_at_levels | trending=follow_trend | hold_only_if vol_collapse|funding_extreme|btc_div>2flags | trade>be_flat\n"
        f'JSON:{{"d":"BUY|SELL|HOLD","b":0-20,"p":0.05-0.15,"r":0.001-0.002,"sa":1.2-2.0,"rf":["fakeout","thin","btc_div","funding","vol"],"sg":0-5}}'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=200,
                       temperature=0.3, cache_prefix="borderline")

    # Sanitize — accept both compact (d,b,p,r,sa) and verbose keys
    decision = result.get("d", result.get("decision", "HOLD"))
    if decision not in ("BUY", "SELL"):
        decision = "HOLD"
    result["decision"] = decision
    result["conviction_boost"] = max(0, min(20, float(result.get("b", result.get("conviction_boost", 0)))))
    result["position_pct"] = max(0.05, min(0.15, float(result.get("p", result.get("position_pct", 0.08)))))
    result["risk_pct"] = max(0.001, min(0.002, float(result.get("r", result.get("risk_pct", 0.0015)))))
    result["stop_atr"] = max(1.2, min(2.0, float(result.get("sa", result.get("stop_atr", 1.5)))))
    result.setdefault("warnings", result.get("rf", []))
    result.setdefault("signals_agreeing", int(result.get("sg", 0)))
    result.setdefault("reason", f"sigs:{result.get('signals_agreeing',0)}")
    if not isinstance(result.get("warnings"), list):
        result["warnings"] = [str(result["warnings"])] if result["warnings"] else []

    return result


# ============================================================
# DECIDE SIZING (conviction >= 70)
# ============================================================

def decide_sizing(
    coin: str,
    conviction_score: float,
    regime: str,
    equity: float,
    atr_pct: float,
    portfolio_heat: float,
    trend_strength: str = "neutral",
    # ── NEW ──
    btc_correlation: float = 0.0,
    support_distance_pct: float = 5.0,
    recent_coin_wr: float = 0.0,
    recent_coin_trades: int = 0,
) -> dict:
    """
    AI decides position sizing with richer context.
    v2: BTC correlation, support distance, coin-specific stats.
    """
    heat_pct = portfolio_heat * 100
    
    # Kelly-inspired sizing: smaller positions for small accounts
    # Risk 1-2% of capital per trade, max 8% position with 2-3x leverage
    account_tier = "micro" if equity < 200 else "small" if equity < 1000 else "normal"
    tier_config = {
        "micro": {"max_pos_pct": 0.50, "max_lev": 6, "risk_pct": 0.03},  # $20 margin floor
        "small": {"max_pos_pct": 0.40, "max_lev": 6, "risk_pct": 0.03},
        "normal": {"max_pos_pct": 0.35, "max_lev": 12, "risk_pct": 0.025},
    }
    tc = tier_config[account_tier]
    max_pos = equity * tc["max_pos_pct"]

    # Regime-based leverage ceiling (account-tier capped)
    regime_lev_cap = {
        "trending_up": min(5, tc["max_lev"]),
        "trending_down": min(3, tc["max_lev"]),
        "sideways": min(3, tc["max_lev"]),  # Reduced from 4 — don't lever in sideways
        "high_vol": min(2, tc["max_lev"]),
        "crisis": min(2, tc["max_lev"]),
    }
    max_lev = regime_lev_cap.get(regime.lower(), tc["max_lev"])

    # Coin experience factor
    if recent_coin_trades >= 10:
        if recent_coin_wr >= 0.45:
            exp = f"proven({recent_coin_wr:.0%}WR/{recent_coin_trades}t)"
        else:
            exp = f"poor({recent_coin_wr:.0%}WR) CAUTION"
    else:
        exp = f"unproven({recent_coin_trades}t)"

    prompt = (
        f"SIZE:{coin} {regime} conv={conviction_score:.0f} ATR={atr_pct:.1f}% eq=${equity:.0f} heat={heat_pct:.0f}%\n"
        f"supp_dist={support_distance_pct:.1f}% btc_corr={btc_correlation:+.2f} trend={trend_strength} coin={exp}\n"
        f"LEV:3-{max_lev}x RULES:higher_lev=higher_conv_needed | min_3x=$10margin_floor | supp>2%_or_heat>10%=reduce | liq_risk:can_stop_save\n"
        f'JSON:{{"p":0.05-0.12,"l":1-{max_lev},"r":0.001-0.002,"sa":1.2-2.0,"rf":["sideways","trending","high_conv","low_conv","tight_supp","wide_supp","high_heat","low_vol","high_vol","unproven","proven","poor_wr"]}}'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=120,
                       temperature=0.2, cache_prefix="sizing")

    # Accept both compact (p,l,r,sa,rf) and verbose keys
    result["position_pct"] = max(0.05, min(0.50, float(result.get("p", result.get("position_pct", 0.10)))))
    result["leverage"] = max(6, min(max_lev, int(float(result.get("l", result.get("leverage", 2))))))  # 6x floor for $20 margin
    result["risk_pct"] = max(0.001, min(0.002, float(result.get("r", result.get("risk_pct", 0.0015)))))
    result["stop_atr"] = max(1.2, min(2.0, float(result.get("sa", result.get("stop_atr", 1.5)))))
    rf = result.get("rf", result.get("reason", ""))
    result["reason"] = ",".join(rf) if isinstance(rf, list) else str(rf)[:40]

    return result


# ============================================================
# PRE-TRADE VALIDATION (NEW)
# ============================================================

def validate_trade(
    coin: str,
    side: str,
    entry_price: float,
    stop_price: float,
    size_usd: float,
    equity: float,
    leverage: int,
    regime: str,
    atr_pct: float,
    tp_levels: list | None = None,
    composite_score: float = 0.0,
    signal_confidence: float = 0.0,
    unified_confidence: float = 0.0,
    signal_reason: str = "",
) -> dict:
    """
    AI safety check before submitting. Catches obvious mistakes AND evaluates
    signal quality — rejecting trades where the setup has no real edge.
    Called right before _submit_action.
    Returns: {ok: bool, flags: [str], suggestion: str}
    """
    tp_str = f"TP:{[t.get('price','?') for t in (tp_levels or [])]}" if tp_levels else "no_TP"
    stop_pct = abs(entry_price - stop_price) / entry_price * 100 if entry_price > 0 else 5
    size_pct = size_usd / equity * 100 if equity > 0 else 20

    # Detect stop placement error before asking AI
    stop_bad = (side == "BUY" and stop_price >= entry_price) or (side == "SELL" and stop_price <= entry_price)

    # Build signal quality context
    qual_parts = []
    if composite_score:
        qual_parts.append(f"comp={composite_score:+.2f}")
    if signal_confidence:
        qual_parts.append(f"sig_conf={signal_confidence:.0f}%")
    if unified_confidence:
        qual_parts.append(f"uni_conf={unified_confidence:.0f}%")
    if signal_reason:
        # Truncate reason to most relevant part
        short_reason = signal_reason[:80] if len(signal_reason) > 80 else signal_reason
        qual_parts.append(f"reason={short_reason}")
    quality_str = " ".join(qual_parts) if qual_parts else "no_signal_data"

    # Dynamic price formatting: more decimals for sub-dollar coins
    price_fmt = ".6f" if entry_price < 1 else ".4f" if entry_price < 100 else ".2f"
    prompt = (
        f"VAL:{coin} {side} entry={entry_price:{price_fmt}} stop={stop_price:{price_fmt}}({stop_pct:.1f}%) "
        f"size=${size_usd:.0f}({size_pct:.1f}%of${equity:.0f}) lev={leverage}x regime={regime} ATR={atr_pct:.1f}% {tp_str} "
        f"SIG:{quality_str}\n"
        f"RULES:{side}=stop_{'below' if side=='BUY' else 'above'}_entry | flag_stop_bad_only_if_wrong_side | "
        f"flag_tight_if<0.5% | flag_size>25% | flag_lev_only_if_dangerous | flag_tp_if<0.3%_or_below_stop | "
        f"REJECT_if_signal_quality_too_low_for_profitable_trade(flag:weak_signal)\n"
        f'JSON:{{"ok":true|false,"fl":["tight","large","lev_high","stop_bad","tp_bad","weak_signal"],"sg":""}}'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=120,
                       temperature=0.1, cache_prefix="validate")

    ok = result.get("ok", True)
    if not isinstance(ok, bool):
        ok = str(ok).lower() != "false"
    result["ok"] = ok
    result.setdefault("flags", [])
    result.setdefault("suggestion", "")

    # -- Deterministic sanity overrides (v4: AI-driven leverage, smart guardrails) --
    # Parse compact keys (fl, sg) or verbose (flags, suggestion)
    result["flags"] = result.get("fl", result.get("flags", []))
    result["suggestion"] = result.get("sg", result.get("suggestion", ""))
    if not isinstance(result.get("flags"), list):
        result["flags"] = [str(result["flags"])] if result["flags"] else []

    # HARD FLOORS (never overridable):
    #   - Stop on wrong side, stop >10%, size >25%: ALWAYS reject
    # LEVERAGE GUARDRAILS:
    #   - >10x: REJECT always (outer safety bound)
    #   - >3x in high_vol/crisis: REJECT (regime incompatible with leverage)
    #   - Everything else: AI DECIDES
    if stop_bad:
        result["ok"] = False
        result["flags"] = list(set(result.get("flags", []) + ["stop_placement_bad"]))
    if stop_pct > 10:
        result["ok"] = False
        result["flags"] = list(set(result.get("flags", []) + ["stop_too_wide"]))
    if size_pct > 25:
        result["ok"] = False
        result["flags"] = list(set(result.get("flags", []) + ["size_too_large"]))

    # Leverage guardrails: hard outer bounds, AI decides everything else
    if leverage > 10:
        result["ok"] = False
        result["flags"] = list(set(result.get("flags", []) + ["leverage_too_high"]))
        result["suggestion"] = f"{leverage}x exceeds 10x hard cap"

    # ── Signal quality guard: reject garbage setups that can't be profitable ──
    # Trust the AI validator's judgment. If AI flags weak_signal, reject.
    # Removed deterministic fallback — it was blocking AI-overridden trades
    # where the AI plan has 80%+ confidence but deterministic metrics are low.
    if "weak_signal" in result.get("flags", []):
        result["ok"] = False
        result["suggestion"] = result.get("suggestion", "") + " signal_quality_too_low"
    elif leverage > 3 and regime.lower() in ("high_vol", "crisis"):
        result["ok"] = False
        result["flags"] = list(set(result.get("flags", []) + ["leverage_too_high"]))
        result["suggestion"] = f"{leverage}x rejected in {regime} regime"

    # For all other leverage cases: trust the AI's assessment.
    # If AI flagged leverage_too_high, respect it.
    ai_flags = result.get("flags", [])
    ai_also_found_bad_stop = bool(set(ai_flags) & {"stop_placement_bad"})
    if ai_also_found_bad_stop and not stop_bad:
        result["ok"] = False  # AI caught something deterministic missed

    # If AI rejected and it's not from a deterministic check above,
    # trust the AI's judgment completely (v4: AI knows best for leverage decisions)
    if not result["ok"] and not stop_bad:
        is_deterministic = any(
            f in result.get("flags", []) for f in ["stop_too_wide", "size_too_large"]
        ) or (leverage > 5) or (leverage > 3 and regime.lower() in ("high_vol", "crisis"))
        if is_deterministic:
            pass  # already rejected above
        else:
            # AI's own call -- keep the rejection (no more subjective override)
            pass
    return result


# ============================================================
# POST-TRADE ANALYSIS (NEW)
# ============================================================

def analyze_fill(
    coin: str,
    side: str,
    planned_entry: float,
    actual_entry: float,
    planned_size: float,
    actual_size: float,
    slippage_pct: float,
    time_to_fill: float,
) -> dict:
    """
    AI analyzes whether fill quality was good.
    Returns: {rating: good|ok|bad, was_slippage_normal: bool, note: str}
    """
    prompt = (
        f"Fill:{coin} {side} planned:${planned_entry:.2f} actual:${actual_entry:.2f}\n"
        f"Slippage:{slippage_pct:.2f}% time:{time_to_fill:.0f}s "
        f"size:{planned_size:.4f}→{actual_size:.4f}\n"
        f"Rate fill: good(<0.1%slip)|ok(<0.5%)|bad(>0.5%) "
        f'JSON:{{"rating":"good|ok|bad","was_slippage_normal":true|false,"note":"10-30 chars"}}'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=100,
                       temperature=0.1, cache_prefix="fill")

    result.setdefault("rating", "ok")
    result.setdefault("was_slippage_normal", True)
    result.setdefault("note", "")
    return result


# ============================================================
# PRICE PREDICTION (NEW — for strong signals)
# ============================================================

def predict_short_term(
    coin: str,
    side: str,
    entry_price: float,
    candles_1h: list | None = None,
    mids: dict | None = None,
    regime: str = "sideways",
    atr: float = 0,
) -> dict:
    """
    4h forward price prediction. Identifies traps and fakeouts.
    Only called for strong signals (conviction >= 75) before execution.
    Returns: {direction: up|down|flat, confidence: 0-100, target_pct: float,
              trap_risk: 0-100, reasoning: str}
    """
    mids = mids or {}
    mid = float(mids.get(coin, 0))

    # Build OHLCV summary from 1h candles
    if candles_1h and len(candles_1h) >= 4:
        recent = candles_1h[-4:]
        ohlcv_parts = []
        for c in recent:
            ohlcv_parts.append(
                f"${float(c.get('o',c.get('open',0))):.2f}→${float(c.get('c',c.get('close',0))):.2f}"
                f"(h:${float(c.get('h',c.get('high',0))):.2f} l:${float(c.get('l',c.get('low',0))):.2f})"
            )
        ohlcv = " | ".join(ohlcv_parts)
    else:
        ohlcv = f"last:${mid:.2f}"

    atr_pct = (atr / entry_price * 100) if entry_price > 0 and atr > 0 else 2.0

    prompt = (
        f"Predict:{coin} 4h ahead. Signal:{side} entry:${entry_price:.2f} regime:{regime}\n"
        f"ATR:{atr_pct:.1f}% 1h_candles:{ohlcv[:250]}\n\n"
        f"Consider: trend strength, volume pattern (rising/falling), "
        f"range compression (BB squeeze→breakout), momentum divergence.\n"
        f"Is this signal a trap? (entering into reversal, thin volume, manipulation)\n\n"
        f'JSON:{{"direction":"up|down|flat","confidence":0-100,'
        f'"target_pct":-5.0-5.0,"trap_risk":0-100,"reasoning":"20-50 chars"}}'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=180,
                       temperature=0.3, cache_prefix="predict")

    result.setdefault("direction", "flat")
    result.setdefault("confidence", 50)
    result.setdefault("target_pct", 0.0)
    result.setdefault("trap_risk", 50)
    result.setdefault("reasoning", "ai_prediction")
    return result


# ============================================================
# REGIME OVERRIDE (unchanged from v1, just uses new cache)
# ============================================================

def decide_regime_override(regime_signals: str, market_data: str = "") -> dict:
    """AI can override the hardcoded regime detection if it disagrees."""
    prompt = (
        f"Market regime signals:\n{regime_signals[:400]}\n"
        f"Data: {market_data or 'none'}\n\n"
        f"Which regime most accurate? "
        f'JSON:{{"regime":"trending_up|trending_down|sideways|volatile","confidence":0-100,"reason":"short"}}'
    )

    result = _call_llm(prompt, model=MODEL_DEEP, max_tokens=120,
                       temperature=0.3, cache_prefix="regime")

    valid = {"trending_up", "trending_down", "sideways", "volatile"}
    if result.get("regime") not in valid:
        result["regime"] = "sideways"
    result["confidence"] = max(0, min(100, int(result.get("confidence", 50))))
    result.setdefault("reason", "ai_regime")
    return result


# ============================================================
# STATS & UTILITIES
# ============================================================

def get_stats() -> dict:
    """Return cache and cost stats."""
    return {
        "cache_entries": len(_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": _cache_hits / max(1, _cache_hits + _cache_misses),
        "total_cost": round(_cache_total_cost, 6),
        "estimated_daily": round(_cache_total_cost / max(1, (time.time() - (os.path.getmtime(CACHE_FILE) if os.path.exists(CACHE_FILE) else time.time()))) * 86400, 6),
    }


def flush_cache():
    """Clear all cached AI responses."""
    global _cache
    _cache = {}
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)


# ============================================================
# AI MARKET SCANNER — replaces deterministic forager selection
# ============================================================

def _scan_batch(batch: list[dict], equity: float, market_context: str,
               recent_str: str, batch_id: int) -> list[dict]:
    """Scan one batch of candidates with AI — designed for ThreadPoolExecutor."""
    lines = []
    vwap_ob_count = 0  # Count overbought (>+1%) coins for direction forcing
    vwap_os_count = 0  # Count oversold (<-1%) coins
    for c in batch:
        vwap_val = c.get('vwap_dist', 0) or 0
        if vwap_val > 1.0:
            vwap_ob_count += 1
        elif vwap_val < -1.0:
            vwap_os_count += 1
        oi_str = f" OI={c.get('oi_delta',0):+.1f}%" if c.get('oi_delta', 0) != 0 else ""
        fund_str = f" fund={c.get('funding',0):.4f}" if c.get('funding', 0) != 0 else ""
        vwap_str = f" VWAP={c.get('vwap_dist',0):+.1f}%" if c.get('vwap_dist', 0) != 0 else ""
        # CVD data
        _cvd = c.get('_cvd', {}) or {}
        cvd_str = ""
        if _cvd.get('trend') in ('rising','falling') and _cvd.get('conf',0) > 5:
            cvd_str = f" CVD:{_cvd['trend']}@{_cvd['conf']}%"
        # Range compression + 1h proximity
        _rp = c.get('_range_pct', 0)
        _1h = c.get('_1h_prox', {}) or {}
        prox_str = ""
        if _1h.get('to_high', 99) < 5 or _1h.get('to_low', 99) < 5:
            hi = _1h.get('to_high', 99)
            lo = _1h.get('to_low', 99)
            prox_str = f" 1h_h:{hi:.1f}% 1h_l:{lo:.1f}%"
        if _rp > 0:
            prox_str += f" rng:{_rp:.1f}%"
        # ── 4h range context (Aug 9): where is price in 4h range? ──
        _4h = c.get('_4h_prox', {}) or {}
        if _4h.get('pct_high', 99) < 99 or _4h.get('pct_low', 99) < 99:
            hi4 = _4h.get('pct_high', 99)
            lo4 = _4h.get('pct_low', 99)
            rp4 = _4h.get('range_pos', 50)
            prox_str += f" 4h:{lo4:+.1f}%/-{hi4:.1f}%@{rp4:.0f}%"
        lines.append(
            f"{c['coin']}: ${c.get('price',0):.2f} comp={c.get('composite',0):+.2f} "
            f"reg={c.get('regime','?')} vol={c.get('volatility',0):.1f}% "
            f"m1={c.get('mom_1m',0):+.1f}% m5={c.get('mom_5m',0):+.1f}% m15={c.get('mom_15m',0):+.1f}% m1h={c.get('mom_1h',0):+.1f}% "
            f"hint={c.get('signal_hint','?')}{vwap_str}{oi_str}{fund_str}{cvd_str}{prox_str}"
        )
    prompt = (
        f'JSON:{{\"picks\":[{{\"coin\":\"X\",\"dir\":\"long\",\"conf\":80,\"tgt\":2.5,\"stop\":0.9,\"hold\":45,'
        + f'\"entry_type\":\"market\",\"scale\":\"full\",\"exit_at\":\"2.5%\",'
        + f'\"entry_zone\":\"0.295-0.298\",\"urgency\":\"now\",\"invalidation\":\"m5 turns neg\",'
        + f'\"risk_note\":\"tight rng\",\"reason\":\"m5 accelerating\"}},'
        + f'{{\"coin\":\"Z\",\"dir\":\"short\",\"conf\":78,\"tgt\":2.0,\"stop\":0.9,\"hold\":40,'
        + f'\"entry_type\":\"market\",\"scale\":\"full\",\"exit_at\":\"2.0%\",'
        + f'\"entry_zone\":\"0.420-0.425\",\"urgency\":\"now\",\"invalidation\":\"m5 turns pos\",'
        + f'\"risk_note\":\"clear downtrend\",\"reason\":\"m1 sharp drop CVD falling\"}}],'
        + f'\"skip\":[\"BAD1\"],\"market_note\":\"BTC neutral\",\"strategy\":\"prioritize A\",\"skip_reason\":\"trending_down\"}}\n'
        + f"Pick 2-3 MAX. ONLY pick coins with REAL data. If mom5<0.3% flat, SKIP. If comp near 0, SKIP. If enriched=dead, SKIP. Pick NONE if no good candidates. SHORTS=EQUALLY VALID — pick SHORT when mom5 neg+CVD falling. MUST cite data in reason. Eq=${equity:.0f}. Recent:{recent_str} Mkt:{market_context}\n"
        + "\n".join(lines) + "\n"
        + f"RULES: tgt>=1.0 stop=0.5-0.6 hold=30-45m. SHORTS=LONGS equally. FEE 0.42%@6x. "
        + f"LONG: m1>0.5%+m5 pos+CVD rising+comp>0.05. "
        + f"SHORT: m1<-0.5%+m5 neg+CVD falling+comp<-0.05. "
        + f"⚠️ VWAP RULES: NEVER buy when VWAP>+1.5% (overbought). NEVER short when VWAP<-1.5% (oversold). "
        + f"Buy dips (VWAP<-1.5%), short pumps (VWAP>+1.5%). "
        + f"Skip: flat mom5(<0.3%), dead enriched, no CVD, m1h>5% exhausted, comp>0.3 extreme. "
        + f"4h CONTEXT: 4h:+X%/-Y%@Z% means Z% in 4h range (0%=bottom,100%=top). "
        + f"Near 4h high (<5% below)=resistance, short ONLY with mom5 turning down+exhaustion signs. "
        + f"Far above 4h low (>30%)=strong uptrend, fading needs reversal evidence. Flat 4h=mean reversion ok. "
        + f"VWAP SCAN: {vwap_ob_count} overbought:{vwap_os_count} oversold | " 
        + (f"⛔ ONLY PICK SHORTS — all coins overbought, do NOT pick any BUY/LONG" if vwap_ob_count > len(batch) * 0.7 else
           f"⛔ ONLY PICK LONGS — all oversold" if vwap_os_count > len(batch) * 0.7 else
           f"SHORTS preferred (more overbought)" if vwap_ob_count > vwap_os_count else
           f"LONGS preferred (more oversold)" if vwap_os_count > vwap_ob_count else
           f"balanced market") + ". "
        + f"Fields: entry_zone urgency(now/confirm/wait) invalidation risk_note reason entry_type(market/limit) scale(full/half) exit_at.\n"
        + f"OUTPUT keys: picks skip market_note strategy skip_reason. "
        + f"DO NOT use decision/reason keys."
        )

    result = _call_llm(prompt, model=MODEL_PICKS, max_tokens=800,  # bumped from 500 — Haiku cuts off with fence only
                       temperature=0.3, cache_prefix=f"ai_scan_b{batch_id}")
    picks = result.get("picks", [])
    skip_list = result.get("skip", [])
    market_note = result.get("market_note", "")
    strategy = result.get("strategy", "")
    skip_reason = result.get("skip_reason", "") or ""
    
    # ── Debug: log what AI returned ──
    _has_decision = "decision" in result
    _pick_count = len(picks) if isinstance(picks, list) else 0
    _skip_count = len(skip_list) if isinstance(skip_list, list) else 0
    if _has_decision:
        log.warning(f"  ⚠️ AI returned decision/reason format (Haiku bug) — decision={result.get('decision')} reason={result.get('reason','')[:60]}")
    elif _pick_count == 0:
        log.info(f"  🤷 AI returned 0 picks, {_skip_count} skips (batch {batch_id})")
    else:
        log.info(f"  ✅ AI returned {_pick_count} picks, {_skip_count} skips (batch {batch_id})")
    
    # Silent fallback: Haiku may return decision/reason — just skip, forager handles it
    usable = []
    if isinstance(picks, list):
        for p in picks[:2]:
            # ── Handle string picks: Haiku sometimes returns ["COIN"] instead of [{coin:...}]
            if isinstance(p, str):
                usable.append({
                    "coin": p.upper(),
                    "direction": "long",  # default, AI validator will check
                    "confidence": 75,
                    "target_pct": 2.0,
                    "stop_pct": 1.0,
                    "hold_min": 30,
                    "reason": f"ai_string_pick:{p}",
                    "entry_type": "market",
                    "scale": "full",
                })
                log.info(f"  📝 AI string pick: {p} → converted to object (default long, 75% conf)")
                continue
            usable.append({
                "coin": str(p.get("coin", "")).upper(),
                "direction": str(p.get("dir", "")).lower(),
                "confidence": int(p.get("conf", 50)),
                "target_pct": float(p.get("tgt", 1.5)),
                "stop_pct": float(p.get("stop", 1.0)),
                "hold_min": int(p.get("hold", 60)),
                "reason": str(p.get("reason", ""))[:200],
                "entry_price": float(p.get("entry_price", 0)) if p.get("entry_price") else 0.0,
                "entry_type": str(p.get("entry_type", "market")).lower(),
                "scale": str(p.get("scale", "full")).lower(),
                "exit_at": str(p.get("exit_at", ""))[:100],
                "risk_note": str(p.get("risk_note", ""))[:100],
                # ── New: richer AI output fields ──
                "entry_zone": str(p.get("entry_zone", ""))[:50],
                "urgency": str(p.get("urgency", "now")).lower(),
                "invalidation": str(p.get("invalidation", ""))[:100],
                "strategy_note": strategy[:200] if strategy else "",
                "skip_reason": skip_reason[:200],
            })
    return usable, skip_list, market_note

def ai_select_coins(
    candidates: list[dict],
    equity: float,
    market_context: str = "",
    recent_trades: list[dict] | None = None,
) -> list[dict]:
    """AI picks best coins — parallel 2-batch scan for speed."""
    if not candidates:
        return []

    recent_str = ", ".join(f"{t['coin']}:{t['pnl_pct']:+.1f}%({t.get('reason','')[:20]})" for t in (recent_trades or [])[-5:]) if recent_trades else "none"
    
    # Split into 2 parallel batches
    half = max(1, len(candidates) // 2)
    batch1 = candidates[:half]
    batch2 = candidates[half:half*2]
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_picks = []
    all_skips = []
    all_market_notes = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_scan_batch, batch1, equity, market_context, recent_str, 1): 1,
            ex.submit(_scan_batch, batch2, equity, market_context, recent_str, 2): 2,
        }
        for fut in as_completed(futures):
            try:
                batch_picks, batch_skips, batch_note = fut.result()
                all_picks.extend(batch_picks)
                if batch_skips:
                    all_skips.extend(batch_skips)
                if batch_note:
                    all_market_notes.append(batch_note)
            except Exception as e:
                log.warning(f"AI batch {futures[fut]} failed: {e}")
    
    # Return top 3 by confidence
    all_picks.sort(key=lambda p: p.get("confidence", 0), reverse=True)
    return all_picks[:3], all_skips, all_market_notes


def ai_assess_market(
    coin_signals: list[str],
    equity: float,
    positions: int,
    fear_greed: int = 50,
    btc_context: str = "",
    sector_context: str = "",
    recent_trades: str = "",
) -> dict:
    """AI assesses whether current market conditions are tradable.

    Returns {tradable, bias, max_leverage, best_timeframe, max_positions}.
    """

    signals_str = "\n".join(coin_signals[:10]) if coin_signals else "no_signals"
    btc_line = f"\n{btc_context}" if btc_context else ""
    sector_line = f"\n{sector_context}" if sector_context else ""
    trade_line = f"\nRecent: {recent_trades}" if recent_trades else ""

    prompt = (
        f"Assess market. Eq=${equity:.0f} pos={positions} FG={fear_greed}{btc_line}{sector_line}{trade_line}\n"
        f"Signals (top 10):\n{signals_str}\n"
        f"Decide: tradable? bias? max_lev? best_tf? max_pos?\n"
        f"RULES: >3 aligned signals=trade, 1-2=cautious, 0 or mixed=sideways. "
        f"Hot sectors+bullish BTC=raise lev. Cold sectors=neutral. "
        f"Recent losing trades=reduce max_pos. SHORTS valid always.\n"
        f'JSON:{{"tradable":true,"bias":"bullish","max_lev":5,"best_tf":"1h","max_pos":3}}'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=100,
                       temperature=0.1, cache_prefix="ai_market")

    return {
        "tradable": bool(result.get("tradable", True)),
        "bias": result.get("bias", "neutral"),
        "max_leverage": int(result.get("max_lev", 12)),
        "best_timeframe": result.get("best_tf", "1h"),
        "max_positions": int(result.get("max_pos", 3)),
    }


def ai_evaluate_exit(
    coin: str,
    direction: str,
    entry_price: float,
    current_price: float,
    pnl_pct: float,
    hold_seconds: float,
    mom_5m: float,
    mom_15m: float,
    regime: str,
    target_pct: float,
    stop_pct: float,
    price_extremes: str = "",
    peak_pnl: float = 0.0,
    btc_ctx: str = "",
) -> dict:
    """AI evaluates whether to hold or exit an open position.

    Returns {"action": "hold"|"exit", "confidence": 0-100, "reason": str}
    Aug 9: added peak_pnl and btc_ctx — exit model now sees macro + peak reversal context.
    """
    ext_line = f"\n{price_extremes}\n" if price_extremes else ""
    peak_line = f" peak_was={peak_pnl:+.1f}%" if peak_pnl != 0 else ""
    btc_line = f"\nBTC/ETH: {btc_ctx}" if btc_ctx else ""
    prompt = (
        f"EVAL:{coin} {direction} @${current_price:.4f} pnl={pnl_pct:+.1f}%{peak_line} {hold_seconds/60:.0f}min "
        f"mom5={mom_5m:+.1f}% mom15={mom_15m:+.1f}% reg={regime} tgt={target_pct:.1f}%{ext_line}{btc_line}\n"
        f"Think like a trader: if up but fading → lock profit. If losing + momentum against → cut. "
        f"If flat too long → dead, exit. If solidly up + small dip in uptrend → hold. "
        f"If trend broke (mom5 flipped) → exit regardless. Use common sense not thresholds.\n"
        f'JSON:{{"action":"hold|exit","conf":0-100,"reason":"brief"}}'
    )
    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=80,
                       temperature=0.1, cache_prefix="ai_exit_eval")
    return {
        "action": result.get("action", "hold"),
        "confidence": int(result.get("conf", 50)),
        "reason": result.get("reason", ""),
    }


def ai_debate_entry(
    coin: str,
    side: str,
    entry_price: float,
    composite: float,
    regime: str,
    vwap_sigma: float,
    ml_direction: str,
    ml_confidence: float,
    mom_1m: float,
    mom_5m: float,
    mom_15m: float,
    ai_conviction: int,
    ai_target_pct: float,
    ai_stop_pct: float,
    atr_pct: float = 1.5,
    cvd_direction: str = "",
    cvd_strength: float = 0.0,
    market_bias: str = "neutral",
    extra_context: str = "",
) -> dict:
    """Bull vs Bear debate before entry. Returns {verdict, conviction_adj, bull_score, bear_score, reason}.

    Verdict mapping:
      BUY  → full conviction, 20-25% position
      OVERWEIGHT → moderate, 12-18% position
      HOLD  → skip entry (don't trade)
    """
    direction_word = "LONG" if side == "BUY" else "SHORT"
    ml_contradicts = (side == "BUY" and ml_direction == "down") or (side == "SELL" and ml_direction == "up")
    cvd_str = f"CVD:{cvd_direction}@{cvd_strength:.0%}" if cvd_direction else ""

    prompt = (
        f"DEBATE: {coin} {direction_word} @ ${entry_price:.4f}. Decide: BUY, OVERWEIGHT, or HOLD.\n"
        f"DATA: comp={composite:+.2f} regime={regime} VWAP={vwap_sigma:+.1f}σ ATR={atr_pct:.1f}%\n"
        f"momentum: m1={mom_1m:+.2f}% m5={mom_5m:+.2f}% m15={mom_15m:+.2f}%\n"
        f"ML: {ml_direction}@{ml_confidence:.0f}% | AI: {ai_conviction}% tgt={ai_target_pct}% stop={ai_stop_pct}%\n"
        f"market: {market_bias} | {cvd_str} | ml_contradicts={ml_contradicts}\n"
        f"{extra_context}\n"
        f"ARGUE BOTH SIDES then DECIDE:\n"
        f"BULL case: why this {direction_word} works NOW (trend strength, momentum alignment, mean reversion setup)\n"
        f"BEAR case: why it fails (fakeout, exhaustion, counter-trend, weak composite, ML contradiction)\n"
        f"VERDICT: BUY=strong_conviction_bull_wins | OVERWEIGHT=moderate_bull_has_edge | HOLD=bear_wins_or_unclear\n"
        f"RETURN numbers only:\n"
        f'JSON:{{"verdict":"BUY|OVERWEIGHT|HOLD","bull":0-100,"bear":0-100,"reason":"1-line"}}\n'
    )

    result = _call_llm(prompt, model=MODEL_CHEAP, max_tokens=120,
                       temperature=0.3, cache_prefix="ai_debate")

    return {
        "verdict": result.get("verdict", "HOLD"),
        "bull_score": int(result.get("bull", 50)),
        "bear_score": int(result.get("bear", 50)),
        "reason": result.get("reason", ""),
    }

# ═══════════════════════════════════════════════════════════════
# WEB-SEARCH-ENABLED POST-TRADE and EVOLUTION ANALYSIS (Aug 9)
# These are NOT time-critical — 3-10s latency is acceptable.
# ═══════════════════════════════════════════════════════════════

def _call_llm_web(prompt: str, model: str = MODEL_CHEAP,
                  max_tokens: int = 500, temperature: float = 0.3) -> dict:
    """Call LLM with OpenRouter web search plugin enabled.
    Adds 3-10s latency — only use for post-trade/evolution analysis, NOT live trading.
    """
    if not OPENROUTER_KEY:
        return {"analysis": "no_api_key", "suggestions": []}

    system = (
        "Crypto trading analyst. Output ONLY valid JSON. No markdown, no text outside JSON. "
        "You analyze trades to find missed opportunities and better parameters. "
        "Focus on: entry timing, position sizing, missed signals, trend catching. "
        "NEVER suggest being more conservative. Inaction = loss. More trades = more chances. "
        "Always suggest at least one way to trade MORE, not less. "
        "Keys: analysis, suggestions (list of {param, current, suggested, reason}), "
        "missed_opportunities (list of strings), confidence (0-100)."
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "plugins": [{"id": "web", "max_results": 5}],
        "provider": {"order": ["Groq", "DeepInfra", "Together"], "allow_fallbacks": True},
    }).encode()

    import urllib.request
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "X-Title": "hyperdik-analysis",
    })

    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            resp = json.loads(r.read())
            choices = resp.get("choices", [])
            if not choices:
                return {"analysis": "empty_response", "suggestions": []}
            content = (choices[0].get("message") or {}).get("content", "") or ""
            content = content.strip()
            if content.startswith("```"):
                nl = content.find("\n")
                content = content[nl+1:] if nl > 0 else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            return json.loads(content.strip())
    except Exception as e:
        return {"analysis": f"api_error:{type(e).__name__}", "suggestions": []}


def analyze_closed_trade(trade_data: dict, market_context: str = "") -> dict:
    """Analyze a completed trade with web search for market context."""
    if not trade_data:
        return {"analysis": "no_data", "suggestions": []}

    coin = trade_data.get("coin", "?")
    side = trade_data.get("side", "?")
    pnl_pct = trade_data.get("pnl_pct", 0)
    entry_px = trade_data.get("entry_price", 0)
    exit_px = trade_data.get("exit_price", 0)
    hold_secs = trade_data.get("hold_secs", 0)
    regime = trade_data.get("regime", "?")
    conviction = trade_data.get("conviction", 0)
    composite = trade_data.get("composite_score", 0)
    leverage = trade_data.get("leverage", 1)

    prompt = (
        f"ANALYZE THIS COMPLETED {side} TRADE ON {coin}:\n"
        f"Entry: ${entry_px} -> Exit: ${exit_px} | PnL: {pnl_pct:+.2f}% | Held: {hold_secs:.0f}s\n"
        f"Leverage: {leverage}x | Conviction: {conviction}/100 | Composite: {composite:+.3f}\n"
        f"Regime at entry: {regime}\n"
    )
    if market_context:
        prompt += f"\nMARKET STATE AT EXIT:\n{market_context}\n"
    prompt += (
        "\nSearch the web for recent news about this coin. "
        "What went right or wrong? Could we have entered earlier? "
        "Was the size appropriate? Any parameter tweaks to catch more like this? "
        "IMPORTANT: Never suggest trading less or being more conservative. "
        "Focus on: better entry timing, sizing adjustments, trend detection, missed signals."
    )

    result = _call_llm_web(prompt, model=MODEL_PREMIUM, max_tokens=500)
    result["coin"] = coin
    result["pnl_pct"] = pnl_pct
    return result
def run_evolution_analysis(evo_requests: list) -> list:
    """Run AI evolution analysis with web search + premium model.
    Now uses the full evolution engine (v2) — massive data dump, web search, code changes.
    """
    results = []
    
    # Use the new evolution engine for full system context
    try:
        from hyperliquid_evolution import run_full_evolution, apply_changes
        evo = run_full_evolution()
        prompt = evo.get("prompt", "")
    except Exception:
        # Fallback to old prompt if v2 engine fails
        if evo_requests:
            prompt = evo_requests[0].get("prompt", "")
        else:
            return [{"analysis": "no_prompt", "suggestions": []}]
    
    if not prompt:
        return [{"analysis": "empty_prompt", "suggestions": []}]
    
    # Use PREMIUM model — evolution runs rarely, worth the best quality
    result = _call_llm_web(prompt, model=MODEL_PREMIUM, max_tokens=2000, temperature=0.4)
    
    # Apply AI-suggested changes to actual files
    changes = result.get("changes", [])
    if changes:
        try:
            from hyperliquid_evolution import apply_changes
            applied = apply_changes(changes, dry_run=False)
            result["applied"] = applied
            # Auto-commit via git
            import subprocess
            subprocess.run(["git", "-C", str(Path(__file__).parent), "add", "-A"], 
                         capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(Path(__file__).parent), "commit", 
                          "-m", f"Evo: AI-optimized — {len(applied)} changes"],
                         capture_output=True, timeout=10)
        except Exception as e:
            result["applied"] = [f"apply_error: {e}"]
    
    # Restart daemon if needed
    if result.get("restart_required") and changes and result.get("applied"):
        try:
            import subprocess, os
            subprocess.run(["pkill", "-f", "python3.*hyperliquid_daemon"], capture_output=True)
            time.sleep(2)
            daemon_path = Path(__file__).parent / "hyperliquid_daemon.py"
            subprocess.Popen(["python3", "-B", str(daemon_path)],
                           cwd=str(Path(__file__).parent),
                           stdout=open(str(Path(__file__).parent / "logs" / "hyperliquid_daemon.log"), "w"),
                           stderr=open(str(Path(__file__).parent / "logs" / "hyperliquid_daemon_error.log"), "w"))
            result["restarted"] = True
        except Exception as e:
            result["restarted"] = False
            result["restart_error"] = str(e)
    
    results.append(result)
    return results
