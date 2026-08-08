"""
AI Prediction Validator — Pure data in, pure data out.
No reasoning text. Every byte is signal.

Input: prediction layers + market context + enrichment data
Output: structured JSON with actionable fields only
"""

import json, time, os, urllib.request
from dataclasses import dataclass, field
from typing import Optional

VALIDATOR_MODEL = "google/gemini-2.5-flash-lite"
VALIDATOR_FALLBACKS = ["google/gemini-2.5-flash-lite", "qwen/qwen-3.5-flash"]

def _get_api_key():
    for path in ["/home/ubuntu/.hermes/.env", os.path.expanduser("~/.hermes/.env")]:
        try:
            with open(path) as f:
                for line in f:
                    if "OPENROUTER_API_KEY" in line:
                        key = line.split("=", 1)[1].strip().strip("'\"")
                        if key:
                            return key
        except:
            pass
    return os.environ.get("OPENROUTER_API_KEY", "")


def _call_deepseek(prompt: str, max_tokens: int = 200) -> dict:
    api_key = _get_api_key()
    if not api_key:
        return {"a": "wait", "rf": ["no_key"]}

    now = time.time()
    # Rate limiting: 10 calls/min max, 0.8s spacing
    if not hasattr(_call_deepseek, '_times'):
        _call_deepseek._times = []
    _call_deepseek._times = [t for t in _call_deepseek._times if now - t < 60]
    if len(_call_deepseek._times) >= 10:
        return {"a": "wait", "rf": ["rate_limit"]}
    if _call_deepseek._times and now - _call_deepseek._times[-1] < 0.8:
        time.sleep(0.8 - (now - _call_deepseek._times[-1]))
    _call_deepseek._times.append(time.time())

    data = json.dumps({
        "model": VALIDATOR_MODEL,
        "messages": [
            {"role": "system", "content": "Output ONLY valid JSON. No markdown, no text outside JSON. Keys: a,c,t,s,sg,op,rf"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.15,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "HL Validator v2",
            "X-OpenRouter-Cache": "true",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"].get("content", "{}")
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                return json.loads(content)
            return {"a": "wait", "rf": ["bad_type"]}
    except json.JSONDecodeError:
        if isinstance(content, str):
            import re
            start = content.find('{')
            if start >= 0:
                depth = 0
                end = start
                for i in range(start, len(content)):
                    if content[i] == '{': depth += 1
                    elif content[i] == '}':
                        depth -= 1
                        if depth == 0: end = i + 1; break
                if end > start:
                    raw = content[start:end]
                    try: return json.loads(raw)
                    except: pass
                    try:
                        fixed = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', raw)
                        fixed = re.sub(r',\s*}', '}', fixed)
                        return json.loads(fixed)
                    except: pass
        return {"a": "wait", "rf": ["json_err"]}
    except Exception as e:
        return {"a": "wait", "rf": [f"api:{type(e).__name__}"[:20]]}


@dataclass
class AIValidationResult:
    approved: bool
    adjusted_target: float
    adjusted_confidence: float
    action: str  # execute_now, wait, reject, reduce_size, tighten_stop
    size_multiplier: float
    reason: str  # compact for logging: "4sig/1opp fund_risk"
    warning_flags: list = field(default_factory=list)
    signals_agreeing: int = 0
    signals_opposing: int = 0


# ── Risk flag codes (compact, machine-readable) ──
RISK_CODES = {
    "funding": "funding expensive against position",
    "btc_div": "BTC divergence",
    "thin": "thin order book",
    "vol": "volatility collapse",
    "fakeout": "potential fakeout/pump",
    "cascade": "near liquidation cascade",
    "xex": "cross-exchange divergence >1%",
    "oi": "OI delta against position",
    "hurst": "Hurst regime contradicts",
    "accel": "funding accelerating against",
}


def validate_prediction(
    coin: str,
    current_price: float,
    prediction: dict,
    portfolio: dict,
    market_regime: str,
    fear_greed: int,
    funding_rate: float,
    btc_correlation: float = 0.0,
    support_level: float = 0.0,
    resistance_level: float = 0.0,
    position_context: dict = None,
    market_context: str = "",
    order_book_info: str = "",
    atr_info: str = "",
    # ── NEW: rich data fields ──
    funding_accel: str = "",         # "accelerating_long|accelerating_short|decelerating|flat"
    hurst_regime: str = "",          # "trending|mean_reverting|random"
    hurst_H: float = 0.5,
    oi_delta_type: str = "",         # "new_shorts|shorts_covering|new_longs|longs_capitulating"
    oi_delta_desc: str = "",         # compact description
    cross_exchange_div: float = 0.0, # divergence % (HL vs Binance)
    cross_exchange_dir: str = "",    # "hl_premium" or "hl_discount"
    cascade_risk: str = "",          # "none|low|moderate|high|critical"
    htf_alignment: str = "",         # "bullish|bearish|conflicting|neutral"
) -> AIValidationResult:

    ctx = position_context or {}
    is_exit = ctx.get("is_exit", False)
    pos_side = ctx.get("side", "")
    pnl_pct = ctx.get("pnl_pct", 0.0)
    margin_pnl = ctx.get("margin_pnl", 0.0)
    leverage = ctx.get("leverage", 1.0)
    liq_dist = ctx.get("liq_distance", 99.0)

    pred_dir = prediction.get("direction", "?")
    pred_conf = prediction.get("confidence", 0)
    pred_target = prediction.get("target_price", prediction.get("predicted_extreme", current_price))

    # Component scores — read from layers dict directly
    layers = prediction.get("layers", {})
    ml_score = layers.get("ml_ensemble", prediction.get("ml_score", 0))
    # Build compact per-layer breakdown sorted by confidence
    layer_parts = []
    layer_names = {
        "order_book": "ob", "volume_profile": "vp", "cvd_delta": "cvd",
        "bollinger": "boll", "sr_levels": "sr", "multi_tf_technical": "mtf",
        "funding_oi": "foi", "ml_ensemble": "ml", "hurst_regime": "hurst",
        "higher_tf_align": "htf", "oi_delta": "oi", "cross_exchange": "xex",
        "taker_ratio": "tkr", "exhaustion": "exh", "imbalance_trend": "imb",
    }
    bullish = []; bearish = []
    for name, short in layer_names.items():
        ld = layers.get(name, {})
        if isinstance(ld, dict) and ld.get("confidence", 0) > 5:
            conf = ld.get("confidence", 0)
            up = ld.get("up", 33); down = ld.get("down", 33)
            tag = f"{short}={(up-down):+.0f}"
            if up > down + 10: bullish.append(tag)
            elif down > up + 10: bearish.append(tag)
            layer_parts.append(f"{short}={conf:.0f}")
    # Sort by confidence
    sorted_parts = sorted(layer_parts, key=lambda x: float(x.split("=")[1]), reverse=True)[:8]
    lines.append(f"L:{' '.join(sorted_parts)}")
    if bullish:
        lines.append(f"BULL:{' '.join(bullish[:6])}")
    if bearish:
        lines.append(f"BEAR:{' '.join(bearish[:6])}")

    pos_count = portfolio.get("positions_count", 0)
    equity = portfolio.get("equity", 100)
    heat = portfolio.get("heat", 0)

    # ── Build ultra-compact prompt ──
    task = "EVAL_EXIT" if is_exit else "VAL_ENTRY"
    lines = [f"TASK:{task} {coin}@{current_price:.4f} pred:{pred_dir} tgt={pred_target:.4f} conf={pred_conf:.0f}"]

    if is_exit:
        lines.append(f"POS:{pos_side} PnL={pnl_pct:+.1f}% margin={margin_pnl:+.1f}% lev={leverage:.0f}x liq={liq_dist:.1f}%")

    # Horizons
    continuous = prediction.get("continuous", {})
    horizon_drivers = prediction.get("horizon_drivers", {})
    if continuous:
        hparts = []
        for h in ["1m", "15m", "4h"]:
            hd = continuous.get(h, {})
            if hd:
                hparts.append(f"{h}:{hd.get('up',0):.0f}/{hd.get('down',0):.0f}/{hd.get('flat',0):.0f}")
                # Add per-horizon drivers
                drivers = horizon_drivers.get(h, [])
                if drivers:
                    dstr = ",".join(f"{d[0]}={d[1]:+.0f}" for d in drivers[:2])
                    hparts[-1] += f"[{dstr}]"
        if hparts:
            lines.append("H:" + " ".join(hparts))

    # Layer weights / contributions
    lw = prediction.get("layer_weights", {})
    if lw:
        lw_parts = [f"{k[:5]}={v:.0f}" for k, v in sorted(lw.items(), key=lambda x: -x[1])[:6]]
        if lw_parts:
            lines.append(f"LW:{' '.join(lw_parts)}")

    # NEW: rich data lines (only if non-empty)
    if oi_delta_type:
        lines.append(f"OI:{oi_delta_type} {oi_delta_desc}" if oi_delta_desc else f"OI:{oi_delta_type}")
    accel_str = f"FUND:{funding_rate*100:+.4f}%/hr"
    if funding_accel:
        accel_str += f" accel={funding_accel}"
    lines.append(accel_str)
    if hurst_regime:
        lines.append(f"HURST:{hurst_regime} H={hurst_H:.2f}")
    if abs(cross_exchange_div) > 0.1:
        lines.append(f"XEX:{cross_exchange_div:+.2f}% {cross_exchange_dir}")
    if cascade_risk and cascade_risk != "none":
        lines.append(f"CASCADE:{cascade_risk}")
    if order_book_info:
        lines.append(f"OB:{order_book_info}")
    if atr_info:
        lines.append(f"ATR:{atr_info}")
    if htf_alignment:
        lines.append(f"HTF:{htf_alignment}")

    lines.append(f"REGIME:{market_regime} FG:{fear_greed} BTC_corr:{btc_correlation:.2f}")
    lines.append(f"PORT:eq={equity:.0f} pos={pos_count} heat={heat*100:.0f}%")

    if market_context:
        lines.append(f"CTX:{market_context}")

    # ── Ultra-compact rules ──
    if is_exit:
        lines.append("RULES:exit if margin<-5% & op>2 | exit if liq<5% | hold if neutral | close if opp_dir>40% & htf_agree | tight_stop if op>25% & cascade>low | keep_pos > be_flat")
    else:
        lines.append("RULES:reject if sigs<3 | exec if sigs>=4 & conf>50 | reduce if regime_against | reject if funding_extreme & no_accel | trade_regime_dont_freeze")

    lines.append('JSON:{"a":"execute_now|wait|reject|reduce_size|tighten_stop","c":0-100,"t":price,"s":0-1,"sg":0-10,"op":0-10,"rf":["funding","btc_div","thin","vol","fakeout","cascade","xex","oi","hurst","accel"]}')

    prompt = "\n".join(lines)

    # Call AI
    result = _call_deepseek(prompt, max_tokens=200)

    # ── Parse structured output ──
    action = result.get("a", result.get("action", "wait"))
    valid_actions = {"execute_now", "wait", "reject", "reduce_size", "tighten_stop"}
    if action not in valid_actions:
        action = "wait"

    adj_target = float(result.get("t", result.get("adjusted_target", pred_target)))
    adj_conf = max(0, min(100, float(result.get("c", result.get("adjusted_confidence", pred_conf)))))
    size_mult = max(0, min(1.0, float(result.get("s", result.get("size_multiplier", 0.5)))))
    sigs_agree = int(result.get("sg", result.get("signals_agreeing", 0)))
    sigs_oppose = int(result.get("op", result.get("signals_opposing", 0)))

    flags = result.get("rf", result.get("warning_flags", []))
    if not isinstance(flags, list):
        flags = [str(flags)]

    # Build compact reason from data (no LLM text)
    reason_parts = []
    if sigs_agree > 0:
        reason_parts.append(f"{sigs_agree}s/+")
    if sigs_oppose > 0:
        reason_parts.append(f"{sigs_oppose}s/-")
    if flags:
        reason_parts.append(",".join(flags[:3]))
    reason = " ".join(reason_parts) if reason_parts else "ai_validated"

    return AIValidationResult(
        approved=action in ("execute_now", "reduce_size"),
        adjusted_target=adj_target,
        adjusted_confidence=adj_conf,
        action=action,
        size_multiplier=size_mult,
        reason=reason,
        warning_flags=flags,
        signals_agreeing=sigs_agree,
        signals_opposing=sigs_oppose,
    )


def format_validation(coin: str, result: AIValidationResult) -> str:
    emoji_map = {"execute_now": "✅", "reduce_size": "⚠️", "wait": "⏸️", "reject": "❌", "tighten_stop": "🔒"}
    emoji = emoji_map.get(result.action, "❓")
    flags_str = f" [{','.join(result.warning_flags)}]" if result.warning_flags else ""
    sig_info = f" +{result.signals_agreeing}/-{result.signals_opposing}" if result.signals_agreeing or result.signals_opposing else ""
    return (
        f"{emoji} {coin}: {result.action} {sig_info} "
        f"tgt=${result.adjusted_target:.4f} conf={result.adjusted_confidence:.0f}% sz={result.size_multiplier:.0%}{flags_str}"
    )


# ── Self-test ──
if __name__ == "__main__":
    test_pred = {
        "direction": "down", "confidence": 65, "target_price": 75.0,
        "ml_score": 55, "exhaustion_score": 70, "structure_score": 60,
        "flow_score": 45, "ob_thinning": 30, "tick_flow": 20, "cvd_divergence": 50,
    }
    test_portfolio = {"equity": 108, "positions_count": 5, "heat": 0.35}
    result = validate_prediction(
        "SOL", 77.3, test_pred, test_portfolio,
        market_regime="ranging", fear_greed=25, funding_rate=0.0027,
        support_level=74.0, resistance_level=80.0,
    )
    print(format_validation("SOL", result))
