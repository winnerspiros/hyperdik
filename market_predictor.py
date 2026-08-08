"""
Market Predictor — AI-powered, forward-looking market & portfolio forecaster.
- Reads reviewer assessment from strategy_calendar.json
- Pulls extended market data (multi-TF, historical patterns, order book)
- AI makes predictions: near-term (hours), medium-term (days), strategic (week+)
- Sets calendar entries: "Hold till X date", "Target price Y by Z", "Exit if..."
- Can write action files (advisory, not execution) for day_trader to read
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone, timedelta

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PREDICTOR] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "predictor.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Predictor")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")
ACTION_DIR = os.path.join(TRADER_DIR, "data", "predictions")
os.makedirs(ACTION_DIR, exist_ok=True)

# ── lazy imports ─────────────────────────────────────────────────────────────

def _import(name):
    try:
        return __import__(name, fromlist=[""])
    except Exception as e:
        return None

rev_client_mod = _import("revolut_client")
data_enricher_mod = _import("data_enricher")
market_regime_mod = _import("market_regime")
market_analyzer_mod = _import("market_analyzer")
chart_patterns_mod = _import("chart_patterns")
funding_signals_mod = _import("funding_signals")
correlation_tracker_mod = _import("correlation_tracker")
ml_predictor_mod = _import("ml_predictor")
liquidation_data_mod = _import("liquidation_data")

# ── AI brain (same pattern as reviewer, but more strategic) ──────────────────

class PredictorBrain:
    """LLM interface for the predictor — forward-looking, strategic."""

    def __init__(self):
        self.api_key = ""
        for source in [
            os.environ.get("OPENROUTER_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        ]:
            if source:
                self.api_key = source
                break
        if not self.api_key:
            for path in [
                os.path.expanduser("~/.hermes/.env"),
                "/home/ubuntu/.hermes/.env",
            ]:
                try:
                    with open(path) as f:
                        for line in f:
                            if "OPENROUTER_API_KEY" in line:
                                raw = line.split("=", 1)[1].strip()
                                key = raw.strip("'\" ").strip()
                                if key:  # skip empty keys
                                    self.api_key = key
                                    break
                except Exception:
                    pass
                if self.api_key:
                    break
        self.standard_model = "qwen/qwen3-30b-a3b-instruct-2507"
        self.premium_model = "qwen/qwen3-30b-a3b-instruct-2507"

    def _call(self, system_prompt, user_prompt, temperature=0.5, premium=False):
        import urllib.request, ssl
        if not self.api_key:
            return {"error": "no_api_key"}
        ctx = ssl.create_default_context()
        model = self.premium_model if premium else self.standard_model
        data = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2000,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=45) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
                return {"content": content}
        except Exception as e:
            return {"error": str(e)}

# ── Calendar helpers ─────────────────────────────────────────────────────────

def _load_calendar():
    try:
        with open(CALENDAR_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"version": 1, "last_updated": None, "reviewer": {}, "predictor": {}, "calendar": []}

def _save_calendar(cal):
    with open(CALENDAR_PATH, "w") as f:
        json.dump(cal, f, indent=2, default=str)

def _add_calendar_entry(cal, entry_type, title, description, author="predictor"):
    entry = {
        "id": f"{entry_type.upper()}_{int(time.time())}",
        "type": entry_type,
        "title": title,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": author,
    }
    cal.setdefault("calendar", []).append(entry)
    if len(cal["calendar"]) > 50:
        cal["calendar"] = cal["calendar"][-50:]

# ── Multi-timeframe price collector ──────────────────────────────────────────

def _collect_multi_tf_prices(client, symbols):
    """Collect OHLC data at multiple timeframes for prediction."""
    tf_data = {}
    if not client:
        return tf_data

    timeframes = {"1h": 48, "4h": 30, "1d": 14}
    for symbol in symbols[:8]:  # Max 8 coins to avoid rate limits
        sym_data = {}
        for tf, hours in timeframes.items():
            try:
                from market_analyzer import MarketAnalyzer
                ma = MarketAnalyzer(client)
                df = ma.get_candles_df(symbol, interval=tf, hours_back=hours)
                if df is not None and len(df) >= 3:
                    close = df["close"].values.astype(float)
                    high = df["high"].values.astype(float)
                    low = df["low"].values.astype(float)
                    vol = df["volume"].values.astype(float)
                    sym_data[tf] = {
                        "close": close[-1],
                        "open": df["open"].values.astype(float)[-1],
                        "high_24h": high[-24:].max() if len(high) >= 24 else high.max(),
                        "low_24h": low[-24:].min() if len(low) >= 24 else low.min(),
                        "vol_24h": vol[-24:].sum() if len(vol) >= 24 else vol.sum(),
                        "trend": "up" if close[-1] > close[-len(close)//2] else "down",
                        "range_pct": (high[-1] - low[-1]) / low[-1] * 100 if low[-1] > 0 else 0,
                        "candles": len(df),
                    }
            except Exception as e:
                log.debug(f"  {symbol} {tf}: {e}")
        if sym_data:
            tf_data[symbol] = sym_data
    return tf_data

def _collect_technical_signals(client, symbols):
    """Pull MACD, RSI, volume signals from multi-TF data."""
    signals = {}
    if not client:
        return signals
    try:
        from market_analyzer import MarketAnalyzer
        ma = MarketAnalyzer(client)
        for symbol in symbols[:8]:
            sym_sig = {}
            df_1h = ma.get_candles_df(symbol, "1h", hours_back=72)
            if df_1h is not None and len(df_1h) >= 20:
                df = ma.calculate_indicators(df_1h)
                if df is not None:
                    last = df.iloc[-1]
                    # RSI
                    rsi = last.get("rsi", 50)
                    sym_sig["rsi"] = rsi
                    sym_sig["rsi_signal"] = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
                    # MACD
                    macd = last.get("macd", 0)
                    macd_sig = last.get("macd_signal", 0)
                    sym_sig["macd"] = "bullish" if macd > macd_sig else "bearish"
                    # Volume
                    avg_vol = df["volume"].values.astype(float).mean()
                    cur_vol = float(last.get("volume", 0))
                    sym_sig["volume_vs_avg"] = cur_vol / avg_vol if avg_vol > 0 else 1.0
                    # ADX
                    adx = last.get("adx", 25)
                    sym_sig["adx"] = adx
                    sym_sig["trend_strength"] = "strong" if adx >= 25 else "weak"
                    signals[symbol] = sym_sig
    except Exception as e:
        log.warning(f"Technical signals error: {e}")
    return signals

def _collect_chart_patterns(client, symbols):
    """Detect chart patterns for each symbol."""
    patterns = {}
    if not client or not chart_patterns_mod:
        return patterns
    try:
        from market_analyzer import MarketAnalyzer
        ma = MarketAnalyzer(client)
        for symbol in symbols[:8]:
            df = ma.get_candles_df(symbol, "1h", hours_back=120)
            if df is not None and len(df) >= 50:
                close = df["close"].values.astype(float)
                high = df["high"].values.astype(float)
                low = df["low"].values.astype(float)
                detected = chart_patterns_mod.detect_all_patterns(close, high, low)
                if detected:
                    patterns[symbol] = detected
    except Exception as e:
        log.debug(f"Chart patterns: {e}")
    return patterns

# ── Action file system ───────────────────────────────────────────────────────

def write_advisory(symbol, recommendation, target_price, timeframe, reasoning, confidence):
    """Write an advisory action file that day_trader can read."""
    advisory = {
        "symbol": symbol,
        "recommendation": recommendation,  # "hold", "buy_dip", "sell_on_rally", "take_profit", "wait"
        "target_price": target_price,
        "timeframe": timeframe,
        "confidence": confidence,
        "reasoning": reasoning[:200],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "market_predictor",
    }
    path = os.path.join(ACTION_DIR, f"{symbol}_advisory.json")
    with open(path, "w") as f:
        json.dump(advisory, f, indent=2)
    log.info(f"  Advisory written: {path}")
    return advisory

# ── Build prediction prompt ──────────────────────────────────────────────────

def build_prediction_prompt(reviewer_data, calendar_data, multi_tf, technical,
                            chart_patterns, eur_balance,
                            funding_data=None, correlation_text="", liquidation_text=""):
    """Build the AI prompt with full context for prediction."""
    lines = []
    lines.append("You are a CRYPTO MARKET PREDICTOR with deep strategic vision.")
    lines.append("Your job: analyze ALL data, make forward-looking predictions, set strategy.")
    lines.append("")
    lines.append("You can predict: what will happen and when. Set strategic calendar entries.")
    lines.append("Think about: economic calendar events, market cycles, seasonality, patterns.")
    lines.append("")
    lines.append("CRITICAL RULES:")
    lines.append("- Be SPECIFIC with timing: 'hold until 15 July', 'target €0.95 by end of week'")
    lines.append("- Be HONEST about uncertainty — if you don't know, say so with low confidence")
    lines.append("- This is ADVISORY — the reader will decide whether to follow your advice")
    lines.append("- If the data says something will happen, state WHEN and WHY")
    lines.append("- You CAN say 'I don't know' or 'too uncertain to predict' for unclear situations")
    lines.append("")

    # Reviewer context
    if reviewer_data:
        lines.append("=== REVIEWER'S ASSESSMENT ===")
        lines.append(f"Overall: {reviewer_data.get('overall_assessment', 'N/A')}")
        lines.append(f"Confidence: {reviewer_data.get('confidence', 0)}")
        lines.append(f"Portfolio Health: {reviewer_data.get('portfolio_health', 'N/A')}")
        lines.append(f"Market Trend: {reviewer_data.get('market_trend', 'N/A')}")
        lines.append(f"Top Concern: {reviewer_data.get('top_concern', 'N/A')}")
        lines.append(f"Top Opportunity: {reviewer_data.get('top_opportunity', 'N/A')}")
        lines.append(f"Reviewer Notes: {reviewer_data.get('notes', '')[:300]}")
        lines.append("")

    # Current portfolio
    if reviewer_data and reviewer_data.get("positions"):
        lines.append(f"EUR Balance: €{reviewer_data.get('eur_balance', 0):.2f}")
        lines.append(f"Portfolio Value: €{reviewer_data.get('portfolio_value', 0):.2f}")
        lines.append("Current Positions:")
        for p in reviewer_data.get("positions", []):
            lines.append(f"  {p['symbol']}: {p['qty']:.6f} @ €{p['current_price']:.4f} = €{p['value_eur']:.2f}")
        lines.append("")

    # Multi-timeframe
    if multi_tf:
        lines.append("=== MULTI-TIMEFRAME PRICES ===")
        for sym, tfs in sorted(multi_tf.items()):
            tf_strs = []
            for tf_name, tf_data in sorted(tfs.items()):
                tf_strs.append(
                    f"{tf_name}: close=€{tf_data['close']:.4f} trend={tf_data.get('trend','?')} "
                    f"range={tf_data.get('range_pct',0):.1f}% vol={tf_data.get('vol_24h',0)/1e3:.0f}K"
                )
            lines.append(f"  {sym}: {' | '.join(tf_strs)}")
        lines.append("")

    # Technical signals
    if technical:
        lines.append("=== TECHNICAL SIGNALS ===")
        for sym, sig in sorted(technical.items()):
            lines.append(
                f"  {sym}: RSI={sig.get('rsi',50):.0f} ({sig.get('rsi_signal','?')}) "
                f"MACD={sig.get('macd','?')} ADX={sig.get('adx',0):.0f} ({sig.get('trend_strength','?')}) "
                f"vol={sig.get('volume_vs_avg',1):.1f}x avg"
            )
        lines.append("")

    # Chart patterns
    if chart_patterns:
        lines.append("=== CHART PATTERNS DETECTED ===")
        for sym, pats in sorted(chart_patterns.items()):
            pat_strs = [f"{p.get('pattern','?')} ({p.get('direction','?')})" for p in pats[:3]]
            if pat_strs:
                lines.append(f"  {sym}: {', '.join(pat_strs)}")
        lines.append("")

    # Funding rates + OI — sentiment/positioning signal
    if funding_data:
        lines.append("=== FUNDING RATES (Kraken/OKX) ===")
        for s in funding_data:
            lines.append(f"  • {s}")
        lines.append("")

    # Correlation / diversification risk
    if correlation_text:
        lines.append("=== PORTFOLIO CORRELATION ===")
        lines.append(str(correlation_text))
        lines.append("")

    # Liquidation zone estimate from volatility anomalies
    if liquidation_text:
        lines.append("=== LIQUIDATION RISK ===")
        lines.append(str(liquidation_text))
        lines.append("")

    # Existing calendar
    if calendar_data:
        recent = [e for e in calendar_data if isinstance(e, dict)][-5:]
        if recent:
            lines.append("=== RECENT CALENDAR ENTRIES ===")
            for e in recent:
                t = e.get("type", "?")
                title = e.get("title", "?")[:60]
                author = e.get("author", "?")
                lines.append(f"  [{t}] {title} (by {author})")
            lines.append("")

    # Economic calendar reminder
    lines.append("Important: Consider what's coming up:")
    lines.append("- CPI/PPI releases, FOMC meetings, jobs data")
    lines.append("- Bitcoin halving cycles (2028 is next)")
    lines.append("- Weekend low-volume effects")
    lines.append("- Month-end / quarter-end rebalancing")
    lines.append("- Major token unlocks and protocol upgrades")
    lines.append("")

    lines.append("""Now provide your analysis as structured JSON:
```json
{
  "market_outlook": "BULLISH | BEARISH | NEUTRAL | MIXED | UNCERTAIN",
  "outlook_confidence": 0.0-1.0,
  "time_horizon": "near_term | medium_term | uncertain",
  "predictions": [
    {
      "symbol": "BTC",
      "current_price": 0,
      "prediction": "rise | fall | range | unclear",
      "confidence": 0.0-1.0,
      "target_price": null,
      "timeframe": "12h | 2d | 1w | unclear",
      "key_levels": {"support": 0, "resistance": 0},
      "reasoning": "why this prediction"
    }
  ],
  "strategy_actions": [
    {
      "type": "hold_until | wait_for_price | set_stop | take_profit | watch",
      "symbol": "BTC",
      "condition": "what must happen",
      "target": "when or at what price",
      "priority": "high | medium | low",
      "reasoning": "why this action"
    }
  ],
  "active_strategy": "a one-line summary of our current strategy",
  "biggest_risk": "what could go wrong",
  "biggest_opportunity": "what could go right",
  "predictor_notes": "detailed strategic analysis"
}
```""")

    return "\n".join(lines)

def parse_predictions(content):
    """Extract structured JSON from AI response."""
    # Strip markdown code fences if present
    cleaned = content.strip()
    if cleaned.startswith('```'):
        first_newline = cleaned.find('\n')
        if first_newline > 0:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try: parse cleaned string as JSON
    try:
        parsed = json.loads(cleaned)
        # Normalize field names (AI may use 'asset' instead of 'symbol')
        if "predictions" in parsed:
            for p in parsed["predictions"]:
                if "asset" in p and "symbol" not in p:
                    p["symbol"] = p["asset"]
        # Fill defaults so downstream code doesn't crash on missing keys
        parsed.setdefault("strategy_actions", [])
        parsed.setdefault("biggest_risk", "")
        parsed.setdefault("biggest_opportunity", "")
        parsed.setdefault("predictor_notes", "")
        parsed.setdefault("active_strategy", "")
        return parsed
    except json.JSONDecodeError:
        pass

    # Try: find any JSON object containing market_outlook
    json_match = re.search(r'({[\s\S]*?"market_outlook"[\s\S]*?})', cleaned)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "predictions" in parsed:
                for p in parsed["predictions"]:
                    if "asset" in p:
                        p["symbol"] = p["asset"]
            return parsed
        except json.JSONDecodeError:
            pass
    return None

# ── Main prediction cycle ────────────────────────────────────────────────────

def run_prediction():
    """Full prediction cycle — read calendar, collect data, predict, write back."""
    log.info("=" * 60)
    log.info("🔮 MARKET PREDICTOR STARTING")
    log.info("=" * 60)

    # 1. Read the calendar (reviewer's assessment)
    cal = _load_calendar()
    reviewer_data = cal.get("reviewer", {})
    cal_entries = cal.get("calendar", [])

    last_review = reviewer_data.get("last_run", "never")
    reviewer_verdicts = reviewer_data.get("verdicts", {})

    log.info(f"📖 Last review: {last_review}")
    log.info(f"📖 Reviewer assessment: {reviewer_data.get('overall_assessment', 'N/A')}")

    # If no review exists, warn but proceed
    if not reviewer_data:
        log.warning("⚠️ No reviewer assessment found — predictions will be less informed")

    # 2. Connect to Revolut for extended data (still read-only)
    client = None
    if rev_client_mod:
        try:
            client = rev_client_mod.RevolutXClient()
        except Exception:
            pass

    # 3. Get current holdings
    holdings_symbols = [p.get("symbol") for p in reviewer_data.get("positions", [])]
    if not holdings_symbols:
        log.warning("⚠️ No holdings from reviewer — checking directly")
        try:
            if client:
                bal = client.get_balances()
                for entry in bal:
                    cur = entry.get("currency", "")
                    avail = float(entry.get("available", 0))
                    if avail > 0 and cur not in ("EUR", "USDC", "USD", "EURC", "GBP", "USDT"):
                        holdings_symbols.append(cur)
        except Exception:
            pass

    # If no symbols at all, use a default set
    if not holdings_symbols:
        holdings_symbols = ["BTC", "ETH", "XRP", "SOL", "ADA", "DOT", "LINK", "AVAX"]
        log.info(f"Using default watchlist: {holdings_symbols}")

    log.info(f"📊 Holdings to analyze: {holdings_symbols}")

    # 4. Collect prediction data
    eur_balance = reviewer_data.get("eur_balance", 0)
    portfolio_value = reviewer_data.get("portfolio_value", 0)

    log.info("📡 Collecting multi-timeframe data...")
    multi_tf = _collect_multi_tf_prices(client, holdings_symbols)
    if multi_tf:
        syms_ok = list(multi_tf.keys())
        log.info(f"   Got TF data for {len(multi_tf)} symbols: {syms_ok}")

    log.info("📊 Collecting technical signals...")
    technical = _collect_technical_signals(client, holdings_symbols)
    if technical:
        log.info(f"   Got tech signals for {len(technical)} symbols")

    log.info("🔍 Detecting chart patterns...")
    chart_pats = _collect_chart_patterns(client, holdings_symbols)
    if chart_pats:
        log.info(f"   Patterns detected for {len(chart_pats)} symbols")

    log.info("📊 Collecting funding rates + correlation + liquidation risk...")
    funding_data = []
    try:
        if funding_signals_mod and hasattr(funding_signals_mod, "get_funding_data"):
            funding_data = funding_signals_mod.get_funding_data()
    except Exception as e:
        log.warning(f"funding_signals error: {e}")

    correlation_text = ""
    try:
        if correlation_tracker_mod and holdings_symbols:
            correlation_text = correlation_tracker_mod.get_diversification_advice(
                holdings_symbols, {s: 1 for s in holdings_symbols}
            )
    except Exception as e:
        log.warning(f"correlation_tracker error: {e}")

    liquidation_text = ""
    try:
        if liquidation_data_mod and hasattr(liquidation_data_mod, "format_liquidation_for_ai") and holdings_symbols:
            # This function analyzes one symbol at a time — check the most heavily weighted holding (first in list)
            liquidation_text = liquidation_data_mod.format_liquidation_for_ai(holdings_symbols[0])
    except Exception as e:
        log.warning(f"liquidation_data error: {e}")

    # 5. Call AI for predictions
    log.info("🤖 Consulting AI for predictions...")
    brain = PredictorBrain()
    if not brain.api_key:
        log.error("❌ No API key — cannot predict")
        return {"status": "error", "reason": "no_api_key"}

    prompt = build_prediction_prompt(
        reviewer_data, cal_entries, multi_tf, technical,
        chart_pats, eur_balance,
        funding_data=funding_data, correlation_text=correlation_text, liquidation_text=liquidation_text,
    )

    # First call: standard model for predictions
    result = brain._call(
        "You are a crypto market predictor. Analyze deeply, predict with specific timing and prices.",
        prompt,
        temperature=0.4,
        premium=False,
    )

    if "error" in result:
        log.error(f"❌ AI call failed: {result['error']}")
        return {"status": "error", "reason": result["error"]}

    ai_output = result["content"]
    log.info(f"✅ AI prediction received ({len(ai_output)} chars)")

    # 6. Parse predictions
    predictions = parse_predictions(ai_output)
    # Always define these with defaults before conditional block
    outlook = "UNCERTAIN"
    confidence = 0.0
    strat = "N/A"
    if predictions:
        outlook = predictions.get("market_outlook", "UNCERTAIN")
        confidence = predictions.get("outlook_confidence", 0)
        strat = predictions.get("active_strategy", "N/A")
        log.info(f"🔮 Outlook: {outlook} (conf: {confidence:.0%})")
        log.info(f"📋 Strategy: {strat}")
        for pred in predictions.get("predictions", []):
            sym = pred.get("symbol", "?")
            direction = pred.get("prediction", "?")
            conf = pred.get("confidence", 0)
            target = pred.get("target_price", "N/A")
            tf = pred.get("timeframe", "?")
            log.info(f"  {sym:6s}: {direction:8s} conf={conf:.0%} target=€{target} timeframe={tf}")
        for action in predictions.get("strategy_actions", []):
            atype = action.get("type", "?")
            asym = action.get("symbol", "?")
            cond = action.get("condition", "")[:60]
            priority = action.get("priority", "?")
            log.info(f"  ▶️ {atype:15s} {asym:6s} | {cond} [priority: {priority}]")
        log.info(f"  ⚠️ Risk: {predictions.get('biggest_risk', 'N/A')}")
        log.info(f"  ✅ Opportunity: {predictions.get('biggest_opportunity', 'N/A')}")
    else:
        log.warning("⚠️ Could not parse structured predictions")
        log.info(f"Raw response preview: {ai_output[:500]}")

    # 7. Write to calendar
    now_iso = datetime.now(timezone.utc).isoformat()
    cal["predictor"] = {
        "last_run": now_iso,
        "market_outlook": (predictions or {}).get("market_outlook", "UNCERTAIN"),
        "outlook_confidence": (predictions or {}).get("outlook_confidence", 0),
        "time_horizon": (predictions or {}).get("time_horizon", "uncertain"),
        "active_strategy": (predictions or {}).get("active_strategy", ""),
        "predictions": (predictions or {}).get("predictions", []),
        "strategy_actions": (predictions or {}).get("strategy_actions", []),
        "biggest_risk": (predictions or {}).get("biggest_risk", ""),
        "biggest_opportunity": (predictions or {}).get("biggest_opportunity", ""),
        "predictor_notes": (predictions or {}).get("predictor_notes", ""),
    }

    # Add calendar entries for each strategy action
    for action in (predictions or {}).get("strategy_actions", []):
        atype = action.get("type", "note")
        sym = action.get("symbol", "")
        condition = action.get("condition", "")
        target = action.get("target", "")
        reasoning = action.get("reasoning", "")[:100]
        title = f"{atype.upper()}: {sym} — {condition[:40]}"
        desc = f"Symbol: {sym} | Condition: {condition} | Target: {target} | Priority: {action.get('priority', 'low')}\nReasoning: {reasoning}"
        _add_calendar_entry(cal, atype, title, desc)

    # Add a summary entry
    outlook = (predictions or {}).get("market_outlook", "UNCERTAIN")
    strat = (predictions or {}).get("active_strategy", "N/A")
    _add_calendar_entry(
        cal,
        "prediction",
        f"Market Outlook: {outlook} | Strategy: {strat[:50]}",
        f"Confidence: {(predictions or {}).get('outlook_confidence', 0):.0%} | "
        f"Time Horizon: {(predictions or {}).get('time_horizon', '?')} | "
        f"Risk: {(predictions or {}).get('biggest_risk', 'N/A')} | "
        f"Opportunity: {(predictions or {}).get('biggest_opportunity', 'N/A')}",
    )

    cal["last_updated"] = now_iso
    _save_calendar(cal)
    log.info("💾 Predictions written to strategy_calendar.json")

    # 8. Write advisory files for each prediction
    if predictions:
        for pred in predictions.get("predictions", []):
            sym = pred.get("symbol", "")
            direction = pred.get("prediction", "")
            conf = pred.get("confidence", 0)
            target = pred.get("target_price")
            tf = pred.get("timeframe", "")
            reasoning = pred.get("reasoning", "")

            # Map prediction direction to recommendation
            if direction == "rise":
                rec = "hold" if conf >= 0.5 else "wait"
            elif direction == "fall":
                rec = "sell_on_rally" if conf >= 0.5 else "wait"
            else:
                rec = "wait"

            if sym:
                write_advisory(sym, rec, target, tf, reasoning, conf)

    # Also write combined strategy file
    combined = {
        "generated_at": now_iso,
        "market_outlook": (predictions or {}).get("market_outlook", "UNCERTAIN"),
        "active_strategy": (predictions or {}).get("active_strategy", ""),
        "strategy_actions": (predictions or {}).get("strategy_actions", []),
        "biggest_risk": (predictions or {}).get("biggest_risk", ""),
        "biggest_opportunity": (predictions or {}).get("biggest_opportunity", ""),
    }
    with open(os.path.join(ACTION_DIR, "current_strategy.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # 9. Build summary
    predictions = predictions or {}  # ensure it's not None
    summary_lines = []
    summary_lines.append("🔮 MARKET PREDICTION")
    summary_lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    summary_lines.append(f"Outlook: {outlook} (confidence: {confidence:.0%})")
    summary_lines.append(f"Strategy: {strat}")
    summary_lines.append("")
    for pred in (predictions or {}).get("predictions", []):
        sym = pred.get("symbol", "?")
        direction = pred.get("prediction", "?")
        conf = pred.get("confidence", 0)
        target = pred.get("target_price", "N/A")
        tf = pred.get("timeframe", "?")
        summary_lines.append(f"  {sym}: {direction} → target €{target} in {tf} (conf: {conf:.0%})")
    summary_lines.append("")
    summary_lines.append("Strategy Calendar:")
    for action in (predictions or {}).get("strategy_actions", []):
        atype = action.get("type", "?")
        sym = action.get("symbol", "")
        cond = action.get("condition", "")[:60]
        target = action.get("target", "")[:40]
        priority = action.get("priority", "?")
        summary_lines.append(f"  [{priority}] {sym}: {atype} — {cond}")
        if target:
            summary_lines.append(f"         → {target}")
    summary_lines.append("")
    summary_lines.append(f"⚠️ Risk: {predictions.get('biggest_risk', 'N/A')}")
    summary_lines.append(f"✅ Opportunity: {predictions.get('biggest_opportunity', 'N/A')}")
    notes = predictions.get("predictor_notes", "")
    if notes:
        summary_lines.append(f"\n📝 Notes: {notes[:500]}")

    log.info("\n" + "\n".join(summary_lines))

    return {
        "status": "ok",
        "outlook": outlook,
        "confidence": confidence,
        "predictions_count": len((predictions or {}).get("predictions", [])),
        "actions_count": len((predictions or {}).get("strategy_actions", [])),
        "active_strategy": strat,
    }


if __name__ == "__main__":
    result = run_prediction()
    print(json.dumps(result, indent=2))