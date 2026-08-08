#!/usr/bin/env python3
"""
Portfolio Reviewer — AI-powered, read-only portfolio & market assessment.
- Analyzes ALL holdings and their performance
- Reviews macro, technical, on-chain, sentiment, correlation data
- AI evaluates each position: STRONG / HOLD / WARNING / WEAK
- Writes structured assessment to strategy_calendar.json
- NO trading actions — analysis only
"""
import sys, os, json, time, logging, re
from datetime import datetime, timezone
from pathlib import Path

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REVIEWER] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "reviewer.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Reviewer")

CALENDAR_PATH = os.path.join(TRADER_DIR, "data", "strategy_calendar.json")

# ── imports (lazy, with fallbacks) ───────────────────────────────────────────

def _import(name):
    try:
        return __import__(name, fromlist=[""])
    except Exception as e:
        return None

rev_client_mod = _import("revolut_client")
data_enricher_mod = _import("data_enricher")
market_regime_mod = _import("market_regime")
self_review_mod = _import("self_review")
performance_metrics_mod = _import("performance_metrics")
correlation_tracker_mod = _import("correlation_tracker")
onchain_metrics_mod = _import("onchain_metrics")
funding_signals_mod = _import("funding_signals")
chart_patterns_mod = _import("chart_patterns")
defi_monitor_mod = _import("defi_monitor")
whale_tracker_mod = _import("whale_tracker")
news_collector_mod = _import("news_collector")
economic_calendar_mod = _import("economic_calendar")
sentiment_scraper_mod = _import("sentiment_scraper")
mempool_monitor_mod = _import("mempool_monitor")
market_intelligence_mod = _import("market_intelligence")

# ── AI brain ─────────────────────────────────────────────────────────────────

class ReviewerBrain:
    """LLM interface for the reviewer — read-only analysis, no trading decisions."""

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
                                if key:
                                    self.api_key = key
                                    break
                except Exception:
                    pass
                if self.api_key:
                    break
        self.model = "google/gemini-2.5-flash-lite"

    def _call(self, system_prompt, user_prompt, temperature=0.3):
        import urllib.request, ssl
        if not self.api_key:
            return {"error": "no_api_key"}
        ctx = ssl.create_default_context()
        data = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 1500,
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
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
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

def _add_calendar_entry(cal, entry_type, title, description, author="reviewer"):
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
    return entry

# ── Portfolio data collector ─────────────────────────────────────────────────

def _get_revolut_client():
    if not rev_client_mod:
        return None
    try:
        return rev_client_mod.RevolutXClient()
    except Exception as e:
        log.error(f"Failed to create Revolut client: {e}")
        return None

def _collect_portfolio(client):
    if not client:
        return 0.0, {}, [], 0.0
    try:
        balances = client.get_balances()
    except Exception as e:
        log.error(f"Failed to get balances: {e}")
        return 0.0, {}, [], 0.0

    eur = 0.0
    holdings = {}
    for entry in balances:
        cur = entry.get("currency", "")
        avail = float(entry.get("available", 0))
        if avail <= 0:
            continue
        if cur == "EUR":
            eur = avail
        elif cur not in ("USDC", "USD", "EURC", "GBP", "USDT"):
            holdings[cur] = avail

    current_prices = {}
    try:
        raw_tickers = client.get_tickers()
        tickers = raw_tickers if isinstance(raw_tickers, list) else raw_tickers.get("data", [])
        for t in tickers:
            if not isinstance(t, dict):
                continue
            sym = t.get("symbol", "").replace("/USD", "")
            if sym in holdings:
                current_prices[sym] = {
                    "bid": float(t.get("bid", 0) or 0),
                    "ask": float(t.get("ask", 0) or 0),
                    "last": float(t.get("last_price", 0) or 0),
                    "change_24h": t.get("change_24h"),
                    "volume_24h": t.get("volume_24h"),
                }
    except Exception as e:
        log.warning(f"Failed to get tickers: {e}")

    positions = []
    total_value = 0.0
    for sym, qty in sorted(holdings.items()):
        price_info = current_prices.get(sym, {})
        last_price = price_info.get("last") or price_info.get("bid") or 0
        value = last_price * qty
        total_value += value
        positions.append({
            "symbol": sym,
            "qty": qty,
            "current_price": last_price,
            "value_eur": round(value, 2),
            "change_24h": price_info.get("change_24h"),
        })

    return eur, holdings, positions, total_value

def _collect_market_data(holdings_symbols):
    data = {}
    try:
        if market_regime_mod:
            data["regime"] = market_regime_mod.classify_regime()
    except Exception as e:
        data["regime"] = f"error: {e}"
    try:
        if data_enricher_mod:
            data["fear_greed"] = data_enricher_mod.get_fear_greed()
            data["global_data"] = data_enricher_mod.get_global_data()
            data["trending"] = data_enricher_mod.get_trending()
            data["coin_prices"] = data_enricher_mod.get_coin_prices(holdings_symbols)
    except Exception as e:
        log.warning(f"data_enricher error: {e}")
    try:
        if self_review_mod:
            data["self_review"] = self_review_mod.get_ai_review_context("day_trader")
            data["scalper_review"] = self_review_mod.get_ai_review_context("scalper")
    except Exception as e:
        log.warning(f"self_review error: {e}")
    # On-chain (Mayer Multiple + MVRV/Supply/ROI/ActiveAddr) — correct function name
    try:
        if onchain_metrics_mod and hasattr(onchain_metrics_mod, "get_combined_onchain_summary"):
            data["onchain"] = onchain_metrics_mod.get_combined_onchain_summary()
    except Exception as e:
        log.warning(f"onchain_metrics error: {e}")
    # DeFi TVL + stablecoin supply — correct function name
    try:
        if defi_monitor_mod and hasattr(defi_monitor_mod, "format_defi_for_ai"):
            data["defi"] = defi_monitor_mod.format_defi_for_ai()
    except Exception as e:
        log.warning(f"defi_monitor error: {e}")
    # Funding rates + OI (Kraken + OKX) — correct function name
    try:
        if funding_signals_mod and hasattr(funding_signals_mod, "get_funding_data"):
            data["funding"] = funding_signals_mod.get_funding_data()
    except Exception as e:
        log.warning(f"funding_signals error: {e}")
    # Multi-timeframe OHLC (daily+weekly) for held coins
    try:
        if funding_signals_mod and hasattr(funding_signals_mod, "format_multi_price_for_ai") and holdings_symbols:
            data["multi_price"] = funding_signals_mod.format_multi_price_for_ai(holdings_symbols)
    except Exception as e:
        log.warning(f"multi_price error: {e}")
    # Whale movements — large on-chain transfers
    try:
        if whale_tracker_mod:
            data["whales"] = whale_tracker_mod.get_whale_movements()
    except Exception as e:
        log.warning(f"whale_tracker error: {e}")
    # Crypto news headlines
    try:
        if news_collector_mod:
            data["news"] = news_collector_mod.get_crypto_news()
    except Exception as e:
        log.warning(f"news_collector error: {e}")
    # Economic calendar — CPI/Fed/jobs events that move markets
    try:
        if economic_calendar_mod:
            data["economic_events"] = economic_calendar_mod.get_upcoming_economic_events()
    except Exception as e:
        log.warning(f"economic_calendar error: {e}")
    # Portfolio correlation + diversification risk
    try:
        if correlation_tracker_mod and holdings_symbols:
            data["diversification"] = correlation_tracker_mod.get_diversification_advice(
                holdings_symbols, {s: 1 for s in holdings_symbols}
            )
    except Exception as e:
        log.warning(f"correlation_tracker error: {e}")
    # Social/news sentiment score
    try:
        if sentiment_scraper_mod and hasattr(sentiment_scraper_mod, "get_crypto_news_sentiment"):
            data["sentiment"] = sentiment_scraper_mod.get_crypto_news_sentiment()
    except Exception as e:
        log.warning(f"sentiment_scraper error: {e}")
    # Bitcoin mempool — leading volatility indicator
    try:
        if mempool_monitor_mod and hasattr(mempool_monitor_mod, "format_mempool_for_ai"):
            data["mempool"] = mempool_monitor_mod.format_mempool_for_ai()
    except Exception as e:
        log.warning(f"mempool_monitor error: {e}")
    return data

# ── AI assessment ────────────────────────────────────────────────────────────

def build_review_prompt(positions, eur_balance, market_data):
    lines = []
    lines.append("You are a CRYPTO PORTFOLIO REVIEWER.")
    lines.append("")
    lines.append("RESPOND ONLY WITH VALID JSON. NO PROSE BEFORE OR AFTER.")
    lines.append("")
    lines.append('Your response must be ONLY this JSON object:')
    lines.append('```json')
    lines.append('{')
    lines.append('  "overall_assessment": "BULLISH | BEARISH | NEUTRAL | MIXED",')
    lines.append('  "confidence": 0.0-1.0,')
    lines.append('  "market_trend": "brief description",')
    lines.append('  "portfolio_health": "GOOD | FAIR | CONCERNING | POOR",')
    lines.append('  "positions": [')
    lines.append('    {')
    lines.append('      "symbol": "BTC",')
    lines.append('      "verdict": "STRONG|HOLD|WARNING|WEAK",')
    lines.append('      "reason": "brief reason based on data",')
    lines.append('      "suggested_action": "hold | consider adding | consider reducing | prepare to exit"')
    lines.append('    }')
    lines.append('  ],')
    lines.append('  "top_concern": "biggest risk or concern",')
    lines.append('  "top_opportunity": "biggest opportunity",')
    lines.append('  "reviewer_notes": "detailed qualitative notes"')
    lines.append('}')
    lines.append('```')
    lines.append("")
    lines.append("Evaluate each position as one of:")
    lines.append("- STRONG: good entry, good trend, solid fundamentals")
    lines.append("- HOLD: neutral, no strong reason to sell or buy more")
    lines.append("- WARNING: concerns \u2014 weakening trend, bad entry")
    lines.append("- WEAK: poor position \u2014 better to exit when opportunity arises")
    lines.append("")
    lines.append("IMPORTANT: This analysis is for INFORMATION ONLY. Do NOT execute any trades.")
    lines.append("")
    lines.append("Consider this data:")
    lines.append("")
    total_val = sum(p.get("value_eur", 0) for p in positions)
    lines.append(f"EUR Balance: \u20ac{eur_balance:.2f}")
    lines.append(f"Portfolio Value (holdings): \u20ac{total_val:.2f}")
    lines.append(f"Total (EUR + holdings): \u20ac{eur_balance + total_val:.2f}")
    lines.append(f"Number of positions: {len(positions)}")
    lines.append("")
    lines.append("=== POSITIONS ===")
    for p in positions:
        lines.append(f"  {p['symbol']}: {p['qty']:.6f} @ \u20ac{p['current_price']:.4f} = \u20ac{p['value_eur']:.2f}")
    lines.append("")
    if market_data.get("regime"):
        lines.append(f"Market Regime: {market_data['regime']}")
    if market_data.get("fear_greed"):
        fg = market_data["fear_greed"]
        lines.append(f"Fear & Greed: {fg.get('value')} ({fg.get('classification')})")
    if market_data.get("global_data"):
        g = market_data["global_data"]
        lines.append(f"Market Cap: \u20ac{g.get('total_market_cap', 0)/1e12:.2f}T  BTC Dom: {g.get('btc_dominance', 0):.1f}%  Vol: \u20ac{g.get('total_volume', 0)/1e9:.1f}B")
    if market_data.get("coin_prices"):
        lines.append("=== COIN PRICES (CoinGecko) ===")
        for sym, info in market_data["coin_prices"].items():
            chg = info.get("change_24h")
            chg_str = f"{chg:+.2f}%" if chg else "N/A"
            lines.append(f"  {sym}: \u20ac{info.get('price', 0):.4f} 24h:{chg_str} vol:\u20ac{info.get('volume', 0)/1e6:.1f}M")
    if market_data.get("self_review"):
        lines.append("=== SELF-REVIEW (Day Trader) ===")
        lines.append(market_data["self_review"])
    if market_data.get("scalper_review"):
        lines.append("=== SELF-REVIEW (Scalper) ===")
        lines.append(market_data["scalper_review"])
    if market_data.get("onchain"):
        lines.append(str(market_data["onchain"]))
    if market_data.get("defi"):
        lines.append(str(market_data["defi"]))
    if market_data.get("funding"):
        lines.append("=== FUNDING RATES (Kraken/OKX) ===")
        for s in market_data["funding"]:
            lines.append(f"  • {s}")
    if market_data.get("multi_price"):
        lines.append(str(market_data["multi_price"]))
    if market_data.get("whales"):
        lines.append("=== WHALE MOVEMENTS ===")
        for s in market_data["whales"]:
            lines.append(f"  • {s}")
    if market_data.get("news"):
        lines.append("=== CRYPTO NEWS HEADLINES ===")
        for h in market_data["news"][:5]:
            lines.append(f"  • {h}")
    if market_data.get("economic_events"):
        lines.append("=== ECONOMIC CALENDAR (this week) ===")
        for ev in market_data["economic_events"][:5]:
            if isinstance(ev, dict):
                lines.append(f"  • {ev.get('name', '?')}: {ev.get('desc', '')}")
            else:
                lines.append(f"  • {ev}")
    if market_data.get("diversification"):
        lines.append("=== PORTFOLIO DIVERSIFICATION ===")
        lines.append(str(market_data["diversification"]))
    if market_data.get("sentiment"):
        lines.append("=== NEWS SENTIMENT ===")
        lines.append(str(market_data["sentiment"]))
    if market_data.get("mempool"):
        lines.append(str(market_data["mempool"]))
    return "\n".join(lines)

def parse_verdicts(content):
    # Try ```json block
    json_match = re.search(r'```(?:json)?\s*\n?({.*?})\n?```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: find JSON object with overall_assessment
    json_match = re.search(r'({[\s\S]*"overall_assessment"[\s\S]*})', content)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: try to parse entire response as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

# ── Main review cycle ────────────────────────────────────────────────────────

def run_review():
    log.info("=" * 60)
    log.info("📋 PORTFOLIO REVIEWER STARTING")
    log.info("=" * 60)

    client = _get_revolut_client()
    if not client:
        log.warning("No Revolut client — running in market-review-only mode")

    eur_balance, holdings, positions, total_value = _collect_portfolio(client)
    log.info(f"💰 EUR: €{eur_balance:.2f} | Positions: {len(positions)} | Value: €{total_value:.2f}")
    for p in positions:
        log.info(f"   {p['symbol']}: {p['qty']:.6f} × €{p['current_price']:.4f} = €{p['value_eur']:.2f}")

    log.info("📡 Collecting market intelligence...")
    market_data = _collect_market_data(list(holdings.keys()))
    log.info(f"   Regime: {market_data.get('regime', 'N/A')}")
    fg = market_data.get("fear_greed", {})
    log.info(f"   F&G: {fg.get('value', 'N/A')} ({fg.get('classification', 'N/A')})")

    log.info("🤖 Consulting AI for portfolio review...")
    brain = ReviewerBrain()
    if not brain.api_key:
        log.error("❌ No API key — cannot analyze")
        return {"status": "error", "reason": "no_api_key"}

    prompt = build_review_prompt(positions, eur_balance, market_data)
    result = brain._call(
        "You are a crypto portfolio reviewer. Analyze data and respond only with JSON.",
        prompt,
    )

    if "error" in result:
        log.error(f"❌ AI call failed: {result['error']}")
        return {"status": "error", "reason": result["error"]}

    ai_output = result["content"]
    log.info(f"✅ AI response received ({len(ai_output)} chars)")

    verdicts = parse_verdicts(ai_output)
    if verdicts:
        log.info(f"📊 Assessment: {verdicts.get('overall_assessment', 'N/A')}")
        log.info(f"   Confidence: {verdicts.get('confidence', 'N/A')}")
        log.info(f"   Portfolio Health: {verdicts.get('portfolio_health', 'N/A')}")
        log.info(f"   Top concern: {verdicts.get('top_concern', 'N/A')}")
        log.info(f"   Top opportunity: {verdicts.get('top_opportunity', 'N/A')}")
        for pos in verdicts.get("positions", []):
            log.info(f"   {pos.get('symbol', '?'):6s}: {pos.get('verdict', '?'):8s} \u2014 {pos.get('reason', '')[:80]}")
    else:
        log.warning("⚠️ Could not parse structured verdicts from AI response")
        log.info(f"Raw response: {ai_output[:500]}")

    cal = _load_calendar()
    cal["reviewer"] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "overall_assessment": (verdicts or {}).get("overall_assessment", "UNKNOWN"),
        "confidence": (verdicts or {}).get("confidence", 0),
        "portfolio_health": (verdicts or {}).get("portfolio_health", "UNKNOWN"),
        "eur_balance": eur_balance,
        "portfolio_value": total_value,
        "total_value": round(eur_balance + total_value, 2),
        "positions": positions,
        "market_regime": str(market_data.get("regime", "")),
        "fear_greed": fg.get("value"),
        "fear_greed_label": fg.get("classification"),
        "notes": (verdicts or {}).get("reviewer_notes", ""),
        "verdicts": verdicts,
        "top_concern": (verdicts or {}).get("top_concern", ""),
        "top_opportunity": (verdicts or {}).get("top_opportunity", ""),
        "market_trend": (verdicts or {}).get("market_trend", ""),
    }

    assessment = (verdicts or {}).get("overall_assessment", "UNKNOWN")
    portfolio_health = (verdicts or {}).get("portfolio_health", "UNKNOWN")
    _add_calendar_entry(
        cal,
        "review",
        f"Portfolio Review: {assessment} | Health: {portfolio_health}",
        f"EUR: \u20ac{eur_balance:.2f} | Portfolio: \u20ac{total_value:.2f} | Total: \u20ac{eur_balance + total_value:.2f}\n"
        f"Regime: {market_data.get('regime', 'N/A')} | F&G: {fg.get('value', '?')} ({fg.get('classification', '?')})\n"
        f"Top Concern: {(verdicts or {}).get('top_concern', 'N/A')}\n"
        f"Top Opportunity: {(verdicts or {}).get('top_opportunity', 'N/A')}",
    )
    cal["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_calendar(cal)
    log.info("💾 Review written to strategy_calendar.json")

    summary = f"""📋 PORTFOLIO REVIEW
Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
EUR: \u20ac{eur_balance:.2f} | Holdings: \u20ac{total_value:.2f} | Total: \u20ac{eur_balance + total_value:.2f}
Positions: {len(positions)}
Regime: {market_data.get('regime', 'N/A')} | F&G: {fg.get('value', '?')} ({fg.get('classification', '?')})
Assessment: {assessment} (conf: {(verdicts or {}).get('confidence', 0):.0%})
Health: {portfolio_health}"""
    if verdicts:
        summary += f"\nMarket Trend: {verdicts.get('market_trend', 'N/A')}"
    log.info("\n" + summary)

    return {
        "status": "ok",
        "total_value": round(eur_balance + total_value, 2),
        "positions": len(positions),
        "assessment": assessment,
        "confidence": (verdicts or {}).get("confidence", 0),
        "portfolio_health": portfolio_health,
    }


if __name__ == "__main__":
    result = run_review()
    print(json.dumps(result, indent=2))