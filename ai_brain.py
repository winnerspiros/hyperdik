"""
AI Trading Brain — uses LLM to analyze markets and make intelligent trading decisions
Calling OpenRouter API for real AI reasoning every cycle
Hybrid model: cheap flash model for daily decisions, premium model for trade execution
"""
import json
import re
import time
import os
import urllib.request
import ssl
from datetime import datetime

OPENROUTER_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"  # SURVIVAL: Qwen 30B — $0.048/$0.19, correct JSON
PREMIUM_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
MAX_TOKENS_PER_CALL = 800

class AITradingBrain:
    """The actual intelligence behind the trading. Uses LLM to reason about markets."""
    
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
                                # Skip commented lines
                                if line.strip().startswith("#"):
                                    continue
                                raw = line.split("=", 1)[1].strip()
                                self.api_key = raw.strip("'\\\"").strip()
                                if self.api_key:
                                    break
                except:
                    pass
                if self.api_key:
                    break
        self.conversation_log = []
        self.total_calls = 0
        self.total_cost = 0.0
        self.last_analysis = ""

    def _call_llm(self, system_prompt, user_prompt, temperature=0.7, rich_context="", model=None):
        """Call OpenRouter API with the given prompts"""
        if not self.api_key:
            return {"error": "no_api_key"}
        use_model = model or OPENROUTER_MODEL
        if rich_context:
            system_prompt = (
                system_prompt.strip()
                + "\n\n─── COMPREHENSIVE MARKET INTELLIGENCE ───\n"
                "Below is comprehensive market intelligence including technical patterns, "
                "market trends, volatility data, and more. Use ALL of this information "
                "to make your trading decision.\n"
                + rich_context
            )
        ctx = ssl.create_default_context()
        data = json.dumps({
            "model": use_model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "max_tokens": MAX_TOKENS_PER_CALL,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Cache": "true",
            },
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
                self.total_calls += 1
                self.total_cost += 0.0005
                return {"content": content}
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
        except Exception as e:
            return {"error": str(e)}

    def _predict_price_direction(self, market_data):
        dip_mode = market_data.get("dip_analysis", False)
        if dip_mode:
            system_prompt = """You are a dip analyst. A coin has crashed 12%+ in 24h. Decide if this is:
- BULLISH: The dip is a buying opportunity (oversold, bouncing off support, good news)
- BEARISH: The dip will continue (breaking support, bad news, volume dumping)

Be conservative. False bottoms are expensive. Only approve if you're confident.

OUTPUT:
DIRECTION: [bullish/bearish/neutral]
REASON: 1-2 sentences why
"""
        else:
            system_prompt = """You are a crypto market predictor. Based on current data, predict what will happen in the next 1-2 hours for the top EUR pairs.

OUTPUT FORMAT:
DIRECTION: [bullish/bearish/neutral]
PREDICTION: 1-2 sentence price prediction
KEY_LEVELS: support and resistance levels to watch
"""
        tickers = market_data.get("tickers", [])
        ticker_summary = ""
        for t in tickers[:5]:
            sym = t.get("symbol", "?")
            bid = t.get("bid", "?")
            ask = t.get("ask", "?")
            chg = t.get("change_24h", "?")
            ticker_summary += f"{sym}: bid={bid}, ask={ask}, 24h={chg}%\n"
        user_prompt = f"""Current time: {datetime.now().strftime('%H:%M UTC')}
Fear & Greed: {market_data.get('fear_greed', {}).get('value', '?')}
EUR balance: €{market_data.get('eur_balance', 0)}
Holdings: {market_data.get('holdings', {})}

Tickers:
{ticker_summary}

Support levels: {market_data.get('support_levels', {})}

Predict the short-term direction (1-2 hours). Be specific about prices."""
        result = self._call_llm(system_prompt, user_prompt, temperature=0.4)
        if "error" in result:
            return {"direction": "unknown", "reason": result["error"]}
        content = result["content"]
        content_upper = content.upper()
        if "BULLISH" in content_upper:
            direction = "bullish"
        elif "BEARISH" in content_upper:
            direction = "bearish"
        else:
            direction = "neutral"
        return {"direction": direction, "raw": content}

    def _parse_decision(self, content):
        if not content:
            return {"direction": "wait", "reason": "empty"}
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            # Fix: some models return Python dicts with single quotes
            if "'" in cleaned and '"' not in cleaned[:10]:
                cleaned = cleaned.replace("'", '"')
            import json as _json
            data = _json.loads(cleaned)
            # Accept both compact (d,sym,sa,sf,sz,ep,tp,sl,rf,tr) and verbose keys
            direction = (data.get("d") or data.get("direction") or "wait").lower()
            if direction == "sell":
                symbol = data.get("sym") or data.get("symbol", "BTC-EUR")
                if "-" not in symbol: symbol = f"{symbol}-EUR"
                symbol = symbol.replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
                return {
                    "direction": "sell", "symbol": symbol,
                    "sell_all": data.get("sa", data.get("sell_all", True)),
                    "sell_fraction": data.get("sf", data.get("sell_fraction")),
                    "reason": data.get("rf", data.get("reason", "")),
                    "condition": str(data.get("tr", ""))[:60],
                }
            elif direction == "buy":
                symbol = data.get("sym") or data.get("symbol", "BTC-EUR")
                if "-" not in symbol: symbol = f"{symbol}-EUR"
                symbol = symbol.replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
                return {
                    "direction": "buy", "symbol": symbol,
                    "entry": data.get("ep", data.get("entry")),
                    "target": data.get("tp", data.get("target")),
                    "stop": data.get("sl", data.get("stop")),
                    "size": str(data.get("sz", data.get("size", ""))),
                    "conf": int(data.get("conf", 50)),
                    "hold": int(data.get("hold", 0)),
                    "reason": data.get("rf", data.get("reason", "")),
                    "condition": str(data.get("tr", ""))[:60],
                }
            else:
                tr_val = str(data.get("tr", data.get("condition", "")))
                return {
                    "direction": "wait",
                    "reason": str(data.get("rf", data.get("reason", ""))),
                    "condition": tr_val[:60],  # Truncate long tr fields
                }
        except (json.JSONDecodeError, AttributeError):
            pass
        # Essay fallback: AI sometimes returns prose despite JSON-only instruction
        if len(content) > 200 and '{' not in content[:50]:
            return {"direction": "wait", "reason": "ai_essay", "condition": content[:80]}
        # Fallback: old text format
        content_upper = content.upper()
        lines = content.split("\n")
        if "DECISION: BUY" in content_upper:
            symbol = "UNKNOWN"
            for line in lines:
                if "DECISION:" in line.upper() and "BUY" in line.upper():
                    parts = line.upper().split("BUY")
                    if len(parts) > 1 and parts[1].strip():
                        symbol = parts[1].strip().split()[0]
            entry = self._extract_value(content, "ENTRY", lines)
            target = self._extract_value(content, "TARGET", lines)
            stop = self._extract_value(content, "STOP", lines)
            size = self._extract_value(content, "SIZE", lines)
            reason = self._extract_text(content, "REASON", lines)
            if "-" not in symbol:
                symbol = f"{symbol}-EUR"
            symbol = symbol.replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
            return {"direction": "buy", "symbol": symbol, "entry": entry, "target": target, "stop": stop, "size": size, "reason": reason}
        elif "DECISION: SELL" in content_upper:
            symbol = "UNKNOWN"
            for line in lines:
                if "DECISION:" in line.upper() and "SELL" in line.upper():
                    parts = line.upper().split("SELL")
                    if len(parts) > 1 and parts[1].strip():
                        symbol = parts[1].strip().split()[0]
            reason = self._extract_text(content, "REASON", lines)
            size_raw = self._extract_value(content, "SIZE", lines)
            sell_all = True
            sell_fraction = None
            if size_raw:
                lower = size_raw.lower().strip()
                if lower in ("all", ""):
                    sell_all = True
                elif "%" in lower:
                    try:
                        pct = float(lower.replace("%", ""))
                        sell_fraction = pct / 100.0
                        sell_all = False
                    except: pass
                else:
                    try:
                        sell_fraction = float(lower)
                        if sell_fraction >= 1:  # all = 1.0 or more
                            sell_all = True
                            sell_fraction = None
                        else:
                            sell_all = False
                    except: pass
            if "-" not in symbol:
                symbol = f"{symbol}-EUR"
            symbol = symbol.replace("-EUR-EUR", "-EUR").replace("/EUR-EUR", "-EUR")
            return {"direction": "sell", "symbol": symbol, "reason": reason, "sell_all": sell_all, "sell_fraction": sell_fraction}
        else:
            reason = self._extract_text(content, "REASON", lines) if lines else "No reason given"
            condition = self._extract_text(content, "CONDITION", lines) if lines else ""
            return {"direction": "wait", "reason": reason, "condition": condition}

    def _extract_value(self, content, key, lines):
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith(key.upper() + ":"):
                return stripped.split(":", 1)[1].strip()
        return ""

    def _extract_text(self, content, key, lines):
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.upper().startswith(key.upper() + ":"):
                parts = stripped.split(":", 1)
                text = parts[1].strip() if len(parts) > 1 else ""
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    if next_line and next_line[0].isupper() and ":" in next_line and len(next_line.split(":")[0].split()) <= 3:
                        break
                    if next_line and next_line not in text:
                        text += " " + next_line
                return text.strip()
        return ""

    def analyze_market_decision(self, market_data, portfolio, trade_history, rich_context=""):
        """The core AI reasoning function. Two-stage: cheap model → premium model if trade."""
        system_prompt = """JSON-ONLY OUTPUT. No text before/after. No markdown. No essays.
Crypto day trader. Trade even in bad conditions. Small size > no trade.
{"d":"buy|sell|wait","sym":"SYMBOL-EUR","sz":EUR_AMOUNT,"conf":0-100,"rf":["flags"],"tr":"short_code"}
sz = amount in EUR like 5 or 10 (NOT a fraction like 0.01). tr = 1-3 word code.
Wait ONLY if no coin is near support with potential bounce.
Weekend: sz=3-8. Fear: tighter stops. Buy small at supports never freeze.
Pairs: BTC,ETH,SOL,XRP,ADA,DOT,LINK,AVAX,DOGE,APT,ATOM,ARB,OP,INJ,SUI,LTC,FIL,UNI"""

        holdings = portfolio.get("holdings", {})
        eur = portfolio.get("eur_balance", 0)
        tickers = market_data.get("tickers", [])
        support_levels = market_data.get("support_levels", {})
        ticker_summary = ""
        for t in tickers[:8]:
            sym = t.get("symbol", "?")
            bid = t.get("bid", "?")
            ask = t.get("ask", "?")
            chg = t.get("change_24h", "?")
            vol = t.get("volume_24h", "?")
            ticker_summary += f"{sym}: bid={bid}, ask={ask}, 24h={chg}%, vol={vol}\n"
        support_summary = ""
        for sym, lv in list(support_levels.items())[:3]:
            support_summary += f"{sym}: support at {lv.get('support', '?')}, {lv.get('support_distance_pct', '?')}% away\n"
        trade_summary = ""
        if trade_history:
            recent = trade_history[-5:]
            for t in recent:
                trade_summary += f"{t.get('time','?')[:16]} {t.get('side','?')} {t.get('symbol','?')} €{t.get('size','?')} @ {t.get('price','?')}\n"
        pnl_summary = market_data.get("pnl_summary", {})
        pnl_str = ""
        for coin, info in pnl_summary.items():
            if isinstance(info, (int, float)):
                pnl_str += f"  {coin}: €{info:.2f} (raw PnL, no breakdown)\n"
            else:
                try:
                    pnl_str += f"  {coin}: entry=€{info['entry']:.4f}, now=€{info['current']:.4f}, PnL={info['pnl_pct']:+.1f}%\n"
                except (TypeError, KeyError):
                    pnl_str += f"  {coin}: {info} (unexpected format)\n"
        news = market_data.get("news", [])
        news_str = "\n".join(f"  • {h[:100]}" for h in news[:3]) if news else "  No recent news"
        fear_greed = market_data.get("fear_greed", {})
        if isinstance(fear_greed, (int, float)):
            fg_value = str(fear_greed)
            fg_class = "raw"
        else:
            fg_value = fear_greed.get("value", "?")
            fg_class = fear_greed.get("classification", "?")
        super_dips = market_data.get("super_dips", [])
        dip_str = ""
        for d in super_dips[:3]:
            dip_str += f"  {d['symbol']}: -{d['drop']:.0f}% in 24h, near support\n"
        is_weekend = datetime.now().weekday() >= 5
        best_perf = ""
        worst_perf = ""
        for t in sorted(tickers[:20], key=lambda x: float(str(x.get("change_24h", 0)).replace("%","") or 0), reverse=True):
            if not best_perf: best_perf = f"{t.get('symbol','')}={t.get('change_24h','?')}%"
        for t in sorted(tickers[:20], key=lambda x: float(str(x.get("change_24h", 0)).replace("%","") or 0)):
            if not worst_perf: worst_perf = f"{t.get('symbol','')}={t.get('change_24h','?')}%"
        user_prompt = (
            f"T:{datetime.now().strftime('%H:%M')} {'WEEKEND' if is_weekend else 'WEEKDAY'} "
            f"EUR={eur:.2f} FG={fg_value}({fg_class}) H={holdings} {market_data.get('regime_summary','')}\n"
            f"PnL:{pnl_str.replace(chr(10),' ') if pnl_str else 'none'}\n"
            f"TKR:{ticker_summary.replace(chr(10),' ')}\n"
            f"BEST:{best_perf} WORST:{worst_perf}\n"
            f"SUP:{support_summary.replace(chr(10),' ')}\n"
            + (f"NEWS:{news_str[:120]}\n" if news_str and 'No recent' not in news_str else "")
            + (f"DIP:{dip_str.replace(chr(10),' ')}\n" if dip_str else "")
            + f"TRD:{trade_summary.replace(chr(10),' ') if trade_summary else 'none'}"
        )
        # STAGE 1: Quick analysis with cheap model (Gemini Flash Lite)
        result = self._call_llm(system_prompt, user_prompt, temperature=0.8, rich_context=rich_context)
        if "error" in result:
            return {"direction": "wait", "reason": result["error"]}
        content = result["content"]
        self.last_analysis = content
        decision = self._parse_decision(content)
        decision["raw_analysis"] = content
        # STAGE 2: Skip premium model (DeepSeek V4 Pro broken per user — Gemini decision is authoritative)
        # Premium refinement was adding latency without improving decisions
        if decision.get("direction") in ("buy", "sell"):
            decision["premium_skipped"] = True  # DeepSeek V4 Pro broken

        decision["cost"] = self.total_cost
        decision["calls_made"] = self.total_calls
        return decision

    def analyze_position_reduction(self, context):
        """AI decides which position to sell when overleveraged — no forced sells"""
        holdings = context.get("holdings", {})
        current_prices = context.get("current_prices", {})
        pnl_summary = context.get("pnl_summary", {})
        eur = context.get("eur_balance", 0)

        pnl_str = ""
        for coin, info in pnl_summary.items():
            pnl_str += f"  {coin}: {info.get('pnl_pct', 0):+.1f}%\n"

        prompt = f"""You are a portfolio manager. We have {len(holdings)} positions and €{eur:.2f} EUR cash.

POSITIONS:
{pnl_str if pnl_str else 'No data'}

DECIDE: Which position to sell to free up capital? Consider:
- Worst performer (biggest loss)
- Smallest position (least impact)
- Most correlated (reduces risk)

OUTPUT:
SYMBOL: [coin to sell]
REASON: why this one"""
        result = self._call_llm(prompt, "", temperature=0.3)
        if "error" in result:
            return {"direction": "sell", "symbol": list(holdings.keys())[0] if holdings else None, "reason": "error"}
        content = result["content"]
        symbol = ""
        for line in content.split("\n"):
            if line.upper().startswith("SYMBOL:"):
                symbol = line.split(":", 1)[1].strip().split()[0].upper()
        reason = ""
        for line in content.split("\n"):
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        if not symbol and holdings:
            symbol = min(holdings, key=holdings.get)
        return {"direction": "sell", "symbol": f"{symbol}-EUR" if "-" not in symbol else symbol, "reason": reason or "AI recommendation"}

if __name__ == "__main__":
    brain = AITradingBrain()
    print(f"Model: {OPENROUTER_MODEL} (standard) + {PREMIUM_MODEL} (premium)")
    print(f"API key: {'✅' if brain.api_key else '❌'}")