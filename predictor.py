#!/usr/bin/env python3
"""
Pionex Prediction Engine — multi-timeframe technical analysis for AI decisions.

Pulls historical OHLCV from CoinGecko (free, no key needed).
Calculates key levels for precise entries and exits:
  - Daily, weekly, monthly pivot points (S1-S3, R1-R3)
  - Volume Profile (POC — where most volume traded)
  - Support/Resistance from historical price action
  - Fib retracements from major swings
  - ATR for volatility-adjusted stops
  - EMA/SMA crossovers for trend detection

ALL numbers derived from real data. AI decides how to use them.
Nothing hardcoded.
"""
import json, time, ssl, urllib.request, os, sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

TRADER_DIR = "/home/ubuntu/revolut-x-trader"
os.chdir(TRADER_DIR)
sys.path.insert(0, TRADER_DIR)

ctx = ssl.create_default_context()

# Cache
CACHE_OHLCV = Path("/tmp/pionex_ohlcv_cache.json")
OHLCV_TTL = 300  # 5 min for hourly, longer for daily/weekly

def _cache_ohlcv(key: str) -> dict | None:
    try:
        raw = CACHE_OHLCV.read_text()
        cache = json.loads(raw)
    except: return None
    entry = cache.get(key)
    if entry is None or time.time() - entry["ts"] > OHLCV_TTL:
        return None
    return entry["data"]

def _cache_ohlcv_set(key: str, data):
    try:
        cache = {} if not CACHE_OHLCV.exists() else json.loads(CACHE_OHLCV.read_text())
    except: cache = {}
    cache[key] = {"ts": time.time(), "data": data}
    CACHE_OHLCV.write_text(json.dumps(cache))


def _fetch(url: str, timeout: int = 15) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


# =============================================================================
# OHLCV DATA — daily, hourly from CoinGecko
# =============================================================================

COIN_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin",
    "XRP": "ripple", "ADA": "cardano", "AVAX": "avalanche-2", "DOT": "polkadot",
    "LINK": "chainlink", "BCH": "bitcoin-cash", "HYPE": "hyperliquid",
}

def get_ohlcv(coin_id: str, days: int = 90) -> list:
    """Get OHLCV data from CoinGecko. Returns list of [timestamp_ms, open, high, low, close]."""
    cache_key = f"ohlcv_{coin_id}_{days}"
    cached = _cache_ohlcv(cache_key)
    if cached:
        return cached

    # CoinGecko free tier gives hourly for 90 days, daily for longer
    data = _fetch(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?"
        f"vs_currency=usd&days={days}"
    )
    if not data or "prices" not in data:
        return []

    prices = data.get("prices", [])  # [ts, price]
    volumes = data.get("total_volumes", [])  # [ts, vol]

    # Build OHLCV from price points (hourly granularity)
    # Group by day for daily candles, by hour for hourly
    candles = _group_into_ohlcv(prices, volumes, interval_hours=1)

    _cache_ohlcv_set(cache_key, candles)
    return candles


def _group_into_ohlcv(prices: list, volumes: list, interval_hours: int = 1) -> list:
    """Group tick-level price data into OHLCV candles."""
    if not prices:
        return []
    
    interval_ms = interval_hours * 3600 * 1000
    candles = {}
    
    for i, (ts, price) in enumerate(prices):
        bucket = ts // interval_ms * interval_ms
        vol = volumes[i][1] if i < len(volumes) and volumes[i][0] == ts else 0
        
        if bucket not in candles:
            candles[bucket] = {
                "ts": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": vol,
            }
        else:
            c = candles[bucket]
            c["high"] = max(c["high"], price)
            c["low"] = min(c["low"], price)
            c["close"] = price
            c["volume"] += vol
    
    result = []
    for ts in sorted(candles.keys()):
        c = candles[ts]
        result.append([c["ts"], c["open"], c["high"], c["low"], c["close"], c["volume"]])
    
    return result


# =============================================================================
# MULTI-TIMEFRAME LEVELS — daily, weekly, monthly pivot points
# =============================================================================

def calc_pivot_points(candles: list) -> dict:
    """
    Calculate pivot points and support/resistance levels from daily candles.
    Returns S1-S3, R1-R3, pivot, and volume POC.
    """
    if len(candles) < 2:
        return {}
    
    # Use the most recent completed day's OHLC
    # For current day: use previous day's close
    last = candles[-1]  # [ts, open, high, low, close, vol]
    prev = candles[-2] if len(candles) >= 2 else last
    
    h, l, c = prev[2], prev[3], prev[4]  # prev day high, low, close
    
    if h <= 0 or l <= 0:
        return {}
    
    pp = (h + l + c) / 3  # classic pivot

    return {
        "pivot": round(pp, 2),
        "r1": round(2 * pp - l, 2),
        "r2": round(pp + (h - l), 2),
        "r3": round(h + 2 * (pp - l), 2),
        "s1": round(2 * pp - h, 2),
        "s2": round(pp - (h - l), 2),
        "s3": round(l - 2 * (h - pp), 2),
    }


def calc_volume_poc(candles: list, num_buckets: int = 20) -> dict:
    """
    Calculate Volume Profile POC (Point of Control) and value areas.
    Splits price into buckets, finds where most volume traded.
    """
    if len(candles) < 5:
        return {}
    
    all_prices = []
    all_volumes = []
    for c in candles:
        lo, hi, vol = c[3], c[2], c[5]
        if vol > 0 and hi > 0 and lo > 0:
            # Mid-price as representation
            all_prices.append((hi + lo + c[1] + c[4]) / 4)
            all_volumes.append(vol)
    
    if not all_prices:
        return {}
    
    min_p = min(all_prices)
    max_p = max(all_prices)
    
    if min_p == max_p:
        return {"poc": round(min_p, 2)}
    
    bucket_size = (max_p - min_p) / num_buckets
    profile = defaultdict(float)
    
    for price, vol in zip(all_prices, all_volumes):
        bucket = int((price - min_p) / bucket_size)
        bucket = min(bucket, num_buckets - 1)
        bucket_price = min_p + bucket * bucket_size + bucket_size / 2
        profile[round(bucket_price, 2)] += vol
    
    if not profile:
        return {}
    
    sorted_levels = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    poc = sorted_levels[0][0]
    total_vol = sum(v for _, v in sorted_levels)
    
    # Value Area (70% of volume)
    cum = 0
    va_high = va_low = poc
    for price, vol in sorted_levels:
        cum += vol
        if cum <= total_vol * 0.70:
            va_high = max(va_high, price)
            va_low = min(va_low, price)
    
    return {
        "poc": round(poc, 2),
        "va_high": round(va_high, 2),
        "va_low": round(va_low, 2),
        "top_levels": [(p, round(v, 0)) for p, v in sorted_levels[:5]],
    }


def calc_fib_levels(candles: list, lookback_days: int = 30) -> dict:
    """Fibonacci retracement from the most recent major swing."""
    if len(candles) < lookback_days:
        return {}
    
    recent = candles[-lookback_days:]
    highs = [c[2] for c in recent]  # high
    lows = [c[3] for c in recent]   # low
    
    if not highs or not lows:
        return {}
    
    swing_high = max(highs)
    swing_low = min(lows)
    
    if swing_high <= swing_low:
        return {}
    
    diff = swing_high - swing_low
    current = recent[-1][4]  # last close
    
    # Determine if we're in an uptrend or downtrend
    midpoint = (swing_high + swing_low) / 2
    is_uptrend = current > midpoint
    
    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "current": round(current, 2),
        "trend": "up" if is_uptrend else "down",
        "fib_0": round(swing_high, 2),
        "fib_236": round(swing_high - diff * 0.236, 2),
        "fib_382": round(swing_high - diff * 0.382, 2),
        "fib_500": round(swing_high - diff * 0.5, 2),
        "fib_618": round(swing_high - diff * 0.618, 2),
        "fib_786": round(swing_high - diff * 0.786, 2),
        "fib_1": round(swing_low, 2),
    }


def calc_atr(candles: list, period: int = 14) -> float:
    """Average True Range — volatility measure for stop-loss sizing."""
    if len(candles) < period + 1:
        return 0
    
    trs = []
    for i in range(1, len(candles)):
        h, l, prev_c = candles[i][2], candles[i][3], candles[i-1][4]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    
    if not trs:
        return 0
    
    # Simple ATR (no smoothing needed for AI context)
    return round(sum(trs[-period:]) / period, 4)


def calc_moving_averages(candles: list) -> dict:
    """Key moving averages and their current relationship."""
    if len(candles) < 100:
        return {}
    
    closes = [c[4] for c in candles]
    
    def ema(data, period):
        if len(data) < period:
            return 0
        mult = 2 / (period + 1)
        e = sum(data[:period]) / period
        for v in data[period:]:
            e = (v - e) * mult + e
        return round(e, 2)
    
    def sma(data, period):
        if len(data) < period:
            return 0
        return round(sum(data[-period:]) / period, 2)
    
    current = closes[-1] if closes else 0
    
    return {
        "current": round(current, 2),
        "ema_12": ema(closes, 12),
        "ema_26": ema(closes, 26),
        "ema_50": ema(closes, 50),
        "ema_200": ema(closes, 200),
        "sma_20": sma(closes, 20),
        "sma_50": sma(closes, 50),
        "sma_200": sma(closes, 200),
        "ema_12_26_cross": (
            "bullish" if ema(closes, 12) > ema(closes, 26)
            else "bearish" if ema(closes, 12) > 0
            else "neutral"
        ),
    }


def calc_support_resistance(candles: list) -> dict:
    """Find key support and resistance levels from historical price action."""
    if len(candles) < 60:
        return {}
    
    # Method: find swing highs/lows (local extremes)
    highs = [c[2] for c in candles[-60:]]
    lows = [c[3] for c in candles[-60:]]
    
    # Cluster nearby levels
    def cluster_levels(levels, tolerance_pct=0.5):
        if not levels:
            return []
        sorted_lvls = sorted(levels, reverse=True)
        clusters = []
        current = [sorted_lvls[0]]
        for lvl in sorted_lvls[1:]:
            if abs(lvl - current[-1]) / max(lvl, 1) * 100 < tolerance_pct:
                current.append(lvl)
            else:
                clusters.append(sum(current) / len(current))
                current = [lvl]
        clusters.append(sum(current) / len(current))
        return sorted(clusters, reverse=True)
    
    resistances = cluster_levels(highs)
    supports = cluster_levels(lows)
    
    current = candles[-1][4]
    
    # Determine which levels are most relevant (near current price)
    above = [r for r in resistances if r > current * 1.001][:3]
    below = [s for s in supports if s < current * 0.999][:3]
    
    return {
        "current": round(current, 2),
        "resistances_above": [round(r, 2) for r in above],
        "supports_below": [round(s, 2) for s in below],
        "nearest_resistance": round(above[0], 2) if above else None,
        "nearest_support": round(below[0], 2) if below else None,
        "distance_to_resistance_pct": (
            round((above[0] - current) / current * 100, 2) if above else None
        ),
        "distance_to_support_pct": (
            round((current - below[0]) / current * 100, 2) if below else None
        ),
    }


# =============================================================================
# UNIFIED ANALYSIS — multi-coin, multi-timeframe
# =============================================================================

def analyze_coin(base: str, coin_id: str, days: int = 90) -> dict:
    """Full analysis for one coin — all timeframes, all indicators."""
    candles = get_ohlcv(coin_id, days)
    
    if not candles or len(candles) < 20:
        return {"symbol": base, "error": "insufficient_data"}
    
    # Separate daily and hourly candles
    daily_candles = _group_daily(candles)
    
    analysis = {
        "symbol": base,
        "coin_id": coin_id,
        "current_price": round(candles[-1][4], 4),
        "candles_available": len(candles),
        "candles_daily": len(daily_candles),
        "days_of_data": days,
    }
    
    # Pivot points (from daily)
    if len(daily_candles) >= 2:
        analysis["pivots"] = calc_pivot_points(daily_candles)
    
    # Volume profile
    analysis["volume_profile"] = calc_volume_poc(daily_candles)
    
    # Fibonacci
    if len(daily_candles) >= 10:
        analysis["fibonacci"] = calc_fib_levels(daily_candles, min(30, len(daily_candles)))
    
    # ATR (volatility)
    analysis["atr_daily"] = calc_atr(daily_candles, 14)
    analysis["atr_hourly"] = calc_atr(candles, 14) if len(candles) > 14 else 0
    
    # Moving averages
    analysis["moving_averages"] = calc_moving_averages(daily_candles)
    
    # Support/Resistance
    analysis["sr_levels"] = calc_support_resistance(daily_candles)
    
    # Weekly pivot (from last 7 daily candles)
    if len(daily_candles) >= 7:
        week_candles = daily_candles[-7:]
        analysis["weekly_pivot"] = {
            "high": max(c[2] for c in week_candles),
            "low": min(c[3] for c in week_candles),
            "close": week_candles[-1][4],
        }
    
    # Monthly pivot (from last 30 daily candles)
    if len(daily_candles) >= 20:
        month_candles = daily_candles[-20:]
        analysis["monthly_pivot"] = {
            "high": max(c[2] for c in month_candles),
            "low": min(c[3] for c in month_candles),
            "close": month_candles[-1][4],
        }
    
    return analysis


def _group_daily(candles: list) -> list:
    """Group hourly candles into daily candles."""
    daily = {}
    for c in candles:
        ts, op, hi, lo, cl, vol = c
        day = ts // 86400000 * 86400000
        if day not in daily:
            daily[day] = {"ts": day, "open": op, "high": hi, "low": lo, "close": cl, "volume": vol}
        else:
            d = daily[day]
            d["high"] = max(d["high"], hi)
            d["low"] = min(d["low"], lo)
            d["close"] = cl
            d["volume"] += vol
    
    result = []
    for ts in sorted(daily.keys()):
        d = daily[ts]
        result.append([d["ts"], d["open"], d["high"], d["low"], d["close"], d["volume"]])
    return result


# =============================================================================
# FORMAT FOR AI — clear, actionable levels
# =============================================================================

def format_analysis_for_ai(analysis: dict) -> str:
    """Format a single coin's analysis for AI consumption."""
    if "error" in analysis:
        return f"  {analysis['symbol']}: {analysis['error']}"
    
    sym = analysis["symbol"]
    price = analysis.get("current_price", 0)
    lines = [f"\n{'='*50}"]
    lines.append(f"📊 {sym} — ${price:,.2f}")
    lines.append(f"{'='*50}")
    
    # Pivot points
    pivots = analysis.get("pivots", {})
    if pivots:
        lines.append(f"  🎯 PIVOT: ${pivots.get('pivot', '?'):,.2f}")
        lines.append(f"     Resistance: R1=${pivots.get('r1', '?'):,.2f} R2=${pivots.get('r2', '?'):,.2f} R3=${pivots.get('r3', '?'):,.2f}")
        lines.append(f"     Support:    S1=${pivots.get('s1', '?'):,.2f} S2=${pivots.get('s2', '?'):,.2f} S3=${pivots.get('s3', '?'):,.2f}")
    
    # Volume POC
    vp = analysis.get("volume_profile", {})
    if vp.get("poc"):
        lines.append(f"  📈 VOLUME POC: ${vp['poc']:,.2f} (most volume traded here)")
        if vp.get("va_high") and vp.get("va_low"):
            lines.append(f"     Value Area: ${vp['va_low']:,.2f} — ${vp['va_high']:,.2f} (70% of volume)")
        if vp.get("top_levels"):
            top = [(p, int(v)) for p, v in vp["top_levels"][:3]]
            lines.append(f"     Top nodes: {', '.join(f'${p:,.0f}({v:,}vol)' for p, v in top)}")
    
    # ATR
    atr_d = analysis.get("atr_daily", 0)
    atr_h = analysis.get("atr_hourly", 0)
    if atr_d > 0:
        atr_d_pct = atr_d / price * 100 if price > 0 else 0
        lines.append(f"  📏 ATR: daily=${atr_d:,.2f} ({atr_d_pct:.2f}%) hourly=${atr_h:,.4f}")
    
    # Fibonacci
    fib = analysis.get("fibonacci", {})
    if fib.get("swing_high"):
        lines.append(f"  🔢 FIB: {fib.get('trend','?')}trend from ${fib['swing_low']:,.2f} → ${fib['swing_high']:,.2f}")
        for level in ["382", "500", "618", "786"]:
            p = fib.get(f"fib_{level}")
            if p:
                dist = abs(p - price) / price * 100 if price > 0 else 0
                marker = " ⬅️ HERE" if dist < 0.5 else ""
                lines.append(f"     {level}: ${p:,.2f} ({dist:.1f}% away){marker}")
    
    # Moving averages
    ma = analysis.get("moving_averages", {})
    if ma.get("current"):
        lines.append(f"  📉 MOVING AVGs (daily):")
        current = ma["current"]
        for key in ["ema_12", "ema_26", "ema_50", "sma_20", "sma_50", "sma_200"]:
            val = ma.get(key, 0)
            if val > 0:
                dist = (current - val) / val * 100
                above = "above" if current > val else "below"
                lines.append(f"     {key}: ${val:,.2f} (price {abs(dist):.1f}% {above})")
        lines.append(f"     CROSS: {ma.get('ema_12_26_cross', '?')}")
    
    # Support/Resistance
    sr = analysis.get("sr_levels", {})
    if sr.get("nearest_resistance"):
        lines.append(f"  🏔️ KEY LEVELS:")
        lines.append(f"     Nearest resistance: ${sr['nearest_resistance']:,.2f} ({sr.get('distance_to_resistance_pct', '?')}% away)")
        lines.append(f"     Nearest support:    ${sr['nearest_support']:,.2f} ({sr.get('distance_to_support_pct', '?')}% away)")
        if sr.get("resistances_above"):
            lines.append(f"     Resistances: {', '.join(f'${r:,.2f}' for r in sr['resistances_above'])}")
        if sr.get("supports_below"):
            lines.append(f"     Supports:    {', '.join(f'${s:,.2f}' for s in sr['supports_below'])}")
    
    # Multi-timeframe pivots
    wp = analysis.get("weekly_pivot", {})
    if wp:
        lines.append(f"  📅 WEEKLY RANGE: H=${wp['high']:,.2f} L=${wp['low']:,.2f} C=${wp['close']:,.2f}")
    
    mp = analysis.get("monthly_pivot", {})
    if mp:
        lines.append(f"  📆 MONTHLY RANGE: H=${mp['high']:,.2f} L=${mp['low']:,.2f} C=${mp['close']:,.2f}")
    
    return "\n".join(lines)


def analyze_all(symbols: list = None, days: int = 90) -> str:
    """Run full analysis for all major coins. Returns formatted AI context."""
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA"]
    
    lines = ["📊 PREDICTION ANALYSIS — multi-timeframe levels"]
    lines.append("=" * 40)
    lines.append(f"Data: {days} days of OHLCV (CoinGecko) | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    
    for sym in symbols:
        coin_id = COIN_IDS.get(sym, sym.lower())
        try:
            analysis = analyze_coin(sym, coin_id, days)
            lines.append(format_analysis_for_ai(analysis))
        except Exception as e:
            lines.append(f"  {sym}: ERROR {e}")
    
    lines.append("\n⚡ AI TRADING DECISION:")
    lines.append("Use these levels to determine exact entry and exit points.")
    lines.append("Buy near supports, sell near resistances.")
    lines.append("ATR = optimal stop distance. POC = where market agrees on value.")
    lines.append("Cross above/below MAs = trend confirmation.")
    
    return "\n".join(lines)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC", help="Coin to analyze")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--all", action="store_true", help="Analyze all major coins")
    args = parser.parse_args()
    
    if args.all:
        print(analyze_all(["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA"], args.days))
    else:
        analysis = analyze_coin(args.symbol, COIN_IDS.get(args.symbol, args.symbol.lower()), args.days)
        print(format_analysis_for_ai(analysis))
