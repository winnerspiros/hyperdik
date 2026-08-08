"""
Funding Rate + Open Interest signal — free public API data for sentiment
Uses Kraken Futures, OKX, and CoinGecko (multiple sources since Binance and Bybit are blocked).
Extremely negative funding = everyone short = buy signal
Extremely positive funding = everyone long = sell signal
"""
import urllib.request, ssl, json, time

ctx = ssl.create_default_context()

def _fetch(url, timeout=10, data=None):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if data is not None:
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, data=data.encode(), context=ctx, timeout=timeout) as resp:
                return json.loads(resp.read())
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return json.loads(resp.read())
    except:
        return None

def get_funding_data():
    """Get funding rates and OI data from public APIs"""
    signals = []
    
    # 1. Kraken Futures BTC funding (works from this server)
    try:
        data = _fetch("https://futures.kraken.com/derivatives/api/v4/tickers")
        if data and "tickers" in data:
            for t in data["tickers"]:
                sym = t.get("symbol", "")
                if sym == "PF_XBTUSD":
                    rate = float(t.get("fundingRate", 0)) * 100
                    signals.append(f"BTC funding (Kraken): {rate:.4f}%")
                    if rate < -0.05:
                        signals.append("⚠️ BTC funding negative — shorts paying")
                    elif rate > 0.05:
                        signals.append("⚠️ BTC funding positive — longs paying")
                    break
    except: pass
    
    # 2. OKX funding rate (also works)
    try:
        data = _fetch("https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USD-SWAP")
        if data and "data" in data and len(data["data"]) > 0:
            rate = float(data["data"][0].get("fundingRate", 0)) * 100
            signals.append(f"BTC funding (OKX): {rate:.4f}%")
    except: pass
    
    # 3. OKX ETH funding
    try:
        data = _fetch("https://www.okx.com/api/v5/public/funding-rate?instId=ETH-USD-SWAP")
        if data and "data" in data and len(data["data"]) > 0:
            rate = float(data["data"][0].get("fundingRate", 0)) * 100
            signals.append(f"ETH funding (OKX): {rate:.4f}%")
    except: pass
    
    # 4. BTC Open Interest from OKX
    try:
        data = _fetch("https://www.okx.com/api/v5/public/open-interest?instId=BTC-USD-SWAP")
        if data and "data" in data and len(data["data"]) > 0:
            oi = data["data"][0].get("oi", "?")
            signals.append(f"BTC OI (OKX): {oi}")
    except: pass
    
    return signals if signals else ["Funding data: rates unavailable at this time"]

def get_multi_timeframe_prices(symbols):
    """
    Get OHLC data for given symbols from Kraken (multi-timeframe).
    Returns dict: {symbol: {daily: {open, high, low, close, volume}, weekly: {...}}}
    Kraken pair format: XXBTZUSD for BTC/USD
    """
    KRAKEN_PAIRS = {
        "BTC": "XXBTZUSD", "ETH": "ETHXUSD", "XRP": "XXRPZUSD",
        "SOL": "SOLXUSD", "ADA": "ADAXUSD", "DOT": "DOTXUSD",
        "LINK": "LINKXUSD", "DOGE": "XDGXUSD",
    }
    results = {}
    for sym in symbols:
        pair = KRAKEN_PAIRS.get(sym.upper())
        if not pair:
            continue
        try:
            # Daily OHLC (interval=1440)
            data = _fetch(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440")
            if data and "result" in data:
                result = data["result"]
                ohlc_key = [k for k in result.keys() if k != "last"][0]
                candles = result[ohlc_key]
                if candles and len(candles) >= 2:
                    today = candles[-1]
                    yesterday = candles[-2]
                    results[f"{sym}_daily"] = {
                        "open": float(today[1]), "high": float(today[2]),
                        "low": float(today[3]), "close": float(today[4]),
                        "volume": float(today[6]),
                        "prev_close": float(yesterday[4]),
                    }
            # Weekly (interval=10080)
            data_w = _fetch(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=10080")
            if data_w and "result" in data_w:
                result = data_w["result"]
                ohlc_key = [k for k in result.keys() if k != "last"][0]
                candles = result[ohlc_key]
                if candles and len(candles) >= 2:
                    week = candles[-1]
                    results[f"{sym}_weekly"] = {
                        "open": float(week[1]), "high": float(week[2]),
                        "low": float(week[3]), "close": float(week[4]),
                        "volume": float(week[6]),
                    }
        except:
            pass
    return results

def format_multi_price_for_ai(symbols):
    """Format multi-timeframe price context for AI."""
    prices = get_multi_timeframe_prices(symbols)
    if not prices:
        return ""
    lines = ["📊 MULTI-TIMEFRAME PRICES (Kraken):"]
    for sym in ["BTC", "ETH", "XRP", "SOL"]:
        daily = prices.get(f"{sym}_daily")
        weekly = prices.get(f"{sym}_weekly")
        if daily:
            chg = ((daily["close"] - daily["prev_close"]) / daily["prev_close"] * 100) if daily.get("prev_close") else 0
            day_range = daily["high"] - daily["low"]
            lines.append(f"  {sym}: day O={daily['open']:.2f} H={daily['high']:.2f} L={daily['low']:.2f} C={daily['close']:.2f} ({chg:+.1f}%) vol={daily['volume']:.1f}")
            if weekly:
                wk_range = weekly["high"] - weekly["low"]
                lines.append(f"       week O={weekly['open']:.2f} H={weekly['high']:.2f} L={weekly['low']:.2f} C={weekly['close']:.2f}")
    return "\n".join(lines)

if __name__ == "__main__":
    print("=== FUNDING ===")
    for s in get_funding_data():
        print(f"  • {s}")
    print()
    print("=== MULTI-PRICE ===")
    print(format_multi_price_for_ai(["BTC", "ETH", "XRP", "SOL"]))