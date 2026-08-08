#!/usr/bin/env python3
"""
OKX Futures Client — leveraged buy/sell, cross/isolated margin.
API: https://www.okx.com/api/v5/
Auth: HMAC-SHA256 (timestamp + method + requestPath + body)

Creates a clean interface mirroring kraken_client.py so the AI brain
can plug into OKX with minimal changes.
"""
import json, time, hashlib, hmac, base64, os, urllib.request, urllib.parse, ssl
from pathlib import Path

BASE = "https://www.okx.com"
ctx = ssl.create_default_context()

# ── Credentials ──────────────────────────────────────────────────────────────

def _load_creds():
    """Load from env or ~/.okx/creds.json: {key, secret, passphrase}"""
    key = os.environ.get("OKX_API_KEY", "")
    secret = os.environ.get("OKX_SECRET_KEY", "")
    passphrase = os.environ.get("OKX_PASSPHRASE", "")
    if not key:
        try:
            creds = json.loads(Path("/home/ubuntu/.okx/creds.json").read_text())
            key = creds.get("key", "")
            secret = creds.get("secret", "")
            passphrase = creds.get("passphrase", "")
        except:
            pass
    return key, secret, passphrase


# ── Signing ──────────────────────────────────────────────────────────────────

def _sign(timestamp: str, method: str, path: str, body: str = "") -> str:
    """OKX V5 signing: Base64(HMAC-SHA256(timestamp + method + path + body))"""
    _, secret, _ = _load_creds()
    sign_str = timestamp + method.upper() + path + body
    mac = hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()
    return base64.b64encode(mac).decode()


def _public_request(path: str, params: dict = None) -> dict:
    """Send unauthenticated request (public endpoints)."""
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:500]}


def _request(method: str, path: str, body: dict = None) -> dict:
    """Send authenticated request to OKX."""
    key, secret, passphrase = _load_creds()
    if not key or not secret or not passphrase:
        return {"error": "no_credentials", "msg": "Set OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE or ~/.okx/creds.json"}

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int(time.time() * 1000) % 1000:03d}Z"
    body_str = json.dumps(body, separators=(",", ":")) if body else ""

    signature = _sign(timestamp, method, path, body_str)

    headers = {
        "OK-ACCESS-KEY": key,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    url = f"{BASE}{path}"
    data = body_str.encode() if body_str else None

    try:
        if method == "GET" and body:
            # GET with body → append as query
            qs = urllib.parse.urlencode(body)
            url += "?" + qs
            data = None
            req = urllib.request.Request(url, headers=headers)
        else:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)

        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:500]}


# ── ACCOUNT ──────────────────────────────────────────────────────────────────

def get_balance() -> dict:
    """Get futures account balance."""
    return _request("GET", "/api/v5/account/balance")

def get_positions(symbol: str = None) -> dict:
    """Get open positions. symbol format: 'BTC-USDT-SWAP'"""
    params = {"instType": "SWAP"}
    if symbol:
        params["instId"] = symbol
    return _request("GET", "/api/v5/account/positions", params)

def get_account_config() -> dict:
    """Get account configuration (position mode, leverage, etc.)."""
    return _request("GET", "/api/v5/account/config")

def set_position_mode(mode: str = "long_short_mode") -> dict:
    """
    Set position mode.
    mode: 'long_short_mode' (can hold both) or 'net_mode' (one direction only)
    """
    return _request("POST", "/api/v5/account/set-position-mode", {"posMode": mode})

def set_leverage(symbol: str, leverage: int, margin_mode: str = "cross") -> dict:
    """
    Set leverage for a symbol.
    symbol: 'BTC-USDT-SWAP'
    leverage: 1-100
    margin_mode: 'cross' or 'isolated'
    """
    return _request("POST", "/api/v5/account/set-leverage", {
        "instId": symbol,
        "lever": str(leverage),
        "mgnMode": margin_mode,
    })


# ── ORDERS ───────────────────────────────────────────────────────────────────

def place_order(symbol: str, side: str, size: str,
                order_type: str = "market", price: str = None,
                reduce_only: bool = False) -> dict:
    """
    Place a futures order.
    symbol: 'BTC-USDT-SWAP'
    side: 'buy' or 'sell'
    type: 'market' (market order), 'limit', 'post_only'
    size: number of contracts (e.g. '1' = 0.01 BTC for BTC-USDT-SWAP)
          For market buy in USDT terms, use 'sz' and 'tdMode': 'cash'
    price: limit price (ignored for market orders)
    reduce_only: close position only (True = no new position)
    """
    body = {
        "instId": symbol,
        "tdMode": "cross",   # cross margin
        "side": side.lower(),
        "ordType": order_type,
        "sz": str(size),
    }
    if price and order_type != "market":
        body["px"] = str(price)
    if reduce_only:
        body["reduceOnly"] = "true"

    return _request("POST", "/api/v5/trade/order", body)


def cancel_order(symbol: str, order_id: str = None, cl_ord_id: str = None) -> dict:
    """Cancel an order. Provide order_id or cl_ord_id."""
    body = {"instId": symbol}
    if order_id:
        body["ordId"] = order_id
    if cl_ord_id:
        body["clOrdId"] = cl_ord_id
    return _request("POST", "/api/v5/trade/cancel-order", body)


def cancel_all_orders(symbol: str) -> dict:
    """Cancel all open orders on a symbol."""
    return _request("POST", "/api/v5/trade/cancel-order",
                    {"instId": symbol, "ordType": "limit"})

def get_open_orders(symbol: str = None) -> dict:
    """Get open orders. symbol: 'BTC-USDT-SWAP' (optional)."""
    params = {"instType": "SWAP"}
    if symbol:
        params["instId"] = symbol
    return _request("GET", "/api/v5/trade/orders-pending", params)


# ── CONVENIENCE ──────────────────────────────────────────────────────────────

OKX_SYMBOL_MAP = {
    "BTC": "BTC-USDT-SWAP",  "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",  "DOGE": "DOGE-USDT-SWAP",
    "XRP": "XRP-USDT-SWAP",  "AVAX": "AVAX-USDT-SWAP",
    "ADA": "ADA-USDT-SWAP",  "LINK": "LINK-USDT-SWAP",
    "DOT": "DOT-USDT-SWAP",  "SUI": "SUI-USDT-SWAP",
    "BNB": "BNB-USDT-SWAP",  "APT": "APT-USDT-SWAP",
    "ARB": "ARB-USDT-SWAP",  "ICP": "ICP-USDT-SWAP",
    "PEPE": "PEPE-USDT-SWAP", "BONK": "BONK-USDT-SWAP",
    "WIF": "WIF-USDT-SWAP",  "RENDER": "RENDER-USDT-SWAP",
}

CT_VAL = {
    # contract value per 1 sz unit — used to compute actual size
    "BTC": 0.01, "ETH": 0.1, "SOL": 1, "DOGE": 1000, "XRP": 100,
    "AVAX": 1, "ADA": 100, "LINK": 1, "DOT": 1, "SUI": 1,
}


def okx_symbol(base: str) -> str:
    """Convert base ticker to OKX symbol. e.g. 'BTC' → 'BTC-USDT-SWAP'"""
    return OKX_SYMBOL_MAP.get(base.upper(), f"{base.upper()}-USDT-SWAP")


def buy(symbol: str, size: str, order_type: str = "market",
        price: str = None) -> dict:
    """Buy (go long)."""
    return place_order(okx_symbol(symbol), "buy", size, order_type, price)


def sell(symbol: str, size: str, order_type: str = "market",
         price: str = None) -> dict:
    """Sell (go short or close long)."""
    return place_order(okx_symbol(symbol), "sell", size, order_type, price)


def close_position(symbol: str) -> dict:
    """Close entire position for a symbol (market order)."""
    pos = get_positions(okx_symbol(symbol))
    positions = pos.get("data", [])
    for p in positions:
        if p.get("instId") == okx_symbol(symbol):
            pos_size = abs(float(p.get("pos", 0)))
            if pos_size > 0:
                side = "sell" if float(p.get("pos", 0)) > 0 else "buy"
                return place_order(okx_symbol(symbol), side, str(pos_size),
                                   reduce_only=True)
    return {"error": "no_position"}


# ── FORMATTING ──────────────────────────────────────────────────────────────

def format_balance(data: dict) -> str:
    """Human-readable balance from API response."""
    if data.get("code") != "0":
        return f"Error: {data}"
    lines = ["🟢 OKX FUTURES:"]
    for detail in data.get("data", []):
        for bal in detail.get("details", []):
            eq = float(bal.get("eq", 0))
            avail = float(bal.get("availBal", 0))
            upl = float(bal.get("upl", 0))
            if eq > 0.01:
                lines.append(f"  {bal['ccy']}: Equity=${eq:,.2f} Avail=${avail:,.2f} UPL=${upl:+.2f}")
    return "\n".join(lines) if len(lines) > 1 else "🟢 OKX: No balance"


def format_positions(data: dict) -> str:
    """Human-readable positions."""
    if data.get("code") != "0":
        return f"Error: {data}"
    positions = data.get("data", [])
    if not positions:
        return "  No open positions"
    lines = ["  POSITIONS:"]
    for p in positions:
        inst_id = p.get("instId", "?")
        pos_size = float(p.get("pos", 0))
        avg_px = float(p.get("avgPx", 0))
        mark = float(p.get("markPx", 0))
        upl = float(p.get("upl", 0))
        lever = p.get("lever", "?")
        side = "LONG" if pos_size > 0 else "SHORT"
        lines.append(
            f"  {inst_id:20s} {side:5s} x{lever} | "
            f"Size:{abs(pos_size)}  Avg:${avg_px:,.2f}  Mark:${mark:,.2f}  "
            f"PnL:{upl:+,.2f}"
        )
    return "\n".join(lines)


def is_configured() -> bool:
    """Check if API credentials are available."""
    key, secret, passphrase = _load_creds()
    return bool(key and secret and passphrase)


# ── SELF-TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== OKX CLIENT SELF-TEST ===\n")

    # Public data always works
    ticker = _public_request("/api/v5/market/ticker", {"instId": "BTC-USDT-SWAP"})
    if ticker.get("code") == "0":
        t = ticker["data"][0]
        print(f"BTC-USDT-SWAP: Last=${t['last']} Bid=${t['bidPx']} Ask=${t['askPx']} 24hVol={t['volCcy24h']}")

    funding = _public_request("/api/v5/public/funding-rate", {"instId": "BTC-USDT-SWAP"})
    if funding.get("code") == "0":
        fr = funding["data"][0]
        print(f"Funding rate: {fr['fundingRate']}  Next: {fr.get('nextFundingTime','?')}")

    if not is_configured():
        print("\n🔐 AUTH NOT CONFIGURED")
        print("Create ~/.okx/creds.json: {\"key\":\"...\",\"secret\":\"...\",\"passphrase\":\"...\"}")
        print("Or set env: OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE")
        print("\nPublic endpoints work. Credentials needed for trading.")
        sys.exit(0)

    # Balance
    bal = get_balance()
    print(format_balance(bal))

    # Positions
    pos = get_positions()
    print(format_positions(pos))

    # Open orders
    orders = get_open_orders()
    if orders.get("code") == "0" and orders.get("data"):
        print(f"\n  Open orders: {len(orders['data'])}")
        for o in orders["data"][:5]:
            print(f"    {o.get('instId')} {o.get('side')} {o.get('sz')}@{o.get('px','market')} [{o.get('ordId','?')}]")
    else:
        print("\n  No open orders")

    # Config
    cfg = get_account_config()
    if cfg.get("code") == "0":
        d = cfg["data"][0]
        print(f"\n  Acct mode: {d.get('acctLvCtl','?')} | Pos mode: {d.get('posMode','?')}")

    print("\n✅ Self-test complete")
