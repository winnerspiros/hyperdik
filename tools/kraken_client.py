#!/usr/bin/env python3
"""
Kraken Futures Client — leveraged buy/sell, cross/isolated margin.
Replaces Pionex for automated leverage trading.
Uses HMAC-SHA512 signing with base64-decoded private key.
"""
import json, time, hashlib, hmac, base64, os, urllib.request, ssl
from pathlib import Path

BASE = "https://futures.kraken.com/derivatives/api/v3"
ctx = ssl.create_default_context()

# Load credentials from env or config
def _load_creds():
    pub = os.environ.get("KRAKEN_API_KEY", "")
    priv = os.environ.get("KRAKEN_PRIV_KEY", "")
    if not pub:
        try:
            creds = json.loads(Path("/home/ubuntu/.kraken/creds.json").read_text())
            pub = creds.get("public", "")
            priv = creds.get("private", "")
        except:
            pass
    return pub, priv


def _sign(endpoint: str, post_data: str = "", nonce: str = "") -> tuple:
    """Kraken Futures signing: HMAC-SHA512 of nonce + postData + endpoint."""
    pub, priv = _load_creds()
    if nonce:
        n = nonce
    else:
        n = str(int(time.time() * 1000))
    
    # Decode base64 private key
    secret = base64.b64decode(priv)
    
    # Kraken signs: nonce + postData (URL-encoded body) + endpoint
    sign_str = n + post_data + endpoint
    sig = hmac.new(secret, sign_str.encode(), hashlib.sha512).digest()
    sig_b64 = base64.b64encode(sig).decode()
    
    return pub, sig_b64, n


def _request(method: str, endpoint: str, body: dict = None, query: dict = None) -> dict:
    """Send authenticated request to Kraken Futures."""
    pub, priv = _load_creds()
    if not pub or not priv:
        return {"error": "no_credentials"}
    
    nonce = str(int(time.time() * 1000))
    
    # Build post data for signing (URL-encoded form)
    post_data = ""
    if body:
        post_data = urllib.parse.urlencode(body)
    
    # Sign
    pub_key, signature, _ = _sign(endpoint, post_data, nonce)
    
    # Build URL
    url = f"{BASE}{endpoint}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    
    headers = {
        "APIKey": pub,
        "Authent": signature,
        "Nonce": nonce,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = post_data.encode() if post_data else None
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:500]}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# ACCOUNT
# =============================================================================

def get_accounts() -> dict:
    """Get account info — balances, margin, PnL."""
    return _request("GET", "/accounts")


def get_open_positions() -> dict:
    return _request("GET", "/openpositions")


def get_open_orders() -> dict:
    return _request("GET", "/openorders")


def get_fills(last_ts: int = None) -> dict:
    params = {}
    if last_ts:
        params["lastFillTime"] = last_ts
    return _request("GET", "/fills", query=params)


# =============================================================================
# ORDERS — buy/sell with leverage
# =============================================================================

def place_order(symbol: str = "PF_XBTUSD", side: str = "buy",
                order_type: str = "market", size: float = 1,
                limit_price: float = None, stop_price: float = None,
                reduce_only: bool = False, leverage: int = None) -> dict:
    """
    Place a futures order on Kraken.
    
    Args:
        symbol: Futures symbol (e.g. "PF_XBTUSD" for BTC PERP)
        side: "buy" (long) or "sell" (short)
        order_type: "market", "limit", "stop", "takeProfit"
        size: Quantity in contracts
        limit_price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        reduce_only: Only reduce position, don't open new
        leverage: Set leverage before placing (optional)
    """
    body = {
        "orderType": order_type,
        "symbol": symbol,
        "side": side,
        "size": str(size),
    }
    
    if limit_price is not None:
        body["limitPrice"] = str(limit_price)
    if stop_price is not None:
        body["stopPrice"] = str(stop_price)
    if reduce_only:
        body["reduceOnly"] = "true"
    
    # Optional: set leverage first
    if leverage is not None:
        set_leverage(symbol, leverage)
    
    return _request("POST", "/sendorder", body=body)


def cancel_order(order_id: str = None) -> dict:
    """Cancel an order. If no order_id, cancels ALL open orders."""
    if order_id:
        return _request("POST", "/cancelorder", body={"orderId": order_id})
    return _request("POST", "/cancelallorders")


def edit_order(order_id: str, size: float = None, limit_price: float = None,
               stop_price: float = None) -> dict:
    """Edit an existing order."""
    body = {"orderId": order_id}
    if size is not None:
        body["size"] = str(size)
    if limit_price is not None:
        body["limitPrice"] = str(limit_price)
    if stop_price is not None:
        body["stopPrice"] = str(stop_price)
    return _request("PUT", "/editorder", body=body)


# =============================================================================
# LEVERAGE + MARGIN
# =============================================================================

def set_leverage(symbol: str = "PF_XBTUSD", leverage: int = 5) -> dict:
    """Set leverage for a futures pair (typically 1-50x)."""
    return _request("POST", "/leveragepreferences", body={
        "symbol": symbol,
        "maxLeverage": str(leverage),
    })


def batch_order(orders: list) -> dict:
    """Place multiple orders atomically."""
    body = {"json": json.dumps({"orders": orders})}
    return _request("POST", "/batchorder", body=body)


# =============================================================================
# HELPERS
# =============================================================================

SYMBOLS = {
    "BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD",
    "DOGE": "PF_XDGUSD", "ADA": "PF_ADAUSD", "DOT": "PF_DOTUSD",
    "XRP": "PF_XRPUSD", "LINK": "PF_LINKUSD", "AVAX": "PF_AVAXUSD",
}

def kraken_symbol(base: str) -> str:
    return SYMBOLS.get(base.upper(), f"PF_{base.upper()}USD")


def buy(symbol: str = "PF_XBTUSD", size: float = 1,
        limit_price: float = None, leverage: int = None) -> dict:
    """Convenience: open LONG position."""
    return place_order(symbol, "buy", "limit" if limit_price else "market",
                       size, limit_price=limit_price, leverage=leverage)


def sell(symbol: str = "PF_XBTUSD", size: float = 1,
         limit_price: float = None, leverage: int = None) -> dict:
    """Convenience: open SHORT position."""
    return place_order(symbol, "sell", "limit" if limit_price else "market",
                       size, limit_price=limit_price, leverage=leverage)


def close_position(symbol: str = "PF_XBTUSD") -> dict:
    """Close all positions for a symbol using reduce-only market order."""
    pos = get_open_positions()
    for p in pos.get("openPositions", []):
        if p.get("symbol") == symbol:
            size = abs(float(p.get("size", 0)))
            if size > 0:
                side = "sell" if p.get("side") == "long" else "buy"
                return place_order(symbol, side, "market", size, reduce_only=True)
    return {"error": "no_position", "symbol": symbol}


def close_all() -> list:
    """Close ALL open positions."""
    pos = get_open_positions()
    results = []
    for p in pos.get("openPositions", []):
        symbol = p.get("symbol")
        size = abs(float(p.get("size", 0)))
        side = "sell" if p.get("side") == "long" else "buy"
        if size > 0:
            r = place_order(symbol, side, "market", size, reduce_only=True)
            results.append(r)
    return results


# =============================================================================
# FORMAT
# =============================================================================

def format_account(acct: dict) -> str:
    if "error" in acct:
        return f"Kraken error: {acct['error']}"
    lines = ["🦑 KRAKEN FUTURES:"]
    for k in ["type", "currency"]:
        if k in acct.get("accounts", {}):
            a = acct["accounts"][k]
            lines.append(f"  Balance: ${float(a.get('balance',0)):,.2f}")
            lines.append(f"  PnL: ${float(a.get('pnl',0)):+.2f}")
            lines.append(f"  Margin: ${float(a.get('margin',0)):,.2f}")
            break
    fp = acct.get("flexFutures", {})
    if fp:
        lines.append(f"  Flex: {fp.get('currencies',{})}")
    return "\n".join(lines)


def format_positions(pos: dict) -> str:
    if "error" in pos:
        return f"Kraken error: {pos['error']}"
    positions = pos.get("openPositions", [])
    if not positions:
        return "🦑 No open positions"
    lines = ["🦑 OPEN POSITIONS:"]
    for p in positions:
        lines.append(f"  {p['symbol']} {p['side']} size={p['size']} "
                     f"entry=${float(p.get('price',0)):,.2f} "
                     f"PnL=${float(p.get('pnl',0)):+.2f}")
    return "\n".join(lines)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    # Save creds from command line if provided
    import sys
    if len(sys.argv) >= 3:
        creds = {"public": sys.argv[1], "private": sys.argv[2]}
        os.makedirs("/home/ubuntu/.kraken", exist_ok=True)
        Path("/home/ubuntu/.kraken/creds.json").write_text(json.dumps(creds))
        print("✅ Credentials saved to ~/.kraken/creds.json")
    
    print("🦑 Kraken Futures Client — Testing...")
    
    # Test account
    acct = get_accounts()
    print(format_account(acct))
    
    # Test positions
    pos = get_open_positions()
    print(format_positions(pos))
    
    # Test leverage
    print(f"\n  Setting BTC leverage to 5x: {set_leverage('PF_XBTUSD', 5)}")
