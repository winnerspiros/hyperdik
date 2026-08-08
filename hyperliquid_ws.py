#!/usr/bin/env python3
"""
Hyperliquid WebSocket Real-Time Feed — millisecond trade data and order books.

Subscribes to Hyperliquid's WebSocket at wss://api.hyperliquid.xyz/ws
Streams: trades (every fill), order book (L2 updates), user positions.

Stores to hyperliquid_ws_data.json for daemon consumption.
"""
import copy
import json, time, threading, ssl, asyncio
from collections import deque
import websocket  # pip install websocket-client

WS_URL = "wss://api.hyperliquid.xyz/ws"

# Shared state (thread-safe)
_state_lock = threading.Lock()
_latest_data = {
    "trades": {},           # {coin: {"price": ..., "size": ..., "side": ..., "ts": ms}} — latest only
    "trade_history": {},    # {coin: deque([{price, size, side, ts}, ...], maxlen=100)} — rolling window
    "orderbooks": {},       # {coin: {"bids": [...], "asks": [...], "ts": ms}}
    "mids": {},             # {coin: mid_price}
    "positions": [],        # user positions
    "account": {},          # margin summary
    "fills": [],            # user fills (real-time trade confirmations)
    "funding_updates": [],  # user funding payments
    "open_interest": {},    # {coin: {"oi": float, "ts": ms, "timestamp": ms}}
    "timestamp": 0,
    "running": False,
}


def get_latest():
    """Thread-safe snapshot of latest WebSocket data."""
    with _state_lock:
        return copy.deepcopy(_latest_data)


def get_field(field: str):
    """Thread-safe shallow copy of a single field — avoids O(n) deepcopy of entire state.

    Use this when you only need one field (fills, funding_updates, mids, etc.).
    Returns a new list/dict so caller can modify safely.
    """
    with _state_lock:
        val = _latest_data.get(field)
        if val is None:
            return None
        if isinstance(val, list):
            return list(val)
        if isinstance(val, dict):
            return dict(val)
        return val


def trim_field(field: str, keep: int):
    """Thread-safe in-place trim of a list field to last N items.

    Negative keep = last N. Positive = first N.
    """
    with _state_lock:
        val = _latest_data.get(field)
        if isinstance(val, list) and len(val) > abs(keep):
            if keep < 0:
                _latest_data[field] = val[keep:]
            else:
                _latest_data[field] = val[:keep]



def normalize_orderbook(ob: dict) -> dict | None:
    """Normalize Hyperliquid order book to [{px, sz}, ...] format.
    
    HL WS returns levels as [{px, sz, n}, ...] dicts or legacy [[px, sz], ...] lists.
    Returns None if no valid data.
    """
    raw_bids = ob.get("bids", [])
    raw_asks = ob.get("asks", [])
    if not raw_bids or not raw_asks:
        return None
    def _px(b):
        if isinstance(b, dict): return float(b["px"])
        return float(b[0])
    def _sz(b):
        if isinstance(b, dict): return float(b["sz"])
        return float(b[1])
    return {
        "bids": [{"px": _px(b), "sz": _sz(b)} for b in raw_bids],
        "asks": [{"px": _px(a), "sz": _sz(a)} for a in raw_asks],
    }


def _on_message(ws, message):
    try:
        data = json.loads(message)
    except:
        return  # skip non-JSON messages

    # Hyperliquid sometimes sends string messages (e.g. connection status)
    if not isinstance(data, dict):
        return

    channel = data.get("channel", "")
    if not channel:
        return

    with _state_lock:
        _latest_data["timestamp"] = int(time.time() * 1000)

        payload = data.get("data")
        if payload is None:
            return

        if channel == "trades":
            if not isinstance(payload, list):
                return
            for trade in payload:
                if not isinstance(trade, dict):
                    continue
                coin = trade.get("coin", "")
                trade_dict = {
                    "price": trade.get("px", "0"),
                    "size": trade.get("sz", "0"),
                    "side": trade.get("side", ""),
                    "ts": trade.get("time", 0),
                }
                # Latest trade (backward compat)
                _latest_data["trades"][coin] = trade_dict
                # Rolling window for taker ratio / order flow analysis
                if coin not in _latest_data["trade_history"]:
                    from collections import deque
                    _latest_data["trade_history"][coin] = deque(maxlen=100)
                _latest_data["trade_history"][coin].append(trade_dict)

        elif channel == "l2Book":
            # Handle both dict (single update) and list (batch) formats
            books = [payload] if isinstance(payload, dict) else payload
            if not isinstance(books, list):
                return
            for book in books:
                if not isinstance(book, dict):
                    continue
                coin = book.get("coin", "")
                levels = book.get("levels", [[], []])
                _latest_data["orderbooks"][coin] = {
                    "bids": levels[0][:5],
                    "asks": levels[1][:5],
                    "ts": book.get("time", 0),
                }

        elif channel == "allMids":
            if isinstance(payload, dict):
                _latest_data["mids"] = payload.get("mids", {})

        elif channel == "webData2":
            # User-specific updates (positions, orders, fills)
            if isinstance(payload, dict):
                cs = payload.get("clearinghouseState", {})
                if isinstance(cs, dict):
                    _latest_data["positions"] = cs.get("assetPositions", [])
                    _latest_data["account"] = cs.get("marginSummary", {})

        elif channel == "userFills":
            # Real-time fill confirmations
            if isinstance(payload, dict):
                # Capture whether this fill is a liquidation (or self-liquidation)
                # Hyperliquid API uses "liquidation" boolean; also check "isLiquidation"
                is_liq = payload.get("liquidation", False) or payload.get("isLiquidation", False)
                fill = {
                    "coin": payload.get("coin", ""),
                    "px": payload.get("px", "0"),
                    "sz": payload.get("sz", "0"),
                    "side": payload.get("side", ""),
                    "time": payload.get("time", 0),
                    "oid": payload.get("oid", 0),
                    "crossed": payload.get("crossed", False),
                    "liquidation": bool(is_liq),
                    "dir": payload.get("dir", ""),
                    "closedPnl": payload.get("closedPnl", "0"),
                }
                _latest_data["fills"].append(fill)
                # Keep only last 100 fills to prevent memory growth (bump from 50 for monitoring)
                if len(_latest_data["fills"]) > 100:
                    _latest_data["fills"] = _latest_data["fills"][-100:]

        elif channel == "userFunding":
            # Real-time funding payments
            if isinstance(payload, dict):
                fund = {
                    "coin": payload.get("coin", ""),
                    "fundingRate": payload.get("fundingRate", "0"),
                    "premium": payload.get("premium", "0"),
                    "time": payload.get("time", 0),
                }
                _latest_data["funding_updates"].append(fund)
                if len(_latest_data["funding_updates"]) > 20:
                    _latest_data["funding_updates"] = _latest_data["funding_updates"][-20:]

        elif channel == "pong":
            pass

        elif channel == "openInterest":
            if isinstance(payload, dict):
                coin = payload.get("coin", "")
                if coin:
                    _latest_data["open_interest"][coin] = {
                        "oi": float(payload.get("oi", 0)),
                        "ts": payload.get("time", int(time.time() * 1000)),
                        "timestamp": int(time.time() * 1000),
                    }
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        coin = item.get("coin", "")
                        if coin:
                            _latest_data["open_interest"][coin] = {
                                "oi": float(item.get("oi", 0)),
                                "ts": item.get("time", int(time.time() * 1000)),
                                "timestamp": int(time.time() * 1000),
                            }


def _on_error(ws, error):
    pass  # Silently ignore — handled by reconnect


def _on_close(ws, close_status_code, close_msg):
    with _state_lock:
        _latest_data["running"] = False


def _on_open(ws):
    with _state_lock:
        _latest_data["running"] = True

    # Subscribe to real-time data
    coins = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "ADA", "LINK", "SUI", "DOT"]

    # Trades for each coin (staggered to avoid rate limits)
    import time
    for coin in coins:
        ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
        time.sleep(0.15)  # stagger 150ms

    # Order books (staggered)
    for coin in coins:
        ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
        time.sleep(0.15)

    # All mids (price updates every 500ms)
    ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))

    # User data (positions, orders) - requires auth
    ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "webData2", "user": ""}}))

    # User fills — real-time trade confirmations (TP/SL triggers, liquidations)
    ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "userFills", "user": ""}}))

    # User funding — real-time funding payment tracking
    ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "userFunding", "user": ""}}))

    # Open Interest — real-time OI per coin
    for coin in coins:
        ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "openInterest", "coin": coin}}))
        time.sleep(0.15)


def start_ws(address: str):
    """Start WebSocket in background thread. Non-blocking."""

    def _run():
        while True:
            try:
                ws = websocket.WebSocketApp(
                    WS_URL,
                    on_message=_on_message,
                    on_error=_on_error,
                    on_close=_on_close,
                    on_open=_on_open,
                )
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            except Exception as e:
                pass
            time.sleep(3)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── Periodic REST fallback (allMids every 2s if WS not providing data) ─────

def _rest_poll(interval: float = 2.0):
    """Fallback: poll REST allMids if WebSocket is stale."""
    import urllib.request

    while True:
        with _state_lock:
            ws_ts = _latest_data.get("timestamp", 0)
            running = _latest_data.get("running", False)

        now = int(time.time() * 1000)
        if not running or (now - ws_ts > 5000):
            # WS not running or stale - REST fallback
            try:
                data = json.dumps({"type": "allMids"}).encode()
                req = urllib.request.Request(
                    "https://api.hyperliquid.xyz/info",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    result = json.loads(r.read())
                    with _state_lock:
                        if isinstance(result, dict) and result:
                            _latest_data["mids"] = result
                            _latest_data["timestamp"] = now
            except:
                pass
        time.sleep(interval)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Hyperliquid WebSocket feed...")
    print("This runs in background. Ctrl+C to stop.\n")

    start_ws("")

    # Also start REST fallback
    rest_thread = threading.Thread(target=_rest_poll, args=(2.0,), daemon=True)
    rest_thread.start()

    # Print updates every 3 seconds
    try:
        while True:
            data = get_latest()
            mids = data.get("mids", {})
            trades = data.get("trades", {})
            running = data.get("running", False)

            print(f"\033[2J\033[H")  # clear screen
            print(f"⚡ HYPERLIQUID WS {'🟢 LIVE' if running else '🔴 REST FALLBACK'}")
            print(f"   Time: {data.get('timestamp', 0)}")

            # Prices
            print("\n  PRICES:")
            for coin in ["BTC", "ETH", "SOL", "DOGE", "XRP"]:
                if coin in mids:
                    print(f"    {coin:6s} ${float(mids[coin]):,.2f}")
                elif coin in trades:
                    print(f"    {coin:6s} ${trades[coin]['price']} (last trade)")

            # Trades
            print(f"\n  LAST TRADES: {len(trades)} coins")
            for coin, t in list(trades.items())[:5]:
                side = "🟢" if t["side"] == "B" else "🔴"
                print(f"    {coin:6s} {side} ${t['price']} x {t['size']}")

            time.sleep(3)
    except KeyboardInterrupt:
        print("\nStopped.")
