#!/usr/bin/env python3
"""
Hyperliquid Perpetuals Client v2 — uses official SDK for all operations.

Key features discovered from docs:
- usd_class_transfer: move USDC spot ↔ perps (unified mode)
- market_open/market_close: proper market entry/exit with slippage
- schedule_cancel: dead-man's switch (auto-cancel orders on crash)
- bulk_orders with TP/SL grouping: entry + stop + target in one call
- funding history: track funding costs
- metaAndAssetCtxs: real-time funding + open interest + mark price
"""
import json, time, os
from pathlib import Path
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

MAINNET = "https://api.hyperliquid.xyz"

# ── Global state ──
_exchange: Exchange = None
_info: Info = None

# ── ADAPTIVE RATE LIMITER ───────────────────────────────────────────────
# Self-tuning based on actual API responses. Starts aggressive (30ms = 33 req/s),
# automatically throttles down when 429s appear, recovers when API is healthy.
# Exposes pressure_level() for callers to skip non-critical work under load.
#
# Design:
#   - Sliding 30s window tracks 429 vs success rate
#   - Interval scales: 30ms (no errors) → 200ms (100% errors)  
#   - Circuit breaker after 5 consecutive 429s: skip non-critical calls
#   - Recovery: interval decays back to 30ms as errors clear from window
# ─────────────────────────────────────────────────────────────────────────

_last_api_time: float = 0.0
_api_call_count: int = 0
_api_call_reset: float = time.time()

# Adaptive interval — starts fast, adjusts based on real API feedback
_ADAPTIVE_INTERVAL: float = 0.03   # current effective interval (30ms = 33 req/s)
_MIN_INTERVAL: float = 0.03        # floor: 33 req/s when API is healthy
_MAX_INTERVAL: float = 0.20        # ceiling: 5 req/s when API is throttling
_CALL_WINDOW: list = []            # [(timestamp, is_429), ...] sliding 30s window
_WINDOW_SECS: float = 30.0
_CONSECUTIVE_429S: int = 0

def report_api_result(is_429: bool):
    """Feed back API result to the adaptive rate limiter.
    
    Call this from the daemon after any batch of API calls completes.
    Pass True if a 429 was encountered, False for success.
    """
    global _ADAPTIVE_INTERVAL, _CALL_WINDOW, _CONSECUTIVE_429S
    now = time.time()
    _CALL_WINDOW.append((now, is_429))
    # Trim old entries from window
    _CALL_WINDOW = [(t, e) for t, e in _CALL_WINDOW if now - t < _WINDOW_SECS]
    # Calculate error rate in window and adjust interval
    if _CALL_WINDOW:
        err_rate = sum(1 for _, e in _CALL_WINDOW if e) / len(_CALL_WINDOW)
        # Scale: 0% errors → min interval; 33%+ errors → max interval
        scale = min(err_rate * 3.0, 1.0)
        _ADAPTIVE_INTERVAL = _MIN_INTERVAL + (_MAX_INTERVAL - _MIN_INTERVAL) * scale
    # Track consecutive 429s for circuit breaker
    if is_429:
        _CONSECUTIVE_429S += 1
    else:
        _CONSECUTIVE_429S = max(0, _CONSECUTIVE_429S - 1)  # gradual recovery


def pressure_level() -> int:
    """Return 0-100 indicating API pressure. 0=healthy, 50+=throttling, 100=heavy.
    Callers should skip non-critical work above 50."""
    if _MAX_INTERVAL == _MIN_INTERVAL:
        return 0
    pct = (_ADAPTIVE_INTERVAL - _MIN_INTERVAL) / (_MAX_INTERVAL - _MIN_INTERVAL) * 100
    return int(min(pct, 100))


def is_throttled() -> bool:
    """True when API is under heavy load — skip non-critical data gathering."""
    return pressure_level() >= 50 or _CONSECUTIVE_429S >= 3


def _rate_limit():
    """Enforce adaptive minimum interval between API calls."""
    global _last_api_time, _api_call_count, _api_call_reset
    now = time.time()
    # Reset counter per second
    if now - _api_call_reset >= 1.0:
        _api_call_count = 0
        _api_call_reset = now
    # Per-second cap using adaptive interval
    max_per_sec = int(1.0 / _ADAPTIVE_INTERVAL) - 2  # leave 2 call buffer
    if _api_call_count >= max(max_per_sec, 3):  # never drop below 3 req/s
        wait = 1.0 - (now - _api_call_reset)
        if wait > 0:
            time.sleep(wait)
        _api_call_count = 0
        _api_call_reset = time.time()
    # Enforce adaptive minimum interval
    elapsed = now - _last_api_time
    if elapsed < _ADAPTIVE_INTERVAL:
        time.sleep(_ADAPTIVE_INTERVAL - elapsed)
    _last_api_time = time.time()
    _api_call_count += 1
_wallet: Account = None
_main_wallet: str = ""
_api_wallet: str = ""


def _autoload():
    global _exchange, _info, _wallet, _main_wallet, _api_wallet
    if _exchange is not None:
        return
    cfg = Path("/home/ubuntu/.hyperliquid/config.json")
    if not cfg.exists():
        return
    c = json.loads(cfg.read_text())
    _api_wallet = c["api_wallet"]
    _main_wallet = c["main_wallet"]
    _wallet = Account.from_key(c["api_private_key"])
    _info = Info(MAINNET, skip_ws=True)
    _exchange = Exchange(_wallet, MAINNET, account_address=_main_wallet)


def configure(api_wallet: str, private_key: str, main_wallet: str = ""):
    global _wallet, _api_wallet, _main_wallet, _exchange, _info
    _api_wallet = api_wallet
    _main_wallet = main_wallet or api_wallet
    _wallet = Account.from_key(private_key)
    _info = Info(MAINNET, skip_ws=True)
    _exchange = Exchange(_wallet, MAINNET, account_address=_main_wallet)


def set_main_wallet(address: str):
    """Update main wallet and recreate exchange with correct account_address."""
    global _main_wallet, _exchange
    _main_wallet = address
    if _wallet is not None:
        _exchange = Exchange(_wallet, MAINNET, account_address=_main_wallet)


# ── Account ──

def get_user_state(address: str = None) -> dict:
    """Get raw user state from Hyperliquid (positions, margin, etc)."""
    _autoload()
    _rate_limit()
    addr = address or _main_wallet
    return _info.user_state(addr)

def get_account() -> dict:
    """Get unified account state (perps + spot). Spot balance cached 5min to avoid 429s."""
    global _spot_cache
    _autoload()
    _rate_limit()
    state = _info.user_state(_main_wallet)

    # Spot balance — cached 5min. Only changes on trade/deposit, not every cycle.
    now = time.time()
    if '_spot_cache' not in globals():
        _spot_cache = {"spot_usdc": 0.0, "spot_hold": 0.0, "spot_free": 0.0, "balances": [], "ts": 0.0}
    if now - _spot_cache["ts"] > 300:
        try:
            _rate_limit()
            spot = _info.spot_user_state(_main_wallet)
            _spot_cache["spot_usdc"] = sum(float(b["total"]) for b in spot["balances"] if b["coin"] == "USDC")
            _spot_cache["spot_hold"] = sum(float(b.get("hold", 0)) for b in spot["balances"] if b["coin"] == "USDC")
            _spot_cache["spot_free"] = _spot_cache["spot_usdc"] - _spot_cache["spot_hold"]
            _spot_cache["balances"] = spot.get("balances", [])
            _spot_cache["ts"] = now
        except Exception:
            pass  # Use cached values on failure

    spot_usdc = _spot_cache["spot_usdc"]
    spot_hold = _spot_cache["spot_hold"]
    spot_free = _spot_cache["spot_free"]
    perp_eq = float(state["marginSummary"]["accountValue"])
    return {
        "perp_equity": perp_eq,
        "spot_usdc": spot_usdc,
        "spot_hold": spot_hold,
        "spot_free": spot_free,
        "total_equity": max(perp_eq + spot_free, spot_usdc),  # floor at spot_usdc for post-close settlement
        "margin_used": float(state["marginSummary"]["totalMarginUsed"]),
        "withdrawable": float(state.get("withdrawable", 0)),
        "positions": state.get("assetPositions", []),
        "spot_balances": _spot_cache.get("balances", []),
    }


def get_open_orders() -> list:
    _autoload()
    _rate_limit()
    return _info.open_orders(_main_wallet)


def get_fills() -> list:
    _autoload()
    _rate_limit()
    return _info.user_fills(_main_wallet)


_meta_cache: dict | None = None
_meta_cache_ts: float = 0

def _get_sz_decimals(coin: str) -> int:
    """Get szDecimals for a coin from meta, cached for 5 minutes."""
    global _meta_cache, _meta_cache_ts
    now = time.time()
    if _meta_cache is None or now - _meta_cache_ts > 300:
        _autoload()
        _meta_cache = _info.meta()
        _meta_cache_ts = now
    meta = _meta_cache or {}
    for m in meta.get('universe', []):
        if m.get('name') == coin:
            return m.get('szDecimals', 5)
    return 5  # Default: 5 decimal places
def market_open(coin: str, is_buy: bool, size_usd: float, slippage: float = 0.005,
                order_type: str = "Ioc") -> dict:
    """Open a position with IOC limit order (immediate-or-cancel).

    IOC fills whatever crosses at limit price, cancels the rest.
    Our own rounding avoids SDK's _slippage_price precision bug."""
    _autoload()
    _rate_limit()
    mids = _info.all_mids()
    price = float(mids.get(coin, 0))
    if price <= 0:
        return {"status": "err", "error": f"no_price: {coin}"}
    sz = size_usd / price
    sz_dec = _get_sz_decimals(coin)
    sz = round(sz, sz_dec)
    # IOC immediate-fill: limit ABOVE market for buys, BELOW for sells
    # Longs: limit ABOVE market → fills at best available price
    # Shorts: limit BELOW market → fills at best available price
    limit_px = price * (1 + slippage) if is_buy else price * (1 - slippage)
    # Use proper tick-aware rounding from execution module
    from hyperliquid_execution import round_price
    import math
    px_dec = max(0, int(5 - abs(math.log10(max(price, 0.0001)))))
    limit_px = round_price(px_dec, limit_px, is_buy=is_buy)
    import logging
    tif = order_type.capitalize()
    logging.getLogger().warning(f"  📏 market_open: {coin} ${size_usd:.0f} @ ${price:.2f} limit={limit_px} sz={sz:.6f} {tif}")
    try:
        result = _exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": tif}})
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"status": "err", "error": f"order exception: {e}", "trace": tb[:200]}


def market_close(coin: str, sz: float = None) -> dict:
    """Close entire position at market. Auto-detects size."""
    _autoload()
    _rate_limit()
    try:
        return _exchange.market_close(coin, sz=sz)
    except Exception as e:
        return {"status": "err", "error": f"close exception: {e}"}


def order(coin: str, is_buy: bool, sz: float, limit_px: float,
          order_type: str = "gtc", reduce_only: bool = False) -> dict:
    """Place a limit order. order_type: 'gtc', 'ioc', 'alo'."""
    _autoload()
    _rate_limit()
    tif_map = {"gtc": "Gtc", "ioc": "Ioc", "alo": "Alo"}
    return _exchange.order(coin, is_buy, sz, limit_px,
                           {"limit": {"tif": tif_map.get(order_type, "Gtc")}},
                           reduce_only=reduce_only)


def cancel(coin: str, oid: int) -> dict:
    _autoload()
    _rate_limit()
    return _exchange.cancel(coin, oid)


def update_leverage(coin: str, leverage: int) -> dict:
    """Set leverage for a coin. SDK signature: update_leverage(leverage, name)."""
    _autoload()
    _rate_limit()
    return _exchange.update_leverage(leverage, coin)


# ── Transfers ──

def spot_to_perp(amount: float) -> dict:
    """Move USDC from spot to perps (enables leverage trading)."""
    _autoload()
    _rate_limit()
    return _exchange.usd_class_transfer(amount, to_perp=True)


def perp_to_spot(amount: float) -> dict:
    """Move USDC from perps back to spot."""
    _autoload()
    _rate_limit()
    return _exchange.usd_class_transfer(amount, to_perp=False)


# ── Safety ──

def schedule_cancel(time_ms: int = None) -> dict:
    """Set/unset auto-cancel of all orders at future time. None = cancel the schedule."""
    _autoload()
    _rate_limit()
    return _exchange.schedule_cancel(time_ms)


def trigger_order(coin: str, is_buy: bool, sz: float, trigger_px: float,
                  order_type: str = "sl", is_market: bool = True,
                  reduce_only: bool = True) -> dict:
    """Place a trigger order (TP/SL). order_type: 'tp' or 'sl'.
    
    HL SDK: order_type = {"trigger": {"triggerPx": ..., "isMarket": ..., "tpsl": "tp"|"sl"}}
    """
    _autoload()
    _rate_limit()
    return _exchange.order(
        coin, is_buy, sz, trigger_px,
        order_type={"trigger": {"triggerPx": trigger_px, "isMarket": is_market, "tpsl": order_type}},
        reduce_only=reduce_only,
    )


def cancel_order(coin: str, oid: int) -> dict:
    """Cancel an open order by order ID. Hummingbot pattern: cancel stale TP before re-place."""
    _autoload()
    _rate_limit()
    return _exchange.cancel(coin, oid)


# ── Market Data ──

def get_all_mids() -> dict:
    _autoload()
    _rate_limit()
    raw = _info.all_mids()
    # HL returns all values as strings — normalize to float
    return {k: float(v) for k, v in raw.items()}


def get_meta() -> dict:
    _autoload()
    _rate_limit()
    return _info.meta()


def get_asset_ctxs() -> dict:
    """Get funding rates, open interest, mark prices for all coins."""
    _autoload()
    _rate_limit()
    meta, ctxs = _info.meta_and_asset_ctxs()
    return {"meta": meta, "contexts": ctxs}


def get_funding_history(coin: str, start_ms: int, end_ms: int = None) -> list:
    _autoload()
    _rate_limit()
    return _info.funding_history(coin, start_ms, end_ms)


# ── Convenience ──

HL_COINS = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "DOGE": "DOGE",
            "XRP": "XRP", "AVAX": "AVAX", "ADA": "ADA", "LINK": "LINK",
            "DOT": "DOT", "SUI": "SUI", "PEPE": "PEPE", "BONK": "BONK",
            "WIF": "WIF", "RENDER": "RENDER", "UNI": "UNI"}


def buy(coin: str, sz: float, limit_px: float = 0) -> dict:
    """Long. limit_px=0 for market."""
    _autoload()
    _rate_limit()
    c = HL_COINS.get(coin.upper(), coin.upper())
    if limit_px:
        return order(c, True, sz, limit_px, "gtc")
    return _exchange.market_open(c, True, sz)


def sell(coin: str, sz: float, limit_px: float = 0) -> dict:
    """Short. limit_px=0 for market."""
    _autoload()
    _rate_limit()
    c = HL_COINS.get(coin.upper(), coin.upper())
    if limit_px:
        return order(c, False, sz, limit_px, "gtc")
    return _exchange.market_open(c, False, sz)


def close(coin: str) -> dict:
    _autoload()
    _rate_limit()
    return market_close(HL_COINS.get(coin.upper(), coin.upper()))


# ── Formatting ──

def format_account(acc: dict) -> str:
    lines = [
        f"💧 HYPERLIQUID",
        f"  Perps:  ${acc['perp_equity']:,.2f}",
        f"  Spot:   ${acc['spot_usdc']:,.2f}",
        f"  Total:  ${acc['total_equity']:,.2f}",
        f"  Margin: ${acc['margin_used']:,.2f}",
    ]
    for p in acc.get("positions", []):
        pos = p.get("position", p)
        coin = pos.get("coin", "?")
        szi = float(pos.get("szi", 0))
        if abs(szi) > 0.0001:
            entry = float(pos.get("entryPx", 0) or 0)
            pnl = float(pos.get("unrealizedPnl", 0))
            liq = float(pos.get("liquidationPx", 0) or 0)
            lev = pos.get("leverage", {})
            lev_val = lev.get("value", "?") if isinstance(lev, dict) else "?"
            side = "LONG" if szi > 0 else "SHORT"
            lines.append(f"  {coin} {side} x{lev_val} sz={abs(szi):.4f} entry=${entry:,.2f} PnL=${pnl:+,.2f}")
            if liq:
                lines.append(f"    Liq: ${liq:,.2f}")
    return "\n".join(lines)


# ── Self-test ──

if __name__ == "__main__":
    # Keys loaded from ~/.hyperliquid/config.json — see README for setup
    import json, os
    cfg_path = os.path.expanduser("~/.hyperliquid/config.json")
    if os.path.exists(cfg_path):
        cfg = json.loads(open(cfg_path).read())
        configure(cfg["api_wallet"], cfg["api_private_key"], cfg["main_wallet"])
    else:
        print("No config found at ~/.hyperliquid/config.json — skipping self-test")
        import sys; sys.exit(0)

    print("=== HYPERLIQUID CLIENT v2 ===\n")

    # Account
    acc = get_account()
    print(format_account(acc))
    print()

    # Funding + OI data
    print("Market data:")
    data = get_asset_ctxs()
    ctxs = data.get("contexts", [])
    meta = data.get("meta", {}).get("universe", [])
    for i, m in enumerate(meta[:5]):
        if i < len(ctxs):
            ctx = ctxs[i]
            print(f"  {m['name']:6s} funding={float(ctx.get('funding',0))*100:.4f}% "
                  f"OI=${float(ctx.get('openInterest',0)):,.0f} "
                  f"mark=${float(ctx.get('markPx',0) or 0):,.2f}")

    # Open orders
    orders = get_open_orders()
    print(f"\nOpen orders: {len(orders)}")


# ── Oracle prices & divergence ──

_asset_ctx_cache: dict[str, dict] = {}
_asset_ctx_cache_ts: float = 0.0
_ASSET_CTX_TTL = 30  # Refresh every 30s

def get_oracle_price(coin: str) -> float:
    """Get oracle price for a coin (used for liquidations). Cached 30s."""
    global _asset_ctx_cache, _asset_ctx_cache_ts
    now = time.time()
    if not _asset_ctx_cache or (now - _asset_ctx_cache_ts) > _ASSET_CTX_TTL:
        _autoload()
        _rate_limit()
        try:
            ctxs = _info.meta_and_asset_ctxs()
            universe_list = ctxs[0].get("universe", []) if isinstance(ctxs, list) else []
            asset_ctxs = ctxs[1] if isinstance(ctxs, list) and len(ctxs) > 1 else []
            _asset_ctx_cache = {}
            for i, asset in enumerate(universe_list):
                name = asset.get("name", "")
                if name and i < len(asset_ctxs):
                    ctx = asset_ctxs[i]
                    _asset_ctx_cache[name] = {
                        "oracle": float(ctx.get("oraclePx", 0)),
                        "mark": float(ctx.get("markPx", 0)),
                        "mid": float(ctx.get("midPx", 0)),
                        "funding": float(ctx.get("funding", 0)),
                        "oi": float(ctx.get("openInterest", 0)),
                    }
            _asset_ctx_cache_ts = now
        except Exception:
            pass
    ctx = _asset_ctx_cache.get(coin.upper(), {})
    return ctx.get("oracle", 0.0)


def check_oracle_divergence(coin: str, mid_price: float) -> tuple[float, bool]:
    """Check if mid price diverges dangerously from oracle.
    Returns (divergence_pct, is_dangerous).
    Hyperliquid uses oracle for liquidations — if mid > oracle, longs risk liquidation.
    """
    oracle = get_oracle_price(coin)
    if oracle <= 0 or mid_price <= 0:
        return 0.0, False
    div_pct = (mid_price - oracle) / oracle * 100
    # Dangerous: >0.5% divergence (liquidation uses oracle, not mid)
    is_dangerous = abs(div_pct) > 0.5
    return div_pct, is_dangerous

    # Safety switch test
    print("\nSafety switch available: schedule_cancel(time_ms)")
    print("  Set auto-cancel of all orders at future timestamp")
    print("  Use as dead-man's switch if daemon crashes.\n")

    print("✅ v2 complete — using official SDK")
