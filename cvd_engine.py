#!/usr/bin/env python3
"""
Real-Time CVD Engine — Trade-Level Cumulative Volume Delta

Watches the Hyperliquid WebSocket trades stream and accumulates
true buy/sell volume delta (NOT OHLCV proxy).

Key insight: Real trade-level CVD is 3-4x more predictive than
OHLCV-estimated delta because it captures the actual aggressor side.

Used by: continuous_predictor CVD layer (replaces OHLCV proxy when available)
Research: alpha-engine by atg0dd, CryptoFlowEngine
"""
import time
from collections import defaultdict
from typing import Optional

# ── In-memory trade accumulator ─────────────────────────────────────────────
# Each coin gets a rolling buffer of trade deltas
MAX_TRADES_PER_COIN = 500  # Keep last 500 trades for CVD computation
_trade_buffer: dict[str, list[dict]] = defaultdict(list)
# Format: [{"price": float, "size": float, "side": "B"/"S"/"A", "ts": float}, ...]


def ingest_trade(coin: str, price: float, size: float, side: str, ts: float = None):
    """
    Called by WebSocket handler on every trade.
    
    side: "B" = taker buy (aggressive buyer), "S" = taker sell, "A" = unknown
    """
    ts = ts or time.time()
    _trade_buffer[coin].append({
        "price": price,
        "size": size,
        "side": side,
        "ts": ts,
    })
    # Trim to max size
    if len(_trade_buffer[coin]) > MAX_TRADES_PER_COIN:
        _trade_buffer[coin] = _trade_buffer[coin][-MAX_TRADES_PER_COIN:]


def get_trade_cvd(coin: str, lookback_seconds: float = 300) -> dict:
    """
    Compute real CVD from accumulated trade data.
    
    Returns: {
        "cvd": float (cumulative delta over lookback),
        "buy_volume": float,
        "sell_volume": float,
        "net_flow": float (buy_vol - sell_vol),
        "trade_count": int,
        "freshness_seconds": float (age of most recent trade),
        "cvd_trend": "rising"/"falling"/"flat",
        "confidence": 0-100,
    }
    """
    trades = _trade_buffer.get(coin, [])
    now = time.time()
    cutoff = now - lookback_seconds
    
    recent = [t for t in trades if t["ts"] >= cutoff]
    
    if not recent:
        return {"cvd": 0, "buy_volume": 0, "sell_volume": 0, "net_flow": 0,
                "trade_count": 0, "freshness_seconds": 999, "cvd_trend": "flat",
                "confidence": 0}
    
    buy_vol = 0.0
    sell_vol = 0.0
    
    for t in recent:
        sz = t["size"]
        side = t["side"]
        if side == "B":
            buy_vol += sz
        elif side == "S":
            sell_vol += sz
        else:
            # Unknown side — split 50/50
            buy_vol += sz * 0.5
            sell_vol += sz * 0.5
    
    cvd = buy_vol - sell_vol
    net_flow = buy_vol + sell_vol
    freshness = now - (recent[-1]["ts"] if recent else now)
    trade_count = len(recent)
    
    # CVD trend: compare first half vs second half
    mid = len(recent) // 2
    if mid >= 3:
        first_half_buy = sum(t["size"] for t in recent[:mid] if t["side"] == "B")
        first_half_sell = sum(t["size"] for t in recent[:mid] if t["side"] == "S")
        second_half_buy = sum(t["size"] for t in recent[mid:] if t["side"] == "B")
        second_half_sell = sum(t["size"] for t in recent[mid:] if t["side"] == "S")
        
        first_delta = first_half_buy - first_half_sell
        second_delta = second_half_buy - second_half_sell
        
        if second_delta > first_delta * 1.2:
            trend = "rising"
        elif second_delta < first_delta * 0.8:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"
    
    # Confidence: based on trade count and volume
    confidence = min(80, trade_count * 2)
    if net_flow > 0:
        confidence = min(85, confidence * (cvd / net_flow + 1) * 0.5)
    
    return {
        "cvd": round(cvd, 4),
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
        "net_flow": round(net_flow, 4),
        "trade_count": trade_count,
        "freshness_seconds": round(freshness, 1),
        "cvd_trend": trend,
        "confidence": round(confidence, 1),
    }


def cvd_to_prediction(cvd_data: dict) -> dict:
    """
    Convert real CVD data into a directional prediction.
    
    Returns {"up": 0-100, "down": 0-100, "flat": 0-100, "confidence": 0-100}
    """
    if cvd_data["trade_count"] < 5 or cvd_data["confidence"] < 15:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 5}
    
    cvd = cvd_data["cvd"]
    net_flow = cvd_data["net_flow"]
    trend = cvd_data["cvd_trend"]
    
    if net_flow <= 0:
        return {"up": 33, "down": 33, "flat": 34, "confidence": 10}
    
    # Normalize CVD as fraction of total flow: -1 (all sells) to +1 (all buys)
    cvd_ratio = cvd / net_flow
    
    # Base score from CVD ratio
    score = 50 + cvd_ratio * 40  # -1→10, 0→50, +1→90
    
    # Trend adjustment
    if trend == "rising":
        score += 10
    elif trend == "falling":
        score -= 10
    
    score = max(0, min(100, score))
    confidence = min(80, cvd_data["confidence"])
    
    up = score
    down = 100 - score
    flat = max(0, 100 - abs(up - 50) * 2)
    flat = min(flat, 35)
    
    total = up + down + flat
    if total > 0:
        return {"up": round(up / total * 100, 1), "down": round(down / total * 100, 1),
                "flat": round(flat / total * 100, 1), "confidence": round(confidence, 1)}
    return {"up": 33, "down": 33, "flat": 34, "confidence": 5}


# ── Wire into WebSocket ─────────────────────────────────────────────────────

def patch_websocket_for_cvd():
    """
    Monkey-patch the Hyperliquid WebSocket to feed trades into the CVD engine.
    Call once after WebSocket starts.
    
    This watches the trades stream and calls ingest_trade() on every fill.
    """
    try:
        from hyperliquid_ws import _latest_data
        import threading
        
        last_seen = {}  # coin -> last trade timestamp to avoid duplicates
        
        def _cvd_watcher():
            while True:
                try:
                    trade_history = _latest_data.get("trade_history", {})
                    for coin, trades in trade_history.items():
                        if not trades:
                            continue
                        # last_seen stores the highest ms timestamp we've processed for this coin
                        cutoff_ts = last_seen.get(coin, 0)
                        newest_processed = cutoff_ts
                        for t in trades:
                            ts = int(t.get("ts", 0))
                            if ts <= cutoff_ts:
                                continue  # already processed
                            price = float(t.get("price", 0))
                            size = float(t.get("size", 0))
                            side = t.get("side", "A")
                            if price > 0 and size > 0:
                                ingest_trade(coin, price, size, side, ts / 1000.0)
                            if ts > newest_processed:
                                newest_processed = ts
                        if newest_processed > cutoff_ts:
                            last_seen[coin] = newest_processed
                except Exception:
                    pass
                time.sleep(0.5)  # Poll every 500ms
        
        t = threading.Thread(target=_cvd_watcher, daemon=True, name="cvd-watcher")
        t.start()
        return True
    except Exception as e:
        print(f"CVD watcher start failed: {e}")
        return False


# ── Performance Metrics ─────────────────────────────────────────────────────

# Track closed trades for performance calculation
_closed_trades: list[dict] = []
MAX_CLOSED = 100


def record_closed_trade(coin: str, pnl: float, pnl_pct: float, reason: str = ""):
    """Record a closed trade for performance metrics."""
    _closed_trades.append({
        "coin": coin, "pnl": pnl, "pnl_pct": pnl_pct,
        "reason": reason, "ts": time.time(),
    })
    if len(_closed_trades) > MAX_CLOSED:
        _closed_trades.pop(0)


def get_performance_metrics() -> dict:
    """Compute Sharpe-like ratio, win rate, total PnL from recorded trades."""
    trades = _closed_trades
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0,
                "avg_pnl_pct": 0, "sharpe_approx": 0, "best": 0, "worst": 0}
    
    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    pnl_pcts = [t["pnl_pct"] for t in trades]
    avg_pnl = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
    
    # Approximate Sharpe: mean(pnl%) / std(pnl%)
    if len(pnl_pcts) > 1:
        import math
        mean_p = sum(pnl_pcts) / len(pnl_pcts)
        variance = sum((p - mean_p) ** 2 for p in pnl_pcts) / (len(pnl_pcts) - 1)
        std_p = math.sqrt(variance)
        sharpe = mean_p / std_p if std_p > 0 else 0
    else:
        sharpe = 0
    
    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_pct": round(avg_pnl, 2),
        "sharpe_approx": round(sharpe, 2),
        "best": round(max(pnl_pcts), 2) if pnl_pcts else 0,
        "worst": round(min(pnl_pcts), 2) if pnl_pcts else 0,
    }


def performance_summary() -> str:
    """One-line performance summary for daemon log."""
    m = get_performance_metrics()
    if m["total_trades"] == 0:
        return "No closed trades yet"
    return (f"Trades:{m['total_trades']} WR:{m['win_rate']:.0f}% "
            f"PnL:${m['total_pnl']:+.2f} Sharpe:{m['sharpe_approx']:+.2f} "
            f"Best:{m['best']:+.1f}% Worst:{m['worst']:+.1f}%")


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)
    
    # Simulate trade stream
    base_price = 100.0
    for i in range(200):
        base_price += random.uniform(-0.5, 0.7)
        side = "B" if random.random() > 0.45 else "S"
        size = random.uniform(0.1, 5.0)
        ingest_trade("TEST", base_price, size, side)
    
    cvd = get_trade_cvd("TEST", 300)
    print(f"Real CVD: {cvd}")
    
    pred = cvd_to_prediction(cvd)
    print(f"CVD Prediction: ↑{pred['up']:.0f} ↓{pred['down']:.0f} →{pred['flat']:.0f} conf={pred['confidence']:.0f}%")
    
    # Performance metrics
    for i in range(10):
        record_closed_trade("TEST", random.uniform(-2, 5), random.uniform(-5, 10), "test")
    
    print(f"\nPerformance: {performance_summary()}")
    print("\n✓ CVD engine + performance metrics ready")
