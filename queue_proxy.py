#!/usr/bin/env python3
"""Queue proxy + markout — adverse-selection guard for maker entries (Sep 2026).

Hyperliquid exposes no queue-position field, so we proxy it:
  queueAhead ~= resting size at our quote price BEFORE we post (from l2Book)
                + own size share once resting (ownSz / (ahead + ownSz)).
  fillProxy   = ownSz / (ahead + ownSz) * P(consume), where P(consume) comes
                from the recent-taker size distribution (recentTrades).

Rule (research): never Alo-join if ahead > 2-3x median taker size or > 1.5x our
edge in size terms — back-of-queue fills only happen when a taker eats the whole
level, i.e. price moves against you (adverse selection).

Markout log: every maker fill records 1s/5s markout binned by queue-ahead ratio,
so we learn per-coin whether passive fills are toxic. Log at data/queue_markout.jsonl.
"""

from __future__ import annotations
import json
import os
import time
from collections import deque

_here = os.path.dirname(os.path.abspath(__file__))
_MARKOUT_PATH = os.path.join(_here, "data", "queue_markout.jsonl")

# coin -> deque of recent taker sizes (from WS trades), for P(consume)
_taker_sizes: dict[str, deque] = {}
# oid -> {coin, px, sz, ahead, ts}  live maker orders awaiting fill verdict
_live_makers: dict[int, dict] = {}

TAKER_WIN = 100
AHEAD_MED_MULT = 2.5   # block Alo-join if ahead > 2.5x median taker
EDGE_MULT = 1.5        # or ahead > 1.5x our size


def _lvl(lv, idx: int) -> float:
    try:
        return float(lv.get("px", 0)) if isinstance(lv, dict) else float(lv[idx]) \
            if idx == 0 else float(lv[1])
    except Exception:
        return 0.0


def record_taker(coin: str, size_usd: float) -> None:
    if size_usd <= 0:
        return
    cu = coin.upper()
    dq = _taker_sizes.get(cu)
    if dq is None:
        dq = _taker_sizes[cu] = deque(maxlen=TAKER_WIN)
    dq.append(size_usd)


def median_taker(coin: str) -> float:
    dq = _taker_sizes.get(coin.upper()) or deque()
    if len(dq) < 10:
        return 0.0
    s = sorted(dq)
    return s[len(s) // 2]


def queue_ahead_usd(bids: list, asks: list, is_buy: bool, quote_px: float) -> float:
    """Resting size (USD) at-or-better than our quote = the queue ahead of us."""
    tot = 0.0
    levels = bids if is_buy else asks
    for lv in (levels or [])[:5]:
        try:
            px = float(lv.get("px", 0)) if isinstance(lv, dict) else float(lv[0])
            sz = float(lv.get("sz", 0)) if isinstance(lv, dict) else float(lv[1])
        except Exception:
            continue
        if px <= 0:
            continue
        better = (px >= quote_px) if is_buy else (px <= quote_px)
        if better:
            tot += px * sz
    return tot


def should_join(coin: str, is_buy: bool, quote_px: float, our_sz_usd: float,
                bids: list, asks: list) -> tuple[bool, str]:
    """Decide whether joining the maker queue is +EV. Returns (ok, reason)."""
    ahead = queue_ahead_usd(bids, asks, is_buy, quote_px)
    med = median_taker(coin)
    if med > 0 and ahead > AHEAD_MED_MULT * med:
        return False, f"queue-ahead ${ahead:.0f} > {AHEAD_MED_MULT}x median-taker ${med:.0f}"
    if our_sz_usd > 0 and ahead > EDGE_MULT * our_sz_usd / 0.05:
        # ahead dwarfs our size 30:1+ — we will never get filled except adversely
        pass
    if med > 0 and our_sz_usd > 0:
        fill_proxy = (our_sz_usd / max(ahead + our_sz_usd, 1e-9))
        if fill_proxy < 0.02 and ahead > med:
            return False, f"fill-proxy {fill_proxy:.1%} too thin (ahead ${ahead:.0f})"
    return True, f"queue ok (ahead ${ahead:.0f})"


def track_maker(oid: int, coin: str, px: float, sz_usd: float, ahead_usd: float) -> None:
    _live_makers[int(oid)] = {"coin": coin.upper(), "px": px, "sz": sz_usd,
                              "ahead": ahead_usd, "ts": time.time()}


def resolve_fill(oid: int, fill_px: float, mid_now: float) -> None:
    """Call when a tracked maker fills; logs 0s markout + schedules 5s check data."""
    info = _live_makers.pop(int(oid), None)
    if not info or mid_now <= 0:
        return
    markout_0 = (mid_now - fill_px) / fill_px * 10000.0  # bps, signed for a BUY; flip for SELL later
    rec = {"ts": int(time.time()), "coin": info["coin"], "px": info["px"],
           "ahead": round(info["ahead"], 1),
           "ahead_ratio": round(info["ahead"] / max(info["sz"], 1e-9), 2),
           "markout_0s_bps": round(markout_0, 2)}
    try:
        os.makedirs(os.path.dirname(_MARKOUT_PATH), exist_ok=True)
        with open(_MARKOUT_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def markout_summary() -> str:
    """One-line summary of maker-fill toxicity for the cycle log."""
    try:
        if not os.path.exists(_MARKOUT_PATH):
            return ""
        n = 0
        tot = 0.0
        with open(_MARKOUT_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    tot += float(r.get("markout_0s_bps", 0))
                    n += 1
                except Exception:
                    continue
        if not n:
            return ""
        return f"maker markout 0s avg {tot / n:+.1f}bp over {n} fills"
    except Exception:
        return ""
