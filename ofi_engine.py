#!/usr/bin/env python3
"""OFI engine — Order Flow Imbalance short-horizon alpha (Cont-Kukanov-Stoikov 2014).

dP ~= beta * OFI, with beta ~ 1/depth. OFI sums SIGNED top-book events over a
short window: bid-add / ask-cancel = +flow, ask-add / bid-cancel / market-sell = -flow.

Implementation notes for Hyperliquid (no L2 delta feed — snapshot diffing):
- WS l2Book snapshots arrive per tick; diff consecutive top-5 snapshots per coin.
- Bucket into 1s windows, keep a rolling 30s OFI series per coin.
- beta is calibrated per coin per hour by rolling OLS of 5s forward return on OFI
  (falls back to 1/depth heuristic when too few samples).
- get_ofi_signal(coin, mid): returns {ofi, beta, edge_bps, direction, confidence}.
  edge_bps = predicted next-5s move in bps = beta * OFI * 10000 / mid.
"""

from __future__ import annotations
import time
from collections import deque

# coin -> deque[(ts, bid_px, bid_sz, ask_px, ask_sz)] top-of-book snapshots
_book_snaps: dict[str, deque] = {}
# coin -> deque[(bucket_ts, ofi)] 1s OFI buckets, 60s window
_ofi_series: dict[str, deque] = {}
# coin -> list[(ofi, fwd_ret)] calibration samples (rolling, max 120)
_calib: dict[str, deque] = {}
_calib_hour: dict[str, int] = {}
_beta_cache: dict[str, tuple[float, float]] = {}  # coin -> (beta, ts)

SNAP_MAX = 40
OFI_WINDOW_S = 30
OFI_BUCKET_S = 1.0
CALIB_MAX = 120
BETA_TTL_S = 3600  # recalibrate hourly


def _lvl_px(lv) -> float:
    try:
        return float(lv.get("px", 0)) if isinstance(lv, dict) else float(lv[0])
    except Exception:
        return 0.0


def _lvl_sz(lv) -> float:
    try:
        return float(lv.get("sz", 0)) if isinstance(lv, dict) else float(lv[1])
    except Exception:
        return 0.0


def record_book_snapshot(coin: str, bids: list, asks: list) -> None:
    """Record one top-of-book snapshot; diffs against previous to build OFI."""
    if not bids or not asks:
        return
    bpx, bsz = _lvl_px(bids[0]), _lvl_sz(bids[0])
    apx, asz = _lvl_px(asks[0]), _lvl_sz(asks[0])
    if bpx <= 0 or apx <= 0:
        return
    now = time.time()
    cu = coin.upper()
    dq = _book_snaps.get(cu)
    if dq is None:
        dq = _book_snaps[cu] = deque(maxlen=SNAP_MAX)
    prev = dq[-1] if dq else None
    dq.append((now, bpx, bsz, apx, asz))
    if prev is None:
        return
    _, pbpx, pbsz, papx, pasz = prev
    # Cont-Kukanov event contribution at best level:
    # bid-size increase or ask-size decrease = buying flow (+); reverse = (-).
    # Price moves re-anchor the level: bid up = strong +, ask down = strong -.
    ofi = 0.0
    if bpx >= pbpx:
        ofi += bsz - (pbsz if bpx == pbpx else 0.0)
    else:
        ofi -= pbsz
    if apx <= papx:
        ofi -= asz - (pasz if apx == papx else 0.0)
    else:
        ofi += pasz
    odq = _ofi_series.get(cu)
    if odq is None:
        odq = _ofi_series[cu] = deque(maxlen=90)
    bucket = int(now / OFI_BUCKET_S)
    if odq and odq[-1][0] == bucket:
        ts, acc = odq[-1]
        odq[-1] = (ts, acc + ofi)
    else:
        odq.append((bucket, ofi))


def record_mid(coin: str, mid: float) -> None:
    """Record a mid print so 5s-forward returns can calibrate beta."""
    if mid <= 0:
        return
    cu = coin.upper()
    odq = _ofi_series.get(cu)
    if not odq:
        return
    now = time.time()
    cutoff = now - OFI_WINDOW_S
    recent = [(b, v) for b, v in odq if b * OFI_BUCKET_S >= cutoff]
    if not recent:
        return
    ofi_5s = sum(v for _, v in recent[-5:])
    # store (ofi, mid, ts); forward return computed when a later mid arrives
    cdq = _calib.get(cu)
    if cdq is None:
        cdq = _calib[cu] = deque(maxlen=CALIB_MAX)
    hour = int(now / 3600)
    if _calib_hour.get(cu) != hour:
        _calib_hour[cu] = hour
        cdq.clear()
        _beta_cache.pop(cu, None)
    cdq.append((ofi_5s, mid, now))
    # resolve samples older than 5s into (ofi, fwd_ret) pairs in place
    resolved = []
    for _entry in list(cdq):
        if len(_entry) != 3:
            continue  # already-resolved pair — skip, do not re-resolve
        o, m0, t0 = _entry
        if isinstance(m0, float) and now - t0 >= 5.0 and abs(m0) > 0:
            resolved.append((o, (mid - m0) / m0))
    # keep raw queue short; stash resolved on the deque as tuples with flag
    if len(resolved) > 4:
        cdq.clear()
        for pair in resolved[-CALIB_MAX:]:
            cdq.append(pair)


def _beta_ols(cu: str, depth_usd: float) -> float:
    """Rolling OLS beta = Cov(OFI, fwd_ret) / Var(OFI); fallback 1/depth."""
    cdq = _calib.get(cu) or deque()
    pairs = [e for e in cdq if len(e) == 2
             and isinstance(e[0], (int, float)) and isinstance(e[1], float)]
    if len(pairs) >= 10:
        n = len(pairs)
        mo = sum(o for o, _ in pairs) / n
        mr = sum(r for _, r in pairs) / n
        cov = sum((o - mo) * (r - mr) for o, r in pairs) / n
        var = sum((o - mo) ** 2 for o, _ in pairs) / n
        if var > 0:
            b = cov / var
            # clamp: beta must be positive and sane (negative = overfit noise)
            if b > 0:
                return min(b, 0.01)
    # fallback: Kyle-lambda heuristic beta ~ 1/depth (per-unit flow moves less in deep books)
    if depth_usd > 0:
        return min(1.0 / max(depth_usd, 1.0), 0.01)
    return 0.0


def get_ofi_signal(coin: str, mid: float, depth_usd: float = 0.0) -> dict:
    """Return {ofi, beta, edge_bps, direction, confidence} for a coin.

    edge_bps = predicted 5s move in bps. Positive = upward pressure.
    """
    cu = coin.upper()
    now = time.time()
    odq = _ofi_series.get(cu) or deque()
    cutoff = now - OFI_WINDOW_S
    ofi = sum(v for b, v in odq if b * OFI_BUCKET_S >= cutoff)
    cached = _beta_cache.get(cu)
    if cached and now - cached[1] < BETA_TTL_S:
        beta = cached[0]
    else:
        beta = _beta_ols(cu, depth_usd)
        _beta_cache[cu] = (beta, now)
    edge_bps = (beta * ofi / mid * 10000.0) if mid > 0 else 0.0
    direction = "bullish" if edge_bps > 1.0 else ("bearish" if edge_bps < -1.0 else "neutral")
    # confidence from |edge| size: 1bp = weak, 5bp+ = strong on a 5s horizon
    confidence = min(80.0, max(0.0, (abs(edge_bps) - 1.0) * 16.0))
    return {"ofi": ofi, "beta": beta, "edge_bps": edge_bps,
            "direction": direction, "confidence": round(confidence, 1)}


def get_ofi_context(coin: str, mid: float) -> str:
    """Compact AI context string."""
    try:
        s = get_ofi_signal(coin, mid)
        if s["direction"] == "neutral" or s["confidence"] <= 0:
            return ""
        return f"OFI:{s['direction']} edge={s['edge_bps']:+.1f}bp conf={s['confidence']:.0f}"
    except Exception:
        return ""
