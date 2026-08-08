#!/usr/bin/env python3
"""
Hyperliquid Funding Sniper v2 — Z-score funding anomalies + OI gate + macro VIX/SPY filter.

Strategies:
  1. Z-SCORE FUNDING (NEW) — Rolling z-score of each asset's own funding history.
     z > +1.5 → SHORT (funding elevated vs own history, likely to revert)
     z < -1.5 → LONG  (funding depressed, likely to revert)
     This is DYNAMIC — "extreme" is relative to each asset, not a fixed threshold.

  2. ABSOLUTE THRESHOLD (v1) — Fixed rate bands for harvesting.
     Range: 0.01%/hr – 0.0285%/hr (87% – 250% APR) → harvest
     Above 0.0285%/hr (>250% APR) → momentum-follow (death spiral)

  3. OI LIQUIDITY GATE — Only trade when OI ≥ 50% of 90-day rolling mean.
     Prevents trading illiquid or dying markets.

  4. VIX/SPY MACRO GATE — Flat when VIX > 30 or SPY 5-day drawdown > 5%.
     Risk-off during macro stress.

  5. MOMENTUM PENALTY — Reduce confidence if price already moved in harvesting direction.

Built from:
  - Crypto-PerpetualFutures (PietroC21) — z-score funding signal, OI gate, macro gate
  - ref-perp-bot (CShear) — funding sniper with momentum switch
  - master-confluence — funding crowding filter
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# Config
# ============================================================

FUNDING_THRESHOLD = 0.0001         # 0.01%/hr = ~87% APR — minimum to harvest
HIGH_FUNDING_THRESHOLD = 0.0003   # 0.03%/hr = ~263% APR — high confidence
MAX_FUNDING_RATE = 0.000285       # 0.0285%/hr = ~250% APR — beyond this, go WITH trend
MAX_CONFIDENCE = 0.70             # Max confidence from funding alone
MOMENTUM_PENALTY = 0.3            # Reduce confidence if price already moved
MOMENTUM_THRESHOLD = 0.01         # 1% price move triggers momentum filter
MOMENTUM_SIGNAL_CONFIDENCE = 0.50  # Confidence for momentum signals (death spiral)
POSITION_SIZE_USD = 20.0          # Flat size (doubled from $10 — matching new margin/lev)
FUNDING_DB_PATH = "data/funding_sniper_state.json"


# ============================================================
# State
# ============================================================

@dataclass
class FundingSignal:
    """Signal from funding sniper."""
    symbol: str
    side: str  # "BUY" or "SELL"
    mode: str  # "harvest" or "momentum"
    confidence: float
    reason: str
    funding_rate: float
    annual_rate_pct: float
    mark_price: float
    momentum_penalty: float = 0.0
    timestamp: float = 0.0


class FundingSniper:
    """Harvests funding rate payments during extreme rate periods."""

    def __init__(self, db_path: str = FUNDING_DB_PATH):
        self.db_path = db_path
        self.asset_ctx: dict[str, dict] = {}
        self.predicted_fundings: dict[str, float] = {}
        self.price_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.active_funding_positions: set[str] = set()
        self.signals: list[FundingSignal] = []
        self.funding_history: dict[str, list[dict]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                data = json.load(f)
            self.active_funding_positions = set(data.get("active_positions", []))
            self.predicted_fundings = data.get("predicted_fundings", {})

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump({
                "active_positions": list(self.active_funding_positions),
                "predicted_fundings": self.predicted_fundings,
                "updated": time.time(),
            }, f, indent=2)

    def update_context(self, asset: str, ctx: dict) -> None:
        """Update from WebSocket activeAssetCtx."""
        self.asset_ctx[asset] = ctx

        # Track price for momentum filter
        mark_px = float(ctx.get("markPx", 0))
        if mark_px > 0:
            now = time.time()
            self.price_history[asset].append((now, mark_px))
            # Keep last 2 hours
            cutoff = now - 7200
            self.price_history[asset] = [
                (t, p) for t, p in self.price_history[asset] if t > cutoff
            ]

        # Record funding + OI for z-score history
        funding = float(ctx.get("funding", 0))
        oi = float(ctx.get("openInterest", 0))
        if funding != 0 or oi > 0:
            self._record_history(asset, funding, mark_px, "ws_update", open_interest=oi)

    def update_predicted(self, predicted: list[dict]) -> None:
        """Update predicted funding rates from API query."""
        for item in predicted:
            coin = item.get("coin", "")
            rate = item.get("predicted_rate", 0)
            if coin:
                self.predicted_fundings[coin] = rate
        self._save()

    def _check_momentum(self, asset: str, harvesting_side: str) -> float:
        """Check if price already moved in the harvesting direction.
        Returns penalty (0 = no penalty, up to MOMENTUM_PENALTY)."""
        history = self.price_history.get(asset, [])
        if len(history) < 2:
            return 0.0

        oldest_price = history[0][1]
        newest_price = history[-1][1]
        if oldest_price <= 0:
            return 0.0

        price_change = (newest_price - oldest_price) / oldest_price

        # If harvesting SHORT (positive funding), price dropping = already moved
        # If harvesting LONG (negative funding), price rising = already moved
        if harvesting_side == "SELL" and price_change < -MOMENTUM_THRESHOLD:
            return MOMENTUM_PENALTY * min(abs(price_change) / MOMENTUM_THRESHOLD, 1.0)
        elif harvesting_side == "BUY" and price_change > MOMENTUM_THRESHOLD:
            return MOMENTUM_PENALTY * min(price_change / MOMENTUM_THRESHOLD, 1.0)

        return 0.0

    # ============================================================
    # Z-SCORE FUNDING SIGNAL
    # ============================================================

    def _funding_zscore(self, asset: str, lookback: int = 270) -> float:
        """Compute rolling z-score of funding rate vs its own history.

        z > +z_entry → funding elevated vs normal → SHORT (collect)
        z < -z_entry → funding depressed     → LONG  (collect)

        Uses last `lookback` funding rate values. Returns 0.0 if insufficient data.
        """
        history = self.funding_history.get(asset, [])
        rates = [h.get("rate", 0) for h in history[-lookback:]]
        if len(rates) < lookback // 2:
            return 0.0

        import statistics
        mu = statistics.mean(rates)
        sigma = statistics.stdev(rates) if len(rates) > 1 else 0.0
        if sigma == 0:
            return 0.0

        current = rates[-1]
        return (current - mu) / sigma

    # ============================================================
    # OI LIQUIDITY GATE
    # ============================================================

    def _oi_gate(self, asset: str, current_oi: float, lookback: int = 270,
                 min_ratio: float = 0.5) -> tuple[bool, str]:
        """Check if open interest is healthy relative to recent history.

        Returns (allowed, reason).
        """
        if current_oi <= 0:
            return False, "oi_zero"

        history = self.funding_history.get(asset, [])
        oi_values = [h.get("open_interest", 0) for h in history[-lookback:]
                      if h.get("open_interest", 0) > 0]

        if len(oi_values) < lookback // 2:
            return True, "insufficient_oi_history"  # Not enough data → allow

        import statistics
        oi_mean = statistics.mean(oi_values)
        if oi_mean <= 0:
            return True, "oi_mean_zero"

        ratio = current_oi / oi_mean
        if ratio < min_ratio:
            return False, f"oi_low:{ratio:.2f}<{min_ratio}"

        return True, "oi_ok"

    # ============================================================
    # VIX/SPY MACRO RISK GATE
    # ============================================================

    def _macro_gate(self, vix: float | None = None,
                    spy_drawdown: float | None = None,
                    vix_threshold: float = 30.0,
                    spy_dd_threshold: float = 0.05) -> tuple[bool, str]:
        """Macro risk gate: go flat during macro stress.

        Flat when EITHER:
          - VIX > vix_threshold (30 default)
          - SPY drawdown > spy_dd_threshold (5% default)

        Returns (risk_on, reason).
        """
        if vix is not None and vix > vix_threshold:
            return False, f"vix_high:{vix:.1f}"

        if spy_drawdown is not None and spy_drawdown > spy_dd_threshold:
            return False, f"spy_dd:{spy_drawdown*100:.1f}%"

        return True, "macro_ok"

    # ============================================================
    # FULL EVALUATION (z-score + absolute + gates)
    # ============================================================

    def evaluate(self, mids: dict[str, float]) -> list[FundingSignal]:
        """Evaluate all assets for funding opportunities."""
        signals = []
        now = time.time()

        # Evaluate all assets with predicted or WebSocket data
        candidates = set(self.predicted_fundings.keys()) | set(self.asset_ctx.keys())

        for asset in candidates:
            ctx = self.asset_ctx.get(asset, {})
            predicted_rate = self.predicted_fundings.get(asset, 0)
            current_funding = float(ctx.get("funding", 0))
            mark_price = float(ctx.get("markPx", 0)) if ctx else mids.get(asset, 0)
            open_interest = float(ctx.get("openInterest", 0)) if ctx else 0

            rate = predicted_rate if predicted_rate != 0 else current_funding
            if abs(rate) < FUNDING_THRESHOLD:
                continue

            annual_rate = rate * 24 * 365 * 100

            # Death spiral check: above MAX_FUNDING_RATE → go WITH momentum
            if abs(rate) > MAX_FUNDING_RATE:
                momentum_side = "BUY" if rate > 0 else "SELL"
                sig = FundingSignal(
                    symbol=asset,
                    side=momentum_side,
                    mode="momentum",
                    confidence=MOMENTUM_SIGNAL_CONFIDENCE,
                    reason=f"Momentum on {asset}: {rate*100:.4f}%/hr (~{annual_rate:.0f}% APR). "
                           f"{'Longing' if momentum_side == 'BUY' else 'Shorting'} with trend.",
                    funding_rate=rate,
                    annual_rate_pct=round(annual_rate, 1),
                    mark_price=mark_price,
                    timestamp=now,
                )
                signals.append(sig)
                self._record_history(asset, rate, mark_price, "momentum", open_interest=open_interest)
                continue

            # === Harvest mode ===
            # Positive funding = longs pay shorts → go SHORT
            # Negative funding = shorts pay longs → go LONG
            if rate > 0:
                harvesting_side = "SELL"
            else:
                harvesting_side = "BUY"

            magnitude = abs(rate)
            if magnitude >= HIGH_FUNDING_THRESHOLD:
                confidence = MAX_CONFIDENCE
            else:
                ratio = (magnitude - FUNDING_THRESHOLD) / (HIGH_FUNDING_THRESHOLD - FUNDING_THRESHOLD)
                confidence = 0.4 + ratio * (MAX_CONFIDENCE - 0.4)
                confidence = min(confidence, MAX_CONFIDENCE)

            # Momentum penalty
            penalty = self._check_momentum(asset, harvesting_side)
            confidence -= penalty

            if confidence < 0.3:
                continue

            # Skip if already in a funding position
            if asset in self.active_funding_positions:
                continue

            sig = FundingSignal(
                symbol=asset,
                side=harvesting_side,
                mode="harvest",
                confidence=round(confidence, 3),
                reason=f"Funding harvest {asset}: {rate*100:.4f}%/hr (~{annual_rate:.0f}% APR). "
                       f"{'Shorting' if harvesting_side == 'SELL' else 'Longing'} to collect."
                       f"{' [penalized]' if penalty > 0 else ''}",
                funding_rate=rate,
                annual_rate_pct=round(annual_rate, 1),
                mark_price=mark_price,
                momentum_penalty=round(penalty, 3),
                timestamp=now,
            )
            signals.append(sig)
            self._record_history(asset, rate, mark_price, "harvest", open_interest=open_interest)

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        self.signals = signals
        return signals

    def _record_history(self, asset: str, rate: float, price: float, mode: str,
                        open_interest: float = 0.0) -> None:
        """Record funding snapshot for history."""
        self.funding_history[asset].append({
            "time": time.time(),
            "rate": rate,
            "price": price,
            "mode": mode,
            "open_interest": open_interest,
        })
        # Keep last 500 (enough for 270-period z-score lookback)
        if len(self.funding_history[asset]) > 500:
            self.funding_history[asset] = self.funding_history[asset][-500:]

    def evaluate_zscore(self, assets: list[str], mids: dict[str, float],
                        z_entry: float = 1.5, min_z_confidence: float = 0.35,
                        vix: float | None = None,
                        spy_drawdown: float | None = None) -> list[FundingSignal]:
        """Evaluate using z-score funding anomalies.

        For each asset, compute the z-score of current funding rate vs its own
        history. Enter when |z| > z_entry. This is more robust than fixed
        thresholds because "extreme" is relative to each asset's normal range.

        Also applies OI gate and macro gate.
        """
        signals = []
        now = time.time()

        # Macro gate
        risk_on, macro_reason = self._macro_gate(vix, spy_drawdown)
        if not risk_on:
            return signals  # No funding trades during macro stress

        for asset in assets:
            z = self._funding_zscore(asset)
            if abs(z) < z_entry:
                continue

            ctx = self.asset_ctx.get(asset, {})
            current_funding = float(ctx.get("funding", 0))
            mark_price = float(ctx.get("markPx", 0)) if ctx else mids.get(asset, 0)
            open_interest = float(ctx.get("openInterest", 0)) if ctx else 0
            annual_rate = abs(current_funding) * 24 * 365 * 100

            # Skip already active
            if asset in self.active_funding_positions:
                continue

            # OI gate
            oi_allowed, oi_reason = self._oi_gate(asset, open_interest)
            if not oi_allowed:
                continue

            # Determine side: z > 0 = elevated funding → short, z < 0 = depressed → long
            if z > z_entry:
                side = "SELL"
            elif z < -z_entry:
                side = "BUY"
            else:
                continue

            # Confidence from z-score magnitude
            abs_z = abs(z)
            if abs_z > 3.0:
                confidence = 0.70
            elif abs_z > 2.0:
                confidence = 0.55 + (abs_z - 2.0) * 0.15
            else:
                confidence = 0.35 + (abs_z - z_entry) * 0.20

            # Momentum penalty
            penalty = self._check_momentum(asset, side)
            confidence -= penalty

            if confidence < min_z_confidence:
                continue

            sig = FundingSignal(
                symbol=asset,
                side=side,
                mode="zscore",
                confidence=round(confidence, 3),
                reason=f"Funding z-score {asset}: z={z:+.2f} "
                       f"rate={current_funding*100:.4f}%/hr (~{annual_rate:.0f}% APR). "
                       f"{'Shorting' if side == 'SELL' else 'Longing'} to collect."
                       f"{' [penalized]' if penalty > 0 else ''}",
                funding_rate=current_funding,
                annual_rate_pct=round(annual_rate, 1),
                mark_price=mark_price,
                momentum_penalty=round(penalty, 3),
                timestamp=now,
            )
            signals.append(sig)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def register_position(self, asset: str) -> None:
        """Register that we've entered a funding position."""
        self.active_funding_positions.add(asset)
        self._save()

    def close_position(self, asset: str) -> None:
        """Register that a funding position was closed."""
        self.active_funding_positions.discard(asset)
        self._save()

    def get_funding_context(self) -> str:
        """Build AI context string of top funding opportunities."""
        if not self.signals:
            return "No extreme funding rates detected"

        lines = ["EXTREME FUNDING RATES (top opportunities):"]
        for s in self.signals[:5]:
            lines.append(
                f"  {s.symbol}: {s.funding_rate*100:.4f}%/hr "
                f"({s.annual_rate_pct:.0f}% APR) → {s.side} "
                f"[{s.mode}] conf={s.confidence:.2f}"
            )
        return "\n".join(lines)

    def get_top_harvest(self, limit: int = 3) -> list[FundingSignal]:
        """Get top harvest signals (non-momentum, enough confidence)."""
        harvest = [s for s in self.signals if s.mode == "harvest" and s.confidence >= 0.4]
        return harvest[:limit]
