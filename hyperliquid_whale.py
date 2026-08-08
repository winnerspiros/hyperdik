#!/usr/bin/env python3
"""
Hyperliquid Whale Tracker v1 — Monitor trade stream for large wallets,
track performance, generate copy-trade signals on whale convergence.

Built from:
  - ref-perp-bot (CShear) — whale tracking, convergence, discovery
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

WHALE_TRADE_MIN_USD = 50_000       # Minimum USD to consider as whale
SIGNAL_SCORE_THRESHOLD = 0.6       # Whale must have 0.6+ score
CONVERGENCE_WINDOW = 300           # 5 minutes for convergence
CONVERGENCE_BOOST = 0.15           # Confidence boost for multi-whale
SCORE_DECAY_RATE = 0.95            # Daily decay for inactive whales
WHALE_DB_PATH = "data/whales.json"


# ============================================================
# State
# ============================================================

@dataclass
class WhaleWallet:
    """Tracked whale wallet."""
    address: str
    label: str = ""
    win_rate: float = 0.5
    total_pnl: float = 0.0
    trade_count: int = 0
    avg_return_pct: float = 0.0
    score: float = 0.3
    last_seen: float = 0.0
    first_seen: float = 0.0
    best_assets: list[str] = field(default_factory=list)  # Most profitable assets

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "WhaleWallet":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WhaleSignal:
    """Signal from whale activity."""
    symbol: str
    side: str  # "BUY" or "SELL"
    confidence: float
    reason: str
    whale_count: int
    total_size_usd: float
    timestamp: float


class WhaleTracker:
    """Tracks whale wallets and generates copy-trade signals."""

    def __init__(self, db_path: str = WHALE_DB_PATH):
        self.db_path = db_path
        self.wallets: dict[str, WhaleWallet] = {}
        self.asset_whale_trades: dict[str, list[dict]] = defaultdict(list)
        self.volume_tracker: dict[str, float] = defaultdict(float)
        self.trade_count_tracker: dict[str, int] = defaultdict(int)
        self.mid_prices: dict[str, float] = {}
        self.signals: list[WhaleSignal] = []
        self._load()

    def _load(self) -> None:
        """Load whale database from disk."""
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                data = json.load(f)
            for addr, wdata in data.get("wallets", {}).items():
                self.wallets[addr.lower()] = WhaleWallet.from_dict(wdata)

    def _save(self) -> None:
        """Save whale database to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {"wallets": {a: w.to_dict() for a, w in self.wallets.items()},
                "updated": time.time()}
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def update_prices(self, mids: dict[str, float]) -> None:
        """Update mid prices for notional calculations."""
        self.mid_prices = mids

    def process_trade(self, trade: dict) -> WhaleSignal | None:
        """Process a single trade from WebSocket.

        trade: {coin, side, px, sz, time, users: [buyer, seller]}
        """
        coin = trade.get("coin", "")
        price = float(trade.get("px", 0))
        size = float(trade.get("sz", 0))
        size_usd = price * size
        users = trade.get("users", [])

        if len(users) < 2 or size_usd < WHALE_TRADE_MIN_USD:
            return None

        buyer = users[0].lower()
        seller = users[1].lower()

        # Track volume for discovery
        self.volume_tracker[buyer] += size_usd
        self.volume_tracker[seller] += size_usd
        self.trade_count_tracker[buyer] += 1
        self.trade_count_tracker[seller] += 1

        # Check tracked whales
        ts = trade.get("time", time.time() * 1000) / 1000

        for addr, side in [(buyer, "BUY"), (seller, "SELL")]:
            if addr not in self.wallets:
                continue

            whale = self.wallets[addr]
            whale.last_seen = ts
            if whale.first_seen == 0:
                whale.first_seen = ts
            whale.trade_count += 1

            # Record for convergence
            self.asset_whale_trades[coin].append({
                "wallet": addr,
                "side": side,
                "size_usd": size_usd,
                "timestamp": ts,
                "score": whale.score,
            })

            # Prune old (>5 min)
            cutoff = ts - CONVERGENCE_WINDOW
            self.asset_whale_trades[coin] = [
                t for t in self.asset_whale_trades[coin]
                if t["timestamp"] > cutoff
            ]

            # Generate signal if score is high enough
            if whale.score >= SIGNAL_SCORE_THRESHOLD:
                signal = self._generate_signal(whale, coin, side, size_usd, ts)
                if signal:
                    self.signals.append(signal)
                    if len(self.signals) > 100:
                        self.signals = self.signals[-50:]
                    return signal

        return None

    def _generate_signal(self, whale: WhaleWallet, coin: str,
                         side: str, size_usd: float, ts: float) -> WhaleSignal | None:
        """Generate a copy-trade signal from whale activity."""
        size_factor = min(size_usd / 100_000, 1.5)
        confidence = min(whale.score * (0.5 + 0.3 * size_factor), 0.9)

        # Check for convergence (multiple whales same direction)
        convergence = self._check_convergence(coin, side)
        if convergence > 1:
            confidence = min(confidence + CONVERGENCE_BOOST, 0.95)

        if confidence < 0.5:
            return None

        label = whale.label or whale.address[:10] + "..."
        reason = (f"Whale {label} "
                  f"{'bought' if side == 'BUY' else 'sold'} "
                  f"${size_usd:,.0f} {coin}"
                  f"{f' ({convergence} whales converged)' if convergence > 1 else ''}")

        return WhaleSignal(
            symbol=coin,
            side=side,
            confidence=round(confidence, 3),
            reason=reason,
            whale_count=convergence,
            total_size_usd=size_usd,
            timestamp=ts,
        )

    def _check_convergence(self, coin: str, side: str) -> int:
        """Count distinct whales trading same direction recently."""
        recent = self.asset_whale_trades.get(coin, [])
        wallets_same_side = set()
        for t in recent:
            if t["side"] == side:
                wallets_same_side.add(t["wallet"])
        return len(wallets_same_side)

    def discover_whales(self) -> int:
        """Auto-discover new whale wallets from volume history."""
        discovered = 0
        for addr, volume in list(self.volume_tracker.items()):
            if addr in self.wallets:
                continue
            trade_count = self.trade_count_tracker.get(addr, 0)
            if volume > 500_000 and trade_count > 5:
                score = min(volume / 5_000_000, 0.7)
                self.wallets[addr] = WhaleWallet(
                    address=addr,
                    score=score,
                    trade_count=trade_count,
                    last_seen=time.time(),
                    first_seen=time.time(),
                )
                discovered += 1

        if discovered:
            self._save()
        return discovered

    def decay_scores(self) -> None:
        """Decay scores for inactive whales."""
        now = time.time()
        for addr, whale in self.wallets.items():
            days_inactive = (now - whale.last_seen) / 86400
            if days_inactive > 1:
                whale.score *= SCORE_DECAY_RATE ** days_inactive
                if whale.score < 0.1:
                    whale.score = 0.0

    def get_whale_context(self) -> str:
        """Build AI context string of recent whale activity."""
        active = [w for w in self.wallets.values() if w.last_seen > time.time() - 3600]
        if not active:
            return "No recent whale activity"

        top = sorted(active, key=lambda w: w.score, reverse=True)[:5]
        lines = ["RECENT WHALE ACTIVITY (last hour):"]
        for w in top:
            lines.append(f"  {w.label or w.address[:8]}... score={w.score:.2f} "
                        f"trades={w.trade_count} last={int((time.time()-w.last_seen)/60)}m ago")
        return "\n".join(lines)

    def get_recent_signals(self, max_age: float = 300) -> list[WhaleSignal]:
        """Get recent whale signals (last N seconds)."""
        cutoff = time.time() - max_age
        return [s for s in self.signals if s.timestamp > cutoff]
