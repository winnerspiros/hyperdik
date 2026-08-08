"""
Revolut X Trader Self-Review — AI self-learning loop.
Queries the SQLite database to analyze past AI decisions by market condition,
calculates win rates, PnL patterns, and returns formatted self-review strings
for the AI to learn from.
"""
import sqlite3
import os
import math
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union

# ── Path helpers (mirrors trader_db.py) ──────────────────────────────────────

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "trader.db")


def _get_db() -> sqlite3.Connection:
    """Open a read-only connection for self-review queries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%" if val != 0 else "0%"

def _fmt_eur(val: float) -> str:
    return f"€{val:+.2f}"

def _fmt_pnl_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"({sign}{val:.1f}%)"

def _safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _extract_market_condition(raw: str) -> str:
    """
    Normalise the market_conditions column to a simple label.

    The column can be:
      - a plain string like "bullish", "bearish", "ranging"
      - a JSON dict with a "sentiment" list
      - NULL / empty
    """
    if not raw:
        return "unknown"
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        val = raw

    if isinstance(val, dict):
        sentiments = val.get("sentiment", val.get("conditions", []))
        combined = " ".join(s if isinstance(s, str) else str(s) for s in sentiments).lower()
        if any(w in combined for w in ("bearish", "bear", "downtrend", "declining", "sell")):
            return "bearish"
        if any(w in combined for w in ("bullish", "bull", "uptrend", "rising", "buy")):
            return "bullish"
        if any(w in combined for w in ("ranging", "neutral", "sideways", "consolidat")):
            return "ranging"
        return "mixed"
    if isinstance(val, str):
        return val.lower()
    return "unknown"


# ── Public API ───────────────────────────────────────────────────────────────

def get_ai_review_context(strategy: str = "day_trader") -> str:
    """
    Query the last 20 decisions for *strategy* and return a formatted
    self-review string for the AI prompt.

    Returns an empty string when there are no decisions yet.
    """
    conn = _get_db()
    try:
        rows = conn.execute("""
            SELECT direction, symbol, ai_reasoning, market_conditions,
                   price_entry, price_target, price_stop, size,
                   timestamp, cycle
            FROM decisions
            WHERE strategy = ?
            ORDER BY id DESC
            LIMIT 20
        """, (strategy,)).fetchall()
    finally:
        conn.close()

    if not rows:
        return ""

    # ── Augment each decision with PnL from the trades table ────────────
    decisions = [dict(r) for r in rows]
    for d in decisions:
        d["pnl_pct"] = _lookup_trade_pnl_pct(d["symbol"], d["direction"])

    n = len(decisions)

    # Win / loss counts
    wins = [d for d in decisions if d["pnl_pct"] is not None and d["pnl_pct"] > 0]
    losses = [d for d in decisions if d["pnl_pct"] is not None and d["pnl_pct"] <= 0]
    no_result = [d for d in decisions if d["pnl_pct"] is None]

    total_closed = len(wins) + len(losses)
    win_rate = _safe_ratio(len(wins), total_closed) if total_closed else 0.0

    avg_pnl = 0.0
    if total_closed:
        avg_pnl = sum(d["pnl_pct"] for d in wins + losses) / total_closed

    # Best / worst
    best = max(wins, key=lambda d: d["pnl_pct"]) if wins else None
    worst = min(losses, key=lambda d: d["pnl_pct"]) if losses else None

    # ── Group by market_conditions ──────────────────────────────────────
    groups: Dict[str, List[dict]] = {}
    for d in decisions:
        mc = d.get("market_conditions") or "unknown"
        mc_label = _extract_market_condition(mc) if isinstance(mc, str) else "unknown"
        groups.setdefault(mc_label, []).append(d)

    def _grade(wr: float) -> str:
        if wr >= 0.60:
            return "GOOD"
        elif wr >= 0.40:
            return "OK"
        else:
            return "POOR"

    def _suggestion(wr: float, mc: str) -> str:
        if mc == "bearish" and wr < 0.40:
            return "consider skipping bearish buys or reducing position size"
        if mc == "ranging" and wr < 0.40:
            return "avoid trading in ranging markets until conditions shift"
        if mc == "bullish" and wr < 0.50:
            return "tighten stop-losses and only take high-conviction entries"
        if wr < 0.40:
            return "reduce overall risk until win rate improves"
        return ""

    lines: List[str] = []
    lines.append(f"📊 SELF-REVIEW: {strategy}")
    lines.append(f"Last {n} decisions: {_fmt_pct(win_rate)} win rate | "
                 f"Avg PnL: {_fmt_eur(avg_pnl)}{' (trades with PnL data)' if total_closed else ''}")

    if best:
        lines.append(f"Best: {best['direction']} {best['symbol']} on "
                     f"{_format_timestamp(best['timestamp'])} {_fmt_pnl_pct(best['pnl_pct'])}")
    if worst:
        lines.append(f"Worst: {worst['direction']} {worst['symbol']} on "
                     f"{_format_timestamp(worst['timestamp'])} {_fmt_pnl_pct(worst['pnl_pct'])}")

    lines.append("")
    lines.append("By market condition:")

    suggestions = []
    for mc in sorted(groups.keys()):
        g = groups[mc]
        g_wins = [d for d in g if d["pnl_pct"] is not None and d["pnl_pct"] > 0]
        g_total = len([d for d in g if d["pnl_pct"] is not None])
        g_wr = _safe_ratio(len(g_wins), g_total) if g_total else 0.0
        grade = _grade(g_wr)
        tag = f" — {grade}"
        if grade == "POOR":
            tag += " ⚠️"
        elif grade == "GOOD":
            tag += " ✅"
        lines.append(f"  {mc.capitalize()}: {len(g_wins)}/{g_total} wins "
                     f"({_fmt_pct(g_wr)}){tag}")
        sug = _suggestion(g_wr, mc.lower())
        if sug:
            suggestions.append(f"  • {mc.capitalize()}: {sug}")

    if no_result:
        lines.append(f"  (no PnL data: {len(no_result)} decisions pending or unmatched)")

    if suggestions:
        lines.append("")
        lines.append("🔧 SUGGESTIONS:")
        lines.extend(suggestions)

    return "\n".join(lines)


def analyze_in_review(
    conn: sqlite3.Connection,
    decisions: Union[List[Dict], List[sqlite3.Row]],
) -> Dict[str, Union[int, float]]:
    """
    Deeper performance analysis of completed decisions.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to trader.db (used for trade-level lookups if needed).
    decisions : list[dict | sqlite3.Row]
        List of decision rows to analyze.  Each must have at least
        ``symbol``, ``direction``, and optionally ``market_conditions``.

    Returns
    -------
    dict with keys:
        overall_score         int    0–100
        bull_market_score     int    0–100
        bear_market_score     int    0–100
        ranging_market_score  int    0–100
        sharpe_ratio          float  annualised Sharpe (or 0.0)
        max_consecutive_losses int
        total_decisions       int
        pattern_notes         str    Human-readable pattern findings
    """
    # Default result for empty input
    empty: Dict = {
        "overall_score": 0,
        "bull_market_score": 50,
        "bear_market_score": 50,
        "ranging_market_score": 50,
        "sharpe_ratio": 0.0,
        "max_consecutive_losses": 0,
        "total_decisions": 0,
        "pattern_notes": "No decisions to analyze.",
    }
    if not decisions:
        return empty

    rows = [dict(d) for d in decisions]

    # Resolve PnL for each decision
    for d in rows:
        d["pnl_pct"] = _lookup_trade_pnl_conn(conn, d["symbol"], d["direction"])

    closed = [d for d in rows if d["pnl_pct"] is not None]
    total = len(closed)

    if total == 0:
        empty["total_decisions"] = len(rows)
        empty["pattern_notes"] = "No closed trades found among decisions."
        return empty

    # ── Sharpe ratio (annualised, risk-free ≃ 0) ────────────────────────
    pnls = [d["pnl_pct"] for d in closed]
    mean_pnl = sum(pnls) / total
    if total > 1:
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (total - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
    else:
        std_dev = 0.0
    sharpe = (mean_pnl / std_dev * math.sqrt(365)) if std_dev > 0 else 0.0

    # ── Max consecutive losses ──────────────────────────────────────────
    max_consec = 0
    cur_consec = 0
    for d in closed:
        if d["pnl_pct"] <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    # ── Win rate ────────────────────────────────────────────────────────
    wins = [d for d in closed if d["pnl_pct"] > 0]
    win_rate = _safe_ratio(len(wins), total)

    # ── Scores 0–100 ────────────────────────────────────────────────────
    #   Base score from win rate (0–60 pts)
    #   Bonus from Sharpe (0–20 pts)
    #   Penalty from max consecutive losses (0–20 pts)
    def _calculate_score(wr: float, sh: float, mcl: int, n: int) -> int:
        base = wr * 60
        # Sharpe bonus: cap at 20 for Sharpe >= 2.0
        sharpe_bonus = max(0, min(20, sh * 10))
        # Consecutive loss penalty
        consec_penalty = min(20, mcl * 5)
        # Small-sample penalty: when n < 5, discount confidence
        sample_factor = min(1.0, n / 5)
        score = (base + sharpe_bonus - consec_penalty) * sample_factor
        return max(0, min(100, int(round(score))))

    overall_score = _calculate_score(win_rate, sharpe, max_consec, total)

    # ── By market condition scores ──────────────────────────────────────
    def _score_for_mc(mc_label: str) -> int:
        mc_closed = [
            d for d in closed
            if _extract_market_condition(d.get("market_conditions") or "") == mc_label
        ]
        if not mc_closed:
            return 50  # neutral default when no data
        mc_wins = [d for d in mc_closed if d["pnl_pct"] > 0]
        mc_wr = _safe_ratio(len(mc_wins), len(mc_closed))
        mc_mcl = 0
        cur = 0
        for d in mc_closed:
            if d["pnl_pct"] <= 0:
                cur += 1
                mc_mcl = max(mc_mcl, cur)
            else:
                cur = 0
        mc_pnls = [d["pnl_pct"] for d in mc_closed]
        mc_mean = sum(mc_pnls) / len(mc_closed)
        mc_var = sum((p - mc_mean) ** 2 for p in mc_pnls) / (len(mc_closed) - 1) if len(mc_closed) > 1 else 0.0
        mc_std = math.sqrt(mc_var) if mc_var > 0 else 0.0
        mc_sharpe = (mc_mean / mc_std * math.sqrt(365)) if mc_std > 0 else 0.0
        return _calculate_score(mc_wr, mc_sharpe, mc_mcl, len(mc_closed))

    bull_score = _score_for_mc("bullish")
    bear_score = _score_for_mc("bearish")
    ranging_score = _score_for_mc("ranging")

    # ── Pattern detection: symbols with poor performance ────────────────
    pattern_notes = _detect_patterns(closed, wins)

    return {
        "overall_score": overall_score,
        "bull_market_score": bull_score,
        "bear_market_score": bear_score,
        "ranging_market_score": ranging_score,
        "sharpe_ratio": round(sharpe, 4),
        "max_consecutive_losses": max_consec,
        "total_decisions": len(rows),
        "pattern_notes": pattern_notes,
    }


# ── Internal helpers ────────────────────────────────────────────────────────

def _lookup_trade_pnl_pct(symbol: str, direction: str) -> Optional[float]:
    """
    Find the most recent closed trade for *symbol* with matching side and
    return its pnl_pct.
    """
    side_map = {"buy": "buy", "sell": "sell",
                 "BUY": "buy", "SELL": "sell",
                 "long": "buy", "short": "sell"}
    side = side_map.get(direction, direction.lower())
    conn = _get_db()
    try:
        row = conn.execute("""
            SELECT pnl_pct FROM trades
            WHERE symbol = ? AND side = ? AND pnl_pct IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 1
        """, (symbol, side)).fetchone()
        return row["pnl_pct"] if row else None
    finally:
        conn.close()


def _lookup_trade_pnl_conn(conn: sqlite3.Connection, symbol: str, direction: str) -> Optional[float]:
    """Same as above but uses an existing connection (for use inside analyze_in_review)."""
    side_map = {"buy": "buy", "sell": "sell",
                 "BUY": "buy", "SELL": "sell",
                 "long": "buy", "short": "sell"}
    side = side_map.get(direction, direction.lower())
    row = conn.execute("""
        SELECT pnl_pct FROM trades
        WHERE symbol = ? AND side = ? AND pnl_pct IS NOT NULL
        ORDER BY exit_time DESC
        LIMIT 1
    """, (symbol, side)).fetchone()
    return row["pnl_pct"] if row else None


def _format_timestamp(ts: str) -> str:
    """Short human-readable date from ISO timestamp."""
    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d %b")
    except (ValueError, TypeError):
        return ts[:10] if ts else "?"


def _detect_patterns(closed: List[dict], wins: List[dict]) -> str:
    """Detect recurring patterns in losing trades."""
    if not closed:
        return "No closed trades."

    losses = [d for d in closed if d["pnl_pct"] <= 0]
    if not losses:
        return "No losing trades — perfect record ❤️"

    notes: List[str] = []

    # By symbol
    from collections import Counter
    sym_total: Dict[str, List[dict]] = {}
    for d in closed:
        sym_total.setdefault(d["symbol"], []).append(d)

    for sym, sym_trades in sorted(sym_total.items()):
        sym_losses = [d for d in sym_trades if d["pnl_pct"] <= 0]
        if len(sym_losses) >= 3:
            wr = _safe_ratio(len([d for d in sym_trades if d["pnl_pct"] > 0]), len(sym_trades))
            notes.append(f"⚠️  {sym}: only {_fmt_pct(wr)} win rate across "
                         f"{len(sym_trades)} trades — consider avoiding")

    # By direction
    dir_total: Dict[str, List[dict]] = {}
    for d in closed:
        dir_total.setdefault(d.get("direction", "?").upper(), []).append(d)

    for dr, dr_trades in sorted(dir_total.items()):
        dr_losses = [d for d in dr_trades if d["pnl_pct"] <= 0]
        if len(dr_losses) >= 3:
            wr = _safe_ratio(len([d for d in dr_trades if d["pnl_pct"] > 0]), len(dr_trades))
            notes.append(f"⚠️  {dr} orders: {_fmt_pct(wr)} win rate ({len(dr_trades)} trades)")

    # Average loss magnitude
    if losses:
        avg_loss = sum(d["pnl_pct"] for d in losses) / len(losses)
        notes.append(f"📉 Avg loss magnitude: {avg_loss:.1f}%")

    # Win size vs loss size
    if wins and losses:
        avg_win = sum(d["pnl_pct"] for d in wins) / len(wins)
        avg_loss = sum(d["pnl_pct"] for d in losses) / len(losses)
        if avg_win > 0 and avg_loss < 0:
            ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            if ratio < 1.0:
                notes.append(f"⚠️  Avg win ({avg_win:.1f}%) < avg loss ({avg_loss:.1f}%): "
                             f"win/loss ratio {ratio:.2f}x — winners too small")
            elif ratio >= 2.0:
                notes.append(f"✅ Avg win ({avg_win:.1f}%) >> avg loss ({avg_loss:.1f}%): "
                             f"win/loss ratio {ratio:.2f}x — good risk management")

    if not notes:
        return "No significant patterns detected."

    return " | ".join(notes)


# ── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== get_ai_review_context() ===")
    print(get_ai_review_context() or "(empty — no decisions in DB)")
    print()

    # Also test analyze_in_review
    conn = _get_db()
    try:
        rows = conn.execute("""
            SELECT direction, symbol, market_conditions, timestamp
            FROM decisions
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()
        result = analyze_in_review(conn, rows)
        print("=== analyze_in_review() ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
    finally:
        conn.close()