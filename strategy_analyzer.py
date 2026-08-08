"""
Strategy Analyzer — provides AI with martingale, grid, and scaling strategies
Derived from: pionex_best_parameters (MartingaleBotTrailing), Crypto_bot (coin ranking)
Integrated into Revolut X bot's AI context for decision support.

NOT a hardcoded strategy — the AI sees this analysis and decides.
"""
import logging
import math
from datetime import datetime

log = logging.getLogger("StrategyAnalyzer")


def analyze_martingale(holdings, current_prices, pnl_summary, eur_balance):
    """
    Analyze if any position is a candidate for martingale scaling.
    Martingale: when a position is in loss, buying more at lower prices
    can reduce average entry and accelerate recovery.

    Reports to AI: which positions are X% down, optimal scale-in size,
    and recovery price targets.

    Based on MartingaleBotTrailing parameters:
    - max_buy_time: 7-9 rounds
    - next_buy_rate: 1.5-1.8x per round
    - start_buy_after_down_rate: 1.5% drop → start scaling
    - sell_after_down_rate: 0.2-0.3% pullback from peak → exit
    """
    if not holdings or not current_prices:
        return ""

    lines = []
    for coin, qty in holdings.items():
        if coin == "EUR":
            continue
        pnl = pnl_summary.get(coin, {})
        if isinstance(pnl, dict):
            pnl_pct = pnl.get("pnl_pct", 0)
        elif isinstance(pnl, (int, float)):
            pnl_pct = pnl
        else:
            continue

        entry_price = pnl.get("entry", 0) if isinstance(pnl, dict) else 0
        current_price = current_prices.get(coin, 0)
        value = qty * current_price if current_price else 0

        # Only analyze positions that are in loss
        if pnl_pct >= 0 or value < 5:
            continue

        # Calculate martingale scale-in
        down_pct = abs(pnl_pct)
        scale_rounds = min(int(down_pct / 1.5), 5)  # One round per 1.5% drop
        if scale_rounds >= 1:
            # Next scale-in amount (1.5x the current position value)
            next_scale = round(value * 1.5, 2)
            if next_scale <= eur_balance * 0.5:  # Don't risk more than 50% of EUR
                # Recovery price to break even after scaling
                avg_entry = (entry_price * qty + current_price * (next_scale / current_price)) / (qty + next_scale / current_price) if entry_price and current_price else 0
                recovery_target = round(avg_entry * 1.01, 6) if avg_entry else 0  # 1% above avg to profit

                lines.append(f"  {coin}: -{down_pct:.1f}% | scale-in {next_scale:.2f} EUR available | recover @ {recovery_target}")

    if not lines:
        return ""

    result = "📉 MARTINGALE SCALE-IN OPPORTUNITIES:\n" + "\n".join(lines)
    result += "\n  AI decides: scale in, hold, or cut loss based on ALL context."
    return result


def analyze_grid_setup(tickers, support_levels, eur_balance):
    """
    Analyze grid trading opportunities: can we place multiple limit orders
    at different price levels below current market?

    Based on the grid concept: buy at support, sell at resistance.
    The AI can decide to use a grid approach or not.
    """
    if not tickers or eur_balance < 20:
        return ""

    lines = []
    for t in tickers[:5]:
        sym = t.get("symbol", "")
        base = sym.split("/")[0]
        if not base:
            continue
        bid = float(t.get("bid", 0))
        if bid <= 0:
            continue

        support = support_levels.get(base, {})
        if not support:
            continue

        support_price = support.get("support", 0)
        distance = support.get("support_distance_pct", 0)
        if not support_price or distance <= 0:
            continue

        # Grid setup: 2 levels below current price
        grid_levels = []
        for i in range(1, 3):
            grid_price = round(bid * (1 - distance * i / 100), max(6, int(-math.log(bid, 10)) + 4))
            grid_levels.append(grid_price)

        if grid_levels:
            lines.append(f"  {base}: grid @ {grid_levels[0]}, {grid_levels[1]} | {distance:.1f}% spacing")

    if not lines:
        return ""

    result = "🔲 GRID OPPORTUNITIES:\n" + "\n".join(lines)
    result += "\n  AI decides: place grid orders, single limit, or wait."
    return result


def analyze_position_exits(pnl_summary, current_prices):
    """
    Analyze partial exit opportunities — sell in stages as price recovers.
    Based on MartingaleBotTrailing's tiered sell approach.

    When a position is in profit, suggest selling in 2-3 tranches
    rather than all at once, to capture more upside.
    """
    if not pnl_summary:
        return ""

    lines = []
    for coin, pnl in pnl_summary.items():
        if isinstance(pnl, dict):
            pnl_pct = pnl.get("pnl_pct", 0)
            entry = pnl.get("entry", 0)
            current = pnl.get("current", 0)
        elif isinstance(pnl, (int, float)):
            pnl_pct = pnl
            entry = current = 0
        else:
            continue

        if pnl_pct > 0 and entry > 0 and current > 0:
            # Tiered exits: sell 50% at +3%, 30% at +5%, 20% at +8%
            tier1 = round(entry * 1.03, 6)
            tier2 = round(entry * 1.05, 6)
            tier3 = round(entry * 1.08, 6)
            lines.append(f"  {coin}: +{pnl_pct:.1f}% | partial exits @ {tier1} (50%), {tier2} (30%), {tier3} (20%)")

    if not lines:
        return ""

    result = "🎯 PARTIAL EXIT PLAN:\n" + "\n".join(lines)
    result += "\n  AI decides: partial or full exit based on market outlook."
    return result


def recommend_coin_ranking(tickers, holdings, eur_balance):
    """
    Rank coins by volatility and volume — which ones are most tradeable.
    Based on Crypto_bot's coin selection approach.
    """
    if not tickers:
        return ""

    ranked = []
    for t in tickers[:15]:
        sym = t.get("symbol", "")
        base = sym.split("/")[0]
        if not base:
            continue
        bid = float(t.get("bid", 0))
        ask = float(t.get("ask", 0))
        vol = float(t.get("volume_24h", 0) or 0)
        chg = float(t.get("change_24h", 0) or 0)

        if bid <= 0:
            continue

        # Score: volatility + volume + tight spread
        spread_pct = ((ask - bid) / bid * 100) if bid > 0 and ask > 0 else 1
        vol_score = min(vol / 100000, 10)  # Cap at 10
        chg_score = min(abs(chg) / 2, 5)  # Cap at 5
        spread_score = max(0, 5 - spread_pct * 10)  # Penalize wide spreads
        total_score = vol_score + chg_score + spread_score

        ranked.append((total_score, base, sym, bid, chg))

    ranked.sort(reverse=True)

    lines = []
    for score, base, sym, price, chg in ranked[:5]:
        hold = "📌" if base in holdings else "   "
        lines.append(f"  {hold} {base:8s} score={score:.1f} price={price:.4f} 24h={chg:+.2f}%")

    if not lines:
        return ""

    result = "🏆 COIN RANKING (volatility+volume+spread):\n" + "\n".join(lines)
    result += "\n  AI decides: prioritize high-score coins for new trades."
    return result


def get_full_strategy_context(holdings, current_prices, pnl_summary, eur_balance, tickers, support_levels):
    """Get all strategy analysis for AI context"""
    parts = []
    parts.append(analyze_martingale(holdings, current_prices, pnl_summary, eur_balance))
    parts.append(analyze_grid_setup(tickers, support_levels, eur_balance))
    parts.append(analyze_position_exits(pnl_summary, current_prices))
    parts.append(recommend_coin_ranking(tickers, holdings, eur_balance))
    return "\n".join(p for p in parts if p)


if __name__ == "__main__":
    # Test
    test_holdings = {"XRP": 10, "ADA": 5}
    test_prices = {"XRP": 0.96, "ADA": 0.22}
    test_pnl = {"XRP": {"entry": 1.00, "current": 0.96, "pnl_pct": -4.0},
                "ADA": {"entry": 0.21, "current": 0.22, "pnl_pct": 4.8}}
    test_tickers = [{"symbol": "XRP/EUR", "bid": "0.96", "ask": "0.97", "volume_24h": "50000000", "change_24h": "-2.5"},
                    {"symbol": "ADA/EUR", "bid": "0.22", "ask": "0.221", "volume_24h": "30000000", "change_24h": "1.2"}]
    test_support = {"XRP": {"support": 0.94, "support_distance_pct": 2.1}}
    ctx = get_full_strategy_context(test_holdings, test_prices, test_pnl, 82.03, test_tickers, test_support)
    print(ctx)