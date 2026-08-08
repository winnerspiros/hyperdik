"""
Lightweight Backtesting Engine — takes a strategy function, runs over OHLCV,
produces equity curve, Sharpe, max DD, win rate, total return.
Single file, no DB, no pandas dependency (pure Python + numpy if available).
Compatible with 1h/5m OHLCV from Revolut X.

Usage:
    from backtest_engine import BacktestEngine

    def my_strategy(i, ohlcv):
        # i = current index, ohlcv = list of dicts with open/high/low/close/volume
        # Return 1.0 (buy), -1.0 (sell), 0.0 (hold), or None (no signal)
        if ohlcv[i]["close"] > ohlcv[i]["open"]:
            return 1.0
        return 0.0

    engine = BacktestEngine(initial_capital=1000.0, fee=0.0009)
    results = engine.run(ohlcv_data, my_strategy)
    print(results)
"""
import math

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class BacktestEngine:
    """Minimal vectorized-ish backtester. ~200 lines."""

    def __init__(self, initial_capital=1000.0, fee=0.0009, position_pct=1.0):
        self.initial_capital = initial_capital
        self.fee = fee
        self.position_pct = position_pct  # fraction of capital to risk per trade

    def run(self, ohlcv, strategy_fn, warmup=50, verbose=False):
        """
        Parameters
        ----------
        ohlcv : list[dict]
            Each dict has keys: open, high, low, close, volume, timestamp (str/int).
            Must be sorted oldest → newest.
        strategy_fn : callable
            fn(index, ohlcv_list) -> 1.0 (buy), -1.0 (sell), 0.0 (hold), None (skip)
        warmup : int
            Skip first N candles for indicator warmup.
        verbose : bool

        Returns
        -------
        dict with:
            - equity_curve : list[float]
            - trades : list[dict]
            - total_return_pct : float
            - sharpe_ratio : float
            - max_drawdown_pct : float
            - win_rate : float
            - num_trades : int
        """
        capital = self.initial_capital
        position = 0.0  # units of asset held
        equity_curve = []
        trades = []
        entry_price = 0.0

        for i in range(len(ohlcv)):
            candle = ohlcv[i]
            price = candle["close"]

            # Skip warmup
            if i < warmup:
                equity_curve.append(capital + position * price)
                continue

            signal = strategy_fn(i, ohlcv)

            # --- Process signals ---
            if signal is not None and signal > 0.5 and capital > 0:
                # BUY — use all available capital
                cost = capital * self.position_pct
                bought = (cost * (1 - self.fee)) / price if price > 0 else 0
                if bought > 0:
                    position += bought
                    entry_price = price
                    capital -= cost
                    if verbose:
                        print(f"  BUY  @ {price:.4f}  qty={bought:.6f}  cap={capital:.2f}")
                    trades.append({
                        "type": "buy", "price": price, "qty": bought,
                        "timestamp": candle.get("timestamp", i),
                        "capital_at_entry": capital,
                        "index": i,
                    })

            elif signal is not None and signal < -0.5 and position > 0:
                # SELL — close all
                proceeds = position * price * (1 - self.fee)
                pnl = proceeds - (trades[-1]["qty"] * trades[-1]["price"] if trades else 0)
                pnl_pct = (price - entry_price) / entry_price * 100 if entry_price else 0
                capital += proceeds
                if verbose:
                    print(f"  SELL @ {price:.4f}  qty={position:.6f}  cap={capital:.2f}  PnL={pnl_pct:+.2f}%")
                trades[-1]["exit_price"] = price
                trades[-1]["exit_index"] = i
                trades[-1]["pnl"] = pnl
                trades[-1]["pnl_pct"] = pnl_pct
                trades[-1]["exit_timestamp"] = candle.get("timestamp", i)
                position = 0.0
                entry_price = 0.0

            # Equity curve
            equity = capital + position * price
            equity_curve.append(equity)

        # Close any open position at last price
        if position > 0 and len(ohlcv) > 0:
            last_price = ohlcv[-1]["close"]
            proceeds = position * last_price * (1 - self.fee)
            capital += proceeds
            position = 0.0
            equity_curve[-1] = capital

        # --- Compute metrics ---
        total_return_pct = (capital - self.initial_capital) / self.initial_capital * 100

        # Sharpe ratio (annualized from daily returns or per-candle)
        if HAS_NUMPY:
            eq = np.array(equity_curve, dtype=float)
            returns = np.diff(eq) / eq[:-1]
            sharpe = self._sharpe(returns)
            max_dd = self._max_drawdown(eq)
        else:
            returns = []
            for j in range(1, len(equity_curve)):
                if equity_curve[j - 1] > 0:
                    returns.append(
                        (equity_curve[j] - equity_curve[j - 1]) / equity_curve[j - 1]
                    )
            sharpe = self._sharpe(returns)
            max_dd = self._max_drawdown_list(equity_curve)

        # Win rate from trades that have exit
        closed = [t for t in trades if "pnl" in t]
        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        win_rate = wins / len(closed) * 100 if closed else 0.0

        # Build per-trade PnL list for performance_metrics.py
        trade_pnls = [t.get("pnl", 0) for t in closed]

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "closed_trades": closed,
            "trade_pnls": trade_pnls,
            "final_capital": capital,
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate": round(win_rate, 1),
            "num_trades": len(closed),
        }

    def _sharpe(self, returns):
        if len(returns) < 2:
            return 0.0
        mu = float(np.mean(returns)) if HAS_NUMPY else sum(returns) / len(returns)
        sigma = float(np.std(returns, ddof=1)) if HAS_NUMPY else math.sqrt(
            sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
        )
        if sigma == 0:
            return 0.0
        # Assume ~96 candles/day for 15m, 24 for 1h, 6 for 4h
        periods_per_year = 24 * 365  # 1h candles
        return (mu / sigma) * math.sqrt(periods_per_year)

    def _max_drawdown(self, eq):
        """numpy max drawdown"""
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        return float(np.min(dd))

    def _max_drawdown_list(self, eq):
        """pure Python max drawdown"""
        peak = eq[0]
        max_dd = 0.0
        for val in eq:
            if val > peak:
                peak = val
            dd = (val - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
        return max_dd


# ── Simple helper to load OHLCV from CSV ────────────────────────────────────
def load_ohlcv_csv(path, date_col="timestamp", fmt="%Y-%m-%d %H:%M:%S"):
    """Load OHLCV CSV into list of dicts. No pandas needed."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": row.get(date_col, ""),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
    return rows


if __name__ == "__main__":
    # Quick self-test
    from datetime import datetime, timedelta
    import random
    random.seed(42)

    # Generate synthetic OHLCV
    ohlcv = []
    price = 100.0
    for i in range(500):
        change = random.uniform(-2, 2)
        price += change
        ohlcv.append({
            "timestamp": (datetime(2024, 1, 1) + timedelta(hours=i)).isoformat(),
            "open": price - change,
            "high": price + abs(change) * 0.5,
            "low": price - abs(change) * 0.5,
            "close": price,
            "volume": random.uniform(1000, 10000),
        })

    def mean_reversion(i, data):
        """Simple mean reversion: buy below 20-period SMA, sell above"""
        if i < 20:
            return 0.0
        sma = sum(d["close"] for d in data[i - 20 : i]) / 20
        price = data[i]["close"]
        if price < sma * 0.98:
            return 1.0
        elif price > sma * 1.02:
            return -1.0
        return 0.0

    engine = BacktestEngine(initial_capital=1000.0)
    results = engine.run(ohlcv, mean_reversion, warmup=20, verbose=True)
    print("\n=== RESULTS ===")
    for k, v in results.items():
        if k in ("equity_curve", "trades", "closed_trades", "trade_pnls"):
            print(f"  {k}: {type(v).__name__} [{len(v)} items]")
        else:
            print(f"  {k}: {v}")
    print("✅ Backtest engine self-test passed")