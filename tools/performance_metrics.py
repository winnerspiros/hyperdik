"""
Performance Analytics — standard trading metrics from trade PnL history.
Sharpe, Sortino, Calmar, Profit Factor, Win Rate, Max Drawdown, etc.
"""
import numpy as np
from typing import List, Dict, Optional


def calculate_metrics(pnls: List[float], pnl_pcts: Optional[List[float]] = None,
                      risk_free_rate: float = 0.0) -> Dict:
    """
    Calculate comprehensive performance metrics from trade PnLs.
    
    Args:
        pnls: list of PnL values (positive = win, negative = loss)
        pnl_pcts: optional list of PnL percentages
        risk_free_rate: annual risk-free rate (e.g. 0.05 for 5%)
        
    Returns:
        dict with all metrics
    """
    arr = np.array(pnls, dtype=float)
    if len(arr) == 0:
        return {"error": "no trades"}
    
    pcts = np.array(pnl_pcts, dtype=float) if pnl_pcts else arr / (np.max(np.abs(arr)) + 1)
    
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    n_total = len(arr)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_total if n_total > 0 else 0
    
    total_pnl = float(np.sum(arr))
    avg_pnl = float(np.mean(arr))
    avg_win = float(np.mean(wins)) if n_wins > 0 else 0
    avg_loss = float(np.mean(losses)) if n_losses > 0 else 0
    med_pnl = float(np.median(arr))
    std_pnl = float(np.std(arr)) if n_total > 1 else 0
    
    # Win/Loss ratio
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 999
    
    # Profit factor
    gross_profit = float(np.sum(wins)) if n_wins > 0 else 0
    gross_loss = abs(float(np.sum(losses))) if n_losses > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    # Max consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    cur_wins = 0
    cur_losses = 0
    for p in arr:
        if p > 0:
            cur_wins += 1
            cur_losses = 0
            max_consec_wins = max(max_consec_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_consec_losses = max(max_consec_losses, cur_losses)
    
    # Sharpe (annualized from daily returns approximation)
    if std_pnl > 0 and n_total > 1:
        # Assume each trade represents ~1 day of holding
        daily_return = avg_pnl / (np.mean(np.abs(arr)) + 1)
        daily_std = std_pnl / (np.mean(np.abs(arr)) + 1)
        sharpe = (daily_return - risk_free_rate / 365) / daily_std * np.sqrt(365)
    else:
        sharpe = 0
    
    # Sortino (downside deviation only)
    downside = pcts[pcts < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = float(np.mean(pcts) / np.std(downside)) * np.sqrt(365)
    else:
        sortino = 0
    
    # Calmar (return / max drawdown)
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak)
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0
    calmar = total_pnl / abs(max_dd) if max_dd != 0 else 0
    
    # Expected value of next trade
    edge = win_rate * avg_win + (1 - win_rate) * avg_loss if avg_loss != 0 else avg_win
    
    return {
        "total_trades": n_total,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate_pct": round(win_rate * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "median_pnl": round(med_pnl, 2),
        "std_pnl": round(std_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "max_drawdown": round(max_dd, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "edge_per_trade": round(edge, 2),
    }


def format_metrics_for_ai(metrics: Dict) -> str:
    """Format performance metrics as a short string for AI context"""
    if not metrics or "error" in metrics:
        return ""
    
    pf_emoji = "✅" if metrics["profit_factor"] >= 1.5 else ("⚠️" if metrics["profit_factor"] >= 1 else "❌")
    sharpe_emoji = "✅" if metrics["sharpe_ratio"] >= 1 else ("⚠️" if metrics["sharpe_ratio"] >= 0 else "❌")
    
    return (
        f"📊 PERFORMANCE: {metrics['total_trades']} trades | "
        f"{pf_emoji} PF {metrics['profit_factor']}x | "
        f"Win {metrics['win_rate_pct']}% | "
        f"{sharpe_emoji} Sharpe {metrics['sharpe_ratio']} | "
        f"PnL €{metrics['total_pnl']:+.2f} | "
        f"Avg €{metrics['avg_pnl']:+.2f}"
    )


def estimate_required_trades(trade_pnls: List[float],
                              confidence: float = 0.95) -> Dict:
    """
    Use bootstrap sampling to estimate how many trades needed
    for reliable performance estimation.
    """
    if len(trade_pnls) < 5:
        return {"required": 30, "current": len(trade_pnls), "message": "Need at least 30 trades for reliable stats"}
    
    arr = np.array(trade_pnls)
    n_bootstrap = 1000
    means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    
    ci_low = np.percentile(means, (1 - confidence) / 2 * 100)
    ci_high = np.percentile(means, (1 + confidence) / 2 * 100)
    
    return {
        "required": max(30, len(trade_pnls)),
        "current": len(trade_pnls),
        "mean_pnl": float(np.mean(arr)),
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "confidence_interval_pct": confidence * 100,
        "is_reliable": len(trade_pnls) >= 30,
    }


if __name__ == "__main__":
    # Example
    import json
    test_pnls = [10, -5, 15, -3, 8, -2, 12, -4, 6, -1, 20, -8, 5, -2, 3]
    test_pcts = [3.2, -1.5, 4.1, -0.8, 2.5, -0.6, 3.8, -1.2, 1.9, -0.3, 5.0, -2.1, 1.5, -0.5, 0.9]
    
    m = calculate_metrics(test_pnls, test_pcts)
    print(json.dumps(m, indent=2))
    print()
    print(format_metrics_for_ai(m))
    print()
    print(estimate_required_trades(test_pnls))