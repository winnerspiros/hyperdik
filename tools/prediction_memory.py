"""
Prediction Memory — lightweight persistence for AI predictions.
Stores predictions to disk, survives restarts, tracks outcomes.
"""
import json, os, time
from datetime import datetime, timezone

PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# In-memory cache (lazy-loaded, small footprint)
_cache = {}
_cache_loaded = False


def _load_all():
    """Lazy-load all predictions from disk once."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    for fname in os.listdir(PREDICTIONS_DIR):
        if fname.endswith("_advisory.json"):
            try:
                with open(os.path.join(PREDICTIONS_DIR, fname)) as f:
                    data = json.load(f)
                _cache[data.get("symbol", fname.split("_")[0])] = data
            except Exception:
                pass
    _cache_loaded = True


def save_prediction(coin: str, direction: str, target_pct: float, stop_pct: float,
                    confidence: float, hold_min: int, reason: str = "", 
                    entry_price: float = 0, source: str = "ai_picks"):
    """Save an AI prediction to disk for later outcome tracking."""
    pred_data = {
        "symbol": coin.upper(),
        "direction": direction,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "confidence": confidence,
        "hold_min": hold_min,
        "reason": (reason or "")[:200],
        "entry_price": entry_price,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "resolved": False,
        "outcome": None,
        "actual_pnl": None,
    }
    pred_file = os.path.join(PREDICTIONS_DIR, f"{coin.upper()}_advisory.json")
    try:
        with open(pred_file, 'w') as f:
            json.dump(pred_data, f, indent=2)
        _cache[coin.upper()] = pred_data
    except Exception:
        pass


def resolve_prediction(coin: str, pnl_pct: float):
    """Mark a prediction as resolved with its actual PnL outcome."""
    _load_all()
    coin = coin.upper()
    pred = _cache.get(coin)
    if not pred:
        pred_file = os.path.join(PREDICTIONS_DIR, f"{coin}_advisory.json")
        if os.path.exists(pred_file):
            try:
                with open(pred_file) as f:
                    pred = json.load(f)
            except Exception:
                return
    
    if pred and not pred.get("resolved"):
        was_correct = (pred.get("direction") == "long" and pnl_pct > 0) or \
                      (pred.get("direction") == "short" and pnl_pct > 0) or \
                      pnl_pct == 0  # Flat = inconclusive
        pred["resolved"] = True
        pred["resolved_at"] = datetime.now(timezone.utc).isoformat()
        pred["outcome"] = "correct" if was_correct else "wrong" if pnl_pct != 0 else "flat"
        pred["actual_pnl"] = round(pnl_pct, 4)
        
        pred_file = os.path.join(PREDICTIONS_DIR, f"{coin}_advisory.json")
        try:
            with open(pred_file, 'w') as f:
                json.dump(pred, f, indent=2)
            _cache[coin] = pred
        except Exception:
            pass


def get_prediction_stats():
    """Return summary stats: total predictions, accuracy, avg PnL."""
    _load_all()
    resolved = [p for p in _cache.values() if p.get("resolved")]
    if not resolved:
        return {"total": 0, "resolved": 0, "accuracy": 0, "avg_pnl": 0}
    
    correct = sum(1 for p in resolved if p.get("outcome") == "correct")
    total_pnl = sum(p.get("actual_pnl", 0) for p in resolved)
    
    return {
        "total": len(resolved),
        "resolved": len(resolved),
        "accuracy": round(correct / len(resolved) * 100, 1) if resolved else 0,
        "avg_pnl": round(total_pnl / len(resolved), 3) if resolved else 0,
    }
