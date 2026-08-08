"""
Train and persist ML models for Hyperliquid price prediction.
Uses historical candle data from Hyperliquid API.
Trains XGBoost direction classifier + calibrates LSTM weights.
Saves to data/ml_models/ for the daemon to load.

Run: python3 train_ml_models.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

os.makedirs("data/ml_models", exist_ok=True)

from hyperliquid.info import Info
from hyperliquid.utils import constants

_INFO = None

def _get_info():
    global _INFO
    if _INFO is None:
        _INFO = Info(constants.MAINNET_API_URL, skip_ws=True)
    return _INFO

def _fetch_candles_fast(coin, interval="1h", limit=500):
    interval_ms = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
    for attempt in range(5):
        try:
            info = _get_info()
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - limit * interval_ms.get(interval, 3_600_000)
            candles = info.candles_snapshot(coin, interval, start_ms, end_ms)
            if isinstance(candles, list):
                return candles
            print(f"  Unexpected response for {coin}: {type(candles)}")
            return []
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = (attempt + 1) * 4
                print(f"  Rate limited for {coin} (attempt {attempt+1}/5), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Candle fetch failed for {coin}: {e}")
                return []
    print(f"  All retries exhausted for {coin}")
    return []

TRADABLE_COINS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "DOT", "ADA", "DOGE", "XRP", "SUI"]
from ml_predictor import DirectionPredictor, LSTMPredictor, build_features


def _normalize_candles(candles):
    if not candles:
        return []
    first = candles[0]
    if "close" in first and "open" in first:
        return candles
    return [
        {
            "close": float(c.get("c", c.get("close", 0))),
            "open": float(c.get("o", c.get("open", 0))),
            "high": float(c.get("h", c.get("high", 0))),
            "low": float(c.get("l", c.get("low", 0))),
            "volume": float(c.get("v", c.get("volume", 0))),
        }
        for c in candles
    ]


def train_coin_models(coin):
    print(f"\n{'='*50}")
    print(f"Training {coin}...")

    raw_1h = _fetch_candles_fast(coin, "1h", 500)
    raw_15m = _fetch_candles_fast(coin, "15m", 500)

    candles_1h = _normalize_candles(raw_1h)
    candles_15m = _normalize_candles(raw_15m)

    if not candles_1h or len(candles_1h) < 100:
        print(f"  Not enough 1h candles for {coin} ({len(candles_1h) if candles_1h else 0}) -- skipping")
        return None

    print(f"  Training XGBoost on {len(candles_1h)} 1h candles...")
    xgb = DirectionPredictor()
    result = xgb.train(candles_1h, test_split=0.2)

    if "error" in result:
        print(f"  XGBoost training failed: {result['error']}")
    else:
        acc = result.get("accuracy", 0)
        print(f"  XGBoost trained: accuracy={acc:.1%} feature_count={result.get('feature_count', 0)}")
        xgb.save(f"data/ml_models/{coin}_xgb.json")
        print(f"  Saved: data/ml_models/{coin}_xgb.json")

    print(f"  Calibrating LSTM on {len(candles_15m)} 15m candles...")
    lstm = LSTMPredictor(seq_len=10, hidden_size=8)

    closes_15m = [float(c.get("close", c.get("c", 0))) for c in candles_15m]
    acc = 0.5
    total = 0
    if len(closes_15m) >= 50:
        correct = 0
        for i in range(20, len(closes_15m) - 1):
            window = candles_15m[max(0, i-10):i+1]
            if len(window) < 10:
                continue
            pred = lstm.predict_next(window)
            if "error" in pred:
                continue
            actual = closes_15m[i+1]
            pred_dir = pred["direction"]
            actual_dir = "up" if actual > closes_15m[i] else "down"
            if pred_dir == actual_dir:
                correct += 1
            total += 1

        if total > 0:
            acc = correct / total
            print(f"  LSTM calibrated: directional accuracy={acc:.1%} over {total} samples")
        else:
            print(f"  LSTM: no valid samples")

    lstm_weights = {
        "Wf": lstm.Wf.tolist(), "Wi": lstm.Wi.tolist(),
        "Wc": lstm.Wc.tolist(), "Wo": lstm.Wo.tolist(),
        "Uf": lstm.Uf.tolist(), "Ui": lstm.Ui.tolist(),
        "Uc": lstm.Uc.tolist(), "Uo": lstm.Uo.tolist(),
        "bf": lstm.bf.tolist(), "bi": lstm.bi.tolist(),
        "bc": lstm.bc.tolist(), "bo": lstm.bo.tolist(),
        "W_out": lstm.W_out.tolist(), "b_out": lstm.b_out.tolist(),
    }
    with open(f"data/ml_models/{coin}_lstm.json", "w") as f:
        json.dump(lstm_weights, f)
    print(f"  Saved: data/ml_models/{coin}_lstm.json")

    return {"coin": coin, "xgb_accuracy": result.get("accuracy", 0), "lstm_accuracy": acc}


def main():
    print("=" * 60)
    print("ML MODEL TRAINING PIPELINE")
    print("=" * 60)
    start = time.time()

    results = {}
    for coin in TRADABLE_COINS:
        try:
            r = train_coin_models(coin)
            if r:
                results[coin] = r
        except Exception as e:
            print(f"  {coin} FAILED: {e}")
        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE ({time.time() - start:.0f}s)")
    print(f"{'='*60}")
    for coin, r in sorted(results.items()):
        xgb_acc = r.get("xgb_accuracy", 0)
        lstm_acc = r.get("lstm_accuracy", 0.5)
        print(f"  {coin:6s}: XGBoost={xgb_acc:.1%}  LSTM={lstm_acc:.1%}")

    manifest = {
        "trained_at": time.time(),
        "coins": list(results.keys()),
        "models": {c: {"xgb": f"data/ml_models/{c}_xgb.json",
                        "lstm": f"data/ml_models/{c}_lstm.json"}
                   for c in results},
    }
    with open("data/ml_models/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved: data/ml_models/manifest.json")
    print(f"  {len(results)}/{len(TRADABLE_COINS)} coins trained")


if __name__ == "__main__":
    main()
