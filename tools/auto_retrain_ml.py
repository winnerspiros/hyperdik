#!/usr/bin/env python3
"""
Auto-retrain all 177 XGBoost models with recent 1h candle data.
Runs from cron or manual. Saves to data/ml_models/.
Daemon auto-reloads on mtime change — no restart needed.

Usage: python3 auto_retrain_ml.py [--coins BTC,ETH,SOL] [--limit N]
"""

import sys, os, json, time, argparse
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

def _fetch_candles(coin, interval="1h", limit=500):
    interval_ms = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
    for attempt in range(5):
        try:
            info = _get_info()
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - limit * interval_ms.get(interval, 3_600_000)
            candles = info.candles_snapshot(coin, interval, start_ms, end_ms)
            if isinstance(candles, list) and len(candles) >= 100:
                return candles
            return []
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (attempt + 1) * 5
                print(f"  {coin}: rate limited (attempt {attempt+1}/5), waiting {wait}s...")
                time.sleep(wait)
            elif "422" in err:
                return []  # Coin doesn't have this interval
            else:
                print(f"  {coin}: fetch failed: {e}")
                return []
    return []


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


def retrain_coin(coin, save_dir="data/ml_models"):
    from ml_predictor import DirectionPredictor

    raw = _fetch_candles(coin, "1h", 500)
    candles = _normalize_candles(raw)

    if len(candles) < 100:
        print(f"  {coin}: SKIP — only {len(candles)} candles")
        return None

    xgb = DirectionPredictor()
    result = xgb.train(candles, test_split=0.2)

    if "error" in result:
        print(f"  {coin}: FAIL — {result['error']}")
        return None

    acc = result.get("accuracy", 0)
    path = os.path.join(save_dir, f"{coin}_xgb.json")
    xgb.save(path)
    print(f"  {coin}: ✓ acc={acc:.1%} ({result.get('train_samples', 0)} samples) → {path}")
    return {"coin": coin, "accuracy": acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coins", type=str, default="", help="Comma-separated coin list (default: all from universe_coins.json)")
    p.add_argument("--limit", type=int, default=0, help="Max coins to train")
    p.add_argument("--delay", type=float, default=0.3, help="Delay between coins (default 0.3s)")
    args = p.parse_args()

    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    else:
        try:
            with open("data/universe_coins.json") as f:
                coins = json.load(f)
        except FileNotFoundError:
            print("ERROR: data/universe_coins.json not found. Use --coins.")
            sys.exit(1)

    if args.limit:
        coins = coins[:args.limit]

    print(f"Retraining {len(coins)} coins (delay={args.delay}s)...")
    start = time.time()
    results = {}
    trained = 0
    skipped = 0

    for i, coin in enumerate(coins):
        try:
            r = retrain_coin(coin)
            if r:
                results[coin] = r
                trained += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  {coin}: ERROR — {e}")
            skipped += 1
        # Rate-limit: short delay between coins
        if i < len(coins) - 1:
            time.sleep(args.delay)

    elapsed = time.time() - start
    print(f"\nDone: {trained} trained, {skipped} skipped in {elapsed:.0f}s")

    # Write manifest
    manifest = {
        "retrained_at": time.time(),
        "retrained_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coins_trained": trained,
        "coins_skipped": skipped,
        "models": {c: f"data/ml_models/{c}_xgb.json" for c in results},
    }
    with open("data/ml_models/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → data/ml_models/manifest.json")


if __name__ == "__main__":
    main()
