"""
Kronos Integration Module for Hyperliquid Trading Bot.
Uses Kronos-mini (4.1M params) to predict OHLCV trajectories.
Replaces/augments AI decider direction and target signals.

Usage:
    from kronos_predictor import KronosSignal, predict_coins
    
    signals = predict_coins(['APE', 'INJ', 'AVAX'], candle_data)
    for sig in signals:
        print(f"{sig.coin}: {sig.direction} {sig.target_pct:+.2f}% stop={sig.stop_pct:.2f}%")
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Lazy-loaded globals
_predictor = None
_tokenizer = None
_model = None
_loaded = False
_load_error = None

KRONOS_LOOKBACK = 150    # Context candles (reduced from 400 for speed)
KRONOS_PRED_LEN = 12     # Predict 1 hour (12 x 5min)


@dataclass
class KronosSignal:
    coin: str
    direction: str          # "LONG", "SHORT", "HOLD"
    confidence: float       # 0-100
    target_pct: float       # Predicted max gain %
    stop_pct: float         # Recommended stop (based on predicted low)
    final_pct: float        # Predicted gain at end of horizon
    max_dd_pct: float       # Max predicted drawdown
    volatility: float       # Predicted price range / current price
    current_price: float
    predicted_price: float


def _ensure_loaded():
    """Lazy-load Kronos model (first call takes ~5s, subsequent calls instant)."""
    global _predictor, _tokenizer, _model, _loaded, _load_error
    
    if _loaded:
        return True
    if _load_error:
        return False
    
    try:
        # Add Kronos source to path
        kronos_dir = '/tmp/Kronos'
        if kronos_dir not in sys.path:
            sys.path.insert(0, kronos_dir)
        
        from model import Kronos, KronosTokenizer, KronosPredictor
        
        log.info("Loading Kronos tokenizer...")
        _tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
        
        log.info("Loading Kronos-mini model...")
        _model = Kronos.from_pretrained('NeoQuasar/Kronos-mini')
        
        _predictor = KronosPredictor(_model, _tokenizer, device='cpu', max_context=512)
        _loaded = True
        log.info(f"Kronos loaded: {sum(p.numel() for p in _model.parameters()):,} params")
        return True
        
    except Exception as e:
        _load_error = str(e)
        log.warning(f"Kronos load failed: {e}")
        return False


def _candles_to_df(candles: list, with_amount: bool = True) -> pd.DataFrame:
    """Convert Hyperliquid candle dicts to Kronos-compatible DataFrame."""
    rows = []
    for c in candles:
        rows.append({
            'open': float(c.get('o', c.get('open', 0))),
            'high': float(c.get('h', c.get('high', 0))),
            'low': float(c.get('l', c.get('low', 0))),
            'close': float(c.get('c', c.get('close', 0))),
            'volume': float(c.get('v', c.get('volume', 0))),
            'timestamp': pd.to_datetime(c.get('t', 0), unit='ms') if isinstance(c.get('t'), (int, float)) else pd.Timestamp.now(),
        })
    df = pd.DataFrame(rows)
    df['amount'] = df['volume'] * df['close']
    return df


def _extract_signal(coin: str, pred_df: pd.DataFrame, current_price: float) -> KronosSignal:
    """Extract trading signal from Kronos prediction DataFrame."""
    pred_close = pred_df['close'].values
    pred_high = pred_df['high'].values
    pred_low = pred_df['low'].values
    
    final_close = pred_close[-1]
    max_close = pred_close.max()
    min_close = pred_close.min()
    max_high = pred_high.max()
    min_low = pred_low.min()
    
    final_pct = (final_close - current_price) / current_price * 100
    max_gain_pct = (max_high - current_price) / current_price * 100
    max_dd_pct = (min_low - current_price) / current_price * 100
    range_pct = (max_high - min_low) / current_price * 100  # volatility
    
    # Direction determination
    if final_pct > 0.15:
        direction = "LONG"
    elif final_pct < -0.15:
        direction = "SHORT"
    else:
        direction = "HOLD"
    
    # Confidence: function of direction strength and consistency
    # Higher when: strong directional move, low variance in prediction
    close_std = np.std(pred_close) / current_price * 100
    if close_std > 0:
        direction_strength = abs(final_pct) / close_std
    else:
        direction_strength = 10
    
    # Confidence 0-100
    confidence = min(95, max(5, direction_strength * 15 + 20))
    
    # Target: use predicted max high with a buffer
    target_pct = max_gain_pct * 0.85  # 85% of predicted max as target
    
    # Stop: use predicted min low with a 10% buffer below
    stop_pct = abs(max_dd_pct) * 1.1
    
    return KronosSignal(
        coin=coin,
        direction=direction,
        confidence=round(confidence, 1),
        target_pct=round(target_pct, 2),
        stop_pct=round(stop_pct, 2),
        final_pct=round(final_pct, 2),
        max_dd_pct=round(max_dd_pct, 2),
        volatility=round(range_pct, 2),
        current_price=current_price,
        predicted_price=round(final_close, 6),
    )


def predict_single(coin: str, candles: list) -> Optional[KronosSignal]:
    """Predict for a single coin. Returns KronosSignal or None on failure."""
    if not _ensure_loaded():
        return None
    
    if len(candles) < KRONOS_LOOKBACK:
        log.warning(f"Kronos: {coin} has {len(candles)} candles, need {KRONOS_LOOKBACK}")
        return None
    
    try:
        df = _candles_to_df(candles)
        current_price = df['close'].iloc[-1]
        
        x_df = df.iloc[-KRONOS_LOOKBACK:][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_ts = df.iloc[-KRONOS_LOOKBACK:]['timestamp'].reset_index(drop=True)
        last_ts = df.iloc[-1]['timestamp']
        y_ts = pd.Series(pd.date_range(
            start=last_ts + pd.Timedelta(minutes=5),
            periods=KRONOS_PRED_LEN, freq='5min'
        ))
        
        pred_df = _predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=KRONOS_PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False
        )
        
        return _extract_signal(coin, pred_df, current_price)
        
    except Exception as e:
        log.warning(f"Kronos predict failed for {coin}: {type(e).__name__}: {e}")
        return None


def predict_batch(coin_candles: dict) -> dict:
    """Batch predict for multiple coins. {coin: candles} -> {coin: KronosSignal}."""
    if not _ensure_loaded():
        return {}
    
    results = {}
    coins = list(coin_candles.keys())
    
    # Filter coins with enough data
    valid_coins = []
    dfs_list, xts_list, yts_list = [], [], []
    
    for coin in coins:
        candles = coin_candles[coin]
        if len(candles) < KRONOS_LOOKBACK:
            continue
        
        try:
            df = _candles_to_df(candles)
            x_df = df.iloc[-KRONOS_LOOKBACK:][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
            x_ts = df.iloc[-KRONOS_LOOKBACK:]['timestamp'].reset_index(drop=True)
            last_ts = df.iloc[-1]['timestamp']
            y_ts = pd.Series(pd.date_range(
                start=last_ts + pd.Timedelta(minutes=5),
                periods=KRONOS_PRED_LEN, freq='5min'
            ))
            
            dfs_list.append(x_df)
            xts_list.append(x_ts)
            yts_list.append(y_ts)
            valid_coins.append(coin)
        except Exception as e:
            log.warning(f"Kronos prep failed for {coin}: {e}")
    
    if not valid_coins:
        return {}
    
    try:
        t0 = time.time()
        all_preds = _predictor.predict_batch(
            df_list=dfs_list, x_timestamp_list=xts_list, y_timestamp_list=yts_list,
            pred_len=KRONOS_PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False
        )
        elapsed = time.time() - t0
        log.info(f"Kronos batch: {len(valid_coins)} coins in {elapsed:.1f}s ({elapsed/len(valid_coins):.2f}s/coin)")
        
        for coin, pred_df in zip(valid_coins, all_preds):
            current_price = dfs_list[valid_coins.index(coin)]['close'].iloc[-1]
            results[coin] = _extract_signal(coin, pred_df, current_price)
            
    except Exception as e:
        log.warning(f"Kronos batch predict failed: {type(e).__name__}: {e}")
    
    return results
