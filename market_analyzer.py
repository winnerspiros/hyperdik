"""
Market Analyzer — technical analysis indicators built on Revolut X data
Uses numpy/pandas + ta library for calculations
"""
import numpy as np
import pandas as pd
import time
import json
from revolut_client import RevolutXClient, RevolutXError

class MarketAnalyzer:
    def __init__(self, client: RevolutXClient):
        self.client = client
    
    def get_candles_df(self, symbol, interval="1h", hours_back=72):
        """Fetch candles and return as DataFrame with indicators"""
        now = int(time.time() * 1000)
        since = now - (hours_back * 3600 * 1000)
        
        raw = self.client.get_candles(symbol, interval=interval, since=since, until=now)
        if "data" not in raw:
            return None
        
        df = pd.DataFrame(raw["data"])
        df["start"] = pd.to_datetime(df["start"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.sort_values("start")
        return df
    
    def calculate_indicators(self, df):
        """Add technical indicators to a candle DataFrame using TA-Lib"""
        if df is None or len(df) < 20:
            return df
        
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        opens = df["open"].values.astype(float)
        
        try:
            import talib
            
            # RSI (14-period) — proven winner
            df["rsi"] = talib.RSI(closes, timeperiod=14)
            df["rsi"] = df["rsi"].fillna(50)
            
            # ADX (Average Directional Index) — trend strength, used by ADXMomentum strat
            df["adx"] = talib.ADX(highs, lows, closes, timeperiod=14)
            df["plus_di"] = talib.PLUS_DI(highs, lows, closes, timeperiod=14)
            df["minus_di"] = talib.MINUS_DI(highs, lows, closes, timeperiod=14)
            
            # MACD — standard
            macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist
            
            # EMAs — trend direction
            df["ema_9"] = talib.EMA(closes, timeperiod=9)
            df["ema_21"] = talib.EMA(closes, timeperiod=21)
            df["ema_50"] = talib.EMA(closes, timeperiod=min(50, len(closes)))
            if len(closes) >= 200:
                df["ema_200"] = talib.EMA(closes, timeperiod=200)
            
            # Bollinger Bands
            bb_upper, bb_mid, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            df["bb_upper"] = bb_upper
            df["bb_mid"] = bb_mid
            df["bb_lower"] = bb_lower
            
            # ATR — volatility
            df["atr"] = talib.ATR(highs, lows, closes, timeperiod=14)
            
            # SAR (Parabolic SAR) — trend reversal, used by ADXMomentum
            df["sar"] = talib.SAR(highs, lows, acceleration=0.02, maximum=0.2)
            
            # MOM (Momentum) — rate of change
            df["mom"] = talib.MOM(closes, timeperiod=14)
            
            # STOCH (Stochastic Oscillator) — overbought/oversold
            slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd
            
            # WILLR (Williams %R) — alternative RSI
            df["willr"] = talib.WILLR(highs, lows, closes, timeperiod=14)
            
            # PPO (Percentage Price Oscillator) — MACD alternative
            df["ppo"] = talib.PPO(closes, fastperiod=12, slowperiod=26, matype=0)
            
            # OBV (On-Balance Volume) — volume confirmation
            df["obv"] = talib.OBV(closes, volumes)
            
            # Volume indicators
            df["volume_sma_20"] = talib.SMA(volumes, timeperiod=20)
            df["volume_ratio"] = volumes / df["volume_sma_20"].replace(0, 0.0001)
            
            # Price action
            df["price_change_pct"] = df["close"].pct_change().fillna(0)
            
            # Support/resistance
            lookback = min(20, len(closes))
            df["recent_high"] = pd.Series(highs).rolling(lookback).max().values
            df["recent_low"] = pd.Series(lows).rolling(lookback).min().values
            
        except ImportError:
            # Fallback to basic pandas calculations
            df = self._calculate_indicators_fallback(df, closes, highs, lows, volumes)
        
        return df
    
    def _calculate_indicators_fallback(self, df, closes, highs, lows, volumes):
        """Fallback when TA-Lib is not available"""
        # RSI (14-period)
        delta = pd.Series(closes).diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)
        
        # EMAs
        close_series = df["close"]
        df["ema_9"] = close_series.ewm(span=9).mean()
        df["ema_21"] = close_series.ewm(span=21).mean()
        df["ema_50"] = close_series.ewm(span=min(50, len(closes))).mean()
        if len(closes) >= 200:
            df["ema_200"] = close_series.ewm(span=200).mean()
        
        # MACD
        ema12 = close_series.ewm(span=12).mean()
        ema26 = close_series.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        sma20 = close_series.rolling(20).mean()
        std20 = close_series.rolling(20).std()
        df["bb_mid"] = sma20
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        
        # ATR
        tr = pd.DataFrame({
            "hl": highs - lows,
            "hc": np.abs(highs - np.roll(closes, 1)),
            "lc": np.abs(lows - np.roll(closes, 1)),
        }).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # Volume
        df["volume_sma_20"] = pd.Series(volumes).rolling(20).mean()
        df["volume_ratio"] = volumes / df["volume_sma_20"].replace(0, 0.0001)
        
        df["price_change_pct"] = df["close"].pct_change().fillna(0)
        
        lookback = min(20, len(closes))
        df["recent_high"] = pd.Series(highs).rolling(lookback).max().values
        df["recent_low"] = pd.Series(lows).rolling(lookback).min().values
        
        return df
    
    def analyze_market_snapshot(self, symbol="BTC-USD"):
        """Get a comprehensive market analysis for a symbol"""
        # Get ticker
        tickers = self.client.get_tickers([symbol])
        ticker = tickers["data"][0] if tickers.get("data") else None
        
        # Get order book
        ob = self.client.get_order_book(symbol, limit=10)
        
        # Get candles at multiple timeframes
        dfs = {}
        for tf in ["15m", "1h", "4h", "1d"]:
            hours = {"15m": 24, "1h": 72, "4h": 168, "1d": 720}
            df = self.get_candles_df(symbol, tf, hours[tf])
            if df is not None:
                dfs[tf] = self.calculate_indicators(df)
        
        # Current metrics
        result = {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
        }
        
        if ticker:
            result["bid"] = float(ticker["bid"])
            result["ask"] = float(ticker["ask"])
            result["mid"] = float(ticker["mid"])
            result["last_price"] = float(ticker["last_price"])
            result["spread"] = float(ticker["ask"]) - float(ticker["bid"])
            result["spread_pct"] = result["spread"] / result["mid"] * 100
        
        # Order book imbalance
        if ob.get("data"):
            asks = sum(float(l["q"]) for l in ob["data"].get("asks", []))
            bids = sum(float(l["q"]) for l in ob["data"].get("bids", []))
            total = asks + bids
            result["order_book_imbalance"] = (bids - asks) / max(total, 0.0001) if total > 0 else 0
        
        # Multi-timeframe analysis
        for tf, df in dfs.items():
            latest = df.iloc[-1] if len(df) > 0 else None
            if latest is not None:
                tf_key = tf
                result[f"{tf_key}_close"] = float(latest["close"])
                result[f"{tf_key}_rsi"] = float(latest.get("rsi", 50))
                result[f"{tf_key}_macd"] = float(latest.get("macd", 0))
                result[f"{tf_key}_macd_signal"] = float(latest.get("macd_signal", 0))
                result[f"{tf_key}_ema_9"] = float(latest.get("ema_9", 0))
                result[f"{tf_key}_ema_21"] = float(latest.get("ema_21", 0))
                result[f"{tf_key}_ema_50"] = float(latest.get("ema_50", 0))
                result[f"{tf_key}_bb_upper"] = float(latest.get("bb_upper", 0))
                result[f"{tf_key}_bb_lower"] = float(latest.get("bb_lower", 0))
                result[f"{tf_key}_atr"] = float(latest.get("atr", 0))
                result[f"{tf_key}_volume_ratio"] = float(latest.get("volume_ratio", 1))
                result[f"{tf_key}_change"] = float(latest.get("price_change_pct", 0))
                
                # Trend direction
                ema9 = latest.get("ema_9", 0)
                ema21 = latest.get("ema_21", 0)
                ema50 = latest.get("ema_50", 0)
                if ema9 and ema21 and ema50:
                    result[f"{tf_key}_trend"] = "bullish" if ema9 > ema21 > ema50 else ("bearish" if ema9 < ema21 < ema50 else "mixed")
        
        return result
    
    def get_market_opportunities(self, top_pairs=30):
        """Scan multiple pairs for trading opportunities"""
        pairs_raw = self.client.get_currency_pairs()
        active_pairs = [k for k, v in pairs_raw.items() 
                       if v.get("status") == "active" and k.endswith("/USD")]
        
        # Sort by volume potential (just take first N pairs)
        scan_pairs = active_pairs[:top_pairs]
        
        opportunities = []
        for pair in scan_pairs:
            try:
                sym = pair.replace("/", "-")
                snapshot = self.analyze_market_snapshot(sym)
                
                # Score the pair
                score = self._score_opportunity(snapshot)
                if score["total"] > 30:  # Minimum threshold
                    opportunities.append(score)
            except Exception as e:
                continue
        
        opportunities.sort(key=lambda x: x["total"], reverse=True)
        return opportunities[:10]
    
    def _score_opportunity(self, snapshot):
        """Score a trading opportunity based on technical analysis"""
        scores = {}
        direction = {}
        
        for tf in ["15m", "1h", "4h"]:
            s = 0
            rsi = snapshot.get(f"{tf}_rsi", 50)
            macd = snapshot.get(f"{tf}_macd", 0)
            macd_signal = snapshot.get(f"{tf}_macd_signal", 0)
            vol_ratio = snapshot.get(f"{tf}_volume_ratio", 1)
            change = snapshot.get(f"{tf}_change", 0)
            trend = snapshot.get(f"{tf}_trend", "mixed")
            
            # RSI signals
            if rsi < 30:  # Oversold
                s += 25
                direction[tf] = "buy"
            elif rsi > 70:  # Overbought
                s += 25
                direction[tf] = "sell"
            elif rsi < 40:  # Near oversold
                s += 15
                direction[tf] = "buy"
            elif rsi > 60:  # Near overbought
                s += 15
                direction[tf] = "sell"
            
            # MACD crossover
            if macd > macd_signal and macd - macd_signal > 0:
                s += 20
                if direction.get(tf) != "sell":
                    direction[tf] = "buy"
            elif macd < macd_signal and macd_signal - macd > 0:
                s += 20
                if direction.get(tf) != "buy":
                    direction[tf] = "sell"
            
            # Volume confirmation
            if vol_ratio > 1.5:
                s += 15
            elif vol_ratio > 1.2:
                s += 8
            
            # Trend alignment
            if trend == "bullish":
                if direction.get(tf) != "sell":
                    direction[tf] = "buy"
                s += 10
            elif trend == "bearish":
                if direction.get(tf) != "buy":
                    direction[tf] = "sell"
                s += 10
            
            scores[tf] = s
        
        # Determine overall direction (majority vote, weighted by timeframe)
        tf_weights = {"15m": 0.2, "1h": 0.3, "4h": 0.5}
        buy_score = sum(tf_weights[tf] for tf, d in direction.items() if d == "buy")
        sell_score = sum(tf_weights[tf] for tf, d in direction.items() if d == "sell")
        
        if buy_score > sell_score and buy_score > 0.4:
            overall = "buy"
        elif sell_score > buy_score and sell_score > 0.4:
            overall = "sell"
        else:
            overall = "neutral"
        
        total = sum(scores.values())
        return {
            "symbol": snapshot.get("symbol"),
            "price": snapshot.get("last_price"),
            "direction": overall,
            "total": total,
            "scores": scores,
            "rsi_1h": snapshot.get("1h_rsi", 50),
            "rsi_4h": snapshot.get("4h_rsi", 50),
            "trend_1h": snapshot.get("1h_trend", "mixed"),
            "trend_4h": snapshot.get("4h_trend", "mixed"),
            "volume_ratio": snapshot.get("1h_volume_ratio", 1),
            "spread_pct": snapshot.get("spread_pct", 0),
            "book_imbalance": snapshot.get("order_book_imbalance", 0),
            "atr_1h": snapshot.get("1h_atr", 0),
        }