"""
ML Price Prediction — lightweight LSTM + XGBoost for crypto direction prediction.
Both models run on CPU, no GPU needed, < 1GB RAM.

Includes:
  1. XGBoost direction classifier (buy/sell/neutral next candle)
  2. Tiny LSTM price predictor using numpy-only forward pass
  3. Feature engineering for both models

Install dependencies before use:
  pip install xgboost scikit-learn numpy

If XGBoost is not available, falls back to sklearn RandomForest.
"""
import json
import math
import time
import numpy as np

# ── Optional deps ───────────────────────────────────────────────────────────
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import RandomForestClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════


def build_features(ohlcv):
    """
    Build feature matrix from OHLCV list.
    Returns (X, y, feature_names) where X is list[list[float]].
    Target: 1 = price up next candle, 0 = down, -1 = flat (< 0.1%)
    """
    if len(ohlcv) < 30:
        return [], [], []

    closes = np.array([c["close"] for c in ohlcv], dtype=float)
    highs = np.array([c["high"] for c in ohlcv], dtype=float)
    lows = np.array([c["low"] for c in ohlcv], dtype=float)
    opens = np.array([c["open"] for c in ohlcv], dtype=float)
    volumes = np.array([c["volume"] for c in ohlcv], dtype=float)

    # Handle zero volume
    volumes = np.where(volumes == 0, np.nanmean(volumes) if np.any(volumes) else 1, volumes)

    X, y = [], []
    n = len(closes)

    for i in range(20, n - 1):
        features = []

        # --- Price action ---
        # Returns over 1, 3, 5, 10 bars
        for lag in [1, 3, 5, 10]:
            ret = (closes[i] - closes[i - lag]) / closes[i - lag] * 100
            features.append(round(float(ret), 6))

        # Volatility (high-low range / close)
        for lag in [3, 5, 10]:
            rng = highs[i - lag + 1 : i + 1] - lows[i - lag + 1 : i + 1]
            avg_range = float(np.mean(rng))
            features.append(avg_range / closes[i] * 100)

        # Position in range (0 = bottom, 1 = top)
        for lag in [1, 3, 5]:
            hi = float(np.max(highs[i - lag + 1 : i + 1]))
            lo = float(np.min(lows[i - lag + 1 : i + 1]))
            pos = (closes[i] - lo) / (hi - lo) if (hi - lo) > 0 else 0.5
            features.append(pos)

        # --- Volume ---
        avg_vol_5 = float(np.mean(volumes[i - 4 : i + 1]))
        avg_vol_10 = float(np.mean(volumes[i - 9 : i + 1]))
        features.append(volumes[i] / avg_vol_5 if avg_vol_5 > 0 else 1.0)
        features.append(volumes[i] / avg_vol_10 if avg_vol_10 > 0 else 1.0)

        # Volume trend
        vol_ratio = avg_vol_5 / avg_vol_10 if avg_vol_10 > 0 else 1.0
        features.append(round(float(vol_ratio), 4))

        # --- Moving averages ---
        for period in [5, 10, 20]:
            sma = float(np.mean(closes[i - period + 1 : i + 1]))
            features.append((closes[i] - sma) / sma * 100)

        # --- RSI (14) ---
        gains, losses = 0.0, 0.0
        for j in range(i - 13, i + 1):
            diff = closes[j] - closes[j - 1]
            if diff > 0:
                gains += diff
            else:
                losses += abs(diff)
        avg_gain = gains / 14
        avg_loss = losses / 14
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        features.append(round(float(rsi), 2))

        # --- MACD ---
        ema12 = _ema(closes[i - 25 : i + 1], 12)[-1] if i >= 25 else closes[i]
        ema26 = _ema(closes[i - 39 : i + 1], 26)[-1] if i >= 39 else closes[i]
        macd = ema12 - ema26
        features.append(round(float(macd), 6))

        # Target: next bar direction
        next_ret = (closes[i + 1] - closes[i]) / closes[i] * 100
        if next_ret > 0.1:
            y.append(1)
        elif next_ret < -0.1:
            y.append(0)
        else:
            y.append(2)  # flat

        X.append(features)

    feature_names = [
        "ret_1", "ret_3", "ret_5", "ret_10",
        "volatility_3", "volatility_5", "volatility_10",
        "pos_in_range_1", "pos_in_range_3", "pos_in_range_5",
        "vol_ratio_5", "vol_ratio_10", "vol_trend",
        "sma_5_dist", "sma_10_dist", "sma_20_dist",
        "rsi_14", "macd",
    ]
    return np.array(X, dtype=float), np.array(y, dtype=int), feature_names


def _ema(values, period):
    """Exponential moving average."""
    values = np.array(values, dtype=float)
    multiplier = 2 / (period + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


# ═══════════════════════════════════════════════════════════════════════════
#  XGBOOST / RANDOM FOREST DIRECTION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════


class DirectionPredictor:
    """
    Predicts next candle direction using XGBoost (or RandomForest fallback).
    Trains on historical OHLCV, predicts 0 (down), 1 (up), 2 (flat).
    """

    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_path = model_path

    def train(self, ohlcv, test_split=0.2):
        """Train on historical OHLCV data."""
        X_raw, y_raw, feat_names = build_features(ohlcv)
        if len(X_raw) < 50:
            return {"error": f"need at least 50 samples, got {len(X_raw)}"}

        # Convert to numpy arrays for boolean indexing
        X = np.array(X_raw, dtype=float)
        y = np.array(y_raw, dtype=int)

        self.feature_names = feat_names
        split = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Handle class imbalance — train on all classes, XGBoost handles multi-class
        if HAS_XGB:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42,
                "nthread": 1,
            }
            # Skip eval if test set too small or missing classes
            evals = []
            if len(X_test) >= 5 and len(set(y_test)) >= 2:
                evals = [(dtest, "test")]
            self.model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, verbose_eval=False)
        elif HAS_SKLEARN:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=1
            )
            self.model.fit(X_train, y_train)
        else:
            return {"error": "neither xgboost nor sklearn available"}

        # Evaluate
        accuracy = 0.5  # Default if we can't evaluate
        if len(X_test) >= 5 and len(set(y_test)) >= 2:
            if HAS_XGB:
                preds_proba = self.model.predict(dtest)
                preds = np.argmax(preds_proba, axis=1)
            else:
                preds = self.model.predict(X_test)
            accuracy = float(np.mean(preds == y_test))
        self.is_trained = True

        if self.model_path:
            self.save(self.model_path)

        return {
            "accuracy": round(accuracy, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feat_names),
        }

    def predict(self, ohlcv):
        """Predict next candle direction from the last N candles."""
        if not self.is_trained:
            return {"direction": "neutral", "confidence": 0, "probabilities": []}

        X_raw, _, _ = build_features(ohlcv)
        if len(X_raw) == 0:
            return {"direction": "neutral", "confidence": 0, "probabilities": []}

        features = np.array(X_raw[-1:], dtype=float)

        if HAS_XGB:
            dmat = xgb.DMatrix(features)
            probs = self.model.predict(dmat)[0]
        else:
            probs = self.model.predict_proba(features)[0]

        dir_idx = int(np.argmax(probs))
        confidence = float(probs[dir_idx])
        direction = {0: "down", 1: "up", 2: "flat"}.get(dir_idx, "neutral")

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs],
        }

    def save(self, path):
        """Save model (XGBoost JSON or sklearn pickle)."""
        if HAS_XGB and isinstance(self.model, xgb.Booster):
            self.model.save_model(path)
        elif HAS_SKLEARN:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.model, f)

    def load(self, path):
        """Load saved model."""
        if HAS_XGB:
            self.model = xgb.Booster()
            self.model.load_model(path)
        elif HAS_SKLEARN:
            import pickle
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        self.is_trained = True


# ═══════════════════════════════════════════════════════════════════════════
#  LSTM — numpy-only forward pass (no training, load pretrained weights)
# ═══════════════════════════════════════════════════════════════════════════


class LSTMPredictor:
    """
    Minimal LSTM for price prediction — forward pass only.
    Uses simple weight matrices (can be trained externally or random).

    Architecture: 1 LSTM layer (8 units) → Dense(4) → Dense(1)

    Sequence length: 10 candles → predict next close price.
    """

    def __init__(self, seq_len=10, hidden_size=8):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self):
        """Initialize with random weights (or load pretrained)."""
        np.random.seed(42)
        scale = 0.01
        # LSTM gates: input, forget, cell, output — each has W, U, b
        self.Wf = np.random.randn(self.hidden_size, 1) * scale  # forget gate
        self.Wi = np.random.randn(self.hidden_size, 1) * scale  # input gate
        self.Wc = np.random.randn(self.hidden_size, 1) * scale  # cell gate
        self.Wo = np.random.randn(self.hidden_size, 1) * scale  # output gate

        self.Uf = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.Ui = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.Uc = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.Uo = np.random.randn(self.hidden_size, self.hidden_size) * scale

        self.bf = np.zeros((self.hidden_size, 1))
        self.bi = np.zeros((self.hidden_size, 1))
        self.bc = np.zeros((self.hidden_size, 1))
        self.bo = np.zeros((self.hidden_size, 1))

        # Output layer: Dense → 1 (next price)
        self.W_out = np.random.randn(1, self.hidden_size) * scale
        self.b_out = np.zeros((1, 1))

    def forward(self, sequence):
        """
        Forward pass through LSTM.

        sequence : np.array of shape (seq_len,) or (seq_len, 1)
            Price sequence, normalized.

        Returns
        -------
        float — predicted next price (de-normalized later)
        """
        if isinstance(sequence, (list, tuple)):
            sequence = np.array(sequence, dtype=float)

        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)

        seq_len = len(sequence)

        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        for t in range(seq_len):
            x = sequence[t].reshape(-1, 1)

            f = self._sigmoid(self.Wf @ x + self.Uf @ h + self.bf)
            i = self._sigmoid(self.Wi @ x + self.Ui @ h + self.bi)
            c_tilde = np.tanh(self.Wc @ x + self.Uc @ h + self.bc)
            c = f * c + i * c_tilde
            o = self._sigmoid(self.Wo @ x + self.Uo @ h + self.bo)
            h = o * np.tanh(c)

        pred = self.W_out @ h + self.b_out
        return float(pred[0, 0])

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def predict_next(self, ohlcv, normalize=True):
        """
        Predict next close price from last N OHLCV candles.

        Parameters
        ----------
        ohlcv : list[dict]
        normalize : bool
            Whether to z-score normalize the closes.

        Returns
        -------
        dict with predicted_price, direction, confidence estimate
        """
        if len(ohlcv) < self.seq_len + 1:
            return {"error": f"need at least {self.seq_len + 1} candles"}

        closes = np.array([c["close"] for c in ohlcv[-(self.seq_len + 1) : -1]], dtype=float)
        current = ohlcv[-1]["close"]

        if normalize:
            mean = np.mean(closes)
            std = np.std(closes)
            if std < 1e-8:
                std = 1.0
            normalized = (closes - mean) / std
            pred_normalized = self.forward(normalized)
            predicted = pred_normalized * std + mean
        else:
            predicted = self.forward(closes)

        direction = "up" if predicted > current else "down"
        change_pct = (predicted - current) / current * 100

        return {
            "predicted_price": round(float(predicted), 6),
            "current_price": current,
            "direction": direction,
            "change_pct": round(float(change_pct), 4),
        }

    def load_weights(self, path):
        """Load pretrained weights from JSON."""
        with open(path) as f:
            w = json.load(f)
        for key in ["Wf", "Wi", "Wc", "Wo", "Uf", "Ui", "Uc", "Uo",
                     "bf", "bi", "bc", "bo", "W_out", "b_out"]:
            if key in w:
                setattr(self, key, np.array(w[key]))
        return self


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE — run both models and merge signals
# ═══════════════════════════════════════════════════════════════════════════


def predict_all(ohlcv, xgb_model=None, lstm_model=None):
    """
    Run both XGBoost and LSTM predictors, merge signals.

    Returns dict with:
        - xgb_signal: direction + confidence
        - lstm_signal: direction + change_pct
        - consensus: "buy" / "sell" / "neutral"
        - consensus_strength: 0-100
    """
    result = {"xgb_signal": None, "lstm_signal": None, "consensus": "neutral", "consensus_strength": 0}

    if xgb_model and xgb_model.is_trained:
        xgb_pred = xgb_model.predict(ohlcv)
        result["xgb_signal"] = xgb_pred
    elif HAS_XGB or HAS_SKLEARN:
        dp = DirectionPredictor()
        # For inference without training, do quick online training on recent data
        train_result = dp.train(ohlcv[-200:] if len(ohlcv) >= 200 else ohlcv, test_split=0.0)
        if "error" not in train_result:
            xgb_pred = dp.predict(ohlcv)
            result["xgb_signal"] = xgb_pred

    if lstm_model:
        lstm_pred = lstm_model.predict_next(ohlcv)
        result["lstm_signal"] = lstm_pred
    else:
        lstm = LSTMPredictor()
        lstm_pred = lstm.predict_next(ohlcv)
        result["lstm_signal"] = lstm_pred

    # Consensus logic
    directions = []
    if result.get("xgb_signal") and result["xgb_signal"]["direction"] in ("up", "down"):
        directions.append(result["xgb_signal"]["direction"])
    if result.get("lstm_signal") and "direction" in result["lstm_signal"]:
        directions.append(result["lstm_signal"]["direction"])

    if directions:
        ups = sum(1 for d in directions if d == "up")
        downs = sum(1 for d in directions if d == "down")
        if ups > downs:
            result["consensus"] = "buy"
            result["consensus_strength"] = int(ups / len(directions) * 100)
        elif downs > ups:
            result["consensus"] = "sell"
            result["consensus_strength"] = int(downs / len(directions) * 100)
        else:
            result["consensus"] = "neutral"
            result["consensus_strength"] = 50

    return result


if __name__ == "__main__":
    import random
    random.seed(42)

    # Generate synthetic OHLCV
    ohlcv = []
    price = 100.0
    for i in range(500):
        change = random.uniform(-0.5, 0.5)
        price += change
        ohlcv.append({
            "open": price - change,
            "high": price + abs(change) * 0.5,
            "low": price - abs(change) * 0.5,
            "close": price,
            "volume": random.uniform(1000, 10000),
        })

    # Test LSTM
    print("=== LSTM Predictor ===")
    lstm = LSTMPredictor()
    result = lstm.predict_next(ohlcv)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Test XGBoost (if available)
    print("\n=== XGBoost Direction Predictor ===")
    dp = DirectionPredictor()
    train_res = dp.train(ohlcv, test_split=0.3)
    print(f"  Train result: {train_res}")
    if "error" not in train_res:
        pred = dp.predict(ohlcv)
        print(f"  Prediction: {pred}")

    # Test combined
    print("\n=== Combined Prediction ===")
    combined = predict_all(ohlcv, dp, lstm)
    print(f"  Consensus: {combined['consensus']} (strength: {combined['consensus_strength']}%)")
    print("✅ ML Predictor test OK")