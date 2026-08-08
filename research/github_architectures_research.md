# GitHub Trading Bot Architecture Research

> Compiled for Revolut X Trader upgrade — focus: persistent memory, skill/strategy library, backtesting, performance tracking, multi-strategy coordination, and AI decision evaluation.

---

## 1. REVOLUT X SPECIFIC BOTS

### 1.1 Mikyner/BTC-Revolut-X-limit-grid-bot
- **URL**: https://github.com/Mikyner/BTC-Revolut-X-limit-grid-bot
- **Stars**: 0 | **Language**: Python | **License**: MIT
- **Description**: BTC/EUR grid bot using GTC limit orders (0% maker fees) with Flask dashboard, SQLite DB, Telegram notifications, Docker support.

**Features we DON'T have:**
- ✅ **SQLite database** for persistent state (orders, wallet, settings, completed trades)
- ✅ **Grid engine** with sliding-window capital recycling
- ✅ **GTC limit orders** for 0% maker fee execution
- ✅ **Compounding** mode (automatically adjusts order size as profit grows)
- ✅ **Web dashboard** (Flask-based, live monitoring)
- ✅ **Proper wallet tracking** (EUR reserved in open orders, reconciliation logic)
- ✅ **Database-backed settings** (change grid params without restart)

**Key code patterns to adopt:**

```python
# Pattern 1: SQLite-based persistence (bot/database.py style)
# Instead of JSON files, use a proper SQLite schema with auto-migration

def init_db():
    conn = sqlite3.connect("trading_bot.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exchange_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            venue_order_id TEXT UNIQUE,
            side TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL,
            quantity REAL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filled_at TIMESTAMP,
            pnl REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wallet (
            id INTEGER PRIMARY KEY,
            eur_balance REAL DEFAULT 0,
            btc_balance REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT,
            action TEXT,
            confidence REAL,
            ai_reasoning TEXT,
            expected_outcome REAL,
            actual_outcome REAL,
            pnl REAL,
            evaluated INTEGER DEFAULT 0,
            evaluation TEXT
        )
    """)

# Pattern 2: Filled-order processing with PnL tracking
def _process_filled_order(db_order, order_detail, settings):
    """Proper accounting: track basis, calculate PnL, manage linked orders"""
    fill_price = float(order_detail.get("average_fill_price") or db_order["price"])
    fill_qty = float(order_detail.get("filled_quantity") or 0)
    fill_amount = float(order_detail.get("filled_amount") or 0)

    db.mark_order_filled(venue_id, fill_price, fill_qty, fill_amount)
    wallet = db.get_wallet()

    if db_order["side"] == "buy":
        # Deduct from EUR balance, add to BTC
        new_eur = wallet["eur_balance"] - fill_amount
        new_btc = wallet["btc_balance"] + fill_qty
        db.update_wallet(new_eur, new_btc)

        # Immediately place matching SELL limit order one grid step above
        target_sell = round(db_order["price"] + grid_step, 2)
        _place_sell_order(db_order, fill_qty, target_sell)
    elif db_order["side"] == "sell":
        # Track PnL by referencing linked buy order
        buy_order = db.get_order(db_order["linked_buy_order_id"])
        eur_received = fill_amount
        eur_spent = buy_order["fill_amount"] * (fill_qty / buy_order["quantity"])
        profit = eur_received - eur_spent
        new_eur = wallet["eur_balance"] + eur_received
        new_btc = wallet["btc_balance"] - fill_qty
        db.update_wallet(new_eur, new_btc)
        db.record_completed_trade(db_order["symbol"], profit, eur_spent, eur_received)
```

---

### 1.2 GlazKrovi/Sabot
- **URL**: https://github.com/GlazKrovi/Sabot
- **Stars**: 0 | **Language**: Go
- **Description**: Bot using official Revolut X API, golden cross strategy.

**Relevance**: Go implementation — not directly usable but demonstrates the Revolut X API integration pattern. Uses the exact same signing mechanism (Ed25519 with timestamp+method+path+query+body). Not a priority.

---

### 1.3 jasonviipers/revolut-x-crypto (Skill/Wrapper)
- **URL**: https://github.com/jasonviipers/revolut-x-crypto
- **Stars**: 2 | **Language**: Python (skill doc)
- **Description**: Documentation + Python client for Revolut X API.

**Key value**: Complete API reference covering all endpoints with Python examples. Better organized than our current client — has proper authentication, endpoint list, and request examples for all 16 endpoints.

**Patterns to adopt:**
- Separate client class with clean method signatures (`place_order`, `get_order`, `cancel_order`)
- Strong typing with Optional/List/Dict hints
- Single `_request()` gateway method with unified error handling
- Ed25519 key loading + signing properly factored

---

### 1.4 IgnacyKrasnodebski1/crypto-alerts
- **URL**: https://github.com/IgnacyKrasnodebski1/crypto-alerts
- **Stars**: 0 | **Language**: JavaScript
- **Description**: Multi-agent crypto alerts (BTC/ETH/XRP) → Revolut X discord notifications.

**Relevance**: Low (JS alerts bot). Not useful for our Python stack.

---

## 2. WELL-ARCHITECTED CRYPTO TRADING BOTS

### 2.1 freqtrade/freqtrade (⭐ 52k — THE gold standard)
- **URL**: https://github.com/freqtrade/freqtrade
- **Language**: Python | **License**: GPL-3.0
- **Description**: Free, open source crypto trading bot. SQLite persistence, backtesting, hyperopt (ML optimization), web UI, Telegram/Discord/Webhook RPC, 40+ exchange support, FreqAI (reinforcement learning).

**Features we DON'T have that we MUST adopt:**

#### A) SQLAlchemy Persistence Layer (`freqtrade/persistence/`)
Files: `models.py`, `trade_model.py`, `base.py`, `pairlock.py`, `key_value_store.py`, `wallet_history.py`, `custom_data.py`

Full SQLAlchemy ORM with:
- `Trade` model (entry/exit, PnL, fees, duration, stop-loss, take-profit)
- `Order` model (every order with status tracking, full CCXT data)
- `PairLock` (lock pairs out after losses — cool-down periods)
- `WalletHistory` (balance tracking over time)
- `KeyValueStore` (arbitrary key-value storage for custom data)
- `_CustomData` (user-defined persistent data)
- Auto-migration system (`db_migration.py`, `migrations.py`)

```python
# Pattern: Trade persistence model (scale this for our bot)
class Trade(ModelBase):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    pair = Column(String, nullable=False)
    is_open = Column(Boolean, default=True)
    open_rate = Column(Float)
    close_rate = Column(Float)
    open_date = Column(DateTime)
    close_date = Column(DateTime)
    stake_amount = Column(Float)
    amount = Column(Float)
    open_order_id = Column(String)
    stop_loss = Column(Float)
    stop_loss_abs = Column(Float)
    take_profit = Column(Float)
    take_profit_abs = Column(Float)
    final_profit_ratio = Column(Float)  # PnL %
    final_profit = Column(Float)         # PnL in quote
    exit_reason = Column(String)
    strategy = Column(String)            # Which strategy made this trade
    ai_decision_id = Column(Integer)     # Link back to AI decision log

    # Relationships
    orders = relationship("Order", backref="trade", lazy="dynamic")

    def calc_profit_ratio(self, rate=None):
        """Calculate profit ratio from open to close"""
        if self.close_rate:
            return (self.close_rate - self.open_rate) / self.open_rate
        return 0
```

#### B) Backtesting Engine (`freqtrade/optimize/backtesting.py` — 1980 lines)
Complete trade simulation with realistic fills, slippage, fee modeling, multi-timeframe support. Tracks every candidate entry/exit, manages open positions, produces detailed statistics. Designed so the same strategy class runs both live and backtest.

#### C) Hyperopt (ML Parameter Optimization) (`freqtrade/optimize/hyperopt*.py`)
Uses Scikit-Optimize / Optuna to search strategy parameter space. Multiple loss functions: Sharpe, Sortino, Calmar, ProfitFactor, Max Drawdown.

```python
# Pattern: Hyperopt loss function (we need this to evaluate strategies)
class HyperoptLoss:
    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int, ...) -> float:
        """
        Calculate loss to minimize. Lower = better strategy.
        Combines profit, drawdown, trade count, win rate.
        """
        total_profit = results["profit_ratio"].sum()
        total_duration = results["trade_duration"].mean()
        winning_trades = results[results["profit_ratio"] > 0]
        losing_trades = results[results["profit_ratio"] <= 0]

        profit_factor = (winning_trades["profit_ratio"].sum() /
                        abs(losing_trades["profit_ratio"].sum())
                        ) if len(losing_trades) else 999

        # Target: maximize profit, minimize drawdown, ensure profit factor
        return -total_profit * profit_factor * (len(winning_trades) / max(trade_count, 1))
```

#### D) FreqAI (`freqtrade/freqai/`)
Full machine learning pipeline:
- `FreqaiDataKitchen` — feature engineering, train/test splits, label creation, prediction storage
- Reinforcement Learning agents (PPO via Stable-Baselines3)
- `BaseReinforcementLearningModel` with custom `MyRLEnv` calculating reward from position PnL
- Auto-retraining schedules, model expiry, model persistence on disk

#### E) Strategy Interface (`freqtrade/strategy/interface.py`)
Clean separation between strategy (what/when to trade) and execution (how to trade):
```python
class IStrategy:
    def populate_indicators(self, dataframe, metadata) -> DataFrame:
        """TA/ML indicators"""
    def populate_entry_trend(self, dataframe, metadata) -> DataFrame:
        """Buy signals"""
    def populate_exit_trend(self, dataframe, metadata) -> DataFrame:
        """Sell signals"""
    def confirm_trade_entry(self, pair, ...) -> bool:
        """Final check before buying"""
    def custom_stoploss(self, pair, current_time, current_rate, ...) -> float:
        """Dynamic stop-loss"""
    def custom_take_profit(self, pair, ...) -> float:
        """Dynamic take-profit"""
```

#### F) Protection Manager (`freqtrade/plugins/protections/`)
Built-in risk protections:
- `CooldownPeriod` — wait N periods between trades
- `LowProfitPairs` — stop trading pairs with low profitability
- `MaxDrawdownProtection` — stop all trading if portfolio drops X%
- `StoplossGuard` — stop if too many stop-losses triggered recently
- `ProtectionManager` — orchestrates all protections

---

### 2.2 hummingbot/hummingbot (⭐ 19k)
- **URL**: https://github.com/hummingbot/hummingbot
- **Language**: Python | **License**: Apache 2.0
- **Description**: High-frequency crypto trading bot framework. Market making, arbitrage, liquidity mining.

**What we can use:**
- **Orderbook imbalance detection** — reads bid/ask depth ratios as signal
- **Pure market making** logic (could complement our scalper)
- **Backtesting framework** via `quants-lab` subproject
- **VWAP / TWAP execution** — smart order routing

Less directly applicable since HFT is overkill for our server, but the orderbook analytics module is worth extracting.

---

### 2.3 ROMEROPS7/ultra-scalping-bot (⭐ 5)
- **URL**: https://github.com/ROMEROPS7/ultra-scalping-bot
- **Language**: Python | **License**: MIT
- **Description**: Multi-strategy scalping bot combining Freqtrade, Passivbot, Hummingbot, Jesse AI patterns. Complete backtesting, ML optimizer, risk manager, Telegram alerts.

**Features we DON'T have:**

#### A) Clean Multi-Strategy Architecture (core/strategies.py)
Base class pattern with pure composition:
```python
class BaseStrategy:
    def __init__(self, config):
        self.config = config
        self.name = "base"
    def analyze(self, df: pd.DataFrame) -> TradeSetup:
        raise NotImplementedError

# Concrete strategies
class EmaRsiAtrStrategy(BaseStrategy):
    """Triple EMA + RSI + ATR"""
    name = "ema_rsi_atr"

class MomentumScalperStrategy(BaseStrategy):
    """Jesse AI-style quick momentum plays"""
    name = "momentum_scalper"

class GridScalpingStrategy(BaseStrategy):
    """Passivbot-inspired grid scalping"""
    name = "grid_scalping"

# Strategy registry
def get_strategy(config):
    strategies = {
        "ema_rsi_atr": EmaRsiAtrStrategy,
        "momentum_scalper": MomentumScalperStrategy,
        "grid_scalping": GridScalpingStrategy,
    }
    return strategies[config.strategy.active_strategy](config)
```

#### B) Structured TradeSetup signal (core/strategies.py)
Instead of raw dicts, use a proper signal object:
```python
class TradeSetup:
    def __init__(self, signal: Signal, entry_price: float = 0,
                 stop_loss: float = 0, take_profit: float = 0,
                 confidence: float = 0, strategy_name: str = ""):
        self.signal = signal
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence  # 0-1 float
        self.strategy_name = strategy_name
```

#### C) Confidence Scoring (strategies.py)
Each signal carries a composite confidence, not just direction:
```python
confidence = min(0.9, 0.5 + (0.1 if vol_ok else 0) +
                 (0.1 if rsi < 50 else 0) +
                 (0.1 if macd_hist > 0 else 0))
```

#### D) Full Backtesting Engine (core/backtester.py)
```python
class Backtester:
    def __init__(self, config):
        self.strategy = get_strategy(config)
        self.risk_manager = RiskManager(config)
        self.trades = []
        self.open_trades = {}
        self.equity_curve = []
        self.balance = config.backtest.initial_balance

    def run(self, df, symbol="BTC/USDT"):
        df = Indicators.calculate_all(df, self.config)
        for i in range(60, len(df)):
            window = df.iloc[:i+1]
            setup = self.strategy.analyze(window)
            if setup.signal != Signal.HOLD and not self._has_open(symbol):
                size = self.risk_manager.calculate_position_size(
                    self.balance, setup.entry_price, setup.stop_loss
                )
                trade = BacktestTrade(symbol, setup.signal, setup.entry_price,
                                     size, df.index[i], setup.sl, setup.tp, setup.strategy_name)
                self.open_trades[symbol] = trade
            self._check_exits(window, symbol, df.index[i])
        return self._metrics()
```

#### E) ML Signal Predictor (core/ml_optimizer.py)
Gradient Boosting-based signal prediction that learns from past data:
```python
class MLSignalPredictor:
    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.05):
        self.model = GradientBoostingClassifier(...)
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_features(self, df):
        """Price returns, MA ratios, volatility, RSI, volume ratios, momentum, BB position"""
        features = pd.DataFrame()
        features["return_1"] = df["close"].pct_change(1)
        features["return_3"] = df["close"].pct_change(3)
        features["sma_ratio_5_20"] = df["close"].rolling(5).mean() / df["close"].rolling(20).mean()
        features["volatility_5"] = df["close"].rolling(5).std() / df["close"].rolling(5).mean()
        features["rsi"] = ...  # calculated RSI
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        return features.dropna()

    def train(self, df):
        X = self._extract_features(df)
        y = (df["close"].shift(-1) > df["close"]).astype(int)  # forward return > 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
        self.model.fit(X_train, y_train)
        self.is_trained = True
```

#### F) Risk Manager (core/risk_manager.py)
Comprehensive risk with streak tracking:
```python
class RiskManager:
    def __init__(self, config):
        self.daily_pnl = 0.0
        self.peak_balance = config.initial_capital
        self.consecutive_losses = 0
        self.trade_count = 0
        self.trades_today = 0
        self.last_trade_time = None

    def can_trade(self) -> Tuple[bool, str]:
        """Check all risk gates. Returns (allowed: bool, reason: str)"""
        if self.trades_today >= self.rc.daily_trade_limit:
            return False, "Daily trade limit reached"
        if self.daily_pnl / self.peak_balance <= -self.rc.max_daily_loss_pct:
            return False, f"Max daily loss exceeded ({self.daily_pnl:.2f})"
        if self.consecutive_losses >= self.rc.max_consecutive_losses:
            return False, f"{self.consecutive_losses} consecutive losses"
        return True, "ok"

    def calculate_position_size(self, balance, entry_price, stop_loss) -> float:
        """Kelly-approximate sizing based on risk per trade"""
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        if risk_per_unit <= 0:
            return 0
        max_risk = balance * self.rc.max_position_size_pct
        return max_risk / (risk_per_unit * entry_price)
```

#### G) Dataclass-based Config (config/settings.py)
Clean typed config, not bare dicts:
```python
@dataclass
class ScalpingConfig:
    symbols: list = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1m"
    leverage: int = 1
    initial_capital: float = 1000.0
    # Indicators
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 55
    rsi_period: int = 14
    # Risk
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.005

@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.05
    max_consecutive_losses: int = 5
    daily_trade_limit: int = 50
```
---

### 2.4 jptsantossilva/BEC (⭐ 64)
- **URL**: https://github.com/jptsantossilva/BEC
- **Language**: Python | **License**: MIT
- **Description**: Binance spot trading bot with automated backtesting, strategy selection, **market-phase analysis**, Streamlit dashboard, Telegram alerts.

**Features we DON'T have:**

#### A) Market Phase Analysis
BEC classifies market into phases (bull, bear, ranging, volatile) and selects strategies accordingly. This is EXACTLY what we need for our day trader + scalper coordination problem.

#### B) Multi-Exchange SQLite Schema (`bec/db/exchange_schema.py`, `backtesting_schema.py`, `live_execution_schema.py`)
Complete database schema with migrations, exchange-agnostic symbol normalization:
- `Exchanges` table (multi-exchange metadata, buy/sell enable flags)
- `Orders` table (with client_order_id, fill tracking, reconciliation indexes)
- `Positions` table
- `Backtesting_Results`, `Backtesting_Trades`
- `Signals_Log` — logs every signal for post-hoc analysis
- `Auto_Switch_Signals` — automated strategy switching
- `Symbols_By_Market_Phase` — which coins are in which phase

#### C) Available Strategy Enum + Backtesting Framework
```python
class strategy(Enum):
    EMA_CROSS = "ema_cross"
    EMA_CROSS_WITH_MARKET_PHASES = "ema_cross_with_market_phases"
    MARKET_PHASES = "market_phases"
    CONSECUTIVE_CANDLES = "consecutive_candles"
    MULTI_EMA_CROSS = "multi_ema_cross"
    RSI_UPTREND = "rsi_uptrend"
    BREAKOUT = "breakout"
```

#### D) Streamlit Dashboard
Real-time monitoring, strategy selection, backtest visualization, position tracking. Could be adapted to show our bot's state.

---

### 2.5 hackobi/AI-Scalpel-Trading-Bot (⭐ 447)
- **URL**: https://github.com/hackobi/AI-Scalpel-Trading-Bot
- **Language**: Python | **License**: GPL-3.0
- **Description**: Python trading bot with ML-driven strategy optimization.

**What we can use:**
- ML model training pipeline for strategy parameter optimization
- Feature engineering for market data (already somewhat covered by ultra-scalping-bot)
- Auto-optimization loop (train → evaluate → refine)

Low priority since ultra-scalping-bot covers similar territory with cleaner code.

---

## 3. AI-POWERED TRADING BOTS

### 3.1 antconsales/QuantMind (⭐ 5)
- **URL**: https://github.com/antconsales/QuantMind
- **Language**: Python | **License**: MIT
- **Description**: LLM + Reinforcement Learning trading bot designed for Raspberry Pi. Uses sentiment analysis + PPO RL agent.

**Features we DON'T have:**

#### QuantMind's Innovation
- **LLM for market sentiment** → feeds into RL agent's state space (we already do LLM but standalone)
- **RL agent** (Stable-Baselines3 PPO) trained to decide buy/sell/hold based on: price, balance, position, **sentiment score**
- **ResourceManager** — dynamic model quantization, RAM/CPU throttling for low-resource environments (we run on a low-resource server too!)
- **Flask API** status endpoints with system monitoring
- **Fine-tuning** pipeline: collect data → fine-tune LLM → improve sentiment → better RL decisions

```python
# Pattern: Combining LLM output with RL state (from QuantMind)
class TradingEnv(gym.Env):
    def __init__(self, ...):
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,))
        # State: [price_norm, balance_norm, position_norm, sentiment,
        #         rsi_norm, volume_norm, volatility_norm]

    def step(self, action):
        # action: 0=hold, 1=buy, 2=sell
        if action == 1:  # buy
            reward = (next_price - current_price) * qty - fee
        elif action == 2:  # sell
            reward = (current_price - prev_entry) * qty - fee
        else:
            reward = -small_decay  # penalty for holding
        return obs, reward, done, info
```

**Relevance to our problems:**
- The RL/LLM combo addresses "AI makes decisions without knowing what worked before"
- ResourceManager pattern is directly applicable to our low-resource server

---

### 3.2 gurkinator/AI-Crypto-TradingBot (⭐ 1)
- **URL**: https://github.com/gurkinator/AI-Crypto-TradingBot
- **Language**: Python (21k lines Streamlit app)
- **Description**: AI-driven KuCoin trading bot. Swing + Day modes, position memory, strategy learning, background scanner, Top 3 optimization.

**Features we DON'T have:**

#### A) Persistent Position Memory (`open_positions.json`)
Survives hard restarts — open positions are tracked in JSON and reconciled on restart. Even though it uses JSON (not ideal), the reconciliation logic is solid.

#### B) Dual Mode Architecture (Swing + Day)
**THIS IS EXACTLY OUR PROBLEM.** The bot runs Swing and Day trading modes independently with separate:
- Capital frames (separate budget tracking)
- Position memory (each mode knows its own positions)
- Live control (start/stop each mode independently)
- Strategy profiles (different indicators/rules per mode)

#### C) Self-Learning Memory System
Screenshots show "Self-learning memory" feature:
- Strategy records tracked over time
- Win/loss tracked per strategy per market condition
- "Top 3" strategy ranking based on recent performance
- Background scanner continuously evaluates strategies

#### D) Strategy Scanner with Walk-Forward Analysis
```python
# Pattern: Background strategy scanning with ranking
# Each strategy gets scored on:
#   - Fitness (combined metric)
#   - Sharpe Ratio
#   - Sortino Ratio
#   - Max Drawdown
#   - Out-of-sample result (walk-forward validation)
# Top 3 strategies are recommended and tracked
```

#### E) Bayesian/Optuna Optimization
Scan parameter space using Bayesian optimization and Optuna, with reliable walk-forward analysis.

**Key insight for us:** This is the closest existing project to what we need. The dual-mode (Swing + Day) architecture directly maps to our DayTrader + Scalper coordination problem. The self-learning memory is exactly what "AI evaluates past decisions" would look like.

---

## 4. SPECIFIC RECOMMENDATIONS FOR OUR BOT

### Priority 1 — Database Persistence (replace JSON files)
**Source**: Mikyner grid bot + Freqtrade + BEC

Replace `/tmp/revolut_trader_state.json` with SQLite:
- `trades` table (all completed trades with PnL, strategy, timestamp)
- `decisions` table (AI decisions with reasoning, expected outcome, actual outcome)
- `positions` table (open positions per strategy — day trader + scalper)
- `wallet` table (balance tracking over time)
- `strategy_performance` table (per-strategy stats: win rate, avg profit, sharpe)
- `settings` table (key-value config, editable at runtime)

### Priority 2 — Decision Evaluation System
**Source**: gurkinator bot + BEC Signals_Log

Every AI decision goes to `decisions` table:
```
timestamp | symbol | action | confidence | reasoning | market_conditions |
expected_outcome | actual_outcome | pnl | was_profitable | evaluation_score
```

Then periodically:
1. Run `SELECT AVG(pnl), win_rate FROM decisions WHERE strategy='day_trader' GROUP BY market_condition`
2. Feed results back to AI: "Last 10 decisions in ranging markets: 30% win rate. Adjust confidence threshold."
3. Auto-disable strategies with >60% loss rate over trailing 50 trades

### Priority 3 — Multi-Strategy Architecture
**Source**: ultra-scalping-bot + Freqtrade

Adopt the `BaseStrategy` + `TradeSetup` pattern:
- Each strategy (technical, momentum, grid, AI) inherits from `BaseStrategy`
- `StrategyOrchestrator` runs all strategies, combines signals with configurable weights
- Each signal carries `strategy_name` for attribution tracking
- Config: `{"strategies": {"day_trader": {"weight": 0.6}, "scalper": {"weight": 0.4}}}`

### Priority 4 — Backtesting Integration
**Source**: ultra-scalping-bot + Freqtrade + BEC

Add a `backtest.py` that:
- Downloads historical data from Revolut X's OHLCV endpoint
- Runs our EXACT strategies (same code paths as live)
- Produces: equity curve, win rate, Sharpe, Sortino, max drawdown, profit factor
- Results go to DB for comparison

### Priority 5 — Strategy Learning / ML
**Source**: ultra-scalping-bot ml_optimizer.py + QuantMind

Lightweight gradient boosting model:
- Features: price returns, volume ratio, RSI, volatility, Bollinger %B
- Label: did price go up next candle? (+0.5% threshold)
- Train weekly on last 3 months of data
- Output: predicted direction probability → weigh into AI decision confidence

### Priority 6 — Market Phase Awareness (Day Trader ↔ Scalper Coordination)
**Source**: BEC market-phase analysis

Classify current market as one of:
- **Trending Bull** → favor day trader (long)
- **Trending Bear** → favor day trader (short) or sit out
- **Ranging** → favor scalper (mean reversion, grid)
- **Volatile** → both active but tighter stops, reduce size
- **Low Vol** → neither (no edge)

Scalper and day trader share this classification and adjust behavior accordingly.

---

## 5. EXISTING PROJECT AUDIT (revolut-x-trader)

### What we DO have (good):
- ✅ Working Revolut X API client with Ed25519 signing
- ✅ Market analyzer (multi-timeframe indicators)
- ✅ AI brain (OpenRouter / LLM integration)
- ✅ Risk manager (basic position sizing, PnL tracking)
- ✅ Strategy evaluation (TechnicalStrategy)
- ✅ Day trader loop (5min cycle)
- ✅ Scalper loop
- ✅ Monitoring / heartbeat

### What's MISSING (as confirmed by this research):
- ❌ **No database** — JSON in /tmp/ that gets wiped
- ❌ **No decision log** — AI decisions not persisted for analysis
- ❌ **No backtesting** — can't test strategies against historical data
- ❌ **No strategy registry** — strategies are functions, not class-based, hard to compose
- ❌ **No performance tracking** — no per-strategy win rate, Sharpe, or drawdown
- ❌ **No market phase detection** — day trader and scalper don't coordinate
- ❌ **No strategy learning** — no ML, no historical pattern matching
- ❌ **No trade-position linking** — can't trace a sell back to which buy made it
- ❌ **No walk-forward analysis** — can't validate strategy robustness

---

**Report compiled by analyzing 12+ GitHub repos via API. Key architectural patterns extracted and annotated with code snippets for direct adoption.**