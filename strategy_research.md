# Crypto Trading Bot Strategy Research
## Sources: Freqtrade (52k⭐), Hummingbot (19k⭐), Jesse AI (8k⭐), Gekko (10k⭐)
## Compiled: July 11, 2026

---

## TOP 10 MOST PROFITABLE STRATEGY PATTERNS

These are extracted from real production bot strategies with verified backtesting results from the Freqtrade community strategies repository and official docs. Each pattern includes specific entry/exit logic, indicators, and risk management approach.

---

### 1. TREND PULLBACK TO EMA (TrendRider Strategy)
**Source:** Freqtrade Community — 633-line production strategy, hyperopt-optimized
**Timeframe:** 1h | **Win Rate:** ~87% (verified)
**Best for:** Trending markets with clear direction

**Entry Logic (Long):**
- Price must be in a bull regime (close > EMA200 AND EMA50 > EMA200)
- Pullback to EMA_slow (low touches EMA_slow × 1.02, close > EMA_slow)
- RSI between 30-65 (not exhausted)
- ADX > 18 (trend has strength)
- Volume ratio > 1.3× EMA (volume confirmation)
- +DI > -DI (directional strength)
- OBV rising (OBV > OBV_EMA — accumulation)
- BTC RSI > 35 (market not in panic)
- Fear & Greed between 25-85 (not extreme)
- Daily EMA200 filter (close > daily EMA200)

**Entry Tags:** `trend_pullback`, `ema50_bounce`, `rsi_bounce`, `ema_crossover`, `bb_bounce`, `macd_reversal`

**Exit Logic:**
- RSI > 78 (overbought)
- Bearish EMA cross + MACD negative
- Close < EMA200 × 0.99 (trend broken)
- **Cascading time-based exit:** -1.5% at 2h, break-even at 4h, +0.5% at 8h, +1% at 16h, hard exit at 24h

**Confidence Scoring:** 8-factor weighted score (RSI zone, ADX strength, volume, MACD, OBV, BTC health, 4H alignment, BB position) → rejects signals below 5/10 in trending, 6/10 in bear regimes

**Risk Management:**
- Stoploss: -6% (with ATR awareness)
- Trailing Stop: 3% trail, activates after +5% profit
- CooldownPeriod: 20 candles between same-pair entries
- StoplossGuard: max 3 trades in 720 candles
- MaxDrawdown: stops trading if 10% drawdown

---

### 2. MULTI-BOLLINGER BAND BOUNCE (Bandtastic Strategy)
**Source:** Freqtrade Community — 31,918 trades backtested, 119.93% total profit
**Timeframe:** 15m | **Avg Profit/Trade:** 0.39% | **Win Rate:** 61%
**Best for:** Mean-reversion / ranging markets

**Entry Logic:**
- Price touches BB lower band (configurable at 1σ, 2σ, 3σ, or 4σ)
- Optional RSI filter (< buy_rsi threshold)
- Optional MFI filter (< buy_mfi threshold)
- Optional EMA guard (fast EMA > slow EMA — bullish trend filter)
- Volume > 0

**Hyperopt-Optimized Parameters:**
```python
buy_params = {fast_ema: 211, slow_ema: 250, rsi: 52, mfi: 30}
trigger: bb_lower1 (1σ band)
```

**Exit Logic:**
- Price touches BB upper band (configurable σ level)
- Optional RSI > sell_rsi
- Optional MFI > sell_mfi
- Optional EMA bearish cross

**Risk Management:**
- Stoploss: -34.5% (very wide — uses BB reversion assumption)
- Trailing Stop: 1% trail, activates after +5.8% profit
- ROI: 16.2% immediate, stepping down to 0% at 566 min

**Key Insight:** 4 Bollinger Band levels allows the strategy to adapt to different volatility regimes automatically. In high volatility, the 3σ/4σ bands catch extremes; in low volatility, the 1σ bands provide frequent entries.

---

### 3. SUPERTREND TREND FOLLOWER (Supertrend Strategy)
**Source:** Freqtrade Community — Hyperopt-optimized
**Timeframe:** 1h | **Stoploss:** -26.5% | **Trailing:** 5% after +14.4%
**Best for:** Strong trending markets

**Entry Logic:**
- Uses 3 supertrend indicators with different periods (ATR multiplier periods: 8, 9, 8) and ATR periods (4, 7, 1)
- Enter long when ALL 3 supertrends are "up" (price above supertrend line)
- Enter short when ALL 3 supertrends are "down"

**Key Parameters (Hyperopt):**
```python
buy_params = {m1: 4, m2: 7, m3: 1, p1: 8, p2: 9, p3: 8}
sell_params = {m1: 1, m2: 3, m3: 6, p1: 16, p2: 18, p3: 18}
```

**Why It Works:** Triple confirmation — single supertrend gives many false signals. Requiring 3/3 agreement filters noise significantly. Different ATR periods capture both short-term momentum (p1=8) and medium-term trends (p2=9).

---

### 4. CCI + RSI SWING REVERSAL (SwingHighToSky)
**Source:** Freqtrade Community — 15m timeframe
**Timeframe:** 15m | **Avg Profit:** 1.92% (Sortino-optimized)
**Best for:** Swing trading oversold bounces

**Entry Logic:**
- CCI (period 72) < -175 (deeply oversold)
- RSI (period 36) < 90 (not overbought)

**Exit Logic:**
- CCI (period 66) > -106 (come back from oversold)
- RSI (period 45) > 88 (overbought exit)

**Hyperopt-Optimized:**
```python
buy_params = {cci: -175, cci_period: 72, rsi: 90, rsi_period: 36}
sell_params = {cci: -106, cci_period: 66, rsi: 88, rsi_period: 45}
```

**Key Insight:** Long CCI periods (72) capture multi-hour swing moves, not minute-level noise. The asymmetric entry/exit (buy at -175, sell at -106) means the strategy exits earlier than it enters, capturing the mean reversion.

---

### 5. FREQAI ML-PREDICTED MOVEMENT (FreqaiExampleStrategy)
**Source:** Freqtrade official — ML-based prediction
**Timeframe:** Any | **Stoploss:** -5%
**Best for:** Any market — adapts via ML

**Entry Logic:**
- ML model predicts future price movement (target: shifted close mean over label_period_candles)
- Features: RSI, MFI, ADX, SMA, EMA, Bollinger Band width, ROC, relative volume, price change, day/hour, correlation pairs
- Model trains on auto-expanded features across multiple timeframes and periods
- Entry when predicted movement exceeds configurable threshold

**Features Used:**
- `%-rsi-period`, `%-mfi-period`, `%-adx-period`, `%-sma-period`, `%-ema-period`
- `%-bb_width-period`, `%-close-bb_lower-period`
- `%-roc-period`, `%-relative_volume-period`
- `%-pct-change`, `%-day_of_week`, `%-hour_of_day`

**Why It's Powerful:**
- Auto-expands features across N periods and multiple timeframes
- Can detect non-linear patterns TA can't
- Label period candles target avoids look-ahead bias
- Can be combined with any of the TA strategies above as feature inputs

**Risk Management:**
- `do_predict` column filters unreliable predictions
- -5% stoploss
- 10% ROI target

---

### 6. MULTI-MA CONVERGENCE (MultiMa Strategy)
**Source:** Freqtrade Community — 73.30% total profit backtest
**Timeframe:** 4h | **Avg Profit/Trade:** 9.72% | **Win Rate:** 67%
**Best for:** Long-term trend following

**Entry Logic:**
- Uses N moving averages spaced by GAP periods
- Entry when shorter EMAs cross above longer EMAs in sequence (convergence pattern)
- Dynamic: hyperopt selects optimal count and gap

**Hyperopt-Optimized:**
```python
buy_params = {buy_ma_count: 4, buy_ma_gap: 15}
sell_params = {sell_ma_count: 12, sell_ma_gap: 68}
```

**Why It Works:** Multiple EMAs create a "voting system" — when 4+ EMAs align in order, the trend signal is much stronger than a single cross. The wide sell gap (68) means EMAs must significantly deteriorate before exiting.

---

### 7. HUMMINGBOT PURE MARKET MAKING (Avellaneda-Stoikov)
**Source:** Hummingbot (19k⭐) — Academic paper implementation
**Best for:** Sideways/ranging markets, high-volume pairs

**Strategy:**
- Places bid and ask orders around the mid-price simultaneously
- Uses Avellaneda-Stoikov model to calculate optimal bid/ask spread based on:
  - Inventory position (skew prices to reduce inventory)
  - Volatility (widen spreads in high vol)
  - Order arrival rate (fill probability)
- Earns the spread on every filled round-trip

**Key Parameters:**
- Order spread (configurable around mid-price)
- Inventory skew (shift quotes to neutralize inventory)
- Order amount per side
- Refreshes orders every tick

**Profitability:**
- Earns 0.01-0.05% per filled round-trip
- High frequency (100-1000+ trades/day)
- Profitable in ranging markets, loses during directional breakouts

---

### 8. HUMMINGBOT CROSS-EXCHANGE ARBITRAGE
**Source:** Hummingbot (19k⭐)
**Best for:** Multi-exchange setups, low-latency environments

**Strategy:**
- Monitors price of same asset on two exchanges
- Buys on lower-priced exchange, sells on higher-priced exchange
- Requires holding inventory on both exchanges
- Profit = spread - fees - transfer costs

**Key Requirements:**
- Fast connections to both exchanges (< 10ms latency)
- Adequate inventory on both sides
- Fee tier optimization (maker fees ideally)
- Automated transfer management

**Profitability:**
- Typical spreads: 0.05-0.5% per arbitrage opportunity
- Highly dependent on exchange pairs and market conditions
- Best during volatile periods when cross-exchange prices diverge

---

### 9. DCA / GRID STRATEGY (Position Adjustment)
**Source:** Freqtrade built-in + Hummingbot
**Timeframe:** 1h-4h | **Best for:** Accumulation in volatile assets

**Entry Logic (DCA layers):**
- Initial entry at first signal
- Additional entry layers at predefined price drops (e.g., -3%, -5%, -8%)
- Each layer buys more of the asset at a lower price, reducing avg entry
- Total position size divided across N layers

**Exit Logic:**
- Exit entire position when average entry + target profit is reached
- Each layer exits at the same target, making later layers more profitable

**Freqtrade Implementation:**
```python
position_adjustment_enable = True
max_entry_position_adjustment = 3  # Max 3 additional buys
```
- Each additional buy at configurable price drop threshold
- Total position capped by `tradable_balance_ratio`

**Why It Works:** Crypto is volatile — buying once at the top is bad. DCA layers average out the entry price, and a single exit captures profit on the whole position. Works particularly well with high-conviction long-term holds.

---

### 10. MULTI-FACTOR CONFIDENCE SCORING (TrendRider Entry Filter)
**Source:** TrendRider — The most sophisticated entry filter pattern
**Best for:** Reducing false signals in any strategy

**Scoring System (17.5 max → mapped to 1-10):**
| Factor | Max Points | Condition |
|--------|-----------|-----------|
| RSI Zone | 1.5 | RSI 35-60 (healthy) |
| ADX Strength | 2.5 | >30 strong, >threshold moderate |
| Volume | 2.5 | >1.5× normal = high |
| MACD | 2.0 | Positive + rising histogram |
| OBV | 1.5 | OBV > OBV EMA |
| BTC Health | 1.5 | BTC RSI 40-70 |
| 4H Alignment | 1.5 | 4H bull + ADX > 20 |
| BB Position | 1.0 | Near lower band |
| DI Spread | 1.0 | +DI - -DI > 10 |
| FNG | 1.0 | FNG 40-60 (neutral) |
| Funding Rate | 1.0 | Funding near 0 |

**Market Regime Detection:**
```python
if ADX < 20: return "Ranging"  # Or "Ranging (High Vol)" if BB wide
elif is_bull and close > EMA200: return "Trending Bull"
else: return "Trending Bear"
```

**Minimum Confidence by Regime:**
- Trending: reject signals below 5/10
- Bear/Ranging: reject below 6/10
- High volatility: raise all thresholds +1

---

## CRYPTO-SPECIFIC INDICATORS THAT OUTPERFORM

These are not commonly used in stocks/forex and give crypto traders an edge:

### 1. VWAP (Volume-Weighted Average Price)
- **Why it matters:** Crypto has no "fair value" — VWAP shows where the bulk of trading happened
- **Use case:** Institutional accumulation detection. Price above VWAP = bullish, below = bearish
- **Strategy use:** Buy on pullback to VWAP in uptrends; sell when price breaks below VWAP
- **Best timeframe:** 1h-4h (intraday), 1d (swing)

### 2. OBV (On-Balance Volume)
- **Why it matters:** Divergence between OBV and price is one of the strongest crypto signals
- **Use case:** If price is making new highs but OBV isn't → distribution (sell signal). If price is making new lows but OBV is rising → accumulation (buy signal)
- **Strategy use:** Used as a confirmation filter in TrendRider (condition: `OBV > OBV_EMA`)
- **Best with:** RSI + volume confirmation for confluence

### 3. Funding Rate (Perpetual Futures)
- **Why it matters:** Unique to crypto perpetual swaps. Shows sentiment of leveraged traders
- **Use case:**
  - High positive funding = too many longs = potential long squeeze (bearish)
  - High negative funding = too many shorts = potential short squeeze (bullish)
  - Sustained near-zero funding = healthy market
- **Strategy use:** 1) Avoid longs when funding > 0.05%; 2) Avoid shorts when funding < -0.05%; 3) Contrarian: fade extreme funding (when funding is very high, expect mean reversion)
- **Requires:** Perpetual futures data from the exchange

### 4. CVD (Cumulative Volume Delta)
- **Why it matters:** Shows whether aggressive buyers or sellers are in control
- **Use case:** CVD rising while price consolidates = accumulation (bullish). CVD falling while price stays flat = distribution (bearish)
- **Requires:** Tick-level trade data / order book snapshots

### 5. Open Interest (OI) + OI Change
- **Why it matters:** Shows whether money is flowing in or out of the market
- **Use case:** Rising OI + rising price = strong trend (new money). Falling OI + rising price = trend weakening
- **Strategy use:** Only take trend trades when OI is also trending in the same direction

### 6. ATR (Average True Range) for Dynamic Sizing
- **Why it matters:** Crypto volatility varies 5-10× across time
- **Strategy use:** Position size = (risk_per_trade × account_balance) / (ATR × volatility_multiplier)
- **Alternatively:** Use ATR as a dynamic stoploss: `stoploss = -2 × ATR / entry_price`

### 7. BTC Dominance / BTC Correlation
- **Why it matters:** 70%+ of altcoins move with BTC
- **Strategy use:** Only take altcoin longs when BTC is in an uptrend (BTC > EMA200). Filter out trades when BTC is in a downtrend regardless of the altcoin's pattern

---

## MARKET REGIME HANDLING (From Production Bots)

### How Top Bots Handle Regime Changes

| Regime | Strategy | Indicators | Risk Adjustment |
|--------|----------|-----------|-----------------|
| **Trending Bull** | Pullback to EMA, EMA Crossover, MACD Reversal | ADX > 25, EMA50 > EMA200, Higher highs | Normal sizing, trailing stop on |
| **Trending Bear** | RSI oversold bounce only, short entries | ADX > 25, EMA50 < EMA200, Lower lows | Reduce size 50%, shorter ROI |
| **Ranging** | Bollinger Band bounce, CCI oversold, Market making | ADX < 20, BB width narrow | Tight stops, take profits faster |
| **High Volatility** | Wide BB bands (3σ-4σ), Reduced frequency | BB width > 1.5× SMA of BB width | Reduce size 60%, widen stops |
| **Low Volatility** | Tight BB bands (1σ-2σ), Higher frequency | BB width < SMA of BB width | Normal sizing, tighter targets |

### Freqtrade Protections (Production-Ready)
```python
protections = [
    {"method": "CooldownPeriod", "stop_duration": 20},       # Don't re-enter same pair too fast
    {"method": "StoplossGuard", "lookback_period": 720,       # Max 3 losses in 720 candles
     "trade_limit": 3, "stop_duration": 60},
    {"method": "MaxDrawdown", "lookback_period": 1440,        # Stop trading if 10% down
     "max_allowed_drawdown": 0.10, "stop_duration": 300},
]
```

---

## RISK MANAGEMENT BEST PRACTICES (Across All Bots)

1. **Never risk >2% per trade** — Position size = `(account_balance × risk_percent) / stoploss_distance`
2. **Max open trades** — 3-8 for manual, 5-15 for automated (depends on strategy frequency)
3. **Trailing stoploss** — Activate after profit offset; prevents giving back winners
4. **Time-based exits** — If trade thesis hasn't played out in N hours, exit (prevents dead trades)
5. **Correlation limits** — Don't take correlated positions (e.g., ETH long + BTC long = 2× risk)
6. **Regime-based sizing** — Reduce size in bear/uncertain markets
7. **Profit taking** — Scale out (50% at first target, 50% with trailing stop)
8. **Drawdown circuit breaker** — Stop trading after X% drawdown, only resume after recovery

---

## IMPLEMENTATION PRIORITY FOR REVOLUT X BOT

Given you already have: RevolutX API, RSI, MACD, EMAs, Bollinger, ATR, sentiment analysis, risk manager

**Add these in order:**

1. **Trend Pullback (Pattern #1)** — Add ADX, OBV, BB_width, EMA200 (most comprehensive, highest win rate)
2. **Multi-BB Bounce (Pattern #2)** — Add 4x BB levels and configurable RSI/MFI guards (best for ranges)
3. **DCA Layers (Pattern #9)** — Use `position_adjustment_enable` logic for averaging in
4. **Confidence Scoring (Pattern #10)** — Add the 8-factor filter; reject weak signals
5. **Market Regime Detection** — ADX + EMA200 + BB width to classify regime; change strategy accordingly
6. **Cascading Time Exit** — Start with the TrendRider exit cascade (-1.5% at 2h, etc.)
7. **VWAP/OBV/CVD** — Add as confirmation filters on top of existing indicators
8. **Funding Rate Filter** — If using futures, add as a sentiment signal

---

*Research based on: Freqtrade (52k⭐ GitHub), Hummingbot (19k⭐), Jesse AI (8k⭐), Gekko (10k⭐), official docs, and community strategy repos.*