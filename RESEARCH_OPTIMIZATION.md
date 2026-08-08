# Hyperliquid AI Trading Bot — Optimization Research
# Compiled 2026-07-14 from GitHub, SDK docs, forums, trading literature
# ==============================================================================

## SOURCES RESEARCHED
- hyperliquid-python-sdk (official)
- hyperliquid (Rust SDK by nktkas)
- JKorf/HyperLiquid.Net (C# SDK)
- chainstacklabs/hyperliquid-trading-bot (grid bot)
- Superior-Trade/superior-skills (211 stars)
- titouannwtt/freqtrade-ultimate (11 stars, 58K lines)
- ryomenhaider/RegimeIQ (microstructure)
- Co-Messi/HyperData-Terminal
- djienne/CROSS_EXCHANGE_DELTA_NEUTRAL
- Hyperliquid GitBook docs

## IMPROVEMENTS TO IMPLEMENT (Priority Order)

### TIER 1 — MUST HAVE (High Impact / Low Effort)
----------------------------------------------------------------------

#### 1. Liquidation & External Close Detection
Source: Freqtrade Ultimate
What: Detect when Hyperliquid force-closes a position or ADL triggers.
These events happen WITHOUT our order and we need to react immediately.
Implementation: After every cycle, compare positions vs known state. If a
position disappeared without our close action, log LIQUIDATION alert.
Status: NEW — add to position monitor

#### 2. Funding Rate Crowding Signal
Source: Multiple (DeltaNeutral, Freqtrade, VOOI)
What: When Hyperliquid funding rate is extremely positive (e.g. >0.01% per 8h),
it means longs are crowded — high probability of mean reversion / long squeeze.
Conversely, very negative = shorts crowded = short squeeze risk.
Use this as a CONTRARIAN signal: avoid going with the crowd.
Implementation: Query Hyperliquid's own funding rate (via metaAndAssetCtxs).
If funding > 0.01%: bias SHORT. If < -0.01%: bias LONG.

#### 3. Dry-Run Replay Mode
Source: Freqtrade Ultimate
What: Run the ACTUAL bot engine (not simplified backtester) on historical
1-minute data with real funding rates and position management.
Implementation: Record all API responses for a period, then replay them
through the daemon in "replay mode" to test strategy without real money.
Status: NEW — build replay engine

#### 4. Staggered WebSocket Subscriptions
Source: Freqtrade Ultimate
What: Don't subscribe all symbols at once — stagger by 200ms per symbol
to avoid triggering rate limits.
Implementation: Add time.sleep(0.2) between each subscribe call.
Status: Fix hyperliquid_ws.py

#### 5. Regime Transition Probability
Source: RegimeIQ
What: Not just "current regime = ranging", but "probability of shifting to
trending in next hour = 35%". This helps the AI size positions accordingly.
Implementation: Track regime changes over time. Build transition matrix.
If probability of breakout > 50%, increase position size.

### TIER 2 — HIGH VALUE (Medium Effort)
----------------------------------------------------------------------

#### 6. CVD (Cumulative Volume Delta) Signal
Source: HyperData Terminal, RegimeIQ
What: Track net buying vs selling volume in real-time. Rising CVD with
flat price = accumulation. Falling CVD with flat price = distribution.
Implementation: Already have volume_delta.py — enhance to feed CVD signal
into AI context every cycle.

#### 7. Order Flow Imbalance (OFI) Enhancement
Source: RegimeIQ, grantreed1/Crypto-Order-Flow-Imbalance
What: Current order_book_imbalance.py is basic. Enhance to track:
- OFI trend over last N seconds (not just snapshot)
- OFI divergence from price (price up but OFI negative = bearish divergence)
Implementation: Store OFI history in deque, compute rolling correlation with price.

#### 8. Wallet Anti-Compounding (Auto Profit Withdrawal)
Source: Freqtrade Ultimate
What: When equity exceeds a threshold, auto-withdraw profits to a cold wallet.
Prevents "one bad trade loses everything" syndrome.
Implementation: After each profitable close, if total equity > 150% of
initial deposit, withdraw the excess to main wallet.

#### 9. Position Coordination (Multi-Bot Safety)
Source: Freqtrade Ultimate
What: If running multiple strategies on same account, coordinate them
to avoid conflicting positions (e.g. one goes LONG BTC, other SHORT BTC).
Implementation: Shared state file with active positions. Before opening
new position, check no conflicting position exists.

#### 10. Cycle Profiling & Health Checks
Source: Freqtrade Ultimate
What: Log how long each cycle takes. If cycle > 30s, something is wrong.
Also: detect when bot hasn't placed an order in N cycles (stuck/broken).
Implementation: Add timing wrapper around cycle execution. Alert if
cycle_time > CYCLE_SECONDS * 0.8.

### TIER 3 — ADVANCED (Lower Priority / Higher Effort)
----------------------------------------------------------------------

#### 11. Kyle's Lambda (Price Impact Estimation)
Source: RegimeIQ
What: Estimate how much price moves per $1 of order flow. Helps with
position sizing — large orders in illiquid coins have high impact.
Implementation: From order book depth, compute lambda = (ask[1]-bid[1]) /
(bid_size + ask_size). Use to cap position size.

#### 12. Funding Rate Term Structure
Source: Hyperliquid API
What: Query funding history to see if rates are accelerating or decelerating.
Accelerating positive = FOMO building. Decelerating = crowd unwinding.
Implementation: Get last 3 funding payments, compute slope. Feed to AI.

#### 13. TWAP/VWAP Execution for Large Orders
Source: Freqtrade Ultimate, JKorf
What: For positions >$100, split into smaller chunks over time to minimize
slippage and market impact. Already built into Hyperliquid SDK.
Implementation: Use bulk_orders with time-spaced limit orders.

#### 14. Heatmap-Based Entry Timing
Source: HyperData Terminal
What: Track liquidation clusters. When many liquidations happen at a price
level, it becomes strong support/resistance.
Implementation: Query Hyperliquid liquidation data, build level map.
Feed "liquidation support" and "liquidation resistance" to AI context.

### TIER 4 — LONG-TERM / OPTIONAL
----------------------------------------------------------------------

#### 15. Cross-Exchange Delta Neutral
Source: DeltaNeutral bot, VOOI
What: When Hyperliquid funding differs from other exchanges by >5% APR,
open delta-neutral position to capture the spread.
Complexity: HIGH (need another exchange account, coordination).

#### 16. Sub-Account Strategy Isolation
Source: Superior-Trade
What: Run different strategies on separate Hyperliquid sub-accounts.
Isolates risk per strategy and makes PnL attribution clean.
Complexity: MEDIUM (need sub-account setup).

#### 17. Copy Trading / Mirror Orders
Source: chainstacklabs examples
What: Mirror profitable traders' positions automatically.
Warning: Blindly copying is dangerous. Use as signal only + AI validation.
Complexity: LOW to implement, HIGH risk.

======================================================================
## IMMEDIATE ACTION PLAN

1. Add liquidation detection → position monitor (30 min)
2. Add funding rate crowding signal → AI context (15 min)
3. Fix staggered WebSocket → hyperliquid_ws.py (5 min)
4. Add regime transition tracking → market_regime.py (30 min)
5. Add cycle profiling → daemon (15 min)
6. Enhance CVD/OFI → existing modules (30 min)

TOTAL: ~2 hours of work for TIER 1 improvements.
======================================================================
