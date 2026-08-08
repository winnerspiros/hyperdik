# Hyperliquid Feature Deep-Research — Complete Inventory

Generated 2026-07-15 from:
- Official `hyperliquid-python-sdk` v0.23+ source code (exchange.py, info.py, signing.py, websocket_manager.py, types.py)
- Live queries to `https://api.hyperliquid.xyz/info`
- Platform mechanics reference (`references/hyperliquid-platform-mechanics.md`)
- Codebase analysis of `/home/ubuntu/revolut-x-trader/`
- GitHub community repos (OctoBot, passivbot, moss-trade-bot-skills, hlp-toshogu)

---

## 1. ORDER TYPES & EXECUTION

### 1.1 Trigger Orders (TP/SL)
**What:** Native stop-loss and take-profit trigger orders. Format:
```python
{"trigger": {"triggerPx": float, "isMarket": bool, "tpsl": "tp" | "sl"}}
```
- `isMarket=True` → aggressive IoC when triggered
- `isMarket=False` → places limit order at `limit_px` when triggered
- `tpsl` distinguishes TP from SL (used by grouping)

**Currently used?** NO — our `order()` method in `hyperliquid_client.py` only supports limit orders with `tif`. No trigger order support.

**Effort:** LOW — just need to add a `trigger_order()` method to the client. The SDK already handles the wire format.

**Benefit ($100 acct):** HIGH — this is the #1 missing safety feature. Without trigger TP/SL, positions are naked. For a 2x leveraged account, a 10% move = 20% account loss. Must-have.

---

### 1.2 Reduce-Only Orders
**What:** `reduce_only=True` flag prevents an order from increasing position size. Market uses it for closing orders.

**Currently used?** PARTIAL — our `order()` method accepts `reduce_only` parameter. `market_close()` uses it automatically. But day_trader/scalper do not pass it, meaning position-closing orders could accidentally flip to opposite side.

**Effort:** LOW — just ensure `reduce_only=True` is passed on all closing orders. Already wired in `hyperliquid_client.py:order()`.

**Benefit ($100 acct):** MEDIUM — prevents accidental position-flip on close attempts.

---

### 1.3 Post-Only / Alo (Add-Liquidity-Only)
**What:** `tif: "Alo"` — order is cancelled if it would cross the spread and take liquidity. Guarantees maker fee (cheaper).

**Currently used?** NO — our client only supports `"gtc"` and `"ioc"`. `"alo"` is listed in the tif_map but our `order()` function defaults to Gtc.

**Effort:** LOW — already supported in the SDK. Just need to call `order(..., order_type="alo")`.

**Benefit ($100 acct):** LOW-MED — maker fee is 1.5 bps vs 4.5 bps taker. On a $100 account, difference is <$0.01/trade. But good practice for limit orders at support/resistance levels (avoid taking liquidity at market).

---

### 1.4 IOC (Immediate-or-Cancel)
**What:** `tif: "Ioc"` — fills what it can immediately, cancels the rest. Used by market_open/market_close as aggressive limit orders.

**Currently used?** YES — `market_open()` and `market_close()` use IoC internally. Also available via our `order()` method.

**Benefit ($100 acct):** Already in use for market entries/exits.

---

### 1.5 GTC (Good-Till-Cancel)
**What:** Default time-in-force. Order rests on book until filled or cancelled.

**Currently used?** YES — our `order()` defaults to GTC.

**Benefit ($100 acct):** Standard. Already used.

---

### 1.6 schedule_cancel (Dead-Man's Switch)
**What:** Schedules automatic cancellation of ALL open orders at a future UTC timestamp (min 5s from now). Max 10 triggers/day. Reset at 00:00 UTC. Pass `None` to cancel the schedule.

**Currently used?** PARTIAL — `hyperliquid_client.py:180-183` has the `schedule_cancel(time_ms)` method, but the daemon never calls it.

**Effort:** LOW — one-liner wired into the daemon startup: set schedule 10 minutes ahead, refresh every cycle.

**Benefit ($100 acct):** HIGH — critical safety for automated trading. If daemon crashes or AWS instance dies, all open orders get cancelled within minutes instead of sitting open indefinitely. Prerequisites for unmanned 24/7 operation.

---

### 1.7 bulk_orders (Atomic Batch Orders)
**What:** Submit multiple orders in a single transaction. All or nothing — if one fails, all fail. Supports:
- `"na"` — independent orders
- `"normalTpsl"` — TP/SL linked to a parent entry order (closes position proportionally)
- `"positionTpsl"` — position-level TP/SL grouping
- `PriorityGrouping({"p": int})` — priority-based ordering within batch

**Currently used?** NO — we use `order()`, `market_open()`, `market_close()` individually. Never call `bulk_orders()` or use grouping.

**Effort:** MEDIUM — SDK method exists. Need to build the order list with proper grouping. Key use case: submit entry + TP + SL in one atomic call so either all execute or none do.

**Benefit ($100 acct):** VERY HIGH — this is the #2 most important missing feature after trigger orders. For a small account, every position MUST have TP/SL set immediately on entry. Without atomic entry+TP+SL, a flash crash between entry and TP/SL placement can wipe the account. With `normalTpsl` grouping, the TP/SL is atomically linked — if entry fills partially, TP/SL sizes auto-adjust.

---

### 1.8 TWAP Implementation Patterns
**What:** HL has NO native TWAP order type. The SDK has `user_twap_slice_fills()` (query-only) suggesting TWAP exists at the protocol level but only via specialized tooling. Community implementations split large orders across time client-side.

**Currently used?** NO.

**Effort:** HIGH — requires client-side time-slicing logic with position tracking.

**Benefit ($100 acct):** LOW — account is too small to need TWAP. Slippage on $5-10 orders is negligible.

---

### 1.9 Client Order IDs (cloid)
**What:** Custom `0x`-prefixed hex order IDs for client-side tracking. Available on order/cancel/modify.

**Currently used?** NO.

**Effort:** LOW — just add `cloid` parameter generation.

**Benefit ($100 acct):** LOW — useful for order reconciliation but not critical for small accounts.

---

### 1.10 modify_order
**What:** Modify an existing order's price, size, TIF without cancelling and re-placing. Maintains queue position for non-price changes. Also `bulk_modify_orders_new()` for batch modifications.

**Currently used?** NO — we cancel+replace instead (losing queue position).

**Effort:** MEDIUM — needs tracking of OIDs and proper modify logic. SDK already supports it.

**Benefit ($100 acct):** LOW — for a small account, losing queue position isn't a big deal. More relevant for market-making strategies.

---

### 1.11 bulk_cancel & cancel_by_cloid
**What:** Cancel multiple orders in one transaction. Also supports cancellation by client order ID.

**Currently used?** NO — we cancel one-by-one via `cancel(coin, oid)`.

**Effort:** LOW — SDK methods exist.

**Benefit ($100 acct):** LOW — small account rarely has many open orders.

---

## 2. ACCOUNT FEATURES

### 2.1 Sub-Accounts
**What:** Create unlimited sub-accounts with isolated margin and positions. Transfer USDC between main and sub-accounts. Query sub-account list via `query_sub_accounts()`. Spot sub-account transfers also available.

SDK methods:
- `create_sub_account(name)` — create named sub-account
- `sub_account_transfer(user, is_deposit, usd)` — USDC to/from sub-account
- `sub_account_spot_transfer(user, is_deposit, token, amount)` — spot assets
- `query_sub_accounts(user)` — list sub-accounts

**Currently used?** NO.

**Effort:** MEDIUM — needs separate wallet management but SDK handles it.

**Benefit ($100 acct):** VERY LOW — sub-accounts are for capital segregation. With $85 total, there's nothing to segregate. Only relevant above ~$5K.

---

### 2.2 Vault Deposits / HLP (Hyperliquidity Provider)
**What:** Deposit USDC to the HLP vault (`0xdfc24b...`) which provides liquidity to order books. Earns fees + spread. Query vault equities via `user_vault_equities()`. Transfer via `vault_usd_transfer()`.

HLP status (2026-07-15): ~$213.7M in vault, 0 active positions, all cash. Earning passive yield from market-making.

**Currently used?** NO.

**Effort:** LOW — `vault_usd_transfer()` exists in SDK.

**Benefit ($100 acct):** VERY LOW — yield on $85 is negligible (~5-15% APY = $4-13/yr). Not worth the complexity.

---

### 2.3 Staking HYPE Token
**What:** Delegate HYPE tokens to validators. Query via `user_staking_summary()`, `user_staking_delegations()`, `user_staking_rewards()`, `delegator_history()`. Transfer via `token_delegate(validator, wei, is_undelegate)`.

**Currently used?** NO.

**Effort:** LOW — SDK methods exist.

**Benefit ($100 acct):** VERY LOW — requires holding HYPE token (not USDC). Irrelevant for USDC-margined perp trading.

---

### 2.4 Referral Codes
**What:** `set_referrer(code)` links your account to a referral code (fee rebate). `query_referral_state(user)` shows referral info. User can earn rebates from referred users.

**Currently used?** NO.

**Effort:** LOW — one SDK call.

**Benefit ($100 acct):** LOW — if user hasn't set a referral, they're missing potential fee rebates. Worth checking if already set. If someone referred them, they get a discount. No downside.

---

### 2.5 Builder Codes / Builder Fees
**What:** `approve_builder_fee(builder, max_fee_rate)` authorizes a builder to collect fees. `BuilderInfo` passed on order placement. For builder-deployed DEXes.

**Currently used?** NO.

**Effort:** MEDIUM — requires understanding builder ecosystem.

**Benefit ($100 acct):** ZERO — for DEX builders, not retail traders.

---

### 2.6 Fee Tiers (VIP)
**What:** Progressive fee discounts based on cumulative notional volume:
| Tier | Volume | Maker | Taker |
|------|--------|-------|-------|
| Base | $0 | 0.015% | 0.045% |
| VIP1 | $5M | 0.012% | 0.040% |
| VIP2 | $25M | 0.008% | 0.035% |
| VIP3 | $100M | 0.004% | 0.030% |
| VIP4 | $500M | 0.0% | 0.028% |

Query via `user_fees(address)`.

**Currently used?** NO — we don't query fee tier.

**Effort:** LOW — one SDK call, informational.

**Benefit ($100 acct):** VERY LOW — $100 account will never reach even VIP1 ($5M). But `user_fees()` also returns `activeReferralDiscount` which IS relevant.

---

## 3. ADVANCED MARGIN & POSITION MANAGEMENT

### 3.1 Isolated vs Cross Margin
**What:** `update_leverage(leverage, name, is_cross=True)` sets leverage and margin mode per coin. Cross = shared margin pool. Isolated = per-position margin. Also `update_isolated_margin(amount, name)` to adjust isolated margin.

**Currently used?** YES — `update_leverage()` is called but `is_cross` defaults to `True` in both our client and the daemon (daemon uses cross margin via `BASE_LEVERAGE=2`).

**Effort:** LOW — already works. Just ensure `is_cross=True` is intentional.

**Benefit ($100 acct):** ALREADY USING. Cross margin with 2x leverage is appropriate for small accounts — prevents isolated liquidation on a single position.

---

### 3.2 modify_order (Detailed)
Covered in 1.10 above.

---

### 3.3 cancel_by_cloid
Covered in 1.11 above.

---

### 3.4 schedule_cancel
Covered in 1.6 above.

---

### 3.5 update_leverage per coin
**What:** Set different leverage for different coins. e.g., BTC at 3x, memecoins at 2x.

**Currently used?** PARTIAL — daemon uses `BASE_LEVERAGE=2` for all coins. Not per-coin.

**Effort:** LOW — just call `update_leverage(lev, coin)` before opening position.

**Benefit ($100 acct):** MEDIUM — allows conservative 2x on volatile alts while using 3x on BTC/ETH (lower volatility, more leverage room). Implements per-coin risk scaling.

---

### 3.6 Withdraw
**What:** `withdraw_from_bridge(amount, destination)` — withdraw USDC from Hyperliquid to Arbitrum. Also `usd_transfer(amount, destination)` to send USDC to another HL address. `spot_transfer(amount, destination, token)` for spot tokens.

**Currently used?** NO.

**Effort:** LOW — SDK methods exist. Not for automated use though.

**Benefit ($100 acct):** LOW — not needed for automated trading. Manual withdrawal via web UI is fine.

---

### 3.7 usdClassTransfer (Spot ↔ Perps)
**What:** `usd_class_transfer(amount, to_perp=True/False)` moves USDC between spot and perps wallets.

**Currently used?** YES — `spot_to_perp()` and `perp_to_spot()` in client call this.

**Benefit ($100 acct):** Already implemented. Critical for initial setup (moving deposited USDC from spot to perps).

---

## 4. SPOT MARKET

### 4.1 Spot Trading Pairs
**What:** Hyperliquid has a full spot DEX. Query spot metadata via `spot_meta()`, spot asset contexts via `spot_meta_and_asset_ctxs()`, spot balances via `spot_user_state()`. Spot pairs include HYPE/USDC, PURR/USDC, etc.

**Currently used?** PARTIAL — `get_account()` queries `spot_user_state()` for USDC balance. But we never trade spot pairs.

**Effort:** MEDIUM — would need spot-specific order methods.

**Benefit ($100 acct):** VERY LOW — spot trading without leverage on $85 is pointless. Might as well use Revolut X for spot. Only relevant if user holds HYPE or other HL-native tokens.

---

### 4.2 Spot Margin
**What:** HL spot has its own margin system. Tokens can be enabled for cross-margin borrowing. Query via `spotClearinghouseState`.

**Currently used?** NO.

**Effort:** HIGH — separate system.

**Benefit ($100 acct):** VERY LOW.

---

### 4.3 Spot Withdraw
**What:** `spot_transfer()` sends spot tokens to another address.

**Currently used?** NO.

**Benefit ($100 acct):** VERY LOW.

---

### 4.4 Deposit Addresses
**What:** Hyperliquid deposits are on Arbitrum. No special API — just send USDC (Arbitrum) to the main wallet address.

**Currently used?** N/A — manual process.

**Benefit ($100 acct):** N/A.

---

## 5. WEBSOCKETS / EVENT STREAMS

### 5.1 Complete WebSocket Channel Inventory
HL's WebSocket supports these subscription types:

| Channel | What | Auth Required | Currently Used? |
|---------|------|--------------|-----------------|
| `allMids` | All mid prices (~500ms updates) | No | YES (WS + REST fallback) |
| `l2Book` | L2 order book per coin | No | YES |
| `trades` | All trade fills per coin | No | YES |
| `bbo` | Best bid/offer per coin | No | NO |
| `candle` | Candlestick updates per coin | No | NO |
| `userEvents` | All user events (fills, orders, positions, funding) | Yes (implicit) | NO |
| `userFills` | User's own trade fills | Yes (user address) | NO |
| `orderUpdates` | Order status changes | Yes (implicit) | NO |
| `userFundings` | User funding payments | Yes (user address) | NO |
| `userNonFundingLedgerUpdates` | Deposits, withdrawals, transfers, liquidations | Yes (user address) | NO |
| `webData2` | Combined user data (positions + orders + fills) | Yes (user address) | PARTIAL — subscribed but broken |
| `activeAssetCtx` | Real-time funding rate + OI + mark per coin | No | NO |
| `activeAssetData` | Coin-specific data per user | Yes (user+coin) | NO |

### 5.2 User Data Channels (Critical — MISSING)
These are the killer channels for a trading bot:

**`userFills`**: Real-time notification of every trade fill. No polling needed to detect fills. This enables immediate post-fill actions (place TP/SL, adjust stops).

**`orderUpdates`**: Order status changes (filled, cancelled, partial fill). Real-time order lifecycle tracking.

**`userFundings`**: Real-time funding payment notifications. Track funding costs as they happen.

**`userNonFundingLedgerUpdates`**: All non-trade account activity — deposits, withdrawals, liquidations. Critical for health monitoring.

### 5.3 Current WS Status
Our `hyperliquid_ws.py` subscribes to: `trades`, `l2Book`, `allMids`, `webData2`. BUT the `webData2` subscription is sent WITHOUT the user address (`"user": ""`), meaning it likely doesn't work. The daemon's reference says the "data thread is broken."

**Effort:** LOW — fix the `webData2` subscription by passing the correct user address and add `userFills` + `orderUpdates` subscriptions. SDK supports all these natively.

**Benefit ($100 acct):** VERY HIGH — this is the #3 most important missing feature. Currently we poll REST for fills/orders/positions (slow, rate-limited, expensive). Real-time event streams enable reactive trading: detect fills instantly, adjust stops on the fly, detect liquidations before they happen. For a leveraged account, reaction time matters.

---

## 6. UTILITY ENDPOINTS

### 6.1 usdClass
**What:** HL's internal accounting unit. All balances, P&L, margin are in USD-class USDC. `usd_class_transfer()` moves between spot/perps.

**Currently used?** YES — via `spot_to_perp()` / `perp_to_spot()`.

**Benefit ($100 acct):** Already handled.

---

### 6.2 L1 vs L2 Data
**What:**
- **L1 (BBO)**: Best bid/offer. Available via WS `bbo` channel and `allMids`.
- **L2 (Order Book)**: Full depth. Available via REST `l2_snapshot(coin)` (returns `levels[0]=bids`, `levels[1]=asks` with `{px, sz, n}`) and WS `l2Book` channel.

**Currently used?** PARTIAL — WS `l2Book` subscribed but `bbo` not subscribed. REST `l2_snapshot()` not used.

**Effort:** LOW — add `bbo` WS subscription and REST `l2_snapshot()` for occasional depth snapshots.

**Benefit ($100 acct):** LOW — BBO is useful for precise entry pricing but mids are sufficient for a small account. L2 depth only matters for sizing >$1K.

---

### 6.3 Funding History
**What:** `funding_history(coin, start_ms, end_ms)` — historical funding rates for a coin up to 500 rows. `user_funding_history(user, start_ms, end_ms)` — your own funding payments.

**Currently used?** PARTIAL — `get_funding_history()` exists in client but no daemon code calls it. User funding history not queried.

**Effort:** LOW — one SDK call.

**Benefit ($100 acct):** HIGH — understanding funding costs is critical for leveraged positions. On HL, funding settles EVERY HOUR (not 8h like Binance). With 2x leverage, negative funding adds up fast. The daemon should track user funding payments to include them in P&L.

---

### 6.4 Predicted Funding (Cross-Exchange)
**What:** `predictedFundings` endpoint returns funding rates for ALL venues simultaneously: `[coin, [[venue, {fundingRate, nextFundingTime, intervalHours}], ...]]`. Shows Binance Perp, Bybit Perp, and HL Perp side-by-side.

**Currently used?** NO — not queried and not even wrapped in our client.

**Effort:** LOW — simple REST query, can be cached.

**Benefit ($100 acct):** HIGH — this is actionable intelligence. When HL funding is significantly different from Binance/Bybit, there are arbitrage opportunities. More importantly: if HL funding rate is at the floor (0.00125%/hr) but Bybit shows deeply negative, shorts dominate on HL = bullish pressure. Directional signal for free.

---

### 6.5 Open Interest
**What:** Available in `meta_and_asset_ctxs()` → `openInterest` field and WS `activeAssetCtx` channel.

**Currently used?** YES — `get_asset_ctxs()` includes OI and displays it in the self-test. Daemon uses it.

**Benefit ($100 acct):** Already used.

---

### 6.6 Oracle Prices
**What:** `oraclePx` in `meta_and_asset_ctxs()`. The price used for liquidations. Critical for understanding liquidation risk.

**Currently used?** YES — `get_oracle_price()` and `check_oracle_divergence()` exist in client. Daemon uses them.

**Benefit ($100 acct):** Already used. Critical for liquidation awareness.

---

### 6.7 Mark Prices
**What:** `markPx` in `meta_and_asset_ctxs()`. The fair price used for P&L calculation.

**Currently used?** YES — included in `get_asset_ctxs()` context.

**Benefit ($100 acct):** Already used.

---

### 6.8 user_role / user_rate_limit
**What:** `user_role(user)` returns account type and permissions. `user_rate_limit(user)` returns rate limit configuration and usage. Important for high-frequency operation.

**Currently used?** NO.

**Effort:** LOW — simple queries.

**Benefit ($100 acct):** LOW — rate limit awareness is useful for production bots but unlikely to be hit at our frequency.

---

### 6.9 portfolio (Performance Tracking)
**What:** `portfolio(user)` returns comprehensive portfolio performance: account value history, PnL history, volume metrics across time periods. Essentially built-in performance analytics.

**Currently used?** NO.

**Effort:** LOW — one SDK call, can feed into the evolution/self-review system.

**Benefit ($100 acct):** MEDIUM — replaces some of our manual P&L tracking. Provides official HL-calculated performance numbers.

---

## 7. API / SDK — ADDITIONAL ENDPOINTS

### 7.1 Builder API
**What:** Full builder infrastructure for deploying custom perp and spot markets: `perp_deploy_register_asset()`, `spot_deploy_register_token()`, `spot_deploy_register_spot()`, auction status queries. Not for retail trading.

**Currently used?** NO.

**Benefit ($100 acct):** ZERO.

---

### 7.2 Vault Strategies
**What:** `user_vault_equities(user)` returns equity across all vaults. `vault_usd_transfer()` to deposit/withdraw. No programmatic vault strategy selection — vault is purely passive HLP.

**Currently used?** NO.

**Benefit ($100 acct):** VERY LOW.

---

### 7.3 Copy-Trading / Leaderboard API
**What:** NO native copy-trading or leaderboard API exists in the SDK. No `copyTrading`, `leaderboard`, or `social` endpoints found. These features don't exist on Hyperliquid natively.

**Community alternatives:** OctoBot and other external tools provide copy-trading by mirroring wallet activity, but there's no official HL endpoint.

**Currently used?** NO.

**Benefit ($100 acct):** ZERO — not available.

---

### 7.4 Historical Orders
**What:** `historical_orders(user)` returns up to 2000 most recent historical orders with current status. `query_order_by_oid(user, oid)` and `query_order_by_cloid(user, cloid)` for individual order lookup.

**Currently used?** NO.

**Effort:** LOW.

**Benefit ($100 acct):** LOW — useful for debugging and trade reconciliation but not for live trading.

---

### 7.5 user_twap_slice_fills
**What:** Query TWAP slice fills for a user.

**Currently used?** NO — not relevant.

**Benefit ($100 acct):** ZERO.

---

### 7.6 non-funding Ledger Updates
**What:** `user_non_funding_ledger_updates(user, startTime, endTime)` — all non-funding account changes: deposits, withdrawals, transfers, liquidations.

**Currently used?** NO.

**Effort:** LOW — one query.

**Benefit ($100 acct):** MEDIUM — useful for account audit log and detecting unexpected withdrawals/liquidations.

---

### 7.7 extra_agents / approve_agent
**What:** `extra_agents(user)` lists authorized agents. `approve_agent(name)` creates a new agent key. Used for multi-key security architecture.

**Currently used?** NO — we use a single API wallet.

**Benefit ($100 acct):** VERY LOW.

---

### 7.8 send_asset (Cross-DEX Transfer)
**What:** Transfer assets between different DEX instances (perps, spot, builder DEXes).

**Currently used?** NO — not relevant.

**Benefit ($100 acct):** ZERO.

---

## 8. PRIORITY IMPLEMENTATION ROADMAP ($100, 2x Leverage)

### TIER 1 — MUST IMPLEMENT (Account Safety)

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 1 | **Trigger TP/SL orders** + `bulk_orders` with `positionTpsl` grouping | MEDIUM | Protects every position. Without this, 10% adverse move = 20% account loss. |
| 2 | **schedule_cancel dead-man's switch** | LOW | Survives daemon crashes. No orphan orders. |
| 3 | **Fix WebSocket user channels** (`userFills`, `orderUpdates`, `webData2` with auth) | LOW | Real-time fill/order tracking. Eliminates polling. |

### TIER 2 — HIGH VALUE ($)

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 4 | **Predicted funding cross-exchange** | LOW | Free directional signal. HL vs Binance/Bybit divergence → trade bias. |
| 5 | **User funding history tracking** | LOW | Include funding costs in P&L. With hourly settlements, this matters. |
| 6 | **Per-coin leverage** (BTC 3x, alts 2x) | LOW | Better risk scaling. More upside on low-vol coins. |
| 7 | **reduce_only on ALL closing orders** | LOW | Prevent accidental position flips. |

### TIER 3 — NICE TO HAVE

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 8 | Post-only (Alo) for limit orders | LOW | Cheaper fees at support/resistance levels. |
| 9 | `user_fees()` fee tier + referral check | LOW | Ensure referral discount is active. |
| 10 | `modify_order` for queue-preserving updates | MEDIUM | Better order management. |
| 11 | Portfolio performance via `portfolio()` | LOW | Built-in HL analytics feed. |
| 12 | L2 snapshot for pre-trade depth check | LOW | Better execution on larger positions. |

### TIER 4 — NOT RELEVANT ($100 account)

| Feature | Reason |
|---------|--------|
| Sub-accounts | No capital to segregate |
| HLP vault deposits | Yield too small on $85 |
| HYPE staking | Requires HYPE token, not USDC |
| TWAP | Position sizes too small |
| Spot trading | Revolut X handles spot better |
| Builder API | Not a DEX operator |
| Copy-trading | Not supported natively |

---

## 9. FEATURE COVERAGE SUMMARY

| Category | Total Features | Currently Used | Missing | Missing % |
|----------|---------------|----------------|---------|-----------|
| Order Types | 11 | 3 (GTC, IOC, market) | 8 | 73% |
| Account | 6 | 1 (fee tiers via lev) | 5 | 83% |
| Advanced | 7 | 3 (lev, transfer, market) | 4 | 57% |
| Spot | 4 | 1 (balance query) | 3 | 75% |
| WebSockets | 13 | 3 (trades, l2Book, allMids) | 10 | 77% |
| Utility | 9 | 4 (mids, OI, oracle, mark) | 5 | 56% |
| API/SDK | 8 | 0 | 8 | 100% |
| **TOTAL** | **58** | **15** | **43** | **74%** |

**Bottom line:** We use 26% of Hyperliquid's available features. The 3 most impactful missing features (trigger TP/SL + grouping, schedule_cancel, real-time user event streams) cost LOW/MEDIUM effort and are essential for safe leveraged trading.
