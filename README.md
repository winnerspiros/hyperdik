# Hyperliquid AI Trading Bot

AI-driven perpetual futures trading bot for [Hyperliquid](https://hyperliquid.xyz). Multi-model ensemble architecture with zero-loss exit system, adaptive rate limiting, and multi-timeframe pump/dump chase detection.

**Status**: Live production · **Equity**: Micro (<$100) · **Style**: High-frequency micro-scalping with AI override

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/winnerspiros/hyperliquid-trading-bot.git
cd hyperliquid-trading-bot

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install hyperliquid-python-sdk eth-account xgboost numpy scikit-learn pandas

# 4. Set up API keys
mkdir -p ~/.hyperliquid
cat > ~/.hyperliquid/config.json << 'EOF'
{
  "api_wallet": "0xYOUR_API_WALLET",
  "api_private_key": "0xYOUR_PRIVATE_KEY",
  "main_wallet": "0xYOUR_MAIN_WALLET"
}
EOF

# 5. (Optional) AI model keys — for AI coin selection
mkdir -p ~/.hermes
cat > ~/.hermes/.env << 'EOF'
OPENROUTER_API_KEY=your_key_here
EOF

# 6. Run
python3 -B hyperliquid_daemon.py
```

The bot loads keys from `~/.hyperliquid/config.json`. AI model keys go in `~/.hermes/.env`. Neither file is in the repo.

---

## Architecture

The bot runs as a single long-lived process (`hyperliquid_daemon.py`) with two parallel loops:

```
┌──────────────────────────────────────────────────────┐
│                   MAIN CYCLE (5s)                     │
│                                                      │
│  Account → Positions → Mids → OI → HSL → AI Market   │
│     ↓                                                 │
│  Sector Scan → Coin Rotation → AI Picks               │
│     ↓                                                 │
│  Per Coin: Enrich → Composite → Unified → ML → Gate   │
│     ↓                                                 │
│  AI Sizing → Debate → Execute (with chase detection)  │
│     ↓                                                 │
│  Monitor Positions → Exit Eval → TP/SL Management     │
└──────────────────────────────────────────────────────┘
```

### How a trade flows

**1. Data Gathering** (every cycle)
Real-time prices via WebSocket + REST fallback. Open interest tracked with 5-min caching. Funding rates, CVD (cumulative volume delta), order book depth, and sector rotation all fed into the signal pipeline.

**2. Coin Selection**
The bot scans 40 coins per cycle (rotating window across all 177 perp markets). AI (Llama 4 Maverick) picks the top 3 candidates with confidence scores. These are force-injected into the selection — the AI is the primary decider.

**3. Signal Generation** — 5 independent layers vote:

| Layer | What it measures | Weight |
|---|---|---|
| **Master composite** | Multi-timeframe technicals + regime + momentum | 40% |
| **Enriched signal** | VWAP deviation, order book imbalance, tick peaks, CVD divergence | 25% |
| **Unified predictor** | ML (XGBoost + LSTM) + market structure + liquidity flow | 20% |
| **Funding signal** | Funding rate extremes, funding acceleration, sniper windows | 10% |
| **Whale signal** | Large wallet activity, accumulation/distribution patterns | 5% |

Layers vote BUY, SELL, or HOLD. The weighted vote produces a composite score (-1.0 to +1.0) and regime classification.

**4. Gating** — trades must pass quality filters:
- **VWAP chase block**: No buying far above VWAP (non-trending) or selling far below
- **ML contradiction**: ML confidence ≥30% against the trade blocks it
- **Composite floor**: Composite must meet minimum threshold for the direction
- **EV gate**: Expected value (reward/risk × win rate) must exceed minimum
- **Extreme regime**: No BUY in `extreme_up`, no SELL in `extreme_down`
- **Survival gate**: Unified confidence must reach threshold (bypassable by AI+momentum)

**5. AI Override**
When AI confidence ≥85%, it can fast-track through gates. But the new chase detection (see below) still applies — AI can't force a trade into an already-pumped coin.

**6. Entry Execution** — 3-tier approach:
- **Normal**: Market IOC at current price
- **Chase detected**: Limit order at pullback price, dynamic wait (25-90s depending on move size)
- **Chase blocked**: Refuse entirely if move is too extreme (e.g. 1h pump >12%)

**7. Exit System** — "zero-loss" design:
- **Breakeven lock**: Once net PnL covers roundtrip fees, exchange SL becomes breakeven — trade can't lose
- **Tiered profit lock**: 3 take-profit levels (40%/35%/25% of position) at escalating prices
- **Trailing stop**: After TP2 fills, remaining 25% trails behind price
- **Emergency close**: Underwater + momentum against us → cut immediately
- **Positive timeout**: Net positive after 5 min → close to free capital
- **Negative positions**: Never closed at a loss — held for recovery

---

## Module Map

### Core Loop
| File | Role |
|---|---|
| `hyperliquid_daemon.py` | Main process — both loops, all orchestration (5300+ lines) |
| `hyperliquid_client.py` | Hyperliquid SDK wrapper, adaptive rate limiter, all API calls |

### Signal Pipeline
| File | Role |
|---|---|
| `hyperliquid_strategy.py` | 5-pillar composite signal, regime detection, conviction scoring |
| `unified_predictor.py` | ML + structure + flow + macro → unified BUY/SELL/HOLD |
| `unified_direction.py` | Direction prediction with VWAP mean reversion |
| `continuous_predictor.py` | Continuous learning prediction with online updates |
| `ml_predictor.py` | XGBoost + LSTM model predictions |
| `market_predictor.py` | Extended market data (multi-TF, order book, patterns) |
| `market_regime.py` | Regime classification (trending_up/down, high_vol, sideways, extreme) |
| `market_structure.py` | Support/resistance, market structure breaks |
| `vwap_signal.py` | VWAP deviation signals with mean reversion detection |
| `cvd_engine.py` | Cumulative volume delta — real-time buy/sell pressure |
| `oi_delta.py` | Open interest delta tracking — positioning shifts |
| `funding_signals.py` | Funding rate signal generation |
| `funding_acceleration.py` | Funding rate change velocity |
| `peak_exhaustion_detector.py` | Local top/bottom detection |
| `tick_peak_detector.py` | Tick-level peak detection for micro entries |
| `chart_patterns.py` | Classical chart pattern recognition |
| `sector_rotation.py` | Sector-level rotation scanning |
| `correlation_tracker.py` | Rolling correlation matrix — concentration risk |
| `order_book_imbalance.py` | Bid/ask imbalance from L2 data |
| `volume_profile.py` | Volume profile analysis |
| `volume_delta.py` | Volume delta (buy vs sell volume) |
| `taker_ratio.py` | Taker buy/sell ratio |
| `imbalance_trend.py` | Order book imbalance trending |

### AI Layer
| File | Role |
|---|---|
| `ai_decider.py` | AI coin selection, sizing, exit evaluation, debate (OpenRouter) |
| `ai_brain.py` | AI context building — market assessment, trend analysis |
| `ai_validator.py` | AI trade validation — sanity checks on AI decisions |
| `self_improving_ai.py` | AI self-improvement loop — learns from trade outcomes |

### Execution & Exit
| File | Role |
|---|---|
| `hyperliquid_execution.py` | Multi-tier TP, trailing stops, fee-aware position sizing |
| `hyperliquid_exit.py` | Exit plan builder — stop loss, take profit, trail logic |
| `action_executor.py` | Order execution engine |
| `action_validator.py` | Pre-execution validation with retry |

### Risk
| File | Role |
|---|---|
| `hyperliquid_risk.py` | WEL/TWEL limits, HSL drawdown protection, cooldowns, unstucking |
| `hrp_sizing.py` | Hierarchical Risk Parity position sizing |
| `kelly_sizing.py` | Kelly criterion optimal bet sizing |
| `ev_gate.py` | Expected value gate — blocks negative-EV trades |
| `risk_manager.py` | Portfolio-level risk management |
| `risk_manager_v2.py` | Improved risk manager with regime awareness |

### Specialized Strategies
| File | Role |
|---|---|
| `hyperliquid_whale.py` | Whale wallet tracking — accumulation/distribution |
| `hyperliquid_funding_sniper.py` | Funding rate arbitrage signals |
| `hyperliquid_delta_neutral.py` | Delta-neutral funding rate arbitrage |
| `hyperliquid_evolution.py` | LLM-driven parameter optimization (weekly) |
| `liquidation_monitor.py` | Liquidation cascade risk detection |
| `cryptogat_predictor.py` | Graph attention network predictions |
| `kronos_predictor.py` | Multi-timeframe predictions |
| `cross_exchange.py` | Cross-exchange price comparison (Binance, OKX) |

### Data & Infrastructure
| File | Role |
|---|---|
| `hyperliquid_ws.py` | WebSocket client — real-time trades, order book, candles |
| `context_enricher.py` | Signal enrichment with VWAP, OI, CVD, order book |
| `portfolio_reviewer.py` | Portfolio performance review |
| `performance_metrics.py` | Trade performance tracking |
| `trader_db.py` | SQLite trade database |
| `telegram_alerts.py` | Telegram notification integration |
| `rapid_monitor.py` | Fast price monitor for extreme moves |

---

## Key Mechanisms

### Adaptive Rate Limiter
Self-tuning throttle that adjusts based on actual API responses:
- Starts at 30ms interval (33 req/s) — faster than the old static 50ms
- 30-second sliding window tracks 429 error rate
- Automatically throttles to 200ms (5 req/s) when API is pressured
- Recovers as 429s clear from the window
- Exposes `pressure_level()` and `is_throttled()` for callers
- Circuit breaker after 3+ consecutive 429s skips non-critical data fetching

### Pump/Dump Chase Detection
Multi-timeframe guard against buying tops and selling bottoms:

| Timeframe | Move in trade dir | Action |
|---|---|---|
| 1m | >1% | Wait for 0.3% pullback, 25s |
| 5m | >2% | Wait for 0.5% pullback, 35s |
| 15m | >3.5% | Wait for 0.8% pullback, 50s |
| 1h | >6% | Wait for 1.5% pullback, 70s |
| 4h | >10% | Wait for 2.5% pullback, 90s |
| 1m | >3% | **BLOCK** |
| 5m | >5% | **BLOCK** |
| 15m | >7% | **BLOCK** |
| 1h | >12% | **BLOCK** |
| 4h | >20% | **BLOCK** |

For LONG trades, "move in trade direction" = price went UP (we want to buy dips, not chase pumps).
For SHORT trades, "move in trade direction" = price went DOWN (we want to short bounces, not chase dumps).
Moves AGAINST our direction (LONG into a dip, SHORT into a bounce) pass through — those are ideal entries.

### HSL (Hysteresis Stop Loss)
4-tier equity protection that progressively restricts trading as drawdown increases:
- **GREEN** (>95% of peak): Normal trading, full position sizes
- **YELLOW** (90-95%): Reduced size, tighter stops
- **ORANGE** (80-90%): Half size, only high-conviction trades
- **RED** (<80%): No new positions, close-only mode

### WEL/TWEL Limits
- **TWEL** (Total Weighted Exposure Limit): Max % of equity in positions — scales dynamically with account size
- **WEL** (Weighted Exposure Limit): Per-position cap × correlation penalty — prevents overconcentration

---

## Configuration

### Required: Hyperliquid Keys
```json
// ~/.hyperliquid/config.json
{
  "api_wallet": "0x...",      // API wallet address (for trading)
  "api_private_key": "0x...", // API wallet private key
  "main_wallet": "0x..."      // Main wallet address (for deposits)
}
```

Create the API wallet in Hyperliquid UI under Settings → API Wallets. Use a dedicated trading wallet, not your main wallet. The API wallet only needs trading permissions — withdraw permissions should stay OFF.

### Optional: AI Model Keys
```bash
# ~/.hermes/.env
OPENROUTER_API_KEY=sk-or-v1-...   # For AI coin selection (Llama 4 Maverick)
OPENAI_API_KEY=sk-...             # Alternative AI provider
TOGETHER_API_KEY=...              # Alternative AI provider
```

Without AI keys, the bot runs on deterministic signals only (still trades, just no AI coin picks or market assessment).

### Optional: Exchange Comparisons
```json
// ~/.kraken/creds.json (for Kraken price comparison)
// ~/.okx/config.json (for OKX price comparison)
```

The bot uses cross-exchange data for validation — it's optional and the bot functions without it.

---

## Running 24/7

### Systemd (recommended)
```bash
sudo cp unified-orchestrator.service /etc/systemd/system/
sudo systemctl enable unified-orchestrator
sudo systemctl start unified-orchestrator
```

### Manual with watchdog
```bash
./watchdog.sh   # Auto-restarts if daemon dies
```

### Logs
```bash
tail -f logs/hyperliquid_daemon.log        # Main log
tail -f logs/hyperliquid_daemon_error.log  # Errors only
```

---

## Key Variables

These are tuned for a micro account. Adjust for larger accounts:

| Variable | Value | Location |
|---|---|---|
| Cycle speed | 5 seconds | `CYCLE_SECONDS` in daemon |
| Coins scanned | 40 per cycle | `_ROTATE_WINDOW` in daemon |
| Max positions | 3 | `max_positions` in AI market |
| Default leverage | 6x | `BASE_LEVERAGE` in daemon |
| Position size | 20% of equity | AI sizing output |
| Roundtrip fee | 0.07% at 6x | `ROUNDTRIP_FEE_PCT` × leverage |
| AI models | Llama 4 Maverick (free) + Qwen 30B | `ai_decider.py` |
| ML models | XGBoost (primary) + LSTM | `ml_predictor.py` |
| WebSocket | Real-time trades + L2 book | `hyperliquid_ws.py` |

---

## Safety Notes

- **Never commit API keys**. They live in `~/.hyperliquid/` outside the repo.
- **The bot trades real money**. Test with --dry-run or on testnet first.
- **Micro accounts need aggressive sizing** (20-35% per position). Scale down for larger accounts.
- **The exit system prevents losses but not all losses**. Market gaps, liquidation cascades, and exchange downtime can still cause losses.
- **AI models can be wrong**. The chase detection and data consensus gates exist because AI ≠ truth.