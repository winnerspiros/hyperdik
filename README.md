# HYPERDIK

<p align="center">
  <img src="assets/hyperdik-logo.jpg" alt="HYPERDIK" width="384">
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python"></a>
  <a href="https://hyperliquid.xyz"><img src="https://img.shields.io/badge/Exchange-Hyperliquid-green" alt="Hyperliquid"></a>
  <a href="https://openrouter.ai"><img src="https://img.shields.io/badge/AI-Qwen_235B_%2B_GPT--4.1-purple" alt="AI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
</p>

Autonomous perpetual futures trading bot — 177 markets, 5-second cycles, three AI models.

---

## 🧠 Architecture

```
  ┌───────────┐     ┌────────────┐     ┌───────────┐     ┌────────────┐
  │  📡 DATA  │────▶│  🧠 AI     │────▶│ ⚡ EXECUTE │────▶│  🛡️ EXIT  │
  │           │     │            │     │           │     │            │
  │ WS mids   │     │ Coin picks │     │ Micro-peak│     │ Breakeven  │
  │ OI · CVD  │     │ Sizing     │     │ 6-12× lev │     │ Profit lock│
  │ Orderbook │     │ Exit eval  │     │ Market IOC│     │ AI re-check│
  │ F&G · News│     │ GPT-4.1 🔍 │     │ TP/SL set │     │ Trend-kill │
  └───────────┘     └─────┬──────┘     └───────────┘     └─────┬──────┘
                          │                                    │
                          ▼                                    ▼
                   ┌──────────────┐                   ┌──────────────┐
                   │ 🧬 EVOLUTION │                   │ 📈 ANALYTICS │
                   │              │                   │              │
                   │ Full rewrite │                   │ Post-trade   │
                   │ Auto-deploy  │                   │ Prompt optim │
                   │ Git commit   │                   │ Weight learn │
                   │ Auto-restart │                   │ Accuracy log │
                   └──────────────┘                   └──────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/winnerspiros/hyperdik.git && cd hyperdik
python3 -m venv .venv && source .venv/bin/activate
pip install hyperliquid-python-sdk eth-account xgboost numpy scikit-learn pandas

# Hyperliquid wallet
mkdir -p ~/.hyperliquid && cat > ~/.hyperliquid/config.json << 'EOF'
{"api_wallet": "0xYOUR_API", "api_private_key": "0xYOUR_KEY", "main_wallet": "0xYOUR_MAIN"}
EOF

# OpenRouter API key (optional — enables AI)
export OPENROUTER_API_KEY=your_key_here

python3 -B hyperliquid_daemon.py
```

---

## 🤖 AI Models

| Model | Role | Prompt | Completion |
|---|---|---|---|---|
| **Qwen 235B MoE** | Decisions, sizing, exits | $0.64/M | $1.28/M |
| **Llama 4 Maverick** | Coin picks, market scan | $1.00/M | $3.00/M |
| **Llama 4 Scout** | Fallback (rate-limit) | $0.40/M | $0.80/M |
| **GPT‑4.1** | Evolution, post-trade | $2.00/M | $8.00/M |

*All via OpenRouter. Prompt caching active (90% off repeated system prompts).*

All via OpenRouter. GPT‑4.1 runs web-enabled (5 searches per run, 1M context window) — it reads the entire codebase and rewrites prompts, parameters, and logic autonomously.

### Estimated Daily Costs

| Component | Frequency | ~Cost/day |
|---|---|---|
| Live trading AI (Qwen + Llama) | ~2,880 calls/day | ~$1.10 |
| Post-trade analysis (GPT‑4.1) | Per trade close | ~$0.30 |
| Evolution (GPT‑4.1 + web) | Every ~1.7h (~14/day) | ~$0.50 |
| **Total** | | **~$1.90/day** |

*Prompt caching (90% off repeated system prompts) active on all calls. Web search adds ~$0.005/result.*

---

## ⚙️ How Trades Work

### Entry
1. **AI scans** 40 coins per cycle, picks top 2–3 with direction, confidence, target, stop
2. **5 signal layers** vote: composite · enriched ML · funding · whale · VWAP
3. **7 quality gates**: VWAP extremes · ML contradiction · composite floor · EV gate · regime · survival · chase block
4. **Micro-peak timing**: WebSocket mids at 150ms for up to 4s — catches best tick

### Exit
| Trigger | What happens |
|---|---|
| 🔒 Breakeven lock | Net ≥ fees → SL moved to entry → **can't lose** |
| 💰 Profit tiers | 0.25% · 0.50% · 1.50% · 3.00% → escalating partial closes |
| 🧠 AI re-check | 5 min flat → AI asked before kill (extends to 10 min if yes) |
| 📉 Trend-kill | 4h strongly against → immediate close |
| 🩸 Bleed | Peak dropping + momentum turning → cut |
| 🚨 Liquidation | <1% distance → force escape |

---

## 🧬 Evolution — Autonomous Self-Optimization

Every ~1.7 hours, GPT‑4.1 reads the **entire system state** and autonomously improves it:

- **Reads**: full trade history, per-coin PnL, signal weights, AI prompts, daemon params, recent logs
- **Searches**: 5 web queries for market context, strategy research, coin-specific news
- **Writes**: actual code changes — rewrites AI prompts, modifies parameters, adjusts logic
- **Deploys**: auto git-commits, restarts the daemon if code changed
- **Cost**: ~$0.04 per run (~$0.50/day)

**Prompt optimizer** runs alongside — records which prompt versions produce winning trades, then meta-prompts GPT‑4.1 to generate better prompts. A/B tests variants and graduates winners.

Toggle it all in `config.yaml`.

---

## 📦 Modules

| Layer | Files |
|---|---|
| 🧠 **AI** | `ai_decider.py` · `ai_validator.py` · `market_intel.py` |
| 🔮 **Prediction** | `continuous_predictor.py` · `unified_predictor.py` · `perfect_predictor.py` · `ml_predictor.py` · `cryptogat_predictor.py` · `kronos_predictor.py` |
| 📊 **Signals** | `hyperliquid_strategy.py` · `unified_direction.py` · `vwap_signal.py` · `oi_delta.py` · `cvd_engine.py` · `cvd_divergence.py` · `taker_ratio.py` · `volume_delta.py` · `imbalance_trend.py` · `market_regime.py` · `market_structure.py` · `hurst_detector.py` · `chart_patterns.py` · `funding_acceleration.py` · `hyperliquid_funding_sniper.py` |
| 📡 **Data** | `context_enricher.py` · `market_analyzer.py` · `price_extremes.py` · `correlation_tracker.py` · `cross_exchange.py` · `sector_rotation.py` · `economic_calendar.py` |
| 🛡️ **Risk** | `hyperliquid_risk.py` · `hrp_sizing.py` · `ev_gate.py` |
| ⚡ **Execution** | `hyperliquid_client.py` · `hyperliquid_execution.py` · `action_executor.py` · `hyperliquid_ws.py` |
| 🧬 **Evolution** | `hyperliquid_evolution.py` → GPT‑4.1 autonomous optimizer · `prompt_optimizer.py` → meta-prompt improvement |
| 🔧 **Extensions** | `hyperliquid_delta_neutral.py` · `hyperliquid_learner.py` · `hyperliquid_whale.py` |
| 💀 **Liquidation** | `liquidation_zones.py` · `liquidation_monitor.py` |
| ⛰️ **Peaks** | `peak_exhaustion_detector.py` · `pump_detector.py` |
| 🏠 **Main** | `hyperliquid_daemon.py` — 6,000 lines, the main loop |

📖 Full map in [`STRUCTURE.md`](STRUCTURE.md)

---


---

## 💻 System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 256 MB | 512 MB |
| CPU | 1 vCPU | 2 vCPU |
| Storage | 100 MB | 500 MB |
| Python | 3.10+ | 3.12+ |
| Network | 1 Mbps | 10 Mbps |

**Storage notes**: All files are capped — AI cache (200 entries, ~150KB), logs (500 lines, ~1MB), trade memory (500 entries). WebSocket buffers trimmed to 20 fills / 5 funding updates. No unbounded growth.

## ⚙️ Configuration

All features toggleable in `config.yaml`:

```yaml
ai.enabled:        true   # Master AI switch
trading.enabled:   true   # Set false for dry-run
evolution.enabled: true   # GPT-4.1 autonomous optimizer
analysis.enabled:  true   # Post-trade AI review
entry.micro_peak:  true   # 150ms WS sampling
exit.breakeven_lock: true # Never lose on fees
```

---

## 🎯 Defaults

| What | Value |
|---|---|
| Leverage | 6× (10–12× on full confluence) |
| Position size | 20% equity (35% on confluence) |
| Max positions | 3 |
| Entry | Market IOC · 0.5% slippage · micro-peak 4s |
| Cycle | 5 seconds |
| Markets | 177 perpetuals, rotating 40/cycle |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
