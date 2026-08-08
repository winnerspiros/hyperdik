"""
Rapid Market Monitor — detects fast price moves across multiple sources.
- Polls CoinGecko + Revolut X every 30 seconds
- Tracks price history (5min window = 10 samples)
- Triggers alerts on sudden drops/pumps/volume spikes
- Writes alerts to data/rapid_alerts.json (shared with all agents)
- Writes actionable signals to pending_actions/ for validator pipeline
- Runs as continuous background process alongside day_trader/scalper
"""
import sys, os, time, json, logging, urllib.request, ssl
from datetime import datetime, timezone

TRADER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRADER_DIR)
os.chdir(TRADER_DIR)

LOG_DIR = os.path.join(TRADER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("RapidMonitor")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "rapid_monitor.log"))
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
log = logger

# ── Configuration ────────────────────────────────────────────────────────────
CHECK_INTERVAL = 30          # seconds between checks
HISTORY_SIZE = 10             # keep 10 samples per coin (5min window at 30s)
ALERTS_FILE = os.path.join(TRADER_DIR, "data", "rapid_alerts.json")
PENDING_DIR = os.path.join(TRADER_DIR, "data", "pending_actions")
os.makedirs(PENDING_DIR, exist_ok=True)

# Coins to monitor (CoinGecko IDs + Revolut symbols)
MONITOR_COINS = {
    "bitcoin":       {"symbol": "BTC",  "cg_id": "bitcoin"},
    "ethereum":      {"symbol": "ETH",  "cg_id": "ethereum"},
    "ripple":        {"symbol": "XRP",  "cg_id": "ripple"},
    "solana":        {"symbol": "SOL",  "cg_id": "solana"},
    "chainlink":     {"symbol": "LINK", "cg_id": "chainlink"},
    "internet-computer": {"symbol": "ICP", "cg_id": "internet-computer"},
    "cardano":       {"symbol": "ADA",  "cg_id": "cardano"},
    "dogecoin":      {"symbol": "DOGE", "cg_id": "dogecoin"},
    "polkadot":      {"symbol": "DOT",  "cg_id": "polkadot"},
    "avalanche-2":   {"symbol": "AVAX", "cg_id": "avalanche-2"},
    "arbitrum":      {"symbol": "ARB",  "cg_id": "arbitrum"},
    "aptos":         {"symbol": "APT",  "cg_id": "aptos"},
    "pepe":          {"symbol": "PEPE", "cg_id": "pepe"},
    "bonk":          {"symbol": "BONK", "cg_id": "bonk"},
    "sui":           {"symbol": "SUI",  "cg_id": "sui"},
}

# Price thresholds
DROP_THRESHOLD_PCT = -2.0      # % drop to trigger alert (-2%)
PUMP_THRESHOLD_PCT = 3.0       # % pump to trigger alert (+3%)
EXTREME_DROP_PCT = -4.0        # % drop to trigger pending action
EXTREME_PUMP_PCT = 6.0         # % pump to trigger pending action
VOLUME_SPIKE_THRESHOLD = 2.5   # volume multiplier vs average

# ── CoinGecko fetcher ────────────────────────────────────────────────────────
def _coingecko_fetch(ids):
    """Fetch prices and 24h volume from CoinGecko for given coin IDs."""
    ctx = ssl.create_default_context()
    coins_str = ",".join(ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coins_str}&vs_currencies=eur&include_24hr_vol=true"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.warning(f"CoinGecko fetch error: {e}")
        return {}

# ── Market monitor ───────────────────────────────────────────────────────────
class RapidMonitor:
    def __init__(self):
        self.price_history = {}   # symbol -> [{"time": ts, "price": float, "volume": float}, ...]
        self.recent_alerts = []   # last 20 alerts
        self._revolut_client = None
        self._alert_cooldowns = {}  # symbol -> timestamp of last alert submission
        self._alert_cooldown_secs = 300  # 5 min cooldown between alerts for same coin
        self._cg_backoff = 0  # exponential backoff for CoinGecko 429s

    def _get_revolut_prices(self):
        """Fallback: fetch prices from Revolut X tickers."""
        try:
            if self._revolut_client is None:
                from revolut_client import RevolutXClient
                self._revolut_client = RevolutXClient()
            ticks = self._revolut_client.get_tickers()
            data = ticks if isinstance(ticks, list) else ticks.get("data", [])
            prices = {}
            for t in data:
                sym = t.get("symbol", "").replace("/", "-")
                base = sym.split("-")[0]
                if base in ("EUR", "USDC", "USD"):
                    continue
                bid = float(t.get("bid", 0) or 0)
                ask = float(t.get("ask", 0) or 0)
                mid = (bid + ask) / 2 if bid and ask else 0
                if mid > 0:
                    prices[base] = round(mid, 6)
            return prices
        except Exception as e:
            log.warning(f"Revolut tickers fallback failed: {e}")
            return {}

    def _get_coin_ids(self):
        return [info["cg_id"] for info in MONITOR_COINS.values()]

    def _cg_fetch_with_retry(self, ids, retries=3):
        """Fetch from CoinGecko with exponential backoff."""
        for attempt in range(retries):
            # Apply backoff if we were rate-limited before
            if self._cg_backoff > 0:
                wait = min(self._cg_backoff, 60)
                log.info(f"CoinGecko backoff: waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            data = _coingecko_fetch(ids)
            if data:
                self._cg_backoff = max(0, self._cg_backoff - 5)  # Decay backoff on success
                return data
            # Exponential backoff: 5, 10, 20s
            self._cg_backoff = min(60, max(5, self._cg_backoff * 2) if self._cg_backoff > 0 else 5)
            if attempt < retries - 1:
                time.sleep(self._cg_backoff)
        return {}

    def _check_and_alert(self, symbol, current_price, current_vol, now):
        """Check for rapid moves and generate alerts."""
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return None

        oldest = history[0]
        window_secs = now - oldest["time"]
        if window_secs < 30:
            return None

        # Price change over the tracked window
        old_price = oldest["price"]
        if old_price <= 0:
            return None
        change_pct = (current_price - old_price) / old_price * 100

        # Recent change (last 2 samples = ~60s)
        recent = history[-2] if len(history) >= 2 else None
        recent_change = None
        if recent and recent["price"] > 0:
            recent_change = (current_price - recent["price"]) / recent["price"] * 100

        # Volume check
        avg_vol = sum(h["volume"] for h in history) / len(history) if history else 0
        vol_spike = (current_vol / avg_vol) if avg_vol > 0 else 1.0

        # Determine alert type
        alert = None
        severity = "info"

        if change_pct <= DROP_THRESHOLD_PCT:
            severity = "warning"
            alert = {
                "type": "rapid_drop",
                "symbol": symbol,
                "change_pct": round(change_pct, 2),
                "recent_change_pct": round(recent_change, 2) if recent_change else None,
                "price_old": round(old_price, 6),
                "price_current": round(current_price, 6),
                "vol_spike": round(vol_spike, 2),
                "window_secs": round(window_secs),
                "time": datetime.now(timezone.utc).isoformat(),
            }
        elif change_pct >= PUMP_THRESHOLD_PCT:
            severity = "warning"
            alert = {
                "type": "rapid_pump",
                "symbol": symbol,
                "change_pct": round(change_pct, 2),
                "recent_change_pct": round(recent_change, 2) if recent_change else None,
                "price_old": round(old_price, 6),
                "price_current": round(current_price, 6),
                "vol_spike": round(vol_spike, 2),
                "window_secs": round(window_secs),
                "time": datetime.now(timezone.utc).isoformat(),
            }
        elif vol_spike >= VOLUME_SPIKE_THRESHOLD:
            severity = "info"
            alert = {
                "type": "volume_spike",
                "symbol": symbol,
                "change_pct": round(change_pct, 2),
                "vol_spike": round(vol_spike, 2),
                "price_current": round(current_price, 6),
                "window_secs": round(window_secs),
                "time": datetime.now(timezone.utc).isoformat(),
            }

        # For extreme moves, also submit a pending action
        if alert and alert["change_pct"] is not None:
            now_ts = time.time()
            last_alert = self._alert_cooldowns.get(symbol, 0)
            if now_ts - last_alert < self._alert_cooldown_secs:
                return alert  # Skip submission, but still return alert for tracking
            if change_pct <= EXTREME_DROP_PCT:
                self._submit_signal("alert", symbol, current_price, change_pct, f"Rapid drop {change_pct:.1f}% in {window_secs:.0f}s")
                self._alert_cooldowns[symbol] = now_ts
            elif change_pct >= EXTREME_PUMP_PCT:
                self._submit_signal("alert", symbol, current_price, change_pct, f"Rapid pump {change_pct:.1f}% in {window_secs:.0f}s")
                self._alert_cooldowns[symbol] = now_ts

        return alert

    def _submit_signal(self, signal_type, symbol, price, change_pct, reasoning):
        """Write alert signal to pending_actions for the pipeline."""
        record = {
            "type": signal_type,
            "symbol": symbol,
            "side": "alert",
            "sell_all": False,
            "price": price,
            "order_type": "market",
            "size": None,
            "source": "rapid_monitor",
            "reasoning": f"{reasoning} | {symbol} @ €{price:.4f}",
            "change_pct": round(change_pct, 2),
            "batch_id": f"rapid_{int(time.time())}",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        fname = f"alert_{symbol}_{int(time.time())}.json"
        with open(os.path.join(PENDING_DIR, fname), "w") as f:
            json.dump(record, f, indent=2)
        log.warning(f" SIGNAL: {reasoning} | submitted to pending_actions")

    def _save_alerts(self):
        """Write recent alerts to shared file for all agents."""
        # Keep last 20, but only write MAX new alerts per cycle
        self.recent_alerts = self.recent_alerts[-20:]
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "alerts": self.recent_alerts,
        }
        with open(ALERTS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    # ── Throttle: max 3 alerts to pending_actions per cycle ──
    _alert_cycle_count = 0
    MAX_ALERTS_PER_CYCLE = 3

    def run(self):
        log.info(" Rapid Market Monitor started — checking CoinGecko every 30s")
        cycle = 0
        while True:
            cycle += 1
            now = time.time()
            alert_count_this_cycle = 0  # Reset each cycle
            try:
                ids = self._get_coin_ids()
                cg_data = self._cg_fetch_with_retry(ids)
                if not cg_data:
                    # Fallback: try Revolut X tickers
                    rev_prices = self._get_revolut_prices()
                    if rev_prices:
                        now_ts = time.time()
                        for sym, price in rev_prices.items():
                            if sym not in self.price_history:
                                self.price_history[sym] = []
                            self.price_history[sym].append({
                                "time": now_ts, "price": float(price), "volume": 0
                            })
                            if len(self.price_history[sym]) > HISTORY_SIZE:
                                self.price_history[sym] = self.price_history[sym][-HISTORY_SIZE:]
                            alert = self._check_and_alert(sym, float(price), 0, now_ts)
                            if alert and alert_count_this_cycle < 3:
                                self.recent_alerts.append(alert)
                                alert_count_this_cycle += 1
                                log.info(f" {sym}: {alert['type']} {alert['change_pct']:+.2f}%")
                        self._save_alerts()
                    if cycle % 2 == 0:
                        log.info(f"Heartbeat: cycle {cycle} — {'Revolut fallback' if rev_prices else 'no data'}")
                    time.sleep(CHECK_INTERVAL)
                    continue
                if cycle % 6 == 0:
                    tracked = sum(1 for c in MONITOR_COINS if cg_data.get(c, {}).get("eur"))
                    log.info(f"Heartbeat: cycle {cycle} — tracking {tracked}/{len(MONITOR_COINS)} coins")

                for coin_name, info in MONITOR_COINS.items():
                    symbol = info["symbol"]
                    cg = cg_data.get(coin_name, {})
                    price = cg.get("eur")
                    volume = cg.get("eur_24h_vol", 0)
                    if not price or price <= 0:
                        continue

                    # Store in price history
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    self.price_history[symbol].append({
                        "time": now,
                        "price": float(price),
                        "volume": float(volume) if volume else 0,
                    })
                    # Keep only last N samples
                    if len(self.price_history[symbol]) > HISTORY_SIZE:
                        self.price_history[symbol] = self.price_history[symbol][-HISTORY_SIZE:]

                    # Check for rapid moves
                    alert = self._check_and_alert(symbol, float(price), float(volume or 0), now)
                    if alert:
                        self.recent_alerts.append(alert)
                        direction = "PUMP" if alert["type"] == "rapid_pump" else "DROP" if alert["type"] == "rapid_drop" else "VOLUME"
                        log.info(f" {symbol}: {direction} {alert['change_pct']:+.2f}% in {alert['window_secs']:.0f}s (vol x{alert.get('vol_spike',1):.1f})")

                self._save_alerts()

            except Exception as e:
                log.debug(f"Cycle {cycle} error: {e}")

            time.sleep(CHECK_INTERVAL)


def get_recent_alerts(max_alerts=5):
    """Read recent alerts from shared file (called by day_trader/scalper for AI context)."""
    try:
        with open(ALERTS_FILE) as f:
            data = json.load(f)
        alerts = data.get("alerts", [])
        if not alerts:
            return ""
        lines = ["\n⚠️ RAPID MARKET ALERTS (last 5min):"]
        for a in alerts[-max_alerts:]:
            t = a.get("type", "")
            sym = a.get("symbol", "?")
            chg = a.get("change_pct", 0)
            vol = a.get("vol_spike", 0)
            icon = " PUMP" if t == "rapid_pump" else " DROP" if t == "rapid_drop" else " VOLUME"
            lines.append(f"  {icon} {sym}: {chg:+.2f}% | vol x{vol:.1f}")
        return "\n".join(lines)
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


if __name__ == "__main__":
    monitor = RapidMonitor()
    monitor.run()