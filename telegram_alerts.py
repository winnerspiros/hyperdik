"""
Telegram alert system for Revolut X trading bot.
Sends trade notifications, error alerts, and daily summaries.
"""
import os
import json
import urllib.request
import time
from datetime import datetime, timezone

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "telegram_config.json")

def _load_config():
    """Load Telegram bot config from JSON file"""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except:
        return {}

def _save_config(config):
    """Save Telegram bot config"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

def setup(bot_token: str = None, chat_id: str = None):
    """Configure Telegram alerts. Call once with your bot token and chat ID."""
    config = _load_config()
    if bot_token:
        config["bot_token"] = bot_token
    if chat_id:
        config["chat_id"] = chat_id
    _save_config(config)
    return bool(config.get("bot_token") and config.get("chat_id"))

def _send(text: str, parse_mode: str = "HTML"):
    """Low-level send to Telegram"""
    config = _load_config()
    token = config.get("bot_token")
    chat = config.get("chat_id")
    if not token or not chat:
        return False
    try:
        data = json.dumps({
            "chat_id": chat,
            "text": text[:4000],  # Telegram 4096 char limit
            "parse_mode": parse_mode,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        # Silently fail — don't crash the bot over a Telegram notification
        return False

def alert(message: str):
    """Send a trade alert"""
    return _send(f"🤖 <b>Revolut X Trader</b>\n{message}")

def trade_alert(symbol: str, side: str, size: float, price: float, reason: str = ""):
    """Send a trade execution alert"""
    emoji = "🟢 BUY" if side == "buy" else "🔴 SELL"
    msg = f"{emoji} <b>{symbol}</b>\nSize: €{size:.2f} @ €{price:.4f}"
    if reason:
        msg += f"\nReason: {reason[:200]}"
    return _send(msg)

def error_alert(error_msg: str):
    """Send an error alert"""
    return _send(f"⚠️ <b>Error</b>\n{error_msg[:500]}")

def daily_summary(eur: float, total_value: float, positions: int, trades_today: int, pnl_today: float):
    """Send a daily performance summary"""
    emoji = "🟢" if pnl_today >= 0 else "🔴"
    msg = (
        f"📊 <b>Daily Summary</b>\n"
        f"{emoji} PnL: €{pnl_today:.2f}\n"
        f"💰 EUR: €{eur:.2f}\n"
        f"💼 Portfolio: €{total_value:.2f}\n"
        f"📦 Positions: {positions}\n"
        f"🔄 Trades: {trades_today}"
    )
    return _send(msg)

def heartbeat():
    """Send a heartbeat to confirm the bot is alive"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _send(f"💚 <b>Bot Alive</b>\n{now}")

def test():
    """Test the Telegram setup"""
    success = _send("✅ <b>Telegram Alert Test</b>\nBot is configured and working!")
    if success:
        print("✅ Telegram alert sent successfully!")
    else:
        print("❌ Failed to send. Configure with: setup(bot_token='xxx', chat_id='xxx')")
    return success

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "setup":
        setup(bot_token=sys.argv[2], chat_id=sys.argv[3])
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print(f"Usage: {sys.argv[0]} setup <bot_token> <chat_id>")
        print(f"       {sys.argv[0]} test")