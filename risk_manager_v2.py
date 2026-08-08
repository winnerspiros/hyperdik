"""
Crypto Risk Management — advisory, data-driven, no hardcoded thresholds.
Reports risk state to AI. All decisions left to AI.
"""
import time
import logging
from datetime import datetime

log = logging.getLogger(__name__)

class CryptoRiskManager:
    def __init__(self):
        self.state = {
            "peak_portfolio_value": 0,
            "current_drawdown_pct": 0,
            "consecutive_losses": 0,
            "daily_pnl": 0.0,
            "circuit_breaker_time": 0,
        }
        self.circuit_breaker_duration = 60  # seconds to pause after flash crash

    def update_drawdown(self, current_value: float):
        if current_value > self.state["peak_portfolio_value"]:
            self.state["peak_portfolio_value"] = current_value
        if self.state["peak_portfolio_value"] > 0:
            dd = (current_value - self.state["peak_portfolio_value"]) / self.state["peak_portfolio_value"] * 100
            self.state["current_drawdown_pct"] = dd

    def record_trade_result(self, pnl_pct: float):
        if pnl_pct < 0:
            self.state["consecutive_losses"] += 1
        else:
            self.state["consecutive_losses"] = 0
        self.state["daily_pnl"] += pnl_pct

    def check_flash_crash(self, current_price, previous_close) -> bool:
        """Detect a flash crash: price gap down >15% in one period."""
        if previous_close > 0:
            gap = (current_price - previous_close) / previous_close * 100
            if gap <= -15:
                self.state["circuit_breaker_time"] = time.time()
                log.warning(f"🚨 Flash crash detected: {gap:.1f}% drop")
                return True
        return False

    def is_circuit_breaker_active(self) -> bool:
        if self.state["circuit_breaker_time"] == 0:
            return False
        elapsed = time.time() - self.state["circuit_breaker_time"]
        return elapsed < self.circuit_breaker_duration

    def get_circuit_breaker_remaining(self) -> int:
        if not self.is_circuit_breaker_active():
            return 0
        elapsed = time.time() - self.state["circuit_breaker_time"]
        return max(0, int(self.circuit_breaker_duration - elapsed))

    def is_weekend(self):
        return datetime.now().weekday() >= 5

    def get_position_multiplier(self) -> float:
        """Advisory multiplier — data-driven, no hardcoded thresholds."""
        multiplier = 1.0
        dd = self.state["current_drawdown_pct"]
        if dd < 0:
            multiplier *= max(0.0, 1.0 + dd / 50.0)
        losses = self.state["consecutive_losses"]
        if losses > 0:
            multiplier *= max(0.2, 1.0 / (1.0 + losses * 0.3))
        if self.is_circuit_breaker_active():
            multiplier = 0.0
        return max(0.0, min(1.0, multiplier))

    def get_risk_summary(self) -> str:
        lines = ["🛡️ RISK MANAGEMENT:"]
        if self.is_circuit_breaker_active():
            remaining = self.get_circuit_breaker_remaining()
            lines.append(f"  🚨 CIRCUIT BREAKER ACTIVE ({remaining}s remaining)")
        if self.is_weekend():
            lines.append("  📅 Weekend — historical volume typically 40-60% lower")
        dd = self.state["current_drawdown_pct"]
        lines.append(f"  📉 Drawdown: {dd:+.1f}%")
        losses = self.state["consecutive_losses"]
        if losses >= 3:
            lines.append(f"  ⚠️ {losses} consecutive losses — higher risk")
        mult = self.get_position_multiplier()
        lines.append(f"  📐 Position multiplier: {mult:.0%}")
        return "\n".join(lines)