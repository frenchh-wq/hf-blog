"""risk_management: position sizing, loss limits, drawdown control and the kill switch."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .config import RiskRules


def position_size(equity: float, entry: float, stop: float, risk_per_trade: float, cash: float) -> float:
    """Units such that hitting `stop` loses `risk_per_trade` of equity. No leverage: capped by cash."""
    per_unit = entry - stop
    if per_unit <= 0 or entry <= 0:
        return 0.0
    qty = equity * risk_per_trade / per_unit
    return max(0.0, min(qty, cash / entry * 0.999))


@dataclass
class RiskManager:
    rules: RiskRules
    peak_equity: float = 0.0
    day: date | None = None
    day_start_equity: float = 0.0
    consecutive_errors: int = 0
    killed: bool = False
    kill_reason: str = ""
    halted_today: bool = False
    events: list[str] = field(default_factory=list)

    def update_equity(self, equity: float, today: date) -> None:
        if self.day != today:
            self.day, self.day_start_equity, self.halted_today = today, equity, False
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity and 1 - equity / self.peak_equity >= self.rules.max_drawdown:
            self.kill(f"max drawdown {1 - equity / self.peak_equity:.1%} >= {self.rules.max_drawdown:.0%}")
        if self.day_start_equity and 1 - equity / self.day_start_equity >= self.rules.daily_loss_limit and not self.halted_today:
            self.halted_today = True
            self.events.append(f"daily loss limit hit ({1 - equity / self.day_start_equity:.1%}); no new trades today")

    def record_error(self, err: Exception) -> None:
        self.consecutive_errors += 1
        self.events.append(f"API error {self.consecutive_errors}: {err}")
        if self.consecutive_errors >= self.rules.max_consecutive_errors:
            self.kill(f"{self.consecutive_errors} consecutive API errors")

    def record_ok(self) -> None:
        self.consecutive_errors = 0

    def check_bar(self, last_close: float, prev_close: float) -> None:
        if prev_close and abs(last_close / prev_close - 1) > self.rules.max_bar_move:
            self.kill(f"abnormal price move {last_close / prev_close - 1:+.1%} in one bar")

    def kill(self, reason: str) -> None:
        if not self.killed:
            self.killed, self.kill_reason = True, reason
            self.events.append(f"KILL SWITCH: {reason}")

    def can_open(self, confidence: float) -> tuple[bool, str]:
        if self.killed:
            return False, f"killed: {self.kill_reason}"
        if self.halted_today:
            return False, "daily loss limit reached"
        if confidence <= self.rules.min_confidence:
            return False, f"confidence {confidence:.2f} <= {self.rules.min_confidence}"
        return True, "ok"
