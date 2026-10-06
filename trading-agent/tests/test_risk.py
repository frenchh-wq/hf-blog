from datetime import date

import pytest

from agent.config import RiskRules
from agent.risk import RiskManager, position_size


def test_position_size_risks_two_percent():
    qty = position_size(equity=10_000, entry=100, stop=95, risk_per_trade=0.02, cash=10_000)
    assert qty == pytest.approx(40)  # 40 units * $5 = $200 = 2%


def test_position_size_capped_by_cash_no_leverage():
    qty = position_size(equity=10_000, entry=100, stop=99.9, risk_per_trade=0.02, cash=10_000)
    assert qty * 100 <= 10_000


def test_position_size_invalid_stop():
    assert position_size(10_000, 100, 101, 0.02, 10_000) == 0


def test_confidence_gate():
    rm = RiskManager(RiskRules(min_confidence=1.5))
    assert not rm.can_open(1.5)[0]
    assert rm.can_open(1.6)[0]


def test_daily_loss_limit_halts_new_trades_then_resets():
    rm = RiskManager(RiskRules())
    rm.update_equity(10_000, date(2025, 1, 1))
    rm.update_equity(9_490, date(2025, 1, 1))
    assert not rm.can_open(5)[0] and not rm.killed
    rm.update_equity(9_490, date(2025, 1, 2))
    assert rm.can_open(5)[0]


def test_max_drawdown_kills():
    rm = RiskManager(RiskRules())
    rm.update_equity(10_000, date(2025, 1, 1))
    for d in range(2, 10):
        rm.update_equity(10_000 - d * 200, date(2025, 1, d))
    assert rm.killed and "drawdown" in rm.kill_reason
    assert not rm.can_open(10)[0]


def test_consecutive_api_errors_kill():
    rm = RiskManager(RiskRules(max_consecutive_errors=3))
    rm.record_error(Exception("a")); rm.record_ok()
    rm.record_error(Exception("b")); rm.record_error(Exception("c"))
    assert not rm.killed
    rm.record_error(Exception("d"))
    assert rm.killed


def test_abnormal_move_kills():
    rm = RiskManager(RiskRules(max_bar_move=0.15))
    rm.check_bar(100, 99)
    assert not rm.killed
    rm.check_bar(70, 100)
    assert rm.killed
