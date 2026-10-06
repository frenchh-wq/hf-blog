import numpy as np
import pandas as pd

from agent.backtest import run, walk_forward
from agent.data import synthetic_ohlcv
from agent.strategies import StrategySpec, grid, signals, STRATEGIES

DF = synthetic_ohlcv(1500, seed=3)


def test_all_strategies_produce_valid_signals():
    for name in STRATEGIES:
        spec = next(s for s in grid([name]))
        sig = signals(DF, spec)
        assert len(sig) == len(DF)
        assert sig["confidence"].min() >= 0
        assert sig["entry"].dtype == bool


def test_no_lookahead():
    """Signals at bar t must not change when future bars are appended."""
    spec = StrategySpec(name="ema_cross", params={"fast": 10, "slow": 50})
    a = signals(DF.iloc[:1000], spec)
    b = signals(DF, spec).iloc[:1000]
    pd.testing.assert_frame_equal(a, b)


def test_backtest_risk_per_trade_respected():
    spec = StrategySpec(name="donchian_breakout", params={"n": 20, "exit_n": 10}, stop_atr=2, take_profit_atr=4)
    r = run(DF, spec)
    stops = [t for t in r.trades if t["reason"] == "stop"]
    assert r.trades, "expected trades"
    # a stop-out should lose roughly <= 2% of equity (+ fees/slippage/gaps)
    eq = r.equity
    for t in stops:
        before = eq.loc[:t["entry_time"]].iloc[-2]
        assert -t["pnl"] / before < 0.03


def test_backtest_respects_max_drawdown_kill():
    spec = StrategySpec(name="ema_cross", params={"fast": 10, "slow": 50}, stop_atr=1.5, take_profit_atr=3)
    r = run(DF, spec)
    assert r.metrics["max_drawdown"] < 0.17


def test_walk_forward_returns_ranked_candidates():
    specs = list(grid(["ema_cross"]))[:6]
    out = walk_forward(DF, specs=specs, top=3)
    assert len(out) == 3
    assert out[0]["test_score"] >= out[-1]["test_score"]
