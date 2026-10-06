"""Strategy selection pipeline: grid search -> Claude proposals -> walk-forward validation -> adopt or keep."""
from __future__ import annotations

import logging

import pandas as pd

from . import analysis
from .backtest import run, score, walk_forward
from .config import Settings
from .strategies import StrategySpec

log = logging.getLogger("agent.research")

DEFAULT_SPEC = StrategySpec(name="ema_cross", params={"fast": 20, "slow": 100}, stop_atr=2.5, take_profit_atr=6.0,
                            rationale="default trend-following baseline")


def _bt_kwargs(s: Settings) -> dict:
    return {"rules": s.risk, "balance": s.paper_balance, "fee": s.fee_rate, "slippage": s.slippage, "timeframe": s.timeframe}


def _summ(row: dict) -> dict:
    keep = ("total_return", "sharpe", "max_drawdown", "trades", "win_rate", "profit_factor")
    return {"strategy": row["spec"].key(), "train": {k: row["train"][k] for k in keep},
            "test": {k: row["test"][k] for k in keep}}


def select_strategy(df: pd.DataFrame, settings: Settings, researcher=None, incumbent: StrategySpec | None = None,
                    recent_perf: dict | None = None, extra_specs: list[StrategySpec] | None = None) -> dict:
    """Return {"spec", "result", "source", "changed", "validated", "claude", "leaderboard"}."""
    kw = _bt_kwargs(settings)
    leaderboard = walk_forward(df, top=6, **kw)
    candidates = [(r, "grid") for r in leaderboard]

    claude_out = None
    proposals = list(extra_specs or [])
    if researcher is not None:
        try:
            claude_out, specs = researcher.research(
                analysis.analyze(df).to_dict(), analysis.summary_stats(df), [_summ(r) for r in leaderboard],
                {"strategy": incumbent.key()} if incumbent else None, recent_perf)
            proposals += specs
            log.info("Claude proposed %d strategies: %s", len(specs), claude_out.get("regime_assessment", "")[:200])
        except Exception as e:
            log.warning("Claude research unavailable, continuing with grid search only: %s", e)
    if proposals:
        candidates += [(r, "claude") for r in walk_forward(df, specs=proposals, top=len(proposals), **kw)]

    inc_row = None
    if incumbent is not None:
        inc_row = walk_forward(df, specs=[incumbent], top=1, **kw)[0]

    # Eligible = profitable out-of-sample and within drawdown limits
    eligible = [(r, src) for r, src in candidates
                if r["test"]["total_return"] > 0 and r["test"]["max_drawdown"] < settings.risk.max_drawdown
                and r["test"]["sharpe"] > 0]
    eligible.sort(key=lambda x: x[0]["test_score"] + 0.5 * x[0]["train_score"], reverse=True)

    if not eligible:
        best, src = (inc_row, "incumbent") if inc_row else (walk_forward(df, specs=[DEFAULT_SPEC], top=1, **kw)[0], "default")
        # Nothing is profitable out-of-sample: report the best guess but mark it unvalidated so
        # the trader stands aside instead of trading it.
        return {"spec": best["spec"], "result": best, "source": src, "changed": incumbent is None, "claude": claude_out,
                "validated": False, "leaderboard": [_summ(r) for r in leaderboard]}

    best, src = eligible[0]
    changed = True
    # Only replace the incumbent if the challenger is meaningfully better out-of-sample.
    inc_ok = inc_row is not None and any(r is inc_row or r["spec"].key() == incumbent.key() for r, _ in eligible)
    if inc_row is not None and best["spec"].key() != incumbent.key():
        # An incumbent that still validates keeps its seat unless clearly beaten.
        if inc_ok and best["test_score"] < inc_row["test_score"] + 0.25:
            best, src, changed = inc_row, "incumbent", False
    elif inc_row is not None:
        changed = False
    return {"spec": best["spec"], "result": best, "source": src, "changed": changed, "claude": claude_out,
            "validated": True, "leaderboard": [_summ(r) for r in leaderboard]}


def full_backtest(df: pd.DataFrame, spec: StrategySpec, settings: Settings):
    return run(df, spec, **_bt_kwargs(settings))


__all__ = ["select_strategy", "full_backtest", "DEFAULT_SPEC", "score"]
