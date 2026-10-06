"""strategy_engine: parameterised long-only strategies.

Every strategy returns a frame with:
  entry       bool   - open a long at the next bar's open
  exit        bool   - close the long at the next bar's open
  confidence  float  - signal strength in normalised units (compared against MIN_CONFIDENCE)
  atr         float  - used for stop / take-profit distances
"""
from __future__ import annotations

import itertools
from typing import Callable

import pandas as pd
from pydantic import BaseModel, Field

from . import indicators as ind


class StrategySpec(BaseModel):
    name: str
    params: dict[str, float] = Field(default_factory=dict)
    stop_atr: float = 2.0
    take_profit_atr: float = 4.0
    rationale: str = ""

    def key(self) -> str:
        p = ",".join(f"{k}={v:g}" for k, v in sorted(self.params.items()))
        return f"{self.name}({p}) sl={self.stop_atr:g} tp={self.take_profit_atr:g}"


def _frame(df, entry, exit_, conf) -> pd.DataFrame:
    return pd.DataFrame(
        {"entry": entry.fillna(False).astype(bool), "exit": exit_.fillna(False).astype(bool),
         "confidence": conf.fillna(0.0).clip(lower=0), "atr": ind.atr(df)},
        index=df.index,
    )


def ema_cross(df, fast=20, slow=50):
    c = df["close"]
    f, s = ind.ema(c, int(fast)), ind.ema(c, int(slow))
    a = ind.atr(df)
    above = f > s
    conf = (f - s) / a  # spread in ATR units
    return _frame(df, above, ~above, conf)


def rsi_reversion(df, period=14, lower=30, upper=60, trend=200):
    c = df["close"]
    r = ind.rsi(c, int(period))
    in_uptrend = c > ind.ema(c, int(trend))
    entry = (r < lower) & in_uptrend
    conf = (50 - r) / 10  # RSI 30 -> 2.0
    return _frame(df, entry, r > upper, conf)


def donchian_breakout(df, n=20, exit_n=10):
    lo, hi = ind.donchian(df, int(n))
    exit_lo, _ = ind.donchian(df, int(exit_n))
    c = df["close"]
    entry = c > hi.shift()
    conf = ind.adx(df) / 15  # ADX 22.5 -> 1.5
    return _frame(df, entry, c < exit_lo.shift(), conf)


def bollinger_reversion(df, n=20, k=2.0):
    c = df["close"]
    lower, mid, _ = ind.bollinger(c, int(n), k)
    sd = c.rolling(int(n)).std()
    z = (mid - c) / sd  # distance below the mean in std devs
    return _frame(df, c < lower, c > mid, z)


STRATEGIES: dict[str, Callable[..., pd.DataFrame]] = {
    "ema_cross": ema_cross,
    "rsi_reversion": rsi_reversion,
    "donchian_breakout": donchian_breakout,
    "bollinger_reversion": bollinger_reversion,
}

PARAM_GRID: dict[str, dict[str, list[float]]] = {
    "ema_cross": {"fast": [10, 20, 30], "slow": [50, 100, 200]},
    "rsi_reversion": {"period": [7, 14], "lower": [25, 30, 35], "upper": [55, 65], "trend": [100, 200]},
    "donchian_breakout": {"n": [20, 40, 55], "exit_n": [10, 20]},
    "bollinger_reversion": {"n": [20, 40], "k": [1.5, 2.0, 2.5]},
}
RISK_GRID = {"stop_atr": [1.5, 2.5], "take_profit_atr": [3.0, 6.0]}

STRATEGY_DOCS = {
    "ema_cross": "Trend following: long while EMA(fast) > EMA(slow). confidence = EMA spread / ATR.",
    "rsi_reversion": "Buy RSI dips below `lower` while price > EMA(trend); exit when RSI > `upper`. confidence = (50-RSI)/10.",
    "donchian_breakout": "Buy a close above the prior n-bar high; exit below the prior exit_n-bar low. confidence = ADX/15.",
    "bollinger_reversion": "Buy a close below the lower Bollinger band; exit at the mid band. confidence = z-score below the mean.",
}


def signals(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    if spec.name not in STRATEGIES:
        raise ValueError(f"unknown strategy {spec.name!r}")
    return STRATEGIES[spec.name](df, **spec.params)


def grid(names: list[str] | None = None):
    for name in names or STRATEGIES:
        g = PARAM_GRID[name]
        for vals in itertools.product(*g.values()):
            params = dict(zip(g.keys(), vals))
            if name == "ema_cross" and params["fast"] >= params["slow"]:
                continue
            for sl, tp in itertools.product(RISK_GRID["stop_atr"], RISK_GRID["take_profit_atr"]):
                yield StrategySpec(name=name, params=params, stop_atr=sl, take_profit_atr=tp)
