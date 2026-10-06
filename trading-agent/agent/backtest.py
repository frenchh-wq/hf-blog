"""Event-driven backtester sharing sizing and gates with the live agent."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import RiskRules
from .risk import RiskManager, position_size
from .strategies import StrategySpec, grid, signals

BARS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "1h": 8_760, "4h": 2_190, "1d": 365}


@dataclass
class BacktestResult:
    spec: StrategySpec
    metrics: dict
    equity: pd.Series = field(repr=False)
    trades: list[dict] = field(repr=False, default_factory=list)


def metrics(equity: pd.Series, trades: list[dict], bars_per_year: int) -> dict:
    r = equity.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else 0.0
    dd = float((1 - equity / equity.cummax()).max())
    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [-p for p in pnl if p <= 0]
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(dd, 4),
        "trades": len(trades),
        "win_rate": round(len(wins) / len(pnl), 3) if pnl else 0.0,
        "profit_factor": round(float(sum(wins) / sum(losses)), 3) if losses and sum(losses) > 0 else (float("inf") if wins else 0.0),
        "avg_trade": round(float(np.mean(pnl)), 2) if pnl else 0.0,
        "exposure": round(sum(t["bars"] for t in trades) / max(len(equity), 1), 3),
    }


def run(df: pd.DataFrame, spec: StrategySpec, rules: RiskRules | None = None, balance: float = 10_000.0,
        fee: float = 0.001, slippage: float = 0.0005, timeframe: str = "1h") -> BacktestResult:
    rules = rules or RiskRules()
    sig = signals(df, spec)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    entry_sig, exit_sig = sig["entry"].to_numpy(), sig["exit"].to_numpy()
    conf, atr = sig["confidence"].to_numpy(), sig["atr"].to_numpy()
    rm = RiskManager(rules)
    cash, qty, stop, tp, entry_px, entry_i = balance, 0.0, 0.0, 0.0, 0.0, 0
    trades, eq = [], np.empty(len(df))
    warmup = 50

    def close_pos(i: int, px: float, reason: str):
        nonlocal cash, qty
        fill = px * (1 - slippage)
        proceeds = qty * fill * (1 - fee)
        pnl = proceeds - qty * entry_px * (1 + fee)
        trades.append({"entry_time": str(df.index[entry_i]), "exit_time": str(df.index[i]), "entry": entry_px,
                       "exit": fill, "qty": qty, "pnl": pnl, "reason": reason, "bars": i - entry_i})
        cash += proceeds
        qty = 0.0

    for i in range(len(df)):
        # 1) act at this bar's open on the previous bar's signal
        if i > warmup:
            j = i - 1
            if qty > 0 and exit_sig[j]:
                close_pos(i, o[i], "signal")
            elif qty == 0 and entry_sig[j] and atr[j] > 0:
                ok, _ = rm.can_open(conf[j])
                if ok:
                    px = o[i] * (1 + slippage)
                    s = px - spec.stop_atr * atr[j]
                    equity = cash
                    q = position_size(equity, px, s, rules.risk_per_trade, cash / (1 + fee))
                    if q > 0:
                        qty, entry_px, entry_i, stop, tp = q, px, i, s, px + spec.take_profit_atr * atr[j]
                        cash -= q * px * (1 + fee)
        # 2) intrabar stop / take-profit (stop assumed first if both touched)
        if qty > 0:
            if l[i] <= stop:
                close_pos(i, min(stop, o[i]), "stop")
            elif h[i] >= tp:
                close_pos(i, max(tp, o[i]), "take_profit")
        eq[i] = cash + qty * c[i]
        rm.update_equity(eq[i], df.index[i].date())
        if rm.killed and qty > 0:
            close_pos(i, c[i], "kill_switch")
            eq[i] = cash
    if qty > 0:
        close_pos(len(df) - 1, c[-1], "end")
        eq[-1] = cash
    equity = pd.Series(eq, index=df.index)
    m = metrics(equity, trades, BARS_PER_YEAR.get(timeframe, 8760))
    if rm.killed:
        m["killed"] = rm.kill_reason
    return BacktestResult(spec, m, equity, trades)


def score(m: dict, min_trades: int = 8) -> float:
    """Objective for optimisation: Sharpe, penalised for few trades and deep drawdowns."""
    if m["trades"] < min_trades:
        return -10 + m["trades"] / min_trades
    return m["sharpe"] - 2 * max(0.0, m["max_drawdown"] - 0.10)


def walk_forward(df: pd.DataFrame, specs=None, train_frac: float = 0.7, top: int = 5, **kw) -> list[dict]:
    """Optimise on the first `train_frac` of data, report the best candidates out-of-sample."""
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut - 200 :]  # 200-bar overlap so indicators are warm on test
    results = []
    for spec in specs or grid():
        r = run(train, spec, **kw)
        results.append((score(r.metrics), spec, r.metrics))
    results.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, spec, m in results[:top]:
        oos = run(test, spec, **kw).metrics
        out.append({"spec": spec, "train": m, "test": oos, "train_score": s, "test_score": score(oos, min_trades=3)})
    out.sort(key=lambda x: x["test_score"], reverse=True)
    return out
