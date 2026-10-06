"""Daily report with key metrics, written to results/."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd


def daily_report(store, settings, spec, risk) -> dict:
    eq = store.rows("SELECT ts, equity FROM equity ORDER BY ts")
    sells = store.rows("SELECT ts, pnl, reason, strategy FROM trades WHERE side = 'sell' ORDER BY id")
    equity = eq[-1]["equity"] if eq else settings.paper_balance
    last_ts = pd.Timestamp(eq[-1]["ts"]) if eq else pd.Timestamp.now(tz="UTC")
    cutoff = last_ts - timedelta(days=1)
    prior = [e for e in eq if pd.Timestamp(e["ts"]) <= cutoff]
    start_24h = prior[-1]["equity"] if prior else (eq[0]["equity"] if eq else equity)
    s = pd.Series([e["equity"] for e in eq]) if eq else pd.Series([equity])
    pnl = [t["pnl"] or 0 for t in sells]
    recent = store.rows("SELECT ts, side, qty, price, pnl, reason FROM trades ORDER BY id DESC LIMIT 20")
    trades_24h = [t for t in recent if pd.Timestamp(t["ts"]) > cutoff]
    rep = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": settings.symbol,
        "mode": "live" if settings.live_enabled else "paper",
        "strategy": spec.key() if spec else None,
        "strategy_rationale": spec.rationale if spec else None,
        "equity": equity,
        "pnl_24h": equity - start_24h,
        "pnl_total": equity - (eq[0]["equity"] if eq else equity),
        "max_drawdown": float((1 - s / s.cummax()).max()),
        "closed_trades_total": len(pnl),
        "trades_24h": len(trades_24h),
        "win_rate": round(sum(p > 0 for p in pnl) / len(pnl), 3) if pnl else None,
        "profit_factor": round(sum(p for p in pnl if p > 0) / -sum(p for p in pnl if p < 0), 3) if any(p < 0 for p in pnl) else None,
        "exit_reasons": pd.Series([t["reason"] for t in sells]).value_counts().to_dict() if sells else {},
        "kill_switch": risk.kill_reason if risk.killed else None,
        "recent_trades": recent,
        "last_signal": store.get_state("last_signal"),
        "claude_insights": store.get_state("last_claude_research"),
        "last_review": store.get_state("last_review"),
    }
    settings.results_dir.mkdir(exist_ok=True)
    day = last_ts.strftime("%Y-%m-%d")
    (settings.results_dir / f"report_{day}.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep
