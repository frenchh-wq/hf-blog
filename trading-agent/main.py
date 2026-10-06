"""Autonomous trading agent CLI.

  python main.py backtest [--synthetic]       research + walk-forward backtest, results to results/
  python main.py paper    [--synthetic]       run the agent against live (or replayed) data with a paper broker
  python main.py live                         run with real orders (requires MODE=live + LIVE_CONFIRM)
  python main.py check    [--synthetic]       final check: backtest, risk-control tests, paper replay
  python main.py report                       write today's report from the database
  python main.py dashboard                    serve the dashboard on localhost
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from agent.config import LIVE_CONFIRM_PHRASE, Settings
from agent.data import fetch_ohlcv, make_exchange, synthetic_ohlcv
from agent.execution import CcxtBroker, PaperBroker
from agent.feeds import LiveFeed, ReplayFeed
from agent.notify import Notifier
from agent.storage import Store

log = logging.getLogger("agent")


def researcher_for(s: Settings, enabled: bool):
    if not enabled:
        return None
    try:
        from agent.llm import Researcher
        r = Researcher(s.anthropic_api_key, s.claude_model)
        return r
    except Exception as e:
        log.warning("Claude disabled (%s); falling back to grid search", e)
        return None


def load_history(s: Settings, synthetic: bool, bars: int):
    if synthetic:
        return synthetic_ohlcv(bars)
    return fetch_ohlcv(make_exchange(s.exchange), s.symbol, s.timeframe, bars)


def cmd_backtest(s: Settings, a) -> int:
    from agent import analysis
    from agent.research import full_backtest, select_strategy

    df = load_history(s, a.synthetic, a.bars)
    print(f"{len(df)} bars {df.index[0]} -> {df.index[-1]}")
    print("market:", json.dumps(analysis.analyze(df).to_dict(), indent=1))
    out = select_strategy(df, s, researcher_for(s, not a.no_claude))
    spec = out["spec"]
    res = full_backtest(df, spec, s)
    print("\nleaderboard (grid, walk-forward):")
    for row in out["leaderboard"]:
        print(f"  {row['strategy']:<60} test sharpe={row['test']['sharpe']:>6} ret={row['test']['total_return']:>7.2%}")
    if out["claude"]:
        print("\nClaude:", out["claude"]["regime_assessment"])
    print(f"\nselected [{out['source']}]: {spec.key()}")
    if not out["validated"]:
        print("WARNING: nothing was profitable out-of-sample - the agent would STAND ASIDE on this data.")
    print("out-of-sample:", out["result"]["test"])
    print("full period:  ", res.metrics)
    s.results_dir.mkdir(exist_ok=True)
    res.equity.to_csv(s.results_dir / "backtest_equity.csv", header=["equity"])
    (s.results_dir / "backtest.json").write_text(json.dumps(
        {"spec": spec.model_dump(), "oos": out["result"]["test"], "full": res.metrics, "leaderboard": out["leaderboard"],
         "claude": out["claude"], "trades": res.trades}, indent=2, default=str))
    print(f"\nwrote {s.results_dir}/backtest.json and backtest_equity.csv")
    return 0


def build_trader(s: Settings, a, live: bool):
    from agent.trader import Trader

    store = Store(s.db_path)
    notifier = Notifier(s.telegram_token, s.telegram_chat_id, s.discord_webhook)
    if a.synthetic:
        feed = ReplayFeed(synthetic_ohlcv(a.bars), start=1000)
    else:
        ex = make_exchange(s.exchange, s.exchange_api_key if live else "", s.exchange_api_secret if live else "")
        feed = LiveFeed(ex, s.symbol, s.timeframe)
    if live:
        broker = CcxtBroker(feed.ex, s.symbol)
    else:
        cash, pos = store.get_state("paper_balances", [s.paper_balance, 0.0])
        broker = PaperBroker(cash, s.fee_rate, s.slippage, pos)
    return Trader(s, feed, broker, store, notifier, researcher_for(s, not a.no_claude))


def cmd_paper(s: Settings, a) -> int:
    s.mode = "paper"
    t = build_trader(s, a, live=False)
    t.run(interval=0 if a.synthetic else a.interval, max_ticks=a.ticks)
    from agent.report import daily_report
    rep = daily_report(t.store, s, t.spec, t.risk)
    print(json.dumps({k: rep[k] for k in ("equity", "pnl_total", "closed_trades_total", "win_rate", "max_drawdown", "strategy")}, indent=1, default=str))
    return 0


def cmd_live(s: Settings, a) -> int:
    if not s.live_enabled:
        print(f"Live trading is locked. Set MODE=live and LIVE_CONFIRM=\"{LIVE_CONFIRM_PHRASE}\" in .env,\n"
              "and run `python main.py check` first.", file=sys.stderr)
        return 2
    if a.synthetic:
        print("--synthetic makes no sense for live trading", file=sys.stderr)
        return 2
    t = build_trader(s, a, live=True)
    t.run(interval=a.interval, max_ticks=a.ticks)
    return 0


def cmd_check(s: Settings, a) -> int:
    """<final_check>: full backtest, verify risk controls, paper-trade test."""
    import tempfile
    from pathlib import Path

    import pytest

    print("== 1/3 risk-control & unit tests ==")
    rc = pytest.main(["-q", str(Path(__file__).parent / "tests")])
    if rc != 0:
        print("FAILED: tests")
        return 1
    print("\n== 2/3 full backtest ==")
    cmd_backtest(s, a)
    print("\n== 3/3 paper trading replay ==")
    with tempfile.TemporaryDirectory() as d:
        s.db_path = Path(d) / "check.db"
        s.kill_file = Path(d) / "KILL"
        a.synthetic, a.ticks = True, None
        t = build_trader(s, a, live=False)
        n = t.run(interval=0)
        from agent.report import daily_report
        rep = daily_report(t.store, s, t.spec, t.risk)
        print(f"replayed {n} bars: equity={rep['equity']:.2f} trades={rep['closed_trades_total']} "
              f"win_rate={rep['win_rate']} max_dd={rep['max_drawdown']:.2%} kill={rep['kill_switch']}")
        ok = rep["max_drawdown"] <= s.risk.max_drawdown + 0.02
        print("paper replay drawdown within limits:", ok)
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["backtest", "paper", "live", "check", "report", "dashboard"])
    p.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    p.add_argument("--no-claude", action="store_true", help="grid search only, no Claude calls")
    p.add_argument("--bars", type=int, default=3000)
    p.add_argument("--interval", type=float, default=60.0, help="seconds between ticks")
    p.add_argument("--ticks", type=int, default=None, help="stop after N ticks")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env()
    if a.command == "report":
        from agent.report import daily_report
        from agent.risk import RiskManager
        from agent.strategies import StrategySpec
        store = Store(s.db_path)
        act = store.active_strategy()
        print(json.dumps(daily_report(store, s, StrategySpec(**act[0]) if act else None, RiskManager(s.risk)), indent=2, default=str))
        return 0
    if a.command == "dashboard":
        from agent.dashboard import serve
        serve(s.db_path, s.dashboard_port)
        return 0
    return {"backtest": cmd_backtest, "paper": cmd_paper, "live": cmd_live, "check": cmd_check}[a.command](s, a)


if __name__ == "__main__":
    sys.exit(main())
