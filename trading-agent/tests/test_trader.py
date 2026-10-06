from pathlib import Path

from agent.config import Settings
from agent.data import synthetic_ohlcv
from agent.execution import PaperBroker
from agent.feeds import ReplayFeed
from agent.notify import Notifier
from agent.storage import Store
from agent.trader import Trader


def make(tmp_path: Path, df=None):
    s = Settings(db_path=tmp_path / "t.db", results_dir=tmp_path / "results", kill_file=tmp_path / "KILL")
    store = Store(s.db_path)
    feed = ReplayFeed(df if df is not None else synthetic_ohlcv(1300, seed=5), start=1000)
    return Trader(s, feed, PaperBroker(10_000), store, Notifier()), s


def test_paper_replay_runs_and_records(tmp_path):
    t, s = make(tmp_path)
    n = t.run(interval=0)
    assert t.spec is not None
    # either replays every bar, or stops early because the drawdown kill switch fired
    assert n >= 300 or (t.risk.killed and "drawdown" in t.risk.kill_reason)
    assert t.store.rows("SELECT COUNT(*) c FROM equity")[0]["c"] >= min(n, 300)
    eq = [r["equity"] for r in t.store.rows("SELECT equity FROM equity ORDER BY ts")]
    peak = max(eq)
    assert 1 - eq[-1] / peak < 0.18
    if t.risk.killed:
        assert t.pos is None and t.broker.position == 0
    assert t.store.active_strategy() is not None


def test_kill_file_stops_and_flattens(tmp_path):
    t, s = make(tmp_path)
    t.run(interval=0, max_ticks=5)
    s.kill_file.write_text("stop")
    assert t.tick() is False
    assert t.pos is None
    assert t.broker.position == 0


def test_never_live_without_confirmation():
    s = Settings(mode="live", live_confirm="yes")
    assert not s.live_enabled


def test_stands_aside_when_not_validated(tmp_path, monkeypatch):
    import agent.trader as tr
    from agent.research import DEFAULT_SPEC

    def fake_select(df, settings, researcher=None, incumbent=None, **kw):
        row = {"train": {}, "test": {"sharpe": -1, "total_return": -0.1, "max_drawdown": 0.1}}
        return {"spec": DEFAULT_SPEC, "result": row, "source": "default", "changed": True,
                "validated": False, "claude": None}

    monkeypatch.setattr(tr, "select_strategy", fake_select)
    t, _ = make(tmp_path)
    t.run(interval=0, max_ticks=200)
    assert t.store.rows("SELECT COUNT(*) c FROM trades")[0]["c"] == 0
