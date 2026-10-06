"""SQLite persistence for trades, equity, signals, logs and strategy history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, ts TEXT, mode TEXT, symbol TEXT, side TEXT,
    qty REAL, price REAL, fee REAL, pnl REAL, reason TEXT, strategy TEXT, order_id TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL, cash REAL, position REAL, price REAL);
CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, ts TEXT, strategy TEXT, action TEXT,
    confidence REAL, accepted INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, ts TEXT, level TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS strategies (id INTEGER PRIMARY KEY, ts TEXT, spec TEXT, metrics TEXT,
    source TEXT, active INTEGER);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, args=()):
        with self.conn:
            return self.conn.execute(sql, args)

    def log(self, message: str, level: str = "INFO") -> None:
        self._exec("INSERT INTO logs (ts, level, message) VALUES (?,?,?)", (now(), level, message))

    def trade(self, **t) -> None:
        self._exec("INSERT INTO trades (ts, mode, symbol, side, qty, price, fee, pnl, reason, strategy, order_id) "
                   "VALUES (:ts,:mode,:symbol,:side,:qty,:price,:fee,:pnl,:reason,:strategy,:order_id)",
                   {"ts": now(), "pnl": None, "order_id": None, **t})

    def equity(self, equity: float, cash: float, position: float, price: float, ts: str | None = None) -> None:
        self._exec("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?)", (ts or now(), equity, cash, position, price))

    def signal(self, strategy: str, action: str, confidence: float, accepted: bool, note: str = "") -> None:
        self._exec("INSERT INTO signals (ts, strategy, action, confidence, accepted, note) VALUES (?,?,?,?,?,?)",
                   (now(), strategy, action, confidence, int(accepted), note))

    def save_strategy(self, spec: dict, metrics: dict, source: str, active: bool = True) -> None:
        if active:
            self._exec("UPDATE strategies SET active = 0")
        self._exec("INSERT INTO strategies (ts, spec, metrics, source, active) VALUES (?,?,?,?,?)",
                   (now(), json.dumps(spec), json.dumps(metrics, default=str), source, int(active)))

    def active_strategy(self) -> tuple[dict, dict] | None:
        r = self.conn.execute("SELECT spec, metrics FROM strategies WHERE active = 1 ORDER BY id DESC LIMIT 1").fetchone()
        return (json.loads(r["spec"]), json.loads(r["metrics"])) if r else None

    def set_state(self, key: str, value) -> None:
        self._exec("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value, default=str)))

    def get_state(self, key: str, default=None):
        r = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def rows(self, sql: str, args=()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]
