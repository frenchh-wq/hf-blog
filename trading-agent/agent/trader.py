"""The autonomous trading loop: orchestration, memory (SQLite), execution and monitoring."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from . import analysis
from .config import Settings
from .execution import Broker
from .feeds import ReplayFeed
from .notify import Notifier
from .report import daily_report
from .research import DEFAULT_SPEC, select_strategy
from .risk import RiskManager, position_size
from .storage import Store
from .strategies import StrategySpec, signals

log = logging.getLogger("agent.trader")


class Trader:
    def __init__(self, settings: Settings, feed, broker: Broker, store: Store, notifier: Notifier,
                 researcher=None, research_every_bars: int = 24 * 7):
        self.s, self.feed, self.broker, self.store, self.notify = settings, feed, broker, store, notifier
        self.researcher = researcher
        self.research_every = research_every_bars
        self.mode = "live" if settings.live_enabled else "paper"
        self.risk = RiskManager(settings.risk, peak_equity=store.get_state("peak_equity", 0.0))
        active = store.active_strategy()
        self.spec = StrategySpec(**active[0]) if active else None
        self.pos = store.get_state("position", None)  # {"qty","entry","stop","tp","strategy","ts"}
        self.last_bar = store.get_state("last_bar", None)
        self.bars_since_research = store.get_state("bars_since_research", 10**9)
        self.report_day = store.get_state("report_day", None)
        self.last_signal: dict = {}
        self.validated = store.get_state("validated", True)

    # ---- helpers -------------------------------------------------------
    def _now(self):
        if isinstance(self.feed, ReplayFeed):
            return self.feed.bar().name.to_pydatetime()
        return datetime.now(timezone.utc)

    def _event(self, msg: str, level: str = "INFO", alert: bool = False):
        log.log(logging.getLevelName(level), msg)
        self.store.log(msg, level)
        if alert:
            self.notify.send(f"[{self.mode}] {self.s.symbol}: {msg}")

    def _persist(self):
        self.store.set_state("position", self.pos)
        self.store.set_state("last_bar", self.last_bar)
        self.store.set_state("peak_equity", self.risk.peak_equity)
        self.store.set_state("bars_since_research", self.bars_since_research)
        self.store.set_state("report_day", self.report_day)
        if hasattr(self.broker, "cash") and self.mode == "paper":
            self.store.set_state("paper_balances", [self.broker.cash, self.broker.position])

    # ---- research / self-improvement -----------------------------------
    def research(self, df):
        recent = self.recent_performance()
        out = select_strategy(df, self.s, self.researcher, incumbent=self.spec, recent_perf=recent)
        spec, res = out["spec"], out["result"]
        if self.spec is None or out["changed"]:
            self.spec = spec
            self.store.save_strategy(spec.model_dump(), {"train": res["train"], "test": res["test"]}, out["source"])
            self._event(f"strategy -> {spec.key()} [{out['source']}] OOS sharpe={res['test']['sharpe']} "
                        f"ret={res['test']['total_return']:.2%} dd={res['test']['max_drawdown']:.2%}", alert=True)
        else:
            self._event(f"keeping {self.spec.key()} (no challenger beat it out-of-sample)")
        self.validated = out["validated"]
        self.store.set_state("validated", self.validated)
        if not self.validated:
            self._event("no strategy is profitable out-of-sample: standing aside (no new entries) until next research",
                        "WARNING", alert=True)
        if out.get("claude"):
            self.store.set_state("last_claude_research", out["claude"])
        self.bars_since_research = 0

    def recent_performance(self) -> dict:
        trades = self.store.rows("SELECT side, pnl, reason, strategy, ts FROM trades WHERE side='sell' ORDER BY id DESC LIMIT 50")
        pnl = [t["pnl"] or 0 for t in trades]
        return {"closed_trades": len(pnl), "net_pnl": sum(pnl),
                "win_rate": (sum(p > 0 for p in pnl) / len(pnl)) if pnl else None, "last_trades": trades[:10]}

    # ---- trading -------------------------------------------------------
    def _open(self, price: float, atr: float, confidence: float):
        cash, _ = self.broker.balances()
        equity = self.equity(price)
        stop = price - self.spec.stop_atr * atr
        qty = position_size(equity, price, stop, self.s.risk.risk_per_trade, cash / (1 + self.s.fee_rate))
        if qty * price < 10:  # dust
            self._event(f"entry skipped: size {qty:.6f} too small")
            return
        f = self.broker.market_buy(qty, price)
        self.pos = {"qty": f.qty, "entry": f.price, "stop": f.price - self.spec.stop_atr * atr,
                    "tp": f.price + self.spec.take_profit_atr * atr, "strategy": self.spec.key(), "ts": str(self._now())}
        self.store.trade(ts=self._now().isoformat(timespec="seconds"), mode=self.mode, symbol=self.s.symbol, side="buy", qty=f.qty, price=f.price, fee=f.fee,
                         reason=f"entry conf={confidence:.2f}", strategy=self.spec.key(), order_id=f.order_id)
        self._event(f"BUY {f.qty:.6f} @ {f.price:.2f} stop={self.pos['stop']:.2f} tp={self.pos['tp']:.2f} "
                    f"conf={confidence:.2f}", alert=True)

    def _close(self, price: float, reason: str):
        if not self.pos:
            return
        f = self.broker.market_sell(self.pos["qty"], price)
        pnl = f.qty * f.price - f.fee - self.pos["qty"] * self.pos["entry"] * (1 + self.s.fee_rate)
        self.store.trade(ts=self._now().isoformat(timespec="seconds"), mode=self.mode, symbol=self.s.symbol, side="sell", qty=f.qty, price=f.price, fee=f.fee,
                         pnl=pnl, reason=reason, strategy=self.pos["strategy"], order_id=f.order_id)
        self._event(f"SELL {f.qty:.6f} @ {f.price:.2f} ({reason}) pnl={pnl:+.2f}", alert=True)
        self.pos = None

    def equity(self, price: float) -> float:
        cash, position = self.broker.balances()
        return cash + position * price

    def tick(self) -> bool:
        """One iteration. Returns False when the agent should stop."""
        if self.s.kill_file.exists():
            self.risk.kill("KILL file present")
        try:
            df = self.feed.candles()
            price = self.feed.price()
            self.risk.record_ok()
        except Exception as e:
            self.risk.record_error(e)
            self._event(f"data error: {e}", "WARNING")
            return self._after_risk(None)

        if self.spec is None or self.bars_since_research >= self.research_every:
            self.research(df)

        # Stops / take-profit against the latest price (and bar extremes in replay)
        if self.pos:
            lo = hi = price
            if isinstance(self.feed, ReplayFeed):
                b = self.feed.bar()
                lo, hi = float(b["low"]), float(b["high"])
            if lo <= self.pos["stop"]:
                self._close(min(self.pos["stop"], price) if not isinstance(self.feed, ReplayFeed) else self.pos["stop"], "stop")
            elif hi >= self.pos["tp"]:
                self._close(price if not isinstance(self.feed, ReplayFeed) else self.pos["tp"], "take_profit")

        bar_ts = str(df.index[-1])
        if bar_ts != self.last_bar:  # new closed candle -> evaluate signals
            self.last_bar = bar_ts
            self.bars_since_research += 1
            if len(df) > 1:
                self.risk.check_bar(float(df["close"].iloc[-1]), float(df["close"].iloc[-2]))
            sig = signals(df, self.spec).iloc[-1]
            conf, atr = float(sig["confidence"]), float(sig["atr"])
            if self.pos and sig["exit"]:
                self._close(price, "signal")
                self.store.signal(self.spec.key(), "exit", conf, True)
            elif not self.pos and sig["entry"]:
                ok, why = self.risk.can_open(conf)
                if ok and not self.validated:
                    ok, why = False, "standing aside: strategy not validated out-of-sample"
                self.store.signal(self.spec.key(), "entry", conf, ok, why)
                if ok:
                    try:
                        self._open(price, atr, conf)
                    except Exception as e:
                        self.risk.record_error(e)
                        self._event(f"order failed: {e}", "ERROR", alert=True)
            self.last_signal = {"bar": bar_ts, "entry": bool(sig["entry"]), "exit": bool(sig["exit"]),
                                "confidence": round(conf, 3), "regime": analysis.analyze(df).regime}
            self.store.set_state("last_signal", self.last_signal)

        eq = self.equity(price)
        now = self._now()
        self.risk.update_equity(eq, now.date())
        _, position = self.broker.balances()
        self.store.equity(eq, eq - position * price, position, price, ts=now.isoformat(timespec="seconds"))

        day = str(now.date())
        if self.report_day and day != self.report_day:
            self.daily(df)
        self.report_day = day
        return self._after_risk(price)

    def _after_risk(self, price):
        for e in self.risk.events:
            self._event(e, "WARNING", alert=True)
        self.risk.events.clear()
        if self.risk.killed:
            if self.pos and price:
                try:
                    self._close(price, "kill_switch")
                except Exception as e:
                    self._event(f"FAILED to flatten on kill: {e} - CLOSE MANUALLY", "CRITICAL", alert=True)
            self._persist()
            self._event(f"agent stopped: {self.risk.kill_reason}", "CRITICAL", alert=True)
            return False
        self._persist()
        return True

    def daily(self, df):
        rep = daily_report(self.store, self.s, self.spec, self.risk)
        self._event(f"daily report: equity={rep['equity']:.2f} pnl_24h={rep['pnl_24h']:+.2f} "
                    f"trades={rep['trades_24h']} win_rate={rep['win_rate']}", alert=True)
        if self.researcher is not None and rep["closed_trades_total"] >= 5:
            try:
                review = self.researcher.review(rep)
                self.store.set_state("last_review", {k: v for k, v in review.items() if k != "specs"})
                out = select_strategy(df, self.s, None, incumbent=self.spec, extra_specs=review["specs"])
                if out["changed"]:
                    self.spec = out["spec"]
                    self.store.save_strategy(self.spec.model_dump(), {"test": out["result"]["test"]}, "self_improvement")
                    self._event(f"self-improvement adopted {self.spec.key()}", alert=True)
            except Exception as e:
                self._event(f"self-improvement review failed: {e}", "WARNING")

    def run(self, interval: float = 60.0, max_ticks: int | None = None):
        self._event(f"agent starting in {self.mode.upper()} mode on {self.s.exchange} {self.s.symbol} {self.s.timeframe}", alert=True)
        n = 0
        while True:
            if not self.tick():
                break
            n += 1
            if max_ticks and n >= max_ticks:
                break
            if not self.feed.advance():
                break
            if interval:
                time.sleep(interval)
        return n
