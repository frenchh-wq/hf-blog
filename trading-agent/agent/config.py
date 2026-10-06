"""Settings loaded from environment / .env. Secrets never live in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LIVE_CONFIRM_PHRASE = "I understand this trades real money"


def _f(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


def _s(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).split("#")[0].strip()


@dataclass
class RiskRules:
    risk_per_trade: float = 0.02
    daily_loss_limit: float = 0.05
    max_drawdown: float = 0.15
    min_confidence: float = 1.5
    max_consecutive_errors: int = 5
    max_bar_move: float = 0.15  # a single-candle move larger than this trips the kill switch


@dataclass
class Settings:
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    mode: str = "paper"
    live_confirm: str = ""
    paper_balance: float = 10_000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook: str = ""
    db_path: Path = ROOT / "trading.db"
    results_dir: Path = ROOT / "results"
    kill_file: Path = ROOT / "KILL"
    dashboard_port: int = 8050
    risk: RiskRules = field(default_factory=RiskRules)

    @property
    def live_enabled(self) -> bool:
        return self.mode == "live" and self.live_confirm == LIVE_CONFIRM_PHRASE

    @classmethod
    def from_env(cls) -> "Settings":
        db = Path(_s("DB_PATH", "trading.db"))
        return cls(
            anthropic_api_key=_s("ANTHROPIC_API_KEY"),
            claude_model=_s("CLAUDE_MODEL", "claude-opus-5-5"),
            exchange=_s("EXCHANGE", "binance"),
            symbol=_s("SYMBOL", "BTC/USDT"),
            timeframe=_s("TIMEFRAME", "1h"),
            exchange_api_key=_s("EXCHANGE_API_KEY"),
            exchange_api_secret=_s("EXCHANGE_API_SECRET"),
            mode=_s("MODE", "paper").lower(),
            live_confirm=os.getenv("LIVE_CONFIRM", "").strip(),
            paper_balance=_f("PAPER_STARTING_BALANCE", 10_000),
            fee_rate=_f("FEE_RATE", 0.001),
            slippage=_f("SLIPPAGE", 0.0005),
            telegram_token=_s("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_s("TELEGRAM_CHAT_ID"),
            discord_webhook=_s("DISCORD_WEBHOOK_URL"),
            db_path=db if db.is_absolute() else ROOT / db,
            dashboard_port=int(_f("DASHBOARD_PORT", 8050)),
            risk=RiskRules(
                risk_per_trade=_f("RISK_PER_TRADE", 0.02),
                daily_loss_limit=_f("DAILY_LOSS_LIMIT", 0.05),
                max_drawdown=_f("MAX_DRAWDOWN", 0.15),
                min_confidence=_f("MIN_CONFIDENCE", 1.5),
            ),
        )
