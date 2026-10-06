"""Data feeds the trader polls: live exchange data, or a replay of a historical frame."""
from __future__ import annotations

import pandas as pd

from .data import fetch_ohlcv


class LiveFeed:
    def __init__(self, exchange, symbol: str, timeframe: str, history: int = 1500):
        self.ex, self.symbol, self.tf, self.history = exchange, symbol, timeframe, history
        self._df: pd.DataFrame | None = None

    def candles(self) -> pd.DataFrame:
        if self._df is None:
            self._df = fetch_ohlcv(self.ex, self.symbol, self.tf, self.history)
        else:
            new = fetch_ohlcv(self.ex, self.symbol, self.tf, 10)
            self._df = pd.concat([self._df, new]).loc[lambda d: ~d.index.duplicated(keep="last")].tail(self.history)
        return self._df

    def price(self) -> float:
        return float(self.ex.fetch_ticker(self.symbol)["last"])

    def advance(self) -> bool:
        return True


class ReplayFeed:
    """Steps through a DataFrame one bar per tick; `price()` is the latest bar's close."""

    def __init__(self, df: pd.DataFrame, start: int = 500):
        self.df, self.i = df, start

    def candles(self) -> pd.DataFrame:
        return self.df.iloc[: self.i]

    def price(self) -> float:
        return float(self.df["close"].iloc[self.i - 1])

    def bar(self) -> pd.Series:
        return self.df.iloc[self.i - 1]

    def advance(self) -> bool:
        self.i += 1
        return self.i <= len(self.df)
