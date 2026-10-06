"""Market data: live OHLCV from any ccxt exchange, plus a synthetic generator for offline tests."""
from __future__ import annotations

import time

import ccxt
import numpy as np
import pandas as pd

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def make_exchange(exchange_id: str, api_key: str = "", secret: str = "") -> ccxt.Exchange:
    cls = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": True}
    if api_key and secret:
        params.update(apiKey=api_key, secret=secret)
    return cls(params)


def _to_df(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").astype(float).sort_index()


def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch the most recent `limit` candles, paginating as needed."""
    step = TF_MS[timeframe]
    since = ex.milliseconds() - step * limit
    rows: list[list] = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=min(1000, limit - len(rows)))
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + step
        if since >= ex.milliseconds():
            break
        time.sleep(ex.rateLimit / 1000)
    df = _to_df(rows)
    # Drop the still-forming candle so signals only use closed bars.
    if len(df) and df.index[-1].value // 1_000_000 + step > ex.milliseconds():
        df = df.iloc[:-1]
    return df


def orderbook_imbalance(ex: ccxt.Exchange, symbol: str, depth: int = 20) -> float:
    """(bid volume - ask volume) / total over the top `depth` levels, in [-1, 1]."""
    ob = ex.fetch_order_book(symbol, depth)
    b = sum(q for _, q in ob["bids"][:depth])
    a = sum(q for _, q in ob["asks"][:depth])
    return (b - a) / (b + a) if b + a else 0.0


def synthetic_ohlcv(n: int = 2000, seed: int = 7, start: float = 30_000.0, freq: str = "1h") -> pd.DataFrame:
    """Regime-switching random walk (trend / range / volatile) for offline testing."""
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    i = 0
    while i < n:
        length = rng.integers(80, 300)
        drift, vol = [(0.0006, 0.006), (-0.0006, 0.007), (0.0, 0.004), (0.0, 0.015)][rng.integers(0, 4)]
        rets[i : i + length] = rng.normal(drift, vol, size=min(length, n - i))
        i += length
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(3, 0.5, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
