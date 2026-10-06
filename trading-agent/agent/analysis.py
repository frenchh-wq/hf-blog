"""market_analysis: regime, trend, volatility and key levels."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class MarketState:
    price: float
    regime: str          # trending_up | trending_down | ranging | volatile
    trend_strength: float  # ADX
    volatility: float    # annualised-ish ATR %, per bar
    vol_percentile: float
    rsi: float
    ema_fast_vs_slow: float
    support: list[float]
    resistance: list[float]
    change_24_bars: float

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def key_levels(df: pd.DataFrame, window: int = 5, top: int = 3) -> tuple[list[float], list[float]]:
    """Swing highs/lows clustered near the current price."""
    h, l = df["high"], df["low"]
    swing_hi = h[(h == h.rolling(2 * window + 1, center=True).max())].dropna()
    swing_lo = l[(l == l.rolling(2 * window + 1, center=True).min())].dropna()
    px = float(df["close"].iloc[-1])
    res = sorted({round(x, 2) for x in swing_hi if x > px})[:top]
    sup = sorted({round(x, 2) for x in swing_lo if x < px}, reverse=True)[:top]
    return sup, res


def analyze(df: pd.DataFrame) -> MarketState:
    close = df["close"]
    a = ind.atr(df)
    atr_pct = a / close
    adx = float(ind.adx(df).iloc[-1])
    fast, slow = ind.ema(close, 20), ind.ema(close, 50)
    spread = float((fast.iloc[-1] - slow.iloc[-1]) / close.iloc[-1])
    vol_pct = float((atr_pct.rank(pct=True)).iloc[-1])
    if vol_pct > 0.9:
        regime = "volatile"
    elif adx > 25:
        regime = "trending_up" if spread > 0 else "trending_down"
    else:
        regime = "ranging"
    sup, res = key_levels(df.tail(300))
    lookback = min(24, len(close) - 1)
    return MarketState(
        price=float(close.iloc[-1]),
        regime=regime,
        trend_strength=adx,
        volatility=float(atr_pct.iloc[-1]),
        vol_percentile=vol_pct,
        rsi=float(ind.rsi(close).iloc[-1]),
        ema_fast_vs_slow=spread,
        support=sup,
        resistance=res,
        change_24_bars=float(close.iloc[-1] / close.iloc[-1 - lookback] - 1) if lookback else 0.0,
    )


def summary_stats(df: pd.DataFrame) -> dict:
    r = np.log(df["close"]).diff().dropna()
    return {
        "bars": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "total_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1),
        "bar_vol": float(r.std()),
        "max_bar_move": float(r.abs().max()),
    }
