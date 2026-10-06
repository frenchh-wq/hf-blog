"""trade_execution: a paper broker and a live ccxt spot broker behind one interface."""
from __future__ import annotations

from dataclasses import dataclass

import ccxt


@dataclass
class Fill:
    side: str
    qty: float
    price: float
    fee: float
    order_id: str = ""


class Broker:
    def base_quote(self, symbol: str) -> tuple[str, str]:
        base, quote = symbol.split("/")
        return base, quote.split(":")[0]

    def balances(self) -> tuple[float, float]:  # (cash in quote, position in base)
        raise NotImplementedError

    def market_buy(self, qty: float, price_hint: float) -> Fill:
        raise NotImplementedError

    def market_sell(self, qty: float, price_hint: float) -> Fill:
        raise NotImplementedError


class PaperBroker(Broker):
    """Simulated fills at the reference price +/- slippage, with taker fees."""

    def __init__(self, cash: float, fee: float = 0.001, slippage: float = 0.0005, position: float = 0.0):
        self.cash, self.position, self.fee, self.slippage = cash, position, fee, slippage
        self._n = 0

    def balances(self):
        return self.cash, self.position

    def _id(self) -> str:
        self._n += 1
        return f"paper-{self._n}"

    def market_buy(self, qty, price_hint):
        px = price_hint * (1 + self.slippage)
        cost = qty * px
        fee = cost * self.fee
        if cost + fee > self.cash + 1e-9:
            raise ValueError(f"insufficient paper cash: need {cost + fee:.2f}, have {self.cash:.2f}")
        self.cash -= cost + fee
        self.position += qty
        return Fill("buy", qty, px, fee, self._id())

    def market_sell(self, qty, price_hint):
        qty = min(qty, self.position)
        px = price_hint * (1 - self.slippage)
        fee = qty * px * self.fee
        self.cash += qty * px - fee
        self.position -= qty
        return Fill("sell", qty, px, fee, self._id())


class CcxtBroker(Broker):
    """Live spot execution via ccxt. Only constructed when live trading is explicitly confirmed."""

    def __init__(self, exchange: ccxt.Exchange, symbol: str):
        if not exchange.apiKey:
            raise RuntimeError("live trading requires EXCHANGE_API_KEY / EXCHANGE_API_SECRET")
        self.ex, self.symbol = exchange, symbol
        self.ex.load_markets()
        self.market = self.ex.market(symbol)
        if not self.market.get("spot"):
            raise RuntimeError("live broker supports spot markets only")

    def balances(self):
        base, quote = self.base_quote(self.symbol)
        bal = self.ex.fetch_balance()
        return float(bal.get(quote, {}).get("free") or 0), float(bal.get(base, {}).get("total") or 0)

    def _fill(self, side: str, order: dict, price_hint: float) -> Fill:
        oid = order["id"]
        try:
            order = self.ex.fetch_order(oid, self.symbol)
        except Exception:
            pass
        qty = float(order.get("filled") or order.get("amount") or 0)
        px = float(order.get("average") or order.get("price") or price_hint)
        fee = float((order.get("fee") or {}).get("cost") or 0)
        return Fill(side, qty, px, fee, str(oid))

    def _amount(self, qty: float) -> float:
        amt = float(self.ex.amount_to_precision(self.symbol, qty))
        min_amt = (self.market.get("limits", {}).get("amount", {}) or {}).get("min") or 0
        if amt < min_amt:
            raise ValueError(f"order size {amt} below exchange minimum {min_amt}")
        return amt

    def market_buy(self, qty, price_hint):
        return self._fill("buy", self.ex.create_market_buy_order(self.symbol, self._amount(qty)), price_hint)

    def market_sell(self, qty, price_hint):
        return self._fill("sell", self.ex.create_market_sell_order(self.symbol, self._amount(qty)), price_hint)
