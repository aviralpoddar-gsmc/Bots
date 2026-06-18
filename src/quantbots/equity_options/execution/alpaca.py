"""Alpaca PAPER broker. Real exchange semantics, no real money.

Submits options orders to `paper-api.alpaca.markets`:
  - single leg -> order_class "simple"
  - vertical   -> order_class "mleg" with per-leg position_intent

All orders are LIMIT (never market) at the model's net debit, time_in_force "day".
The base URL is fixed to the paper host; there is deliberately no way to point this
class at the live host (that lives only in live.py behind the approval gate).
"""

from __future__ import annotations

import logging

from .._alpaca_http import PAPER_TRADING_URL, AlpacaHTTP
from .base import BrokerClient, OptionOrder, OrderResult

logger = logging.getLogger(__name__)


def _side(side: str) -> str:
    return "buy" if side.upper() == "BUY" else "sell"


def _intent(side: str, closing: bool = False) -> str:
    buy = side.upper() == "BUY"
    if closing:
        return "buy_to_close" if buy else "sell_to_close"
    return "buy_to_open" if buy else "sell_to_open"


class AlpacaPaperBroker(BrokerClient):
    name = "paper"

    def __init__(self, *, key: str | None = None, secret: str | None = None):
        self._http = AlpacaHTTP(PAPER_TRADING_URL, key=key, secret=secret)

    def _payload(self, order: OptionOrder) -> dict:
        base = {
            "type": "limit",
            "time_in_force": order.time_in_force,
            "qty": str(order.qty),
            "limit_price": str(round(order.limit_price, 2)),
            "client_order_id": order.ticket_id,
        }
        if order.is_multileg:
            base["order_class"] = "mleg"
            base["legs"] = [
                {"symbol": l.symbol, "ratio_qty": str(l.ratio_qty),
                 "side": _side(l.side), "position_intent": _intent(l.side, l.closing)}
                for l in order.legs
            ]
        else:
            leg = order.legs[0]
            base["order_class"] = "simple"
            base["symbol"] = leg.symbol
            base["side"] = _side(leg.side)
        return base

    def submit(self, order: OptionOrder) -> OrderResult:
        resp = self._http.post("/v2/orders", json=self._payload(order)) or {}
        return OrderResult(
            ticket_id=order.ticket_id, broker="paper",
            status=resp.get("status", "submitted"),
            broker_order_id=resp.get("id"),
            filled_price=float(resp["filled_avg_price"]) if resp.get("filled_avg_price") else None,
            raw=resp,
        )

    def account_equity(self) -> float:
        acct = self._http.get("/v2/account") or {}
        return float(acct.get("equity", 0.0))

    def positions(self) -> list[dict]:
        return self._http.get("/v2/positions") or []

    def cancel_all(self) -> None:
        self._http.delete("/v2/orders")

    def close_all(self) -> None:
        self._http.delete("/v2/positions")

    def list_orders(self, *, status: str = "all", limit: int = 100) -> list[dict]:
        return self._http.get("/v2/orders", {"status": status, "limit": limit,
                                             "nested": "true"}) or []

    def is_market_open(self) -> bool:
        clk = self._http.get("/v2/clock") or {}
        return bool(clk.get("is_open"))

    def submit_equity(self, symbol: str, qty: int, side: str) -> dict:
        """Market order on the underlying shares — used for delta-hedging."""
        if qty <= 0:
            return {}
        return self._http.post("/v2/orders", json={
            "symbol": symbol, "qty": str(int(qty)), "side": side.lower(),
            "type": "market", "time_in_force": "day"}) or {}
