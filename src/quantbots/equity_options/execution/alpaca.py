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

    def account_pnl(self) -> dict:
        """Broker-truth P&L (the real scoreboard — never the local ledger). Returns
        current equity, prior-close last_equity, the account base_value (portfolio
        history origin), and total_pnl = equity - base_value. The caller derives
        realized = total_pnl - unrealized(open positions)."""
        acct = self._http.get("/v2/account") or {}
        # Fail fast: a malformed payload must raise, not default to 0.0 — a zero
        # last_equity silently disarms the circuit breaker's equity-cliff check.
        equity = float(acct["equity"])
        last_equity = float(acct["last_equity"])
        hist = self._http.get("/v2/account/portfolio/history",
                              {"period": "all", "timeframe": "1D"}) or {}
        base = hist.get("base_value")
        base = float(base) if base is not None else None
        return {"equity": equity, "last_equity": last_equity, "base_value": base,
                "total_pnl": (equity - base) if base is not None else None}

    def positions(self) -> list[dict]:
        return self._http.get("/v2/positions") or []

    def cancel_all(self) -> None:
        self._http.delete("/v2/orders")

    def close_all(self) -> None:
        self._http.delete("/v2/positions")

    def list_orders(self, *, status: str = "all", limit: int = 100) -> list[dict]:
        return self._http.get("/v2/orders", {"status": status, "limit": limit,
                                             "nested": "true"}) or []

    def list_all_orders(self, *, status: str = "all", page_size: int = 500) -> list[dict]:
        """Full order history. /v2/orders caps each response (max 500), so page
        newest-first via `until` = oldest submitted_at of the previous page. A capped
        single call here once left 65 ledger legs unreconciled — never cap history."""
        out: list[dict] = []
        params = {"status": status, "limit": page_size, "direction": "desc",
                  "nested": "true"}
        prev_until = None
        for _ in range(200):  # hard page cap: fail loudly, never spin inside launchd
            page = self._http.get("/v2/orders", params) or []
            out.extend(page)
            if len(page) < page_size:
                return out
            until = min(o["submitted_at"] for o in page)
            if prev_until is not None and until >= prev_until:
                raise RuntimeError(
                    f"order-history pagination stalled: `until` cursor not decreasing "
                    f"({until!r} after {prev_until!r})")
            prev_until = until
            params = {**params, "until": until}
        raise RuntimeError("order-history pagination exceeded 200 pages — aborting")

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
