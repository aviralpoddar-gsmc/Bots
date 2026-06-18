"""Broker abstraction + the safe-by-default dry-run broker.

`BrokerClient` is the one seam the rest of the package talks to. `DryRunBroker`
implements it without any network call — it just returns the order it *would* have
placed, so `eo recommend`/`eo trade` (default) and tests never touch a real account.
The Alpaca paper client and the live stub subclass `BrokerClient`.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrderLeg:
    symbol: str           # OCC symbol
    side: str             # BUY | SELL
    ratio_qty: int = 1    # legs per structure (1 for verticals/singles)
    closing: bool = False  # True => position_intent is *_to_close (exits), else *_to_open


@dataclass
class OptionOrder:
    underlying: str
    structure: str
    legs: list[OrderLeg]
    qty: int                       # number of structures (contracts)
    limit_price: float             # net per-share debit (>0 = pay)
    time_in_force: str = "day"
    ticket_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_multileg(self) -> bool:
        return len(self.legs) > 1


@dataclass
class OrderResult:
    ticket_id: str
    broker: str                    # dry | paper | live
    status: str                    # intended | submitted | filled | rejected
    broker_order_id: str | None = None
    filled_price: float | None = None
    raw: dict | None = None


class BrokerClient(ABC):
    name = "base"

    @abstractmethod
    def submit(self, order: OptionOrder) -> OrderResult: ...

    @abstractmethod
    def account_equity(self) -> float: ...

    @abstractmethod
    def positions(self) -> list[dict]: ...

    def cancel_all(self) -> None:  # optional; no-op default
        return None

    def close_all(self) -> None:   # optional; no-op default
        return None

    def list_orders(self, *, status: str = "all", limit: int = 100) -> list[dict]:
        return []

    def is_market_open(self) -> bool:
        return True

    def submit_equity(self, symbol: str, qty: int, side: str) -> dict:
        return {}


class DryRunBroker(BrokerClient):
    """No network. Echoes the intended order. The default everywhere."""

    name = "dry"

    def __init__(self, *, equity: float = 100_000.0):
        self._equity = equity

    def submit(self, order: OptionOrder) -> OrderResult:
        legs = ", ".join(f"{l.side} {order.qty}x {l.symbol}" for l in order.legs)
        logger.info("[DRY] %s %s @ net %.2f  (%s)", order.structure, order.underlying,
                    order.limit_price, legs)
        return OrderResult(ticket_id=order.ticket_id, broker="dry", status="intended")

    def account_equity(self) -> float:
        return self._equity

    def positions(self) -> list[dict]:
        return []
