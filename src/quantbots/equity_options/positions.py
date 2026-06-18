"""Reconstruct STRUCTURE positions from the broker's leg-level positions.

Alpaca reports one row per option leg. We group legs by (underlying, expiry) to
rebuild the spreads/singles we actually opened, and compute each structure's live
economics (net cost, current mark, unrealized P&L, DTE, and — for a debit vertical —
its max profit/loss). The exit engine (`manage.py`) and entry de-duplication
(`recommend.py`) both read from here. The BROKER is the source of truth — we never
infer open positions from the local ledger for risk decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from .occ import parse_occ

CONTRACT_MULTIPLIER = 100


@dataclass
class StructureLeg:
    symbol: str
    qty: int                 # signed contracts (+long / -short)
    kind: str
    strike: float
    avg_entry: float
    market_value: float
    unrealized_pl: float


@dataclass
class StructurePosition:
    underlying: str
    expiry: date
    legs: list[StructureLeg] = field(default_factory=list)

    @property
    def dte(self) -> int:
        return (self.expiry - datetime.now(UTC).date()).days

    @property
    def contracts(self) -> int:
        return max((abs(l.qty) for l in self.legs), default=0)

    @property
    def net_cost(self) -> float:
        """Net debit paid to open (>0). cost = avg_entry * qty * mult, signed by qty."""
        return sum(l.avg_entry * l.qty * CONTRACT_MULTIPLIER for l in self.legs)

    @property
    def market_value(self) -> float:
        return sum(l.market_value for l in self.legs)

    @property
    def unrealized_pl(self) -> float:
        return sum(l.unrealized_pl for l in self.legs)

    @property
    def is_vertical(self) -> bool:
        return len(self.legs) == 2 and self.legs[0].kind == self.legs[1].kind

    @property
    def width(self) -> float:
        if not self.is_vertical:
            return 0.0
        return abs(self.legs[0].strike - self.legs[1].strike)

    @property
    def max_profit(self) -> float | None:
        """Max profit of a debit vertical ($). None for a naked long (unbounded)."""
        if self.is_vertical:
            return self.width * CONTRACT_MULTIPLIER * self.contracts - self.net_cost
        return None

    @property
    def short_leg(self) -> StructureLeg | None:
        return next((l for l in self.legs if l.qty < 0), None)

    def short_itm(self, spot: float) -> bool:
        """Is the short leg in-the-money (assignment risk near expiry)?"""
        s = self.short_leg
        if s is None:
            return False
        return (spot < s.strike) if s.kind == "put" else (spot > s.strike)

    def profit_fraction(self) -> float:
        """Fraction of the realizable profit captured: upl / max_profit for a vertical,
        else upl / net_cost for a naked long. 0 when undefined."""
        mp = self.max_profit
        if mp is not None and mp > 0:
            return self.unrealized_pl / mp
        if self.net_cost > 0:
            return self.unrealized_pl / self.net_cost
        return 0.0

    def loss_fraction(self) -> float:
        """Fraction of premium-at-risk currently lost (>0 means losing)."""
        risk = abs(self.net_cost)
        return -self.unrealized_pl / risk if risk > 0 else 0.0


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def structures_from_broker(raw_positions: list[dict]) -> list[StructurePosition]:
    """Group broker leg-positions into StructurePositions. Non-option rows are skipped."""
    groups: dict[tuple[str, date], StructurePosition] = {}
    for p in raw_positions:
        sym = p.get("symbol", "")
        try:
            occ = parse_occ(sym)
        except ValueError:
            continue  # equities / crypto / malformed — not our option legs
        key = (occ.underlying, occ.expiry)
        sp = groups.get(key) or StructurePosition(underlying=occ.underlying, expiry=occ.expiry)
        sp.legs.append(StructureLeg(
            symbol=sym, qty=int(_to_float(p.get("qty"))), kind=occ.kind, strike=occ.strike,
            avg_entry=_to_float(p.get("avg_entry_price")),
            market_value=_to_float(p.get("market_value")),
            unrealized_pl=_to_float(p.get("unrealized_pl")),
        ))
        groups[key] = sp
    return list(groups.values())


def held_underlyings(structures: list[StructurePosition]) -> set[str]:
    return {s.underlying for s in structures if s.contracts > 0}
