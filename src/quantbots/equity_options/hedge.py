"""Intraday delta-hedging for short-vol positions.

Only structures meant to be vol-neutral (iron fly / condor / straddle) get hedged —
directional positions (debit/credit verticals) are intentional bets and are left alone.
For each hedged underlying we compute net option delta from the live chain greeks, net
the shares already held, and trade the difference to flatten. Equity hedge orders go
through `broker.submit_equity`. No-ops when there are no short-vol positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

VOL_STRUCTURES = {"iron_fly", "iron_condor", "straddle", "strangle"}
MIN_HEDGE_SHARES = 1   # don't trade sub-1-share hedge deltas


@dataclass
class HedgeAction:
    underlying: str
    net_option_delta: float   # shares-equivalent (incl. ×100 multiplier)
    current_shares: float
    trade_shares: int         # +buy / -sell to flatten
    side: str


def _vol_underlyings(store) -> set[str]:
    """Underlyings whose OPEN ledger positions include a short-vol structure."""
    out: set[str] = set()
    for t in store.trades():
        if t["structure"] in VOL_STRUCTURES and t["status"] not in ("canceled", "rejected"):
            out.add(t["underlying"])
    # keep only those still open on the broker side is handled by the caller via positions
    return out


def compute_hedges(broker, chain_client, store, *, r: float = 0.04) -> list[HedgeAction]:
    """Net-delta hedge actions for each short-vol underlying (broker = source of truth)."""
    from .occ import parse_occ
    vol_unds = _vol_underlyings(store)
    if not vol_unds:
        return []
    positions = broker.positions()
    # equity share holdings per symbol
    shares: dict[str, float] = {}
    opt_by_und: dict[str, list[dict]] = {}
    for p in positions:
        sym = p.get("symbol", "")
        try:
            occ = parse_occ(sym)
            opt_by_und.setdefault(occ.underlying, []).append({**p, "_occ": occ})
        except ValueError:
            shares[sym] = float(p.get("qty", 0))   # an equity (hedge) position
    actions: list[HedgeAction] = []
    for und in vol_unds:
        legs = opt_by_und.get(und, [])
        if not legs:
            continue
        # delta per option symbol from the live chain
        try:
            chain = {row["symbol"]: row for row in chain_client.get_chain(und, min_dte=0, max_dte=400)}
        except Exception as e:  # noqa: BLE001
            logger.warning("hedge: chain fetch failed for %s (%s)", und, e)
            continue
        net_delta = 0.0
        for leg in legs:
            row = chain.get(leg["symbol"])
            delta = row.get("delta") if row else None
            if delta is None:
                continue
            net_delta += float(leg["qty"]) * float(delta) * 100
        cur = shares.get(und, 0.0)
        trade = -(net_delta + cur)
        n = int(round(trade))
        if abs(n) < MIN_HEDGE_SHARES:
            continue
        actions.append(HedgeAction(underlying=und, net_option_delta=net_delta,
                                   current_shares=cur, trade_shares=n,
                                   side="buy" if n > 0 else "sell"))
    return actions


def apply_hedges(broker, actions: list[HedgeAction], *, dry: bool) -> None:
    for a in actions:
        if dry:
            logger.info("[DRY-HEDGE] %s %s %d shares (net opt Δ=%.0f, held=%.0f)",
                        a.side, a.underlying, abs(a.trade_shares), a.net_option_delta, a.current_shares)
        else:
            broker.submit_equity(a.underlying, abs(a.trade_shares), a.side)
