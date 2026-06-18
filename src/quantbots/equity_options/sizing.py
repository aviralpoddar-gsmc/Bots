"""Position sizing — fractional Kelly on the option P&L distribution, capped.

The parent `sizing.py` pushes a fixed 1/3 toward fair value because on a binary
clone market the price gap already encodes conviction. Options give us a full P&L
distribution, so we can size properly: fractional Kelly (the `edge.kelly` already
computed as E[PnL]/E[PnL^2]) scaled by the config `kelly_fraction`, then clamped by
the dollar cap-stack. For a debit structure max loss = premium, so "dollars at risk"
= contracts * premium, which is exactly what the caps bound.
"""

from __future__ import annotations

import math

from .selection import Candidate


def size_contracts(candidate: Candidate, *, bankroll: float, limits: dict) -> int:
    """Number of contracts to trade for one candidate (0 = skip)."""
    premium = candidate.premium
    if premium <= 0:
        return 0
    # Fractional Kelly fraction of bankroll to put at risk on this position.
    risk_fraction = limits["kelly_fraction"] * candidate.edge.kelly
    kelly_dollars = max(0.0, risk_fraction * bankroll)
    by_kelly = kelly_dollars / premium
    by_cap = limits["max_premium_per_trade"] / premium
    contracts = int(math.floor(min(by_kelly, by_cap)))
    if contracts < 1 or contracts * premium < limits["min_premium"]:
        return 0
    return contracts
