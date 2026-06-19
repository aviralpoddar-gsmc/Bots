"""Exit / roll logic for open option structures.

A position is closed when ANY rule fires (checked in priority order):

  1. **DTE / assignment guard** — within `min_hold_dte` of expiry, close. If the short
     leg is in-the-money near expiry the close is urgent (avoid pin/assignment on the
     short put). Defined-risk verticals can't blow up, but we still don't want to be
     assigned and hold stock over a weekend.
  2. **Profit target** — captured >= `profit_target_frac` of the realizable profit.
     Taking 50-70% of a vertical's max profit early beats grinding the last pennies
     against gamma/assignment risk.
  3. **Stop loss** — lost >= `stop_loss_frac` of the premium at risk.

Closing reverses every leg with a *_to_close intent. The limit is the current mark
(market_value); if it doesn't fill it rests as a working order and the next cycle's
stale-order sweep re-prices it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .execution.base import OptionOrder, OrderLeg
from .positions import StructurePosition

# Exit-rule defaults (overridable via config `manage:`).
DEFAULT_EXIT_RULES = {
    "profit_target_frac": 0.60,   # take 60% of max profit
    "stop_loss_frac": 0.60,       # cut at 60% of premium lost
    "min_hold_dte": 10,           # close inside 10 DTE regardless
    "assignment_dte": 21,         # if short leg ITM and within this DTE, close
}


@dataclass
class ExitDecision:
    structure: StructurePosition
    reason: str


def exit_decisions(structures: list[StructurePosition], *, rules: dict,
                   spots: dict[str, float] | None = None,
                   keep: set[str] | None = None) -> list[ExitDecision]:
    """Decide which open structures to close now and why.

    `keep` = underlyings to RIDE (the validated momentum sleeve); everything else is a
    legacy/non-validated position. When `rules["breakeven_close"]` is set, those legacy
    positions are wound down the moment they are no longer at a loss (uPnL >= 0) — the
    "close on green/break-even" policy — instead of waiting for the full profit target.
    """
    spots = spots or {}
    keep = keep or set()
    out: list[ExitDecision] = []
    for s in structures:
        if s.contracts <= 0:
            continue
        spot = spots.get(s.underlying)
        if s.dte <= rules["min_hold_dte"]:
            out.append(ExitDecision(s, f"dte<={rules['min_hold_dte']} ({s.dte}d)"))
            continue
        if (spot is not None and s.dte <= rules["assignment_dte"] and s.short_itm(spot)):
            out.append(ExitDecision(s, f"short-leg ITM, {s.dte}d to expiry (assignment guard)"))
            continue
        # Legacy wind-down: close non-validated positions as soon as they're green/flat.
        if (rules.get("breakeven_close") and s.underlying not in keep
                and s.unrealized_pl >= 0):
            out.append(ExitDecision(s, f"legacy wind-down at break-even (uPnL {s.unrealized_pl:+.0f})"))
            continue
        if s.profit_fraction() >= rules["profit_target_frac"]:
            out.append(ExitDecision(s, f"profit target {s.profit_fraction():.0%} of max"))
            continue
        if s.loss_fraction() >= rules["stop_loss_frac"]:
            out.append(ExitDecision(s, f"stop loss {s.loss_fraction():.0%} of premium"))
            continue
    return out


def build_close_order(s: StructurePosition) -> OptionOrder:
    """Reverse every leg with a close intent. Limit = current mark (per share)."""
    legs = [OrderLeg(symbol=l.symbol, side=("SELL" if l.qty > 0 else "BUY"),
                     ratio_qty=abs(l.qty) // s.contracts if s.contracts else 1,
                     closing=True)
            for l in s.legs]
    # Closing a debit spread is a SELL for a credit ~= current mark per share.
    limit = abs(s.market_value) / (s.contracts * 100) if s.contracts else 0.01
    return OptionOrder(underlying=s.underlying, structure="close", legs=legs,
                       qty=s.contracts, limit_price=round(max(limit, 0.01), 2))
