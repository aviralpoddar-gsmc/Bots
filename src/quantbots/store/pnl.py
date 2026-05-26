"""PnL formulas.

Manifold CPMM settlement: a YES share is worth `prob`, a NO share is worth
`1 - prob` (both at the current/resolution probability). Realized PnL comes from
EXIT rows; unrealized from marking the remaining net shares at the current price.

Resolution is modelled as a synthetic EXIT (`RESOLUTION_CLOSE`) with
price_after = 1.0 for the winning side or 0.0 for the losing side, so there is no
special-case PnL code — inserting that row realizes the position.

NOTE: the parent repo's sketch defined `realized = exit_proceeds - entry_amount`,
which double-counts the entry cost once a position is partially open
(realized + unrealized != true total PnL). We use a per-share cost basis instead
so `realized + unrealized` always equals the true total. The CPMM share-value
semantics (`proceeds`) are unchanged.

Pure functions, no DB and no optional deps, so they are trivial to unit-test.
"""

from __future__ import annotations

from collections.abc import Callable

from .trades import ENTRY, EXIT_TYPES, group_positions, summarize_position


def position_pnl(
    position_trades: list[dict],
    current_prob: float | None,
    is_resolved: bool = False,
) -> tuple[float, float]:
    """Return (realized, unrealized) PnL for one (market, direction) position."""
    direction = position_trades[0]["direction"]
    entry_amount = sum(t["amount"] for t in position_trades if t["trade_type"] == ENTRY)
    entry_shares = sum(t["shares"] for t in position_trades if t["trade_type"] == ENTRY)
    exits = [t for t in position_trades if t["trade_type"] in EXIT_TYPES]

    def proceeds(shares: float, prob: float) -> float:
        """Value of `shares` of `direction` at probability `prob`."""
        return shares * (prob if direction == "YES" else 1 - prob)

    # Per-share cost basis from entries.
    cost_per_share = entry_amount / entry_shares if entry_shares > 1e-9 else 0.0
    exited_shares = sum(t["shares"] for t in exits)

    realized = sum(proceeds(t["shares"], t["price_after"]) for t in exits) \
        - exited_shares * cost_per_share

    net_shares = entry_shares - exited_shares
    if net_shares <= 1e-9 or current_prob is None:
        unrealized = 0.0
    else:
        unrealized = proceeds(net_shares, current_prob) - net_shares * cost_per_share
    return realized, unrealized


def bot_pnl(trades: list[dict], current_prob: Callable[[str], float | None]) -> dict:
    """Aggregate realized + unrealized PnL across all of a bot's positions.

    `current_prob(market_id)` supplies the mark price for unrealized PnL.
    """
    realized = unrealized = total_invested = 0.0
    open_n = closed_n = 0
    for (market_id, _dir), pos_trades in group_positions(trades).items():
        summary = summarize_position(pos_trades)
        r, u = position_pnl(pos_trades, current_prob(market_id))
        realized += r
        unrealized += u
        total_invested += summary["entry_amount"]
        if summary["status"] == "OPEN":
            open_n += 1
        else:
            closed_n += 1
    return {
        "realized_pnl": round(realized, 4),
        "unrealized_pnl": round(unrealized, 4),
        "pnl": round(realized + unrealized, 4),
        "total_invested": round(total_invested, 4),
        "open_positions": open_n,
        "closed_positions": closed_n,
    }
