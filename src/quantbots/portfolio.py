"""Portfolio allocation — turn a pile of per-market signals into a capital-
efficient book of orders the runner can place at scale.

`sizing.compute_trade` decides *how much* a single market can absorb (its caps).
This module decides, given hundreds or thousands of such candidate orders and a
finite budget, *which* to fund and *how much* of the budget each gets. Pure stdlib
so it stays trivially testable, like `sizing`.

Two ideas do the work:

1. **Rank by expected value per mana, not raw edge.** Betting M mana on YES at
   market price q buys ~M/q shares worth our estimate p in expectation, so the
   expected profit per mana is (p-q)/q; for NO it is (q-p)/(1-q). A NO bet at
   q=0.99 we believe is 0.01 returns ~98x per mana — far better than an
   equal-*edge* bet at mid price. Ranking by EV/mana deploys capital where it
   compounds fastest.

2. **Correlation-aware concentration caps.** 100 "gold exceeds $X" markets are
   one bet on the gold price, not 100 independent ones. A per-group budget cap
   (group = the underlying, via `Strategy.correlation_key`) stops the allocator
   from sinking the whole book into a single correlated view.

The result is a greedy knapsack: walk signals best-EV-first, fund each up to its
own sizing cap, the remaining total budget, and its group's remaining budget.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def ev_per_mana(estimate: float, current_prob: float, direction: str) -> float:
    """Expected profit per mana staked for this order (slippage ignored).

    Positive means +EV. Buying the cheaper side of a large mispricing yields a
    large number because each mana buys many shares of something we think is
    nearly certain.
    """
    q = min(max(current_prob, 1e-4), 1 - 1e-4)
    p = min(max(estimate, 0.0), 1.0)
    if direction == "YES":
        return (p - q) / q
    return (q - p) / (1 - q)


def allocate(
    signals: list[dict[str, Any]],
    *,
    total_budget: float | None,
    per_group_budget: float | None = None,
    min_ev: float = 0.0,
    min_order_mana: float = 1.0,
    max_total_exposure: float | None = None,
    max_group_exposure: float | None = None,
    existing_total: float = 0.0,
    existing_group: dict[Any, float] | None = None,
) -> list[dict[str, Any]]:
    """Select and size orders to maximize expected profit under budget caps.

    Each signal must carry: estimate, current_prob, direction, amount (its
    per-market sizing cap), and optionally `group` (correlation key, default the
    market id). Returns a new list of funded signals, best-EV-first, each with its
    `amount` possibly trimmed to fit a remaining budget and an added
    `ev_per_mana` / `exp_profit` for reporting. Non-destructive: inputs untouched.

    Per-run caps (reset every run):
    - `total_budget` None/<=0 means no overall ceiling per run.
    - `per_group_budget` caps mana per correlation group within this run.
    - `min_ev` drops signals whose EV per mana is below the threshold.

    Across-run exposure caps (bound cumulative position over repeated live runs,
    so a bot that keeps nudging a stubborn market can't deploy unbounded capital):
    - `max_total_exposure` ceilings existing_total + this run's spend.
    - `max_group_exposure` ceilings existing_group[g] + this run's group spend.
    `existing_*` are the already-deployed stakes (from the trade ledger).
    """
    existing_group = dict(existing_group or {})
    ranked = []
    for s in signals:
        # Realized EV = paper EV x P(market actually resolves YES/NO). With ~93% of
        # resolutions cancelling (refund), this is what edge actually pays out, so we
        # rank and gate on it — capital flows to resolvable markets, not paper edge.
        ev = ev_per_mana(s["estimate"], s["current_prob"], s["direction"])
        realized = ev * float(s.get("resolvability", 1.0))
        if realized < min_ev:
            continue
        ranked.append((realized, s))
    # Best realized-EV first; break ties toward larger orders to deploy capital faster.
    ranked.sort(key=lambda t: (t[0], t[1].get("amount", 0)), reverse=True)

    spent_total = 0.0
    spent_group: dict[Any, float] = defaultdict(float)
    kept: list[dict[str, Any]] = []

    for ev, s in ranked:
        group = s.get("group", s["market_id"])
        # Per-run room.
        room_total = math.inf if not total_budget or total_budget <= 0 else total_budget - spent_total
        room_group = math.inf if per_group_budget is None else per_group_budget - spent_group[group]
        # Across-run exposure room (cumulative ledger position + this run so far).
        if max_total_exposure is not None:
            room_total = min(room_total, max_total_exposure - existing_total - spent_total)
        if max_group_exposure is not None:
            room_group = min(
                room_group, max_group_exposure - existing_group.get(group, 0.0) - spent_group[group]
            )
        if room_total <= 0:
            break  # overall budget / total exposure exhausted; nothing more can be funded
        if room_group <= 0:
            continue  # this underlying is full, but others may still have room

        amount = min(float(s["amount"]), room_total, room_group)
        if amount < min_order_mana:
            continue  # too small to bother once trimmed

        amount = int(amount)
        if amount < min_order_mana:
            continue
        funded = {**s, "amount": amount, "ev_per_mana": ev, "exp_profit": ev * amount}
        kept.append(funded)
        spent_total += amount
        spent_group[group] += amount

    return kept


def book_summary(funded: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats for a funded book (for logging / reporting)."""
    staked = sum(s["amount"] for s in funded)
    exp_profit = sum(s.get("exp_profit", 0.0) for s in funded)
    by_group: dict[Any, float] = defaultdict(float)
    for s in funded:
        by_group[s.get("group", s["market_id"])] += s["amount"]
    return {
        "orders": len(funded),
        "staked": staked,
        "exp_profit": exp_profit,
        "exp_roi": (exp_profit / staked) if staked else 0.0,
        "groups": dict(by_group),
    }
