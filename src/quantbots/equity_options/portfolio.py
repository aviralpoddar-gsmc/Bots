"""Portfolio allocation across candidates — greedy by P&L Sharpe, capped.

Mirrors `portfolio.allocate`'s philosophy (rank by a risk-adjusted score, then honor
concentration caps), with options-specific budgets:
  - per-underlying and total premium caps (dollars at risk),
  - a `correlation_key` = the driving COMMODITY, so eight gold names can't become
    eight independent gold bets (the plan's net-delta-in-commodity-factor-space idea,
    enforced here as a per-commodity premium cap),
  - portfolio greek budgets: net vega, net theta, gross gamma.

One structure per underlying per cycle (take its best candidate) keeps the book
legible and avoids overlapping legs on the same name.
"""

from __future__ import annotations

from dataclasses import dataclass

from .selection import Candidate
from .sizing import size_contracts


@dataclass
class Allocation:
    candidate: Candidate
    contracts: int
    premium_total: float          # contracts * candidate.premium
    commodity: str


def allocate(candidates: list[Candidate], *, bankroll: float, limits: dict,
             commodity_of: dict[str, str]) -> list[Allocation]:
    """Greedily allocate the highest-Sharpe candidates subject to all caps.

    `commodity_of` maps underlying ticker -> commodity entity (the correlation key).
    """
    spent_total = 0.0
    spent_underlying: dict[str, float] = {}
    spent_commodity: dict[str, float] = {}
    net_vega = net_theta = gross_gamma = 0.0
    used_underlyings: set[str] = set()
    out: list[Allocation] = []

    per_und = limits["max_premium_per_underlying"]
    per_comm = limits.get("max_premium_per_underlying", per_und)  # commodity cap (reuse per-underlying)
    total_cap = limits["max_total_premium"]

    for c in candidates:                      # already sorted by score desc
        if c.underlying in used_underlyings:  # one structure per underlying
            continue
        commodity = commodity_of.get(c.underlying, c.underlying)
        contracts = size_contracts(c, bankroll=bankroll, limits=limits)
        if contracts < 1:
            continue
        # Shrink contracts to fit the binding premium cap, if any.
        room = min(
            total_cap - spent_total,
            per_und - spent_underlying.get(c.underlying, 0.0),
            per_comm - spent_commodity.get(commodity, 0.0),
        )
        if room < c.premium:
            continue
        contracts = min(contracts, int(room // c.premium))
        if contracts < 1:
            continue
        prem = contracts * c.premium
        # Greek-budget check (scale greeks by chosen contracts).
        cand_vega = c.net_greeks["vega"] * contracts
        cand_theta = c.net_greeks["theta"] * contracts
        cand_gamma = c.net_greeks["gamma"] * contracts
        if abs(net_vega + cand_vega) > limits["max_net_vega"]:
            continue
        if abs(net_theta + cand_theta) > limits["max_net_theta"]:
            continue
        if gross_gamma + abs(cand_gamma) > limits["max_gross_gamma"]:
            continue

        spent_total += prem
        spent_underlying[c.underlying] = spent_underlying.get(c.underlying, 0.0) + prem
        spent_commodity[commodity] = spent_commodity.get(commodity, 0.0) + prem
        net_vega += cand_vega; net_theta += cand_theta; gross_gamma += abs(cand_gamma)
        used_underlyings.add(c.underlying)
        out.append(Allocation(candidate=c, contracts=contracts, premium_total=prem,
                              commodity=commodity))
    return out
