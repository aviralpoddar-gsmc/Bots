"""The Strategy interface — the ONE seam a bot author implements.

A market is a raw Manifold v0 dict (whatever `get_market`/`list_markets`
returns), so strategies have full fidelity. The required contract is tiny:

    estimate(group) -> {market_id: fair_value_probability}

Everything else (client, sizing, execution, ledger, PnL) is shared infrastructure
a bot author never touches. Optional hooks `prefilter` and `group` let a strategy
narrow the market universe and decide which markets are evaluated together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Market = dict[str, Any]


class Strategy(ABC):
    #: Stable identifier, also used as the key in the strategy REGISTRY.
    name: str = "base"

    def __init__(self, **params: Any):
        #: Free-form strategy parameters from config/bots.yaml `params:`.
        self.params = params

    @abstractmethod
    def estimate(self, group: list[Market]) -> dict[str, float]:
        """Return your fair-value probability for each market in `group`.

        Keys are market ids; values are probabilities in (0, 1). Omit a market to
        abstain from trading it. Markets passed together are one `group` (see
        `group`), so multi-market strategies (ladder fits, coherence) can reason
        across them jointly.
        """

    def prefilter(self, markets: list[Market]) -> list[Market]:
        """Narrow the universe before evaluation. Default: keep open, liquid,
        not-about-to-close markets. Override to add strategy-specific filters."""
        out = []
        for m in markets:
            if m.get("isResolved"):
                continue
            if (m.get("totalLiquidity") or 0) < self.params.get("min_liquidity", 50):
                continue
            out.append(m)
        return out

    def group(self, markets: list[Market]) -> list[list[Market]]:
        """Partition markets into groups evaluated together by `estimate`.
        Default: each market is its own group (single-market strategies)."""
        return [[m] for m in markets]
