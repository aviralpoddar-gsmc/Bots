"""The Strategy interface — the ONE seam a bot author implements.

A market is a raw Manifold v0 dict (whatever `get_market`/`list_markets`
returns), so strategies have full fidelity. The required contract is tiny:

    estimate(group) -> {market_id: fair_value_probability}

Everything else (client, sizing, execution, ledger, PnL) is shared infrastructure
a bot author never touches. Optional hooks `prefilter` and `group` let a strategy
narrow the market universe and decide which markets are evaluated together.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

Market = dict[str, Any]


class Strategy(ABC):
    #: Stable identifier, also used as the key in the strategy REGISTRY.
    name: str = "base"

    def __init__(self, **params: Any):
        #: Free-form strategy parameters from config/bots.yaml `params:`.
        self.params = params

    def bind(self, observations: Any) -> None:
        """Optional: give the strategy a read handle to ingested observations.

        `observations` exposes `latest_observation(entity)` and
        `load_observations(...)` (the Store satisfies this). The runner calls it
        once before `estimate`. Strategies that trade on external data (e.g.
        `ensemble`) keep the reference; market-only strategies ignore it.
        """
        return None

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
        now_ms = time.time() * 1000
        out = []
        for m in markets:
            if m.get("isResolved"):
                continue
            # Closed-but-unresolved markets reject bets (403). closeTime is epoch ms;
            # None/0 means no close set, treat as open.
            close = m.get("closeTime")
            if close and close <= now_ms:
                continue
            if (m.get("totalLiquidity") or 0) < self.params.get("min_liquidity", 50):
                continue
            out.append(m)
        return out

    def group(self, markets: list[Market]) -> list[list[Market]]:
        """Partition markets into groups evaluated together by `estimate`.
        Default: each market is its own group (single-market strategies)."""
        return [[m] for m in markets]

    def correlation_key(self, market: Market) -> str:
        """Key identifying which markets share an underlying risk, used by the
        portfolio allocator to cap concentration in correlated bets.

        Default: the market id (every market independent — no grouping). Override
        to return e.g. the underlying entity so all strikes/dates of one quantity
        ("gold price") count against a single per-group budget."""
        return str(market.get("id"))
