"""The Strategy interface — the ONE seam a bot author implements.

A market is a raw Manifold v0 dict (whatever `get_market`/`list_markets`
returns), so strategies have full fidelity. The required contract is tiny:

    estimate(group) -> {market_id: fair_value_probability}

Everything else (client, sizing, execution, ledger, PnL, **commenting**) is
shared infrastructure a bot author never touches. Optional hooks `prefilter` and
`group` let a strategy narrow the market universe and decide which markets are
evaluated together.

**Pipeline guarantee — commenting.** Every successful bet placed by the runner
automatically gets a markdown comment posted on its market explaining the model's
reasoning (universal block: model vs market vs edge vs fill; plus the strategy's
own `explain()` block if implemented). This is on by default for ALL bots — you
do not need to wire anything up. To surface strategy-specific reasoning, override
`explain(market_id)` and populate `self._explanations[market_id]` during
`estimate`. To disable comments (testing only), set `post_comments: false` in the
bot's limits — the runner will log a warning that this is unusual.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

Market = dict[str, Any]


class Strategy(ABC):
    #: Stable identifier, also used as the key in the strategy REGISTRY.
    name: str = "base"

    #: Long-form description shown on the dashboard's bot card. Should explain
    #: *what inefficiency the strategy exploits*, *which market universe it
    #: targets*, and *the edge logic in one or two sentences*. Authored on the
    #: class so the description lives with the code and new bots can't ship
    #: without one. 2-4 sentences is the sweet spot.
    description: str = ""

    def __init__(self, **params: Any):
        #: Free-form strategy parameters from config/bots.yaml `params:`.
        self.params = params
        #: Per-market reasoning recorded during `estimate`, keyed by market_id.
        #: Strategies populate this in `estimate` so `explain` can format the
        #: numbers without re-running the model. Latest-wins on overwrite.
        self._explanations: dict[str, dict[str, Any]] = {}

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

    def explain(self, market_id: str) -> str | None:
        """Optional: markdown reasoning for the model estimate, posted as a
        comment alongside the trade. Default: None (no strategy-specific block;
        the runner still posts the universal model vs. market vs. edge summary).

        Strategies that override should populate `self._explanations[market_id]`
        during `estimate` with whatever numbers they reason from, then format them
        here. Return None if the market wasn't reasoned about (no explanation
        available)."""
        return None
