"""Market-maker — an EXECUTION style, not a new forecasting model.

The fleet already produces calibrated fair values via `Strategy.estimate()`. The
maker does not re-forecast: it WRAPS an existing fair-value strategy (default
`commodity_spot`) and delegates all market understanding — prefilter, grouping,
estimate, correlation, explanation — to it. The maker logic itself (two-sided
resting limit quotes around the fair value, with a TTL, plus fill reconciliation)
lives in `quantbots.maker.run_maker`, which consumes this strategy's estimates.

Keeping the fair value in the source strategy is the whole point: any calibrated
anchor can be turned into a liquidity provider for free. v1 ships fixed-spread
quoting on the source's markets; price skew / toxic-flow widening are Phase 3.

Config (`config/bots.yaml`):

    strategy: market_maker
    params:
      fair_value_source: commodity_spot   # any registered, calibrated strategy
      base_half_spread: 0.04              # quote at fair ± this (whole-percent)
      min_half_spread: 0.02              # never quote tighter than this
      inventory_cap: 200                 # |net YES shares| before quoting one-sided
      quote_ttl_hours: 25                # TTL so stale quotes self-expire (+ cancel)
      max_markets: 10                    # breadth cap (top-N by liquidity)
      source_params: {}                  # optional kwargs forwarded to the source
"""

from __future__ import annotations

from typing import Any

from .base import Market, Strategy


class MarketMakerStrategy(Strategy):
    name = "market_maker"
    description = (
        "Provides two-sided liquidity: posts resting limit quotes a fixed spread "
        "either side of a calibrated fair value (from a wrapped source strategy, "
        "default commodity_spot) so the thin clone AMM gains real book depth, the "
        "price pins to fair value as fills arrive, and the maker earns the spread. "
        "An execution style over an existing anchor, not a new forecasting model."
    )

    def __init__(
        self,
        fair_value_source: str = "commodity_spot",
        base_half_spread: float = 0.04,
        min_half_spread: float = 0.02,
        inventory_cap: float = 200.0,
        quote_ttl_hours: float = 25.0,
        max_markets: int = 10,
        source_params: dict[str, Any] | None = None,
        **params: Any,
    ):
        super().__init__(
            fair_value_source=fair_value_source,
            base_half_spread=base_half_spread,
            min_half_spread=min_half_spread,
            inventory_cap=inventory_cap,
            quote_ttl_hours=quote_ttl_hours,
            max_markets=max_markets,
            source_params=source_params or {},
            **params,
        )
        # Local import avoids any package import-order pitfalls (the registry that
        # defines get_strategy imports this module lazily).
        from quantbots.strategies import get_strategy

        self.source = get_strategy(fair_value_source, **(source_params or {}))
        self.base_half_spread = float(base_half_spread)
        self.min_half_spread = float(min_half_spread)
        self.inventory_cap = float(inventory_cap)
        self.quote_ttl_hours = float(quote_ttl_hours)
        self.max_markets = int(max_markets)

    # --- fair value + universe: delegate entirely to the wrapped source -------

    def bind(self, observations: Any) -> None:
        self.source.bind(observations)

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return self.source.prefilter(markets)

    def group(self, markets: list[Market]) -> list[list[Market]]:
        return self.source.group(markets)

    def correlation_key(self, market: Market) -> str:
        return self.source.correlation_key(market)

    def estimate(self, group: list[Market]) -> dict[str, float]:
        return self.source.estimate(group)

    def explain(self, market_id: str) -> str | None:
        return self.source.explain(market_id)

    # --- maker-specific ------------------------------------------------------

    def half_spread(self, market_id: str | None = None) -> float:
        """Half-spread for a market's quotes. v1: a flat floor-clamped constant.
        Phase 3 widens this by fair-value uncertainty, resolvability, and toxic
        flow (see docs/market-maker.md spread model)."""
        return max(self.min_half_spread, self.base_half_spread)
