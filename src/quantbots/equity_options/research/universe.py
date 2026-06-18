"""The tradable universe + the equity<->commodity mapping.

Thin glue over `config.EquityOptionsConfig`: it resolves each configured underlying's
commodity entity to the yfinance ticker the price feed uses (via the parent's
`research.data_fetch.DEFAULT_UNIVERSE`), so the beta regression and the diffusion
forecast pull the same commodity series.
"""

from __future__ import annotations

from ...research.data_fetch import DEFAULT_UNIVERSE
from ..config import Underlying


def commodity_ticker(commodity: str) -> str | None:
    """yfinance ticker for a commodity entity key (e.g. COPPER -> HG=F)."""
    return DEFAULT_UNIVERSE.get(commodity.upper())


def resolve(u: Underlying) -> dict[str, str | None]:
    """Resolve an underlying's three regression legs to fetchable tickers."""
    return {
        "equity": u.ticker,
        "commodity_entity": u.commodity,
        "commodity_ticker": commodity_ticker(u.commodity),
        "market": u.market_ticker,
    }
