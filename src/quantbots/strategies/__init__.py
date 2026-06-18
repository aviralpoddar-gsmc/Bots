"""Strategy registry.

Resolution is LAZY: heavy strategies (surface_arb -> numpy/scipy, llm -> openai)
are only imported when actually requested, so a bot author can use the core
without installing every optional extra. To add a strategy, implement a
`Strategy` subclass and add one line to `_REGISTRY`.
"""

from __future__ import annotations

import importlib

from .base import Strategy

# name -> "module path:ClassName"
_REGISTRY: dict[str, str] = {
    "mean_reversion": "quantbots.strategies.mean_reversion:MeanReversionStrategy",
    "surface_arb": "quantbots.strategies.surface_arb:SurfaceArbStrategy",
    "ensemble": "quantbots.strategies.ensemble:EnsembleStrategy",
    "enso": "quantbots.strategies.enso:EnsoStrategy",
    "commodity_futures": "quantbots.strategies.commodity_futures:CommodityFuturesStrategy",
    "commodity_spot": "quantbots.strategies.commodity_spot:CommoditySpotStrategy",
    "diffusion_mc": "quantbots.strategies.diffusion_mc:DiffusionMcStrategy",
    "cotton_fundamental": "quantbots.strategies.cotton_fundamental:CottonFundamentalStrategy",
    "cocoa_fundamental": "quantbots.strategies.cocoa_fundamental:CocoaFundamentalStrategy",
    "coffee_consumption": "quantbots.strategies.coffee_consumption:CoffeeConsumptionStrategy",
    # Single-source, price-anchored bots (one data source each) on the shared base.
    "fas_fundamental": "quantbots.strategies.fas_fundamental:FasFundamentalStrategy",
    "fas_balance": "quantbots.strategies.fas_balance:FasBalanceStrategy",
    "wasde_event": "quantbots.strategies.wasde_event:WasdeEventStrategy",
    "cftc_positioning": "quantbots.strategies.cftc_positioning:CftcPositioningStrategy",
    "weather_anomaly": "quantbots.strategies.weather_anomaly:WeatherAnomalyStrategy",
    "nass_crop": "quantbots.strategies.nass_crop:NassCropStrategy",
    "cocoa_atlantic": "quantbots.strategies.cocoa_atlantic:CocoaAtlanticStrategy",
    "drought_cotton": "quantbots.strategies.drought_cotton:DroughtCottonStrategy",
    "cocoa_stocks": "quantbots.strategies.cocoa_stocks:CocoaStocksStrategy",
    "pair_trading": "quantbots.strategies.pair_trading:PairTradingStrategy",
    "ladder_arb": "quantbots.strategies.ladder_arb:LadderArbStrategy",
    "semantic_arb": "quantbots.strategies.semantic_arb:SemanticArbStrategy",
    "term_structure": "quantbots.strategies.term_structure:TermStructureStrategy",
    "stockpile_facts": "quantbots.strategies.stockpile_facts:StockpileFactsStrategy",
    "stockpile_grid_arb": "quantbots.strategies.stockpile_grid_arb:StockpileGridArbStrategy",
    "stockpile_coherence": "quantbots.strategies.stockpile_coherence:StockpileCoherenceStrategy",
    "market_maker": "quantbots.strategies.market_maker:MarketMakerStrategy",
    "news_drift": "quantbots.strategies.news_drift:NewsDriftStrategy",
    "llm": "quantbots.strategies.llm:LLMStrategy",
    # Hosted-inference exception (see docs/mercury-ensemble-calibration.md).
    "mercury_ensemble": "quantbots.strategies.mercury_ensemble:MercuryEnsembleStrategy",
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_strategy(name: str, **params: object) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {available()}")
    module_path, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**params)


__all__ = ["Strategy", "get_strategy", "available"]
