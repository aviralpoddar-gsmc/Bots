"""Source registry — lazy, like the strategy registry.

To add a source: implement a `Source` subclass and add one line to `_REGISTRY`.
Heavy/optional imports stay inside their module so the core isn't burdened.
"""

from __future__ import annotations

import importlib

from .base import Observation, Source

# name -> "module path:ClassName"
_REGISTRY: dict[str, str] = {
    "stooq": "quantbots.sources.stooq:StooqSource",        # commodities / FX / indices / equities
    "worldbank": "quantbots.sources.worldbank:WorldBankSource",  # macro / econ (global, annual)
    "fred": "quantbots.sources.fred:FredSource",           # US macro series (keyless CSV)
    "noaa": "quantbots.sources.noaa:NoaaSource",           # climate indices (ENSO/ONI)
    "rss": "quantbots.sources.rss:RSSSource",              # news / text
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_source(name: str, **params: object) -> Source:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown source {name!r}. Available: {available()}")
    module_path, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**params)


__all__ = ["Source", "Observation", "get_source", "available"]
