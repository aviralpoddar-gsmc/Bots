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
    "llm": "quantbots.strategies.llm:LLMStrategy",
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
