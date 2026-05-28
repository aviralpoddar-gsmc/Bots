"""Climate bot: trades ENSO (Oceanic Niño Index) threshold markets.

Different domain, different source (NOAA), and a different *model* from the
`ensemble` bot: the ONI is an additive, mean-reverting anomaly that can be
negative, so a lognormal model is inappropriate. Instead we use a **Gaussian
persistence** model — assume the future ONI is normally distributed around the
latest reading, with a spread that grows with the horizon:

    P(ONI_future > T) = 1 - Phi( (T - V) / sigma ),   sigma = monthly_vol * sqrt(months)

This is its own self-contained linker (it only understands ONI markets), so it
never steps on the other bots' domains.
"""

from __future__ import annotations

import re
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market, Strategy
from .ladder import parse_threshold

_ONI_RE = re.compile(r"oceanic ni|\boni\b|el ni[nñ]o[- ]?southern", re.I)
ENTITY = "ENSO_ONI"


class EnsoStrategy(Strategy):
    name = "enso"
    description = (
        "Climate-index forecaster for ENSO / Oceanic Niño Index threshold markets. "
        "Models ONI as a Gaussian random walk anchored to the latest NOAA monthly "
        "value, with sigma = monthly_vol × √(months-to-close). Trades "
        "'will ONI exceed X by month Y' markets where the market price diverges "
        "from this persistence-model survival."
    )

    def __init__(self, monthly_vol: float = 0.25, min_sigma: float = 0.2, **params: Any):
        super().__init__(monthly_vol=monthly_vol, min_sigma=min_sigma, **params)
        self.monthly_vol = monthly_vol
        self.min_sigma = min_sigma
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [m for m in markets if not m.get("isResolved") and _ONI_RE.search(m.get("question", ""))]

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        o = self._obs.latest_observation(ENTITY)
        if not o or o.get("value") is None:
            return {}
        current = o["value"]
        out: dict[str, float] = {}
        for m in group:
            if not _ONI_RE.search(m.get("question", "")):
                continue
            parsed = parse_threshold(m.get("question", ""))
            if parsed is None:
                continue
            threshold, direction = parsed
            months = years_to_close(m) * 12.0
            sigma = max(self.monthly_vol * (months ** 0.5), self.min_sigma)
            surv = 1.0 - norm_cdf((threshold - current) / sigma)  # P(future > T)
            p = surv if direction == "exceeds" else 1.0 - surv
            out[m["id"]] = min(max(p, 0.01), 0.99)
        return out
