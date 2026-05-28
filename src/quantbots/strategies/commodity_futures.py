"""Soft-commodities bot: trades agricultural futures *price* threshold markets.

Different market set (ICE/CBOT softs: cotton, sugar, wheat, corn, cocoa) and its
own source basket + linker catalog. Futures prices are positive and roughly
lognormal, so it uses a lognormal threshold model (horizon-scaled vol) — the
same *family* as `ensemble`, but a distinct domain, catalog, and calibration, and
fully self-contained so it doesn't touch other bots' markets.

The catalog maps the precise market phrasing (e.g. "ICE Cotton No. 2 front-month
futures") to a Stooq-backed entity. Add a line per commodity as feeds are added.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market, Strategy
from .ladder import parse_threshold

# question phrase (regex) -> entity (paired with a stooq symbol in sources.yaml)
_CATALOG_RAW: list[tuple[str, str]] = [
    (r"ice cotton no\.?\s*2|cotton no\.?\s*2 (front-month )?futures|\bcotton.*futures", "CME_COTTON"),
    (r"\bsugar\b.*(no\.?\s*11|futures)", "CME_SUGAR"),
    (r"\bcorn\b.*futures|cbot corn", "CME_CORN"),
    (r"\bwheat\b.*futures|cbot wheat", "CME_WHEAT"),
    (r"\bcocoa\b.*futures", "CME_COCOA"),
]
CATALOG = [(re.compile(p, re.I), e) for p, e in _CATALOG_RAW]


class CommodityFuturesStrategy(Strategy):
    name = "commodity_futures"
    description = (
        "Lognormal pricing on ICE/CBOT soft-commodity futures threshold markets "
        "— sugar, cocoa, coffee, cotton, wheat, corn, soybeans. Same diffusion "
        "model as commodity_spot but anchored to the front-month futures "
        "contract from Stooq rather than physical spot."
    )

    def __init__(self, annual_vol: float = 0.30, min_vol: float = 0.05, **params: Any):
        super().__init__(annual_vol=annual_vol, min_vol=min_vol, **params)
        self.annual_vol = annual_vol
        self.min_vol = min_vol
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _entity(self, question: str) -> str | None:
        for pat, entity in CATALOG:
            if pat.search(question):
                return entity
        return None

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [
            m for m in markets
            if not m.get("isResolved") and self._entity(m.get("question", "")) is not None
        ]

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            entity = self._entity(m.get("question", ""))
            if entity is None:
                continue
            parsed = parse_threshold(m.get("question", ""))
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            sigma = max(self.annual_vol * math.sqrt(years_to_close(m)), self.min_vol)
            surv = 1.0 - norm_cdf(math.log(threshold / o["value"]) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            out[m["id"]] = min(max(p, 0.01), 0.99)
        return out
