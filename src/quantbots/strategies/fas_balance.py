"""FAS balance-sheet bot (single source = USDA FAS PSD): quantity markets.

Covers the cotton *quantity* market type — "Will global cotton ending stocks /
production / mill use ... exceed X million bales?", "Will China cotton imports ...",
"Will Brazil cotton exports ...". These map DIRECTLY to USDA FAS PSD (no price
anchor): we read the FAS forecast for that exact quantity & marketing year and
price P(value > threshold) with a small forecast-uncertainty band.

Distinct from fas_fundamental (which prices *price* markets). Same single source.
Abstains when FAS has no forecast for the market's marketing year (far-dated) or
the question doesn't match a covered quantity — only meaningful trades.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ._model import norm_cdf
from .base import Market, Strategy
from .ladder import parse_threshold

# question regex -> balance entity (emitted by processing/signals.compute_fas_balance)
_CATALOG = [
    (re.compile(r"global cotton production", re.I), "SIG_COTTON_WORLD_PRODUCTION"),
    (re.compile(r"global cotton mill use|global cotton consumption", re.I), "SIG_COTTON_WORLD_MILLUSE"),
    (re.compile(r"global cotton ending stocks", re.I), "SIG_COTTON_WORLD_ENDSTOCKS"),
    (re.compile(r"china cotton imports", re.I), "SIG_COTTON_CHINA_IMPORTS"),
    (re.compile(r"brazil cotton exports", re.I), "SIG_COTTON_BRAZIL_EXPORTS"),
]
_YEAR = re.compile(r"\b(20\d{2})\b")


class FasBalanceStrategy(Strategy):
    name = "fas_balance"
    description = (
        "Single-source (USDA FAS PSD) bot for cotton balance-sheet quantity markets "
        "(global production / mill use / ending stocks, China imports, Brazil exports). "
        "Prices P(quantity > threshold) directly from the FAS forecast for that "
        "marketing year, with a forecast-uncertainty band. Abstains beyond FAS's "
        "forecast horizon."
    )

    def __init__(self, rel_std: float = 0.04, **params: Any):
        super().__init__(rel_std=rel_std, **params)
        self.rel_std = rel_std  # FAS forecast revision uncertainty as a fraction of the value
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _entity(self, q: str) -> str | None:
        for pat, ent in _CATALOG:
            if pat.search(q):
                return ent
        return None

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [m for m in markets
                if not m.get("isResolved") and self._entity(m.get("question", "")) is not None]

    def correlation_key(self, market: Market) -> str:
        return self._entity(market.get("question", "")) or str(market.get("id"))

    def _fas_value(self, entity: str, question: str) -> tuple[float, int] | None:
        """FAS value (million bales) for the marketing year the market references."""
        o = self._obs.latest_observation(entity) if self._obs else None
        if not o:
            return None
        pay = o.get("payload")
        if isinstance(pay, str):
            try:
                pay = json.loads(pay)
            except (ValueError, TypeError):
                pay = {}
        by_my = (pay or {}).get("by_my") or {}
        years = [int(y) for y in _YEAR.findall(question)]
        if not years:
            return None
        my = max(years) - 1  # cotton MY ending year Y == PSD market year Y-1
        v = by_my.get(str(my))
        return (float(v), my) if v is not None else None

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            q = m.get("question", "")
            entity = self._entity(q)
            if entity is None:
                continue
            parsed = parse_threshold(q)
            if parsed is None:
                continue
            threshold, direction = parsed
            fv = self._fas_value(entity, q)
            if fv is None:
                continue  # FAS has no forecast for this marketing year -> abstain
            value, my = fv
            sigma = max(self.rel_std * value, 1e-6)
            surv = 1.0 - norm_cdf((threshold - value) / sigma)  # P(value > threshold)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "entity": entity, "fas_value": value, "marketing_year": my,
                "threshold": threshold, "direction": direction, "sigma": sigma, "p": p,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        return (
            f"- USDA FAS forecast ({d['entity'].replace('SIG_COTTON_', '').replace('_', ' ').lower()}, "
            f"MY {d['marketing_year']}): **{d['fas_value']:.1f} M bales**\n"
            f"- Threshold: **{d['threshold']:.1f}** ({d['direction']}), band σ={d['sigma']:.1f} "
            f"→ P({d['direction']})=**{d['p']:.3f}**"
        )
