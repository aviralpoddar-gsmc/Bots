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

    def __init__(self, rel_std: float = 0.04, extrap_widen: float = 0.6,
                 max_extrap_years: int = 4, **params: Any):
        super().__init__(rel_std=rel_std, extrap_widen=extrap_widen,
                         max_extrap_years=max_extrap_years, **params)
        self.rel_std = rel_std  # FAS forecast revision uncertainty as a fraction of the value
        # COVERAGE: FAS only forecasts ~1-2 marketing years out, but the clone lists
        # quantity markets several years ahead. Rather than abstain (leaving them at
        # 0.50), carry the latest FAS balance forward and WIDEN the band per year
        # beyond the forecast horizon — so far-dated markets get a humble-but-real
        # fair value instead of a coin flip. extrap_widen = extra band fraction per
        # extrapolated year; max_extrap_years caps how far we'll carry forward.
        self.extrap_widen = extrap_widen
        self.max_extrap_years = max_extrap_years
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

    def _fas_value(self, entity: str, question: str) -> tuple[float, int, int] | None:
        """FAS value for the marketing year the market references, plus how many
        years it was extrapolated beyond FAS's forecast horizon (0 = exact match).

        Far-dated markets (the clone lists them years out, past FAS's ~1-2yr
        horizon) carry the latest available marketing year's value forward rather
        than abstaining; `estimate` widens the band per extrapolated year."""
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
        if v is not None:
            return float(v), my, 0
        avail = sorted(int(k) for k in by_my)
        if avail and my > avail[-1] and (my - avail[-1]) <= self.max_extrap_years:
            return float(by_my[str(avail[-1])]), my, my - avail[-1]
        return None  # past marketing year, or beyond the carry-forward cap -> abstain

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
                continue  # FAS has no value within the carry-forward horizon -> abstain
            value, my, extrap = fv
            # Widen the band per extrapolated year so far-dated carry-forward fair
            # values stay humble (more uncertain the further past FAS's forecast).
            sigma = max(self.rel_std * value * (1.0 + self.extrap_widen * extrap), 1e-6)
            surv = 1.0 - norm_cdf((threshold - value) / sigma)  # P(value > threshold)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "entity": entity, "fas_value": value, "marketing_year": my,
                "threshold": threshold, "direction": direction, "sigma": sigma,
                "p": p, "extrap": extrap,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        carry = (f" (carried forward {d['extrap']}y past FAS's horizon, band widened)"
                 if d.get("extrap") else "")
        return (
            f"- USDA FAS forecast ({d['entity'].replace('SIG_COTTON_', '').replace('_', ' ').lower()}, "
            f"MY {d['marketing_year']}): **{d['fas_value']:.1f} M bales**{carry}\n"
            f"- Threshold: **{d['threshold']:.1f}** ({d['direction']}), band σ={d['sigma']:.1f} "
            f"→ P({d['direction']})=**{d['p']:.3f}**"
        )
