"""Cocoa bot: vol-calibrated lognormal on ICE cocoa (NY) futures price markets.

Cocoa is the odd one out: it is **not in USDA PSD** (USDA tracks cotton & coffee
but not cocoa — confirmed 404 on the PSD bulk download). World cocoa balance-sheet
data lives with the ICCO (quarterly bulletin: grindings, arrivals, stocks), which
is not a clean keyless feed. So this v1 carries **no fundamental drift** — it is a
zero-drift lognormal anchored to the live ICE cocoa futures feed (`CME_COCOA` from
Stooq), with a cocoa-appropriate vol (~40%; realized 2y has been far higher).

That still adds value over `commodity_futures`' generic 30% vol by (a) using a
cocoa-calibrated vol and horizon guard, and (b) restricting strictly to cocoa
*price* markets so it never overlaps other softs. The fundamental ICCO drift is
documented as the next enhancement in docs/usda-softs-bots.md §4 (Bot 4).
"""

from __future__ import annotations

import math
import re
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market, Strategy
from .ladder import parse_threshold

_COCOA = re.compile(r"\bcocoa\b", re.I)
_PRICE = re.compile(r"futures|price|usd/?t\b|per tonne|/t\b", re.I)
# Reject spread/basis/differential markets (price them off the outright = wrong).
_EXCLUDE = re.compile(
    r"basis|spread|differential|premium|discount|\bminus\b|\bless\b|\bover\b|\bvs\.?\b|grind", re.I
)


class CocoaFundamentalStrategy(Strategy):
    name = "cocoa_fundamental"
    description = (
        "ICE cocoa (NY) futures price-threshold markets priced with a zero-drift "
        "lognormal anchored to the live futures feed and a cocoa-calibrated vol "
        "(~40%). No USDA drift — cocoa is absent from USDA PSD; ICCO grindings/"
        "stocks drift is the planned next step. Restricted strictly to cocoa price "
        "markets that resolve."
    )

    def __init__(
        self,
        annual_vol: float = 0.40,
        min_vol: float = 0.07,
        max_horizon_years: float = 1.5,
        spot_entity: str = "CME_COCOA",
        **params: Any,
    ):
        super().__init__(
            annual_vol=annual_vol, min_vol=min_vol,
            max_horizon_years=max_horizon_years, **params,
        )
        self.annual_vol = annual_vol
        self.min_vol = min_vol
        self.max_horizon_years = max_horizon_years
        self.spot_entity = spot_entity
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _is_cocoa_price(self, q: str) -> bool:
        return bool(_COCOA.search(q) and _PRICE.search(q) and not _EXCLUDE.search(q))

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [
            m for m in markets
            if not m.get("isResolved")
            and self._is_cocoa_price(m.get("question", ""))
            and years_to_close(m) <= self.max_horizon_years
        ]

    def correlation_key(self, market: Market) -> str:
        return "COCOA" if self._is_cocoa_price(market.get("question", "")) else str(market.get("id"))

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            q = m.get("question", "")
            if not self._is_cocoa_price(q):
                continue
            parsed = parse_threshold(q)
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(self.spot_entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            spot = o["value"]
            T = years_to_close(m)
            sigma = max(self.annual_vol * math.sqrt(T), self.min_vol)
            surv = 1.0 - norm_cdf(math.log(threshold / spot) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "spot": spot, "threshold": threshold, "direction": direction,
                "T": T, "sigma": sigma, "p": p,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        return (
            f"- ICE cocoa (NY) spot anchor: **{d['spot']:.0f} USD/t**\n"
            f"- Threshold: **{d['threshold']:.0f}** ({d['direction']})\n"
            f"- Zero-drift lognormal: T={d['T']:.2f}y, σ_eff={d['sigma']:.3f} "
            f"→ P({d['direction']})=**{d['p']:.3f}**"
        )
