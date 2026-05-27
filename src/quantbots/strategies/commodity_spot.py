"""Data-anchored bot: trades *spot-price* threshold markets for the handful of
commodities we have a live price feed for (gold, silver, copper, WTI, Brent).

The clone is dominated by metals/mining markets. Most are operational metrics
(production, demand, AISC, trade flow) with no free feed — those are out of scope
here. This bot trades ONLY genuine spot/futures *price* questions for commodities
Stooq quotes, prices the whole threshold ladder off the latest spot with a
horizon-scaled lognormal model, and lets the runner's sizing turn the gap from
0.50 into orders.

Two correctness traps this bot exists to avoid (see CLAUDE.md "confidently wrong"):

  1. **Wrong metric.** "gold dental-alloy demand", "Equinox Gold AISC", "copper
     production" all contain a commodity word but are NOT its spot price. Pricing
     them against spot is catastrophic. We hard-exclude operational keywords and
     require explicit price phrasing.

  2. **Wrong units / benchmark.** Stooq quotes silver in cents/oz and copper in
     cents/lb, while the markets quote $/oz and $/MT. And "natural gas" markets
     here are EUR/MWh European gas, not Henry Hub $/MMBtu — a different benchmark
     entirely, so natgas is deliberately excluded. Each commodity carries a
     feed->market unit factor; the lognormal only depends on threshold/spot, so
     as long as both are in the same unit the ratio is correct.

Annual vols are realized estimates (FRED daily history): oils ~0.40, copper ~0.21,
gold ~0.16, silver ~0.30. Conservative/wide is safe — it avoids overconfidence on
far-dated strikes. Override per-commodity via `vols` in config.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market, Strategy
from .ladder import parse_threshold

# Operational/structural metrics that merely mention a commodity — NOT its price.
# A question matching any of these is dropped before we ever look at the spot.
_EXCLUDE = re.compile(
    r"\b(demand|production|output|produced|aisc|all-in sustaining|cash cost|"
    r"\bcost\b|dental|alloy|imports?|exports?|reserves?|proved|capacity|"
    r"utili[sz]ation|sales|revenue|inventor(?:y|ies)|stocks?|warrant|shipments?|"
    r"consumption|recycl|scrap|\bmine\b|mining|refinery|smelter|grade|spread|"
    r"premium|discount|ratio|balance|tariff)\b",
    re.I,
)
# The question must actually be about a price/level.
_PRICEY = re.compile(r"\b(spot price|price|futures|settlement|fixing)\b", re.I)

# Unit verification — a commodity matches only if the threshold's quoted unit
# confirms the right benchmark AND currency. This is what keeps us out of traps:
# palladium "koz" (a volume), nickel "CNY per tonne" (yuan, ~7x), "zinc sulfate"
# (a chemical, not LME zinc metal), natural gas "EUR/MWh" (European, not Henry Hub).
# Each rule is (required-unit pattern, forbidden pattern). Both are searched over
# the whole question; required must be present, forbidden must be absent.
_FOREIGN_CCY = r"cny|eur|yuan|rmb|gbp|jpy"
_UNIT_RULES: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    # $/troy ounce (gold, silver, platinum, palladium).
    "OZ": (
        re.compile(r"(per troy|/oz\b|/ozt\b|usd/oz|usd per (?:troy )?(?:oz|ounce)|dollars? per (?:troy )?ounce)", re.I),
        re.compile(rf"\b(koz|moz|kt\b|kilotonne|tonne|metric ton|/mt|{_FOREIGN_CCY})\b", re.I),
    ),
    # USD per metric ton (copper). Reject foreign currency and chemical compounds.
    "MT": (
        re.compile(r"(/mt\b|/t\b|per (?:metric )?ton|per tonne|usd/mt|usd per (?:metric )?ton|usd per tonne)", re.I),
        re.compile(rf"\b(sulfate|sulphate|oxide|carbonate|hydroxide|chloride|koz|{_FOREIGN_CCY})\b", re.I),
    ),
    # USD per barrel (WTI, Brent).
    "BBL": (re.compile(r"(/barrel|/bbl|per barrel)", re.I), re.compile(rf"\b({_FOREIGN_CCY})\b", re.I)),
    # USD per gallon (RBOB gasoline).
    "GAL": (re.compile(r"(/gal|per gallon)", re.I), re.compile(rf"\b({_FOREIGN_CCY})\b", re.I)),
}

# Per-commodity spec: (entity, commodity regex, unit category, feed->market factor, annual vol).
#   factor converts the Stooq feed value into the unit the *market* quotes in:
#     GOLD/PLATINUM/PALLADIUM  $/oz     -> $/oz   x1
#     SILVER       si.f  cents/oz -> $/oz   x0.01
#     COPPER       hg.f  cents/lb -> $/MT   x22.0462   (100 cents/$ ... 2204.62 lb/MT)
#     WTI/BRENT    $/bbl  -> $/bbl  x1
#     GASOLINE     rb.f  $/gal   -> $/gal  x1
_LB_PER_MT = 2204.62
_SPECS: list[tuple[str, str, str, float, float]] = [
    ("GOLD", r"\bgold\b", "OZ", 1.0, 0.16),
    ("SILVER", r"\bsilver\b", "OZ", 0.01, 0.30),
    ("PLATINUM", r"\bplatinum\b", "OZ", 1.0, 0.22),
    ("PALLADIUM", r"\bpalladium\b", "OZ", 1.0, 0.30),
    ("COPPER", r"\bcopper\b", "MT", _LB_PER_MT / 100.0, 0.21),
    ("WTI_OIL", r"\b(wti|west texas)\b", "BBL", 1.0, 0.40),
    ("BRENT_OIL", r"\bbrent\b", "BBL", 1.0, 0.39),
    ("GASOLINE", r"\b(rbob|gasoline)\b", "GAL", 1.0, 0.45),
]
_CATALOG = [
    (ent, re.compile(pat, re.I), unit, factor, vol) for ent, pat, unit, factor, vol in _SPECS
]


class CommoditySpotStrategy(Strategy):
    name = "commodity_spot"

    def __init__(self, vols: dict[str, float] | None = None, min_vol: float = 0.05,
                 max_horizon_years: float = 1.25, **params: Any):
        super().__init__(vols=vols, min_vol=min_vol, max_horizon_years=max_horizon_years, **params)
        self.vol_override = vols or {}
        self.min_vol = min_vol
        # Zero-drift lognormal is validated (FRED backtest) out to ~1y; beyond that
        # secular drift dominates and a NO bet on a far-dated "exceeds" strike can be
        # a trap. Trade only the horizons we've proven calibration on.
        self.max_horizon_years = max_horizon_years
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _spec(self, question: str) -> tuple[str, float, float] | None:
        """Return (entity, feed->market factor, annual_vol) if this is a genuine
        spot-price question for a feedable commodity in the right unit/currency,
        else None."""
        if _EXCLUDE.search(question) or not _PRICEY.search(question):
            return None
        for ent, pat, unit, factor, vol in _CATALOG:
            if not pat.search(question):
                continue
            required, forbidden = _UNIT_RULES[unit]
            # Unit must confirm the benchmark/currency and carry no disqualifier.
            if not required.search(question) or forbidden.search(question):
                return None
            return ent, factor, self.vol_override.get(ent, vol)
        return None

    def prefilter(self, markets: list[Market]) -> list[Market]:
        # super() drops resolved / closed / illiquid; we keep only feedable spot-price
        # markets within the horizon our zero-drift model is calibrated for.
        return [
            m for m in super().prefilter(markets)
            if self._spec(m.get("question", "")) is not None
            and years_to_close(m) <= self.max_horizon_years
        ]

    def correlation_key(self, market: Market) -> str:
        # All strikes/dates of one commodity are a single directional bet on its
        # price — group them so the allocator caps total exposure per commodity.
        spec = self._spec(market.get("question", ""))
        return spec[0] if spec else str(market.get("id"))

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            spec = self._spec(m.get("question", ""))
            if spec is None:
                continue
            entity, factor, annual_vol = spec
            parsed = parse_threshold(m.get("question", ""))
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            spot = o["value"] * factor  # feed value in the market's quoted unit
            sigma = max(annual_vol * math.sqrt(years_to_close(m)), self.min_vol)
            # P(spot_at_close > threshold) under lognormal diffusion (zero drift).
            surv = 1.0 - norm_cdf(math.log(threshold / spot) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            out[m["id"]] = min(max(p, 0.01), 0.99)
        return out
