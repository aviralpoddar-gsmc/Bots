"""NASS bot (single source = USDA NASS QuickStats): crop-condition drift.

Reads ONE signal — ``NASS_COTTON_COND_GE`` (US cotton % good+excellent, the
in-season yield proxy) — and drifts the cotton price anchor: a crop in better
condition than its reference points to a larger harvest → more supply → bearish
(and vice versa). Abstains when the NASS feed is absent (no key) or condition is
near the reference. Cotton only (NASS is US-domestic).
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy


class NassCropStrategy(SignalDriftStrategy):
    name = "nass_crop"
    description = (
        "Single-source (USDA NASS) bot: drifts cotton futures threshold markets on "
        "US crop condition (good+excellent %). Better-than-reference condition → "
        "larger harvest → bearish; worse → bullish. Abstains without the NASS feed "
        "or when condition is near reference."
    )
    CATALOG = [(re.compile(r"\bcotton\b", re.I), "CME_COTTON", 0.24)]

    def __init__(self, cond_ref: float = 50.0, k: float = 0.003, min_dev: float = 5.0,
                 cond_entity: str = "SIG_COTTON_COND_IDX",
                 fallback_entity: str = "NASS_COTTON_COND_GE", **params: Any):
        super().__init__(**params)
        self.cond_ref = cond_ref   # reference good+excellent %
        self.k = k                 # drift per percentage-point deviation
        self.min_dev = min_dev     # meaningful-trade gate: ignore small deviations
        # Prefer the production-weighted per-state index; fall back to the national
        # print if processing hasn't produced the index (e.g. no per-state data yet).
        self.cond_entity = cond_entity
        self.fallback_entity = fallback_entity

    def signal_drift(self, spot: float, price_entity: str, T: float):
        o = self._obs.latest_observation(self.cond_entity)
        if (not o or o.get("value") is None) and self.fallback_entity:
            o = self._obs.latest_observation(self.fallback_entity)
        if not o or o.get("value") is None:
            return None
        cond = o["value"]
        dev = cond - self.cond_ref
        if abs(dev) < self.min_dev:
            return None
        mu = -self.k * dev  # better condition -> more supply -> bearish
        reason = (
            f"US cotton condition {cond:.0f}% good+excellent vs {self.cond_ref:.0f}% ref "
            f"→ {'better crop (bearish)' if dev > 0 else 'worse crop (bullish)'}"
        )
        return mu, {"condition": cond, "cond_dev": dev, "reason": reason}
