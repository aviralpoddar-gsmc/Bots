"""FAS bot (single source = USDA FAS PSD): fundamental fair-value drift.

Reads ONE signal — ``SIG_COTTON_FAS`` (fundamental fair value from world-ex-China
stocks-to-use, computed in processing/signals.py) — and drifts the live futures
price anchor toward it. This is the clean, single-source successor to the earlier
blended cotton_fundamental (which mixed FAS + NASS + price): the NASS crop signal
is now its own bot (nass_crop).
"""

from __future__ import annotations

import math
import re
from typing import Any

from ._signal_base import SignalDriftStrategy, obs_payload


class FasFundamentalStrategy(SignalDriftStrategy):
    name = "fas_fundamental"
    description = (
        "Single-source (USDA FAS PSD) bot: prices cotton futures threshold markets "
        "by drifting the live futures anchor toward a fundamental fair value from "
        "world-ex-China stocks-to-use (elasticity -0.39). Abstains unless the "
        "fundamental gap is meaningful."
    )
    CATALOG = [(re.compile(r"\bcotton\b", re.I), "CME_COTTON", 0.24)]

    def __init__(self, reversion_rate: float = 0.5, fv_entity: str = "SIG_COTTON_FAS",
                 **params: Any):
        super().__init__(**params)
        self.reversion_rate = reversion_rate
        self.fv_entity = fv_entity

    def signal_drift(self, spot: float, price_entity: str, T: float):
        o = self._obs.latest_observation(self.fv_entity)
        if not o or o.get("value") is None or o["value"] <= 0:
            return None
        fair = o["value"]
        mu = self.reversion_rate * math.log(fair / spot)
        pay = obs_payload(o)
        sur = pay.get("sur")
        reason = (
            f"USDA ex-China stocks-to-use {sur:.3f} (z {pay.get('sur_z', 0):+.1f}) "
            f"→ fundamental fair value {fair:.1f} vs spot {spot:.1f}"
            if sur is not None else f"FAS fair value {fair:.1f} vs spot {spot:.1f}"
        )
        return mu, {"fair_fund": fair, "sur": sur, "sur_z": pay.get("sur_z"), "reason": reason}
