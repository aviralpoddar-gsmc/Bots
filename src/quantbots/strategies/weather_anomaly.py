"""Weather bot (single source = open-meteo): growing-region anomaly drift.

Reads ONE signal — ``SIG_COCOA_WX`` (Ivory Coast cocoa-belt drought signal =
negative precip z-score) — and drifts the cocoa price anchor: drought in the
world's dominant cocoa region threatens supply → bullish. Abstains unless the
anomaly is meaningful. Cocoa only on the clone (coffee has no price markets).
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy, obs_payload


class WeatherAnomalyStrategy(SignalDriftStrategy):
    name = "weather_anomaly"
    description = (
        "Single-source (open-meteo) bot: drifts cocoa futures threshold markets on "
        "Ivory Coast cocoa-belt precipitation anomalies (drought → supply risk → "
        "bullish). Abstains when weather is near normal."
    )
    CATALOG = [(re.compile(r"\bcocoa\b", re.I), "CME_COCOA", 0.40)]

    def __init__(self, k: float = 0.03, min_abs: float = 1.0,
                 wx_entity: str = "SIG_COCOA_WX", **params: Any):
        super().__init__(**params)
        self.k = k            # drift per 1 unit of (drought) signal
        self.min_abs = min_abs
        self.wx_entity = wx_entity

    def signal_drift(self, spot: float, price_entity: str, T: float):
        o = self._obs.latest_observation(self.wx_entity)
        if not o or o.get("value") is None:
            return None
        sig = o["value"]  # = -(precip z): positive = drought = bullish
        if abs(sig) < self.min_abs:
            return None
        mu = self.k * sig
        pay = obs_payload(o)
        reason = (
            f"Ivory Coast 30d precip {pay.get('prcp30', 0):.0f}mm, precip z "
            f"{pay.get('precip_z', 0):+.1f} → {'drought (bullish)' if sig > 0 else 'wet (bearish)'}"
        )
        return mu, {"wx_signal": sig, "prcp30": pay.get("prcp30"), "reason": reason}
