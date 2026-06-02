"""Cocoa bot (single source = CPC Atlantic SST): Atlantic-Niño drift.

Reads ONE signal — ``SIG_ATL3_COCOA`` (z-scored tropical-Atlantic SST anomaly, a
proxy for the equatorial-Atlantic "Atlantic Niño") — and drifts the cocoa price
anchor. The equatorial Atlantic modulates Gulf-of-Guinea rainfall on a seasonal
horizon (3–12 months) that matches the cocoa price markets, and is largely
independent of the Pacific ENSO feed the fleet already carries.

⚠️ SIGN IS UNVALIDATED. Warm tropical Atlantic → more West-Africa rainfall, which
can be *bearish* (better crop → more supply) OR *bullish* (excess rain → black-pod
disease → less supply). Default `sign=-1` (warm → bearish) is a hypothesis, not a
result. Walk-forward validate against cocoa futures before flipping enabled:true.
Cocoa only (the Atlantic Niño is a West-Africa driver).
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy


class CocoaAtlanticStrategy(SignalDriftStrategy):
    name = "cocoa_atlantic"
    description = (
        "Single-source (CPC Atlantic SST) bot: drifts cocoa futures threshold "
        "markets on the tropical-Atlantic 'Atlantic Niño' SST anomaly, a seasonal "
        "Gulf-of-Guinea rainfall driver for West-African cocoa (~70% of supply). "
        "Abstains without the feed or when the anomaly is near normal. Sign "
        "unvalidated — ships disabled pending a walk-forward check."
    )
    CATALOG = [(re.compile(r"\bcocoa\b", re.I), "CME_COCOA", 0.40)]

    def __init__(self, sig_entity: str = "SIG_ATL3_COCOA", k: float = 0.02,
                 min_z: float = 0.5, sign: float = -1.0, **params: Any):
        super().__init__(**params)
        self.sig_entity = sig_entity
        self.k = k          # drift per 1σ of SST anomaly (before drift_cap)
        self.min_z = min_z  # conviction floor: ignore near-normal Atlantic
        self.sign = sign    # cocoa response to a WARM anomaly (+warm→bearish if -1)

    def signal_drift(self, spot: float, price_entity: str, T: float):
        o = self._obs.latest_observation(self.sig_entity)
        if not o or o.get("value") is None:
            return None
        z = o["value"]
        if abs(z) < self.min_z:
            return None
        mu = self.sign * self.k * z
        warm = z > 0
        reason = (
            f"Tropical-Atlantic SST anomaly z={z:+.2f} ({'warm' if warm else 'cool'}) "
            f"→ cocoa drift {mu:+.1%}/yr (ATL3 proxy; sign={self.sign:+.0f}, UNVALIDATED)"
        )
        return mu, {"atl3_z": z, "sign": self.sign, "reason": reason}
