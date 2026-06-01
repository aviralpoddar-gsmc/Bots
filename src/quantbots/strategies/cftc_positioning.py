"""CFTC bot (single source = CFTC Commitments of Traders): positioning reversion.

Reads ONE signal — ``SIG_<COM>_CFTC`` (managed-money net-% z-score) — and bets
mean reversion: when speculators are extremely net-long (high z), the marginal
buyer is exhausted and price tends to drift back, so it applies a bearish drift
(and vice versa). Abstains unless positioning is at a meaningful extreme.
Covers cotton and cocoa (the price markets that resolve on the clone).
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy, obs_payload


class CftcPositioningStrategy(SignalDriftStrategy):
    name = "cftc_positioning"
    description = (
        "Single-source (CFTC COT) bot: fades managed-money positioning extremes on "
        "cotton & cocoa futures threshold markets. Extreme net-long z-score → bearish "
        "drift on the live price anchor; extreme net-short → bullish. Abstains when "
        "positioning is near its historical mean (no meaningful signal)."
    )
    CATALOG = [
        (re.compile(r"\bcotton\b", re.I), "CME_COTTON", 0.24),
        (re.compile(r"\bcocoa\b", re.I), "CME_COCOA", 0.40),
    ]

    def __init__(self, k: float = 0.04, min_z: float = 1.0, **params: Any):
        super().__init__(**params)
        self.k = k          # drift per 1 z of positioning extreme
        self.min_z = min_z  # meaningful-trade gate: ignore positioning near the mean

    def signal_drift(self, spot: float, price_entity: str, T: float):
        com = price_entity.replace("CME_", "")
        o = self._obs.latest_observation(f"SIG_{com}_CFTC")
        if not o or o.get("value") is None:
            return None
        z = o["value"]
        if abs(z) < self.min_z:
            return None  # positioning not extreme -> no meaningful trade
        mu = -self.k * z  # fade the crowd
        pay = obs_payload(o)
        reason = (
            f"managed money net {pay.get('netpct', 0)*100:+.1f}% of OI, "
            f"z {z:+.1f} vs 3y → fade ({'long crowded' if z > 0 else 'short crowded'})"
        )
        return mu, {"cot_z": z, "netpct": pay.get("netpct"), "reason": reason}
