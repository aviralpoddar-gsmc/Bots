"""Cotton bot (single source = US Drought Monitor): Texas drought drift.

Reads ONE signal — ``SIG_COTTON_DROUGHT`` (Texas DSCI z-scored vs its same-week-of-
year baseline) — and drifts the cotton price anchor. Texas grows ~40% of US cotton;
drought stress there points to a smaller/lower-quality crop → less supply → bullish.
Drought severity is published weekly with no key and LEADS the NASS crop-condition
print (condition is the observed consequence of moisture), so this is an earlier,
unconventional read on the same supply risk.

⚠️ VALIDATION (2026-06-02, 26y CT=F) REFUTED the naive "+drought → +cotton" supply
thesis: deseasonalized TX drought correlates NEGATIVELY with forward cotton returns
(6mo Spearman -0.26, stable in/out, significant) — most likely a confound, not a US
supply edge. Default sign flipped to -1 to match the data, but the bot stays
enabled:false (mechanism unexplained — a documented dead-end). Cotton only.
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy


class DroughtCottonStrategy(SignalDriftStrategy):
    name = "drought_cotton"
    description = (
        "Single-source (US Drought Monitor) bot: drifts ICE Cotton futures threshold "
        "markets on Texas drought severity (DSCI, deseasonalized z). Worse drought in "
        "the dominant cotton state → tighter supply → bullish. A weekly, keyless, "
        "leading read on crop stress (ahead of the NASS condition print). Abstains "
        "without the feed or when drought is near its seasonal normal."
    )
    CATALOG = [(re.compile(r"\bcotton\b", re.I), "CME_COTTON", 0.24)]

    def __init__(self, sig_entity: str = "SIG_COTTON_DROUGHT", k: float = 0.03,
                 min_z: float = 0.7, sign: float = -1.0, **params: Any):
        super().__init__(**params)
        self.sig_entity = sig_entity
        self.k = k          # drift per 1σ of deseasonalized DSCI (before drift_cap)
        self.min_z = min_z  # conviction floor: ignore near-seasonal-normal drought
        self.sign = sign    # +1: more drought → bullish cotton (hypothesis)

    def signal_drift(self, spot: float, price_entity: str, T: float):
        o = self._obs.latest_observation(self.sig_entity)
        if not o or o.get("value") is None:
            return None
        z = o["value"]
        if abs(z) < self.min_z:
            return None
        mu = self.sign * self.k * z
        reason = (
            f"TX drought DSCI z={z:+.2f} vs week-of-year normal "
            f"({'drier' if z > 0 else 'wetter'}) → cotton drift {mu:+.1%}/yr "
            f"(sign={self.sign:+.0f}, leads NASS condition)"
        )
        return mu, {"drought_z": z, "sign": self.sign, "reason": reason}
