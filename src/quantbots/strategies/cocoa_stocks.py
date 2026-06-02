"""Cocoa bot (single source = ICE certified stocks): deliverable-tightness drift.

Reads ONE signal — ``SIG_COCOA_STOCK_Z`` (z-score of ICE certified cocoa stock vs
its trailing history) — and drifts the cocoa price anchor. Certified stock is the
cocoa graded and deliverable against ICE futures; when it runs LOW relative to
history, physical deliverable supply is tight → bullish; high stock → easing.

It is daily and widely watched, so this is a short-horizon CONFIRMER (a small
capped drift), not slow fundamental alpha — and it is orthogonal to the weather /
Atlantic-SST cocoa signals. Sign (low stock → bullish, i.e. sign=-1 on the level z)
is the standard inventory relationship.

⚠️ VALIDATION (2026-06-02) was INCONCLUSIVE — only ~15 aligned months were usable
(older ICE .xls files use a different layout the source parser skips, so the
backfill is too short/recent and trend-dominated by the 2023-24 stock collapse).
Ships enabled:false; extend the parser to older layouts for a deeper backfill, then
walk-forward vs cocoa futures before enabling. Cocoa only.
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy


class CocoaStocksStrategy(SignalDriftStrategy):
    name = "cocoa_stocks"
    description = (
        "Single-source (ICE certified cocoa stocks) bot: drifts ICE cocoa futures "
        "threshold markets on deliverable-inventory tightness (certified-stock z). "
        "Low certified stock vs history → tight deliverable supply → bullish; high → "
        "easing. A daily, keyless, orthogonal confirmer. Abstains without the feed or "
        "when stock is near its historical norm."
    )
    CATALOG = [(re.compile(r"\bcocoa\b", re.I), "CME_COCOA", 0.40)]

    def __init__(self, sig_entity: str = "SIG_COCOA_STOCK_Z", k: float = 0.03,
                 min_z: float = 0.7, sign: float = -1.0, **params: Any):
        super().__init__(**params)
        self.sig_entity = sig_entity
        self.k = k          # drift per 1σ of certified-stock level (before drift_cap)
        self.min_z = min_z  # conviction floor: ignore near-normal stock
        self.sign = sign    # -1: LOW stock (z<0) → bullish (positive drift)

    def signal_drift(self, spot: float, price_entity: str, T: float):
        o = self._obs.latest_observation(self.sig_entity)
        if not o or o.get("value") is None:
            return None
        z = o["value"]
        if abs(z) < self.min_z:
            return None
        mu = self.sign * self.k * z
        reason = (
            f"ICE certified cocoa stock z={z:+.2f} vs history "
            f"({'tight/low' if z < 0 else 'ample/high'}) → cocoa drift {mu:+.1%}/yr "
            f"(sign={self.sign:+.0f})"
        )
        return mu, {"stock_z": z, "sign": self.sign, "reason": reason}
