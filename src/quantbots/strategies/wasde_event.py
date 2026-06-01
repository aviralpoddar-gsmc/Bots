"""WASDE event overlay (single source = USDA FAS PSD / WASDE): ending-stocks surprise.

The monthly WASDE (World Agricultural Supply and Demand Estimates, ~9th-12th) is
cotton's biggest scheduled catalyst. The tradeable signal is the *revision*: when
USDA marks the world cotton ending-stocks forecast DOWN vs the prior report, the
balance is tightening → bullish; UP → bearish. The clone's quotes often lag the
print, so a short post-release drift on the price anchor captures the move.

Reads the revision-tracked ``SIG_COTTON_WASDE`` series (processing/signals.py,
stamped per report month) and drifts cotton price by the month-over-month surprise.
Abstains until at least two reports exist (the surprise needs a prior to diff
against) — so it activates on the first WASDE after deployment.
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy


class WasdeEventStrategy(SignalDriftStrategy):
    name = "wasde_event"
    description = (
        "Single-source (USDA WASDE) overlay on cotton futures threshold markets: "
        "drifts the price anchor by the month-over-month world ending-stocks "
        "revision surprise (stocks down → bullish). Abstains until a new WASDE "
        "print provides a surprise vs the prior report."
    )
    CATALOG = [(re.compile(r"\bcotton\b", re.I), "CME_COTTON", 0.24)]

    def __init__(self, k: float = 1.0, min_surprise: float = 0.01,
                 wasde_entity: str = "SIG_COTTON_WASDE", **params: Any):
        super().__init__(**params)
        self.k = k                      # drift per 1.0 (100%) ending-stocks surprise
        self.min_surprise = min_surprise  # meaningful-trade gate: ignore tiny revisions
        self.wasde_entity = wasde_entity

    def signal_drift(self, spot: float, price_entity: str, T: float):
        if self._obs is None or not hasattr(self._obs, "load_observations"):
            return None
        rows = self._obs.load_observations(self.wasde_entity, limit=2)  # newest first
        if len(rows) < 2:
            return None  # need a prior report to compute a surprise -> abstain
        new, old = rows[0].get("value"), rows[1].get("value")
        if not new or not old:
            return None
        surprise = (new - old) / old  # relative change in world ending stocks
        if abs(surprise) < self.min_surprise:
            return None
        mu = -self.k * surprise  # stocks revised down -> tighter -> bullish
        reason = (
            f"WASDE world ending stocks {old:.1f}→{new:.1f} M bales ({surprise:+.1%}) "
            f"→ {'tightening (bullish)' if surprise < 0 else 'building (bearish)'}"
        )
        return mu, {"endstocks_new": new, "endstocks_prior": old, "surprise": surprise,
                    "reason": reason}
