"""Shared base for SINGLE-SOURCE, price-anchored commodity bots.

Architecture (see docs/usda-softs-bots.md "Pipeline"):
- The futures **price feed is shared market infrastructure** (the thing being
  traded), NOT a signal source.
- Each bot draws its *alpha* from exactly **one** data source (FAS / NASS / CFTC /
  weather). The bot expresses that source as a single, bounded **drift** applied to
  the live price anchor, then prices threshold markets with a lognormal CDF.
- "Only trade if meaningful": a bot ABSTAINS (returns no estimate) unless its
  signal clears a conviction floor — so trades only happen when the one source
  actually says something. Every funded trade then gets a data-grounded comment
  (see runner + llm/comment.py).

Subclasses provide:
- ``CATALOG``: list of (compiled regex, price_entity, annual_vol) — matches a
  market to a commodity, its shared price anchor, and its vol.
- ``signal_drift(spot, spec, T) -> (mu, detail) | None``: the ONE-source drift.
  Return None to abstain (no meaningful signal). ``detail`` is a dict of the raw
  numbers, surfaced in the trade comment.
- ``signal_entities``: the source entities this bot reads (for documentation/health).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market, Strategy
from .ladder import parse_threshold


def obs_payload(o: dict | None) -> dict:
    """Observation payloads are stored as JSON TEXT — parse to a dict safely."""
    if not o:
        return {}
    p = o.get("payload")
    if isinstance(p, str):
        try:
            return json.loads(p) or {}
        except (ValueError, TypeError):
            return {}
    return p or {}

# Spread/basis markets reference a small differential, not the outright — pricing
# them off the outright is a confidently-wrong trap. Shared across all price bots.
EXCLUDE = re.compile(
    r"basis|spread|differential|premium|discount|\bminus\b|\bless\b|\bover\b|\bvs\.?\b|grind",
    re.I,
)
PRICEY = re.compile(r"futures|price|cents/?lb|usd/?t\b|/lb|per pound|per tonne|/t\b", re.I)


class SignalDriftStrategy(Strategy):
    #: subclass: [(regex, price_entity, annual_vol), ...]
    CATALOG: list[tuple[re.Pattern[str], str, float]] = []

    def __init__(
        self,
        min_vol: float = 0.05,
        max_horizon_years: float = 1.5,
        drift_cap: float = 0.10,
        min_drift: float = 0.005,
        **params: Any,
    ):
        super().__init__(
            min_vol=min_vol, max_horizon_years=max_horizon_years,
            drift_cap=drift_cap, min_drift=min_drift, **params,
        )
        self.min_vol = min_vol
        self.max_horizon_years = max_horizon_years
        self.drift_cap = drift_cap
        self.min_drift = min_drift  # meaningful-trade gate: abstain if |drift| below this
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    # --- commodity matching (shared) ---------------------------------------
    def _spec(self, q: str) -> tuple[str, float] | None:
        """Return (price_entity, annual_vol) if q is an outright price market we cover."""
        if EXCLUDE.search(q) or not PRICEY.search(q):
            return None
        for pat, entity, vol in self.CATALOG:
            if pat.search(q):
                return entity, vol
        return None

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [
            m for m in markets
            if not m.get("isResolved")
            and self._spec(m.get("question", "")) is not None
            and years_to_close(m) <= self.max_horizon_years
        ]

    def correlation_key(self, market: Market) -> str:
        spec = self._spec(market.get("question", ""))
        return spec[0] if spec else str(market.get("id"))

    # --- subclass hook -----------------------------------------------------
    def signal_drift(self, spot: float, price_entity: str, T: float) -> tuple[float, dict] | None:
        """Bounded annualized log-drift from this bot's ONE signal source, plus the
        raw numbers behind it. Return None to ABSTAIN (no meaningful signal)."""
        raise NotImplementedError

    # --- pricing (shared) --------------------------------------------------
    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            q = m.get("question", "")
            spec = self._spec(q)
            if spec is None:
                continue
            price_entity, vol = spec
            parsed = parse_threshold(q)
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(price_entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            spot = o["value"]
            T = years_to_close(m)
            sig = self.signal_drift(spot, price_entity, T)
            if sig is None:
                continue  # meaningful-trade gate: no signal -> no trade
            mu, detail = sig
            mu = max(min(mu, self.drift_cap), -self.drift_cap)
            if abs(mu) < self.min_drift:
                continue  # signal too weak to be meaningful
            fair = spot * math.exp(mu * T)
            sigma = max(vol * math.sqrt(T), self.min_vol)
            surv = 1.0 - norm_cdf(math.log(threshold / fair) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "source": self.name, "spot": spot, "threshold": threshold,
                "direction": direction, "T": T, "sigma": sigma, "mu": mu,
                "fair": fair, "p": p, **detail,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        lines = [
            f"- **{d['source']}** signal (single-source bot)",
            f"- price anchor: **{d['spot']:.2f}**, threshold **{d['threshold']:.2f}** ({d['direction']})",
        ]
        if "reason" in d:
            lines.append(f"- {d['reason']}")
        lines.append(
            f"- drift μ={d['mu']:+.3f}/yr (capped), T={d['T']:.2f}y, σ_eff={d['sigma']:.3f} "
            f"→ fair@close {d['fair']:.2f} → P({d['direction']})=**{d['p']:.3f}**"
        )
        return "\n".join(lines)
