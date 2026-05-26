"""Ensemble strategy — fuse ingested observations into a fair-value probability.

Deterministic and inspectable (no LLM). For each market:

  1. the linker maps it to source entities + a threshold/direction,
  2. each linked numeric observation (a price / macro print) is turned into a
     probability that the quantity clears the threshold, via a lognormal model,
  3. those per-signal probabilities are combined as a weighted average.

The lognormal model: with a current value `V`, threshold `T`, and an assumed
relative volatility `sigma` over the horizon, the survival probability is
`P(future > T) = 1 - Phi( ln(T/V) / sigma )`. We lack per-entity history so
`sigma` is a single tunable parameter (per-source weights tune trust). This is a
starting point — swap in history-based vol or a local-LLM extractor for text
signals later; the Strategy contract is unchanged.

Single-source vs multi-source is just configuration: restrict `entity_map` to one
entity for a specialist bot, or leave the full map for a generalist that fuses
several feeds for the same market.
"""

from __future__ import annotations

import math
import time
from typing import Any

from .base import Market, Strategy
from .linker import link_market


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class EnsembleStrategy(Strategy):
    name = "ensemble"

    def __init__(
        self,
        annual_vol: float = 0.5,
        source_weights: dict[str, float] | None = None,
        entity_map: dict[str, list[str]] | None = None,
        max_ratio: float = 20.0,
        min_vol: float = 0.05,
        **params: Any,
    ):
        super().__init__(
            annual_vol=annual_vol, source_weights=source_weights, entity_map=entity_map,
            max_ratio=max_ratio, min_vol=min_vol, **params,
        )
        # Volatility is horizon-scaled: effective sigma = annual_vol * sqrt(years
        # to close), floored at min_vol so near-dated markets aren't priced at 0/1.
        self.annual_vol = annual_vol
        self.min_vol = min_vol
        self.source_weights = source_weights or {}
        self.entity_map = entity_map
        # Plausibility guard: if a market's threshold and the observed value differ
        # by more than this ratio, the link is almost certainly wrong (e.g. a
        # natural-gas *volume* market matched to the gas *price* feed) — skip it
        # rather than trade a confident bogus estimate.
        self.max_ratio = max_ratio
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _effective_sigma(self, market: Market) -> float:
        """annual_vol scaled by sqrt(years to close), floored at min_vol."""
        close = market.get("closeTime")
        tau = 1.0
        if close:
            tau = (close / 1000.0 - time.time()) / (365.25 * 24 * 3600)
        return max(self.annual_vol * math.sqrt(max(tau, 0.0)), self.min_vol)

    def _signal_prob(
        self, value: float, threshold: float, direction: str, sigma: float
    ) -> float | None:
        """P(quantity clears threshold) for one numeric signal, via lognormal."""
        if value <= 0:
            return None
        if threshold <= 0:
            # A positive quantity essentially always exceeds a non-positive bound.
            surv = 0.99
        else:
            z = math.log(threshold / value) / max(sigma, 1e-6)
            surv = 1.0 - _norm_cdf(z)  # P(future value > threshold)
        return surv if direction == "exceeds" else 1.0 - surv

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}  # not bound to a store -> nothing to fuse
        out: dict[str, float] = {}
        for m in group:
            link = link_market(m, self.entity_map)
            if link is None or link.threshold is None:
                continue
            sigma = self._effective_sigma(m)
            probs: list[float] = []
            weights: list[float] = []
            for entity in link.entities:
                o = self._obs.latest_observation(entity)
                if not o or o.get("value") is None:
                    continue
                value = o["value"]
                # Scale sanity check: drop wildly mismatched threshold/value pairs.
                if link.threshold > 0 and value > 0:
                    ratio = max(value / link.threshold, link.threshold / value)
                    if ratio > self.max_ratio:
                        continue
                p = self._signal_prob(value, link.threshold, link.direction, sigma)
                if p is None:
                    continue
                probs.append(p)
                weights.append(self.source_weights.get(o["source"], 1.0))
            if not probs:
                continue
            fair = sum(p * w for p, w in zip(probs, weights)) / sum(weights)
            out[m["id"]] = min(max(fair, 0.01), 0.99)
        return out
