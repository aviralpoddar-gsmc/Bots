"""mercury_ensemble: a Bayesian-mixture calibration strategy.

⚠️ Uses HOSTED inference (Mercury / Inception Labs) — a sanctioned exception to
the local-only rule. See `docs/mercury-ensemble-calibration.md`.

Where the `llm` strategy asks the model once for a percentile distribution and
reads strikes off the fitted CDF, this asks Mercury N times with a temperature
spread, treats the samples as a posterior predictive, and per strike:

  1. mixes the per-sample probabilities into the predictive mean p̄,
  2. decomposes the spread into aleatoric vs epistemic (sample disagreement),
  3. abstains if the samples split on direction (no consensus to trade on),
  4. shrinks p̄ toward the market price in proportion to disagreement.

The calibrated number is returned through the normal `estimate()` seam, so
sizing, allocation, and execution are untouched.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math

from ._mixture import direction_agreement, fit_normal, mixture, prob_from_normal, shrink
from ..llm.mercury import DEFAULT_MERCURY_MODEL, MercuryLLM
from .base import Market
from .ladder import measurable_key
from .llm import LLMStrategy

logger = logging.getLogger(__name__)


class MercuryEnsembleStrategy(LLMStrategy):
    name = "mercury_ensemble"
    description = (
        "Bayesian-mixture calibration bot. Samples Mercury (Inception Labs) N "
        "times per measurable, pools the forecasts into a posterior-predictive "
        "probability, and decomposes uncertainty into aleatoric vs epistemic. "
        "Edge is calibration: it shrinks toward the market price when samples "
        "disagree and abstains when they split on direction. Hosted inference "
        "is a sanctioned experimental exception to the local-only rule."
    )

    def __init__(
        self,
        model: str | None = None,
        n_samples: int = 20,
        min_quorum: int | None = None,
        temperature_lo: float = 0.4,
        temperature_hi: float = 1.0,
        sample_concurrency: int = 10,
        epistemic_tau: float = 0.04,
        direction_agreement_floor: float = 0.70,
        spread_mult: float = 1.1,
        conf_cap: float = 0.80,
        max_groups: int = 12,
        api_key: str | None = None,
        **params: object,
    ):
        model = model or DEFAULT_MERCURY_MODEL
        super().__init__(
            model=model, spread_mult=spread_mult, conf_cap=conf_cap, max_groups=max_groups,
            n_samples=n_samples, min_quorum=min_quorum, temperature_lo=temperature_lo,
            temperature_hi=temperature_hi, sample_concurrency=sample_concurrency,
            epistemic_tau=epistemic_tau, direction_agreement_floor=direction_agreement_floor,
            **params,
        )
        self.llm = MercuryLLM(model=model, api_key=api_key)
        self.n_samples = n_samples
        self.min_quorum = min_quorum if min_quorum is not None else math.ceil(0.6 * n_samples)
        self.temperature_lo = temperature_lo
        self.temperature_hi = temperature_hi
        self.sample_concurrency = sample_concurrency
        self.epistemic_tau = epistemic_tau
        self.direction_agreement_floor = direction_agreement_floor

    def _temperatures(self) -> list[float]:
        if self.n_samples == 1:
            return [self.temperature_lo]
        step = (self.temperature_hi - self.temperature_lo) / (self.n_samples - 1)
        return [self.temperature_lo + i * step for i in range(self.n_samples)]

    def estimate(self, group: list[Market]) -> dict[str, float]:
        fits = self._sample_fits(group)
        if len(fits) < self.min_quorum:
            logger.warning(
                "mercury_ensemble: %d/%d samples (< quorum %d) on %r — abstaining",
                len(fits), self.n_samples, self.min_quorum, measurable_key(group[0])[:60],
            )
            return {}

        lo, hi = 1.0 - self.conf_cap, self.conf_cap
        out: dict[str, float] = {}
        for m in group:
            if m.get("threshold") is None:
                continue
            current = m["probability"]
            direction = m.get("direction", "exceeds")
            probs = [prob_from_normal(m["threshold"], direction, mu, sigma) for mu, sigma in fits]
            stats = mixture(probs)
            agree = direction_agreement(probs, current, stats.mean)
            if agree < self.direction_agreement_floor:
                continue  # split jury — no consensus to trade on
            sr = shrink(stats.mean, current, stats.epistemic, self.epistemic_tau)
            out[m["id"]] = min(max(sr.estimate, lo), hi)
            self._explanations[m["id"]] = {
                "mean": stats.mean, "epistemic": stats.epistemic, "aleatoric": stats.aleatoric,
                "agreement": agree, "confidence": sr.confidence, "n_eff": len(fits),
                "current_prob": current,
            }
        return out

    def _sample_fits(self, group: list[Market]) -> list[tuple[float, float]]:
        """Fan out N Mercury calls (one temperature each), return the fitted
        (mu, sigma) of every sample that returned valid percentiles."""
        fits: list[tuple[float, float]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.sample_concurrency) as ex:
            futures = [ex.submit(self._ask_percentiles, group, t) for t in self._temperatures()]
            for f in concurrent.futures.as_completed(futures):
                pct = f.result()
                if pct is not None:
                    fits.append(fit_normal(pct, self.spread_mult))
        return fits

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        return (
            f"**Mercury ensemble** (n={d['n_eff']}): posterior-predictive "
            f"**{d['mean']:.3f}** · epistemic (disagreement) variance {d['epistemic']:.4f} · "
            f"aleatoric {d['aleatoric']:.4f} · direction agreement {d['agreement']:.0%} · "
            f"confidence {d['confidence']:.2f} → estimate shrunk from {d['mean']:.3f} toward "
            f"market {d['current_prob']:.3f}."
        )
