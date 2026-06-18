"""Pure mixture math for the mercury_ensemble strategy.

Each Mercury sample yields a normal forecast (mu, sigma) of the underlying
quantity; `prob_from_normal` reads one strike's probability off it. `mixture`
pools N such per-strike probabilities into a posterior-predictive mean and
decomposes the spread into aleatoric (irreducible) and epistemic (sample
disagreement) parts via the law of total variance for a Bernoulli mixture:

    Var = E[p(1-p)]  +  Var(p)
          ^aleatoric    ^epistemic   ,   and  aleatoric + epistemic = p̄(1-p̄).

`shrink` turns epistemic disagreement into a calibrated estimate by pulling the
mean toward the current market price; `direction_agreement` gates trades on a
sample supermajority. All functions are pure (no I/O, deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._model import norm_cdf

PERCENTILE_KEYS = ["p10", "p25", "p50", "p75", "p90"]
# z-scores of the 10/90 and 25/75 percentile pairs of a standard normal.
_Z_10_90 = 2.5631  # p90 - p10 span in sigmas
_Z_25_75 = 1.3490  # p75 - p25 span in sigmas (IQR)


def fit_normal(pct: dict, spread_mult: float) -> tuple[float, float]:
    """Fit a normal to returned percentiles: mu = median; sigma from both the
    10-90 span and the IQR (averaged for robustness), inflated by spread_mult to
    counter local-model overconfidence. Shared by `llm` and `mercury_ensemble`
    so their forecasts are identical and the A/B isolates ensembling."""
    p10, p25, p50, p75, p90 = sorted(float(pct[k]) for k in PERCENTILE_KEYS)
    mu = p50
    sigma_raw = 0.5 * ((p90 - p10) / _Z_10_90 + (p75 - p25) / _Z_25_75)
    return mu, max(sigma_raw, 1e-9) * spread_mult


def prob_from_normal(threshold: float, direction: str, mu: float, sigma: float) -> float:
    """One sample's P(quantity exceeds/below threshold) from its fitted normal."""
    cdf = norm_cdf((threshold - mu) / sigma)  # P(quantity <= threshold)
    return 1.0 - cdf if direction == "exceeds" else cdf


@dataclass(frozen=True)
class MixtureStats:
    mean: float       # posterior-predictive probability p̄
    epistemic: float  # Var across samples — the disagreement signal
    aleatoric: float  # mean within-sample Bernoulli variance (irreducible)


def mixture(probs: list[float]) -> MixtureStats:
    n = len(probs)
    mean = sum(probs) / n
    epistemic = sum((p - mean) ** 2 for p in probs) / n
    aleatoric = sum(p * (1.0 - p) for p in probs) / n
    return MixtureStats(mean=mean, epistemic=epistemic, aleatoric=aleatoric)


def direction_agreement(probs: list[float], current_prob: float, mean: float) -> float:
    """Fraction of samples on the same side of the market price as the mean."""
    bullish = mean >= current_prob
    same = sum(1 for p in probs if (p >= current_prob) == bullish)
    return same / len(probs)


@dataclass(frozen=True)
class ShrinkResult:
    estimate: float
    confidence: float


def shrink(mean: float, current_prob: float, epistemic: float, tau: float) -> ShrinkResult:
    """Pull the mean toward the market price in proportion to disagreement."""
    confidence = max(0.0, min(1.0, 1.0 - epistemic / tau))
    estimate = current_prob + (mean - current_prob) * confidence
    return ShrinkResult(estimate=estimate, confidence=confidence)
