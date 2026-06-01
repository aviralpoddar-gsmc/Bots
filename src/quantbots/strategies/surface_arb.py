"""Phase 2 reference strategy: surface / monotonicity stat-arb (no LLM).

A "measurable" (e.g. "Brent crude on 2026-06-30") is usually listed as a ladder
of threshold markets: "Will it exceed $62?", "...$70?", "...$82?". Their prices
must obey P(>62) >= P(>70) >= P(>82) and lie on one smooth CDF. This strategy:

  1. groups markets by measurable,
  2. fits a normal distribution to the (threshold, survival-prob) points,
  3. reads each strike's fair value off the fitted curve.

Strikes that deviate from the fit (or violate monotonicity) get an estimate that
pulls them back toward the curve; the runner's sizing turns that gap into orders.

Requires the `quant` extra (numpy + scipy). Each market must carry parsed
`threshold` (float) and `direction` ("exceeds"/"below") fields and a `measurable`
group key — see `quantbots.strategies.ladder` for a heuristic parser, or attach
them upstream. Markets lacking them are skipped.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .base import Market, Strategy
from .ladder import attach_ladder_fields, measurable_key


def _fit_normal_to_survival(strikes: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of (mu, sigma) so that 1 - CDF(strike) ~= prob."""
    mu0 = float(np.average(strikes, weights=np.clip(probs, 1e-3, 1)))
    sigma0 = float(np.std(strikes) or max(abs(mu0), 1.0) * 0.25)

    def loss(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        model = 1.0 - norm.cdf(strikes, mu, sigma)
        return float(np.sum((model - probs) ** 2))

    res = minimize(loss, x0=[mu0, np.log(sigma0)], method="Nelder-Mead")
    mu, log_sigma = res.x
    return float(mu), float(np.exp(log_sigma))


class SurfaceArbStrategy(Strategy):
    name = "surface_arb"
    description = (
        "Parametric volatility-surface fit. For each threshold ladder, finds the "
        "normal CDF whose survival best matches the quoted prices (scipy lsq); "
        "strikes off-curve are traded toward the fitted surface. Heavier-weight, "
        "scipy-based counterpart to ladder_arb, useful when the ladder is dense "
        "enough that a smooth parametric fit beats a step-isotonic one."
    )

    def prefilter(self, markets: list[Market]) -> list[Market]:
        markets = super().prefilter(markets)
        # Keep only markets we can place on a ladder.
        return [m for m in (attach_ladder_fields(m) for m in markets)
                if m.get("threshold") is not None]

    def group(self, markets: list[Market]) -> list[list[Market]]:
        groups: dict[str, list[Market]] = {}
        for m in markets:
            groups.setdefault(measurable_key(m), []).append(m)
        # A 1-point ladder can't be fit; let those fall through as singletons.
        return list(groups.values())

    def estimate(self, group: list[Market]) -> dict[str, float]:
        usable = [m for m in group if m.get("threshold") is not None]
        if len(usable) < 3:  # need enough points to fit 2 params meaningfully
            return {}
        strikes = np.array([m["threshold"] for m in usable], dtype=float)
        probs = np.array([m["probability"] for m in usable], dtype=float)
        mu, sigma = _fit_normal_to_survival(strikes, probs)

        out: dict[str, float] = {}
        for m in usable:
            surv = float(1.0 - norm.cdf(m["threshold"], mu, sigma))
            p = surv if m.get("direction", "exceeds") == "exceeds" else 1.0 - surv
            out[m["id"]] = float(np.clip(p, 0.01, 0.99))
        return out
