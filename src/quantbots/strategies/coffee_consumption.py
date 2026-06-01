"""Coffee bot: USDA consumption fundamentals on coffee *demand-growth* markets.

The clone has **no coffee price markets** — only demand/volume/market-size
questions (verified against market_cache). Of those, exactly one template maps to
a USDA-measurable quantity: "global coffee consumption growth rate for the year
ending <date> exceed X%". USDA FAS PSD publishes world coffee domestic consumption
(1000 60-kg bags) back to 1960, so its YoY growth is a directly relevant estimate.

Model: world coffee consumption growth is a low-single-digit, mildly mean-reverting
series — empirically mean ≈ 1.4%/yr, σ ≈ 2.6% (FAS PSD 2010+). We price each
threshold as a Normal CDF around that mean (the multi-year horizon makes the
long-run mean a better central estimate than any single noisy YoY print, though the
latest `PSD_COFFEE_CONS_GROWTH` observation can be blended in via `mu_weight`).

    P(growth > thr) = 1 − Φ( (thr − μ) / σ )

⚠️ Resolvability caveat: consumption/demand markets historically resolve YES/NO
≈0% on this clone (they cancel). The portfolio allocator's resolvability weighting
will therefore size this bot small by design. It is built and correct; whether it
ever deploys meaningful capital depends on these markets actually resolving. See
docs/usda-softs-bots.md §4 (Bot 3).
"""

from __future__ import annotations

from typing import Any

from ._model import norm_cdf
from .base import Market, Strategy
from .ladder import parse_threshold

_KEYS = ("coffee", "consumption", "growth")


class CoffeeConsumptionStrategy(Strategy):
    name = "coffee_consumption"
    description = (
        "Prices 'global coffee consumption growth rate exceed X%' markets off USDA "
        "FAS world coffee consumption (YoY), as a Normal CDF around the long-run "
        "growth mean (~1.4%/yr, σ~2.6%). The only coffee market on the clone that "
        "maps to a USDA-measurable fundamental; demand markets resolve rarely, so "
        "the allocator sizes it conservatively."
    )

    def __init__(
        self,
        mean_growth: float = 1.43,
        sigma_growth: float = 2.59,
        mu_weight: float = 0.0,
        growth_entity: str = "PSD_COFFEE_CONS_GROWTH",
        **params: Any,
    ):
        super().__init__(
            mean_growth=mean_growth, sigma_growth=sigma_growth,
            mu_weight=mu_weight, **params,
        )
        self.mean_growth = mean_growth
        self.sigma_growth = sigma_growth
        self.mu_weight = mu_weight
        self.growth_entity = growth_entity
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _matches(self, q: str) -> bool:
        ql = q.lower()
        return all(k in ql for k in _KEYS)

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [
            m for m in markets
            if not m.get("isResolved") and self._matches(m.get("question", ""))
        ]

    def correlation_key(self, market: Market) -> str:
        return "COFFEE_CONS" if self._matches(market.get("question", "")) else str(market.get("id"))

    def _mu(self) -> float:
        """Central growth estimate: long-run mean, optionally blended with latest YoY."""
        mu = self.mean_growth
        if self.mu_weight > 0 and self._obs is not None:
            o = self._obs.latest_observation(self.growth_entity)
            if o and o.get("value") is not None:
                mu = (1 - self.mu_weight) * self.mean_growth + self.mu_weight * o["value"]
        return mu

    def estimate(self, group: list[Market]) -> dict[str, float]:
        out: dict[str, float] = {}
        mu = self._mu()
        for m in group:
            q = m.get("question", "")
            if not self._matches(q):
                continue
            parsed = parse_threshold(q)
            if parsed is None or self.sigma_growth <= 0:
                continue
            threshold, direction = parsed
            surv = 1.0 - norm_cdf((threshold - mu) / self.sigma_growth)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "mu": mu, "sigma": self.sigma_growth, "threshold": threshold,
                "direction": direction, "p": p,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        return (
            f"- USDA world coffee consumption growth: μ=**{d['mu']:.2f}%/yr**, σ={d['sigma']:.2f}%\n"
            f"- Threshold: **{d['threshold']:.2f}%** ({d['direction']}) "
            f"→ P({d['direction']})=**{d['p']:.3f}**"
        )
