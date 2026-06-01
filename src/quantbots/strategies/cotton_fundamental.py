"""Cotton bot: USDA-fundamentals drift on ICE Cotton No. 2 futures price markets.

Extends the plain zero-drift lognormal of `commodity_futures` with a *bounded*
fundamental drift. The level is anchored to the live ICE Cotton No. 2 futures feed
(`CME_COTTON` from Stooq) — NOT to a USDA price, which sidesteps the farm-price↔ICE
basis problem. The drift nudges that anchor toward a fundamental fair value derived
from the world-**ex-China** stocks-to-use ratio (`PSD_COTTON_FREE_SUR`).

Why ex-China and why only a small drift: see docs/usda-softs-bots.md §2b. Empirically
(FAS PSD 1960+ vs ICE futures, walk-forward OOS) the *world* SUR is uninformative
(R²≈0, China's state reserve is off-market); world-ex-China flips the sign correct
(elasticity ≈ −0.39) and is the only model that beats the zero-drift baseline OOS,
but the fit is weak (R²≈0.03) with a *reliable direction* (≈71% hit-rate). So the
fundamental enters as a capped directional tilt (±5%/yr), never as a hard level.

Fair value:  F_fund = price_ref · (free_sur / sur_ref) ** elasticity
Drift:        mu = clamp( reversion_rate · log(F_fund / spot), ±drift_cap )
Pricing:      F = spot · exp(mu · T);  P(exceeds) = 1 − Φ( ln(thr/F) / (vol·√T) )

An optional in-season crop-condition nudge (NASS good+excellent %) tilts mu further
when that observation is present; it is purely additive and skipped if absent.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market, Strategy
from .ladder import parse_threshold

# Cotton-only: ICE Cotton No. 2 *outright* price markets. Deliberately narrow so
# this bot never overlaps commodity_futures' other softs.
_COTTON = re.compile(r"\bcotton\b", re.I)
_PRICE = re.compile(r"futures|cents/?lb|price|/lb|per pound", re.I)
# Reject spread/basis/differential markets — they reference a few-cent spread, not
# the ~70-cent outright the futures feed quotes. Pricing them off the outright is a
# confidently-wrong trap (e.g. "Cotlook A minus ICE Cotton No.2 basis exceed 18").
_EXCLUDE = re.compile(
    r"basis|spread|differential|premium|discount|\bminus\b|\bless\b|\bover\b|\bvs\.?\b", re.I
)


class CottonFundamentalStrategy(Strategy):
    name = "cotton_fundamental"
    description = (
        "ICE Cotton No. 2 futures price-threshold markets, priced with a "
        "lognormal CDF anchored to the live futures feed and tilted by a bounded "
        "drift from USDA FAS world-ex-China stocks-to-use (elasticity ≈ −0.39, "
        "capped at ±5%/yr) plus an optional NASS crop-condition nudge. Edge is a "
        "better-calibrated fair value than zero-drift, concentrated on the cotton "
        "price markets that actually resolve."
    )

    def __init__(
        self,
        annual_vol: float = 0.24,
        min_vol: float = 0.05,
        max_horizon_years: float = 1.25,
        elasticity: float = -0.39,
        sur_ref: float = 0.487,
        price_ref: float = 68.4,
        reversion_rate: float = 0.5,
        drift_cap: float = 0.05,
        condition_ref: float = 50.0,
        condition_beta: float = 0.0010,
        spot_entity: str = "CME_COTTON",
        sur_entity: str = "PSD_COTTON_FREE_SUR",
        cond_entity: str = "NASS_COTTON_COND_GE",
        **params: Any,
    ):
        super().__init__(
            annual_vol=annual_vol, min_vol=min_vol, max_horizon_years=max_horizon_years,
            elasticity=elasticity, sur_ref=sur_ref, price_ref=price_ref,
            reversion_rate=reversion_rate, drift_cap=drift_cap,
            condition_ref=condition_ref, condition_beta=condition_beta, **params,
        )
        self.annual_vol = annual_vol
        self.min_vol = min_vol
        self.max_horizon_years = max_horizon_years
        self.elasticity = elasticity
        self.sur_ref = sur_ref
        self.price_ref = price_ref
        self.reversion_rate = reversion_rate
        self.drift_cap = drift_cap
        self.condition_ref = condition_ref
        self.condition_beta = condition_beta
        self.spot_entity = spot_entity
        self.sur_entity = sur_entity
        self.cond_entity = cond_entity
        self._obs: Any | None = None

    def bind(self, observations: Any) -> None:
        self._obs = observations

    def _is_cotton_price(self, q: str) -> bool:
        return bool(_COTTON.search(q) and _PRICE.search(q) and not _EXCLUDE.search(q))

    def prefilter(self, markets: list[Market]) -> list[Market]:
        out = []
        for m in markets:
            if m.get("isResolved"):
                continue
            if not self._is_cotton_price(m.get("question", "")):
                continue
            if years_to_close(m) > self.max_horizon_years:
                continue
            out.append(m)
        return out

    def correlation_key(self, market: Market) -> str:
        return "COTTON" if self._is_cotton_price(market.get("question", "")) else str(market.get("id"))

    def _drift(self, spot: float) -> tuple[float, dict[str, Any]]:
        """Bounded fundamental drift + the numbers behind it (for explain())."""
        d: dict[str, Any] = {}
        mu = 0.0
        o_sur = self._obs.latest_observation(self.sur_entity) if self._obs else None
        if o_sur and o_sur.get("value") and o_sur["value"] > 0:
            sur = o_sur["value"]
            f_fund = self.price_ref * (sur / self.sur_ref) ** self.elasticity
            mu = self.reversion_rate * math.log(f_fund / spot)
            d.update(free_sur=sur, f_fund=f_fund)
        o_c = self._obs.latest_observation(self.cond_entity) if self._obs else None
        if o_c and o_c.get("value") is not None:
            cond_tilt = self.condition_beta * (o_c["value"] - self.condition_ref)
            mu += cond_tilt
            d.update(condition=o_c["value"], cond_tilt=cond_tilt)
        mu = max(min(mu, self.drift_cap), -self.drift_cap)
        d["mu"] = mu
        return mu, d

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            q = m.get("question", "")
            if not self._is_cotton_price(q):
                continue
            parsed = parse_threshold(q)
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(self.spot_entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            spot = o["value"]
            mu, dd = self._drift(spot)
            T = years_to_close(m)
            fair = spot * math.exp(mu * T)
            sigma = max(self.annual_vol * math.sqrt(T), self.min_vol)
            surv = 1.0 - norm_cdf(math.log(threshold / fair) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "spot": spot, "threshold": threshold, "direction": direction,
                "T": T, "sigma": sigma, "fair": fair, "p": p, **dd,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        lines = [
            f"- ICE Cotton No.2 spot anchor: **{d['spot']:.2f} cents/lb**",
            f"- Threshold: **{d['threshold']:.2f}** ({d['direction']})",
        ]
        if "free_sur" in d:
            lines.append(
                f"- USDA free (ex-China) stocks-to-use: **{d['free_sur']:.3f}** "
                f"→ fair value {d['f_fund']:.1f} cents/lb"
            )
        if "condition" in d:
            lines.append(f"- NASS crop condition (good+excellent): **{d['condition']:.0f}%**")
        lines.append(
            f"- Drift μ={d['mu']:+.3f}/yr (capped), T={d['T']:.2f}y, σ_eff={d['sigma']:.3f} "
            f"→ fair@close {d['fair']:.1f} → P({d['direction']})=**{d['p']:.3f}**"
        )
        return "\n".join(lines)
