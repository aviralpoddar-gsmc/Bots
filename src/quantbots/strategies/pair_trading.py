"""Pair-trading bot: spot-price threshold markets, priced with a mean-reversion
*drift* derived from a currently-dislocated cointegrated pair.

This is `commodity_spot` with one extra term. `commodity_spot` prices a strike as
a **zero-drift** lognormal — it assumes today's spot is the unbiased forecast of
the spot at close. That is exactly wrong when a cointegrated pair is dislocated:
the spread `s = log a − (α + β·log b)` mean-reverts, so the relative price has a
*predictable* drift back toward its mean. Ignoring it leaves money on the table on
precisely the strikes where we have the most information.

The research layer (`quantbots.research.pairs`) already fits everything we need:
the OLS hedge ratio β, the spread mean μ and std σ_s, and the OU half-life of the
spread. From those, the expected change in the spread over horizon T is

    E[s_T − s_0] = (μ − s_0)·(1 − e^{−θT}),   θ = ln 2 / half_life

i.e. the dislocation decays exponentially at the half-life. We attribute that
reversion to the leg we are pricing (treating its partner as a martingale anchor)
and damp it by `reversion_capture ∈ (0, 1]` to acknowledge that, in reality, both
legs move — so we never bet the full modelled reversion on one side:

    leg = a (regression target):  drift = +κ·(μ − s_0)·(1 − e^{−θT})
    leg = b (regression input):   drift = −κ·(μ − s_0)·(1 − e^{−θT}) / β

The drift is a dimensionless **log-space** shift, so it is applied to the live
feed spot regardless of the unit/benchmark differences between the research panel
(yfinance front-month futures) and the runtime feed (Stooq). Only the *current
z-score* must be unit-consistent, so it is read off the same panel the params were
fit on, never mixed with the live feed.

We trade a leg only when its pair is dislocated past `entry_z` right now — this is
a focused convergence overlay, not a re-implementation of `commodity_spot`. Both
legs of every default pair are commodities we carry a live spot feed for AND that
resolve YES/NO reliably (precious + base metals, crude, gasoline), so the
resolvability filter does not gut the book.

Requires the `research` extra (numpy/pandas/yfinance): `uv sync --extra research`.
If that import or the data fetch fails, the strategy fits no pairs and abstains
everywhere — a safe no-op, never a wrong bet.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from ._model import norm_cdf, years_to_close
from .base import Market
from .commodity_spot import CommoditySpotStrategy
from .ladder import parse_threshold

logger = logging.getLogger(__name__)

_DAYS_PER_YEAR = 365.25

# Cointegrated pairs to monitor. Every entity here is one `commodity_spot` can
# price (live Stooq feed + verified units) AND resolves YES/NO reliably, so the
# convergence trade actually pays out. Ordering is (a, b) = (regression target,
# regression input) and matches `research.pairs.pair_stats(panel, a, b)`. These
# are the resolvable members of the report's cointegration shortlist; the ag,
# defense and macro pairs are deliberately omitted (no spot feed / no reliable
# resolution on the clone).
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("GOLD", "SILVER"),
    ("WTI_OIL", "BRENT_OIL"),
    ("PLATINUM", "PALLADIUM"),
    ("GASOLINE", "WTI_OIL"),
    ("SILVER", "COPPER"),
    ("PLATINUM", "COPPER"),
    ("SILVER", "PLATINUM"),
    ("GOLD", "PLATINUM"),
    ("GOLD", "PALLADIUM"),
    ("SILVER", "PALLADIUM"),
    ("COPPER", "GOLD"),
]


@dataclass
class PairSignal:
    """The active convergence signal attached to one entity (leg of one pair)."""
    a: str
    b: str
    role: str          # "a" or "b": which leg of the pair this entity is
    beta: float        # OLS hedge ratio (log a ≈ alpha + beta·log b)
    half_life: float   # OU half-life of the spread, in days
    z: float           # current spread z-score (signed: +z => leg a rich vs b)
    mu_minus_s0: float # μ − s_0 = −z·σ_s, the total log-reversion target
    corr: float        # return correlation (diagnostic)

    def drift(self, years: float, capture: float) -> float:
        """Expected log-price change of THIS leg over `years`, damped by capture.

        Mean reversion realised so far at horizon T: (μ − s_0)·(1 − e^{−θT}),
        θ = ln2/half_life (per day). For the target leg `a` the spread change maps
        one-to-one onto Δlog a; for input leg `b`, Δs = −β·Δlog b, so it maps with
        a −1/β factor.
        """
        if self.half_life <= 0 or not math.isfinite(self.half_life):
            return 0.0
        theta = math.log(2.0) / self.half_life
        frac = 1.0 - math.exp(-theta * years * _DAYS_PER_YEAR)
        reversion = capture * self.mu_minus_s0 * frac
        return reversion if self.role == "a" else -reversion / self.beta


class PairTradingStrategy(CommoditySpotStrategy):
    name = "pair_trading"
    description = (
        "Convergence overlay on commodity spot-price threshold markets. Prices "
        "each strike with the same horizon-scaled lognormal as commodity_spot, "
        "but injects a mean-reversion drift from a currently-dislocated "
        "cointegrated pair (β / spread / OU half-life fit by the research layer). "
        "Fires only on a leg whose pair is stretched past entry_z, betting the "
        "rich leg down and the cheap leg up as the spread reverts."
    )

    def __init__(
        self,
        *,
        pairs: list[list[str]] | None = None,
        entry_z: float = 1.5,
        reversion_capture: float = 0.5,
        max_half_life_days: float = 90.0,
        min_abs_corr: float = 0.4,
        period: str = "3y",
        lookback_days: int = 750,
        vols: dict[str, float] | None = None,
        min_vol: float = 0.05,
        max_horizon_years: float = 1.25,
        **params: Any,
    ):
        super().__init__(
            vols=vols, min_vol=min_vol, max_horizon_years=max_horizon_years,
            pairs=pairs, entry_z=entry_z, reversion_capture=reversion_capture,
            max_half_life_days=max_half_life_days, min_abs_corr=min_abs_corr,
            period=period, lookback_days=lookback_days, **params,
        )
        self.pairs = [tuple(p) for p in pairs] if pairs else list(DEFAULT_PAIRS)
        self.entry_z = entry_z
        self.reversion_capture = reversion_capture
        self.max_half_life_days = max_half_life_days
        self.min_abs_corr = min_abs_corr
        self.period = period
        self.lookback_days = lookback_days
        # entity -> the most-dislocated active PairSignal touching that entity.
        self._signals: dict[str, PairSignal] = {}
        self._fitted = False

    # -- pair fitting -------------------------------------------------------

    def bind(self, observations: Any) -> None:
        super().bind(observations)
        self._fit_pairs()

    def _fit_pairs(self) -> None:
        """Fit cointegration params from the cached research panel and keep, per
        entity, the most-dislocated pair past entry_z. Any failure (missing
        research extra, no network/cache) leaves `_signals` empty -> abstain."""
        self._fitted = True
        try:
            from ..research.data_fetch import fetch_universe
            from ..research.pairs import align_panel, pair_stats
        except Exception as e:  # noqa: BLE001 — research extra not installed
            logger.warning("pair_trading: research layer unavailable (%s); abstaining", e)
            return
        try:
            panel = fetch_universe(period=self.period)
            panel = align_panel(panel, lookback_days=self.lookback_days, min_coverage=0.85)
        except Exception as e:  # noqa: BLE001 — data fetch failed
            logger.warning("pair_trading: panel fetch failed (%s); abstaining", e)
            return
        if panel.empty:
            logger.warning("pair_trading: empty panel; abstaining")
            return
        self.set_signals_from_panel(panel, pair_stats)
        logger.info(
            "pair_trading: %d active signal(s) past |z|>=%.1f: %s",
            len(self._signals), self.entry_z,
            ", ".join(f"{e}({s.z:+.2f})" for e, s in self._signals.items()) or "none",
        )

    def set_signals_from_panel(self, panel: Any, pair_stats: Any) -> None:
        """Build `_signals` from an aligned price panel. Separated from the fetch
        so it can be unit-tested with a synthetic panel (no network)."""
        candidates: list[PairSignal] = []
        for a, b in self.pairs:
            if a not in panel.columns or b not in panel.columns:
                continue
            ps = pair_stats(panel, a, b)
            if ps is None or not math.isfinite(ps.half_life):
                continue
            if not (0 < ps.half_life <= self.max_half_life_days):
                continue
            if not math.isfinite(ps.corr_returns) or abs(ps.corr_returns) < self.min_abs_corr:
                continue
            if abs(ps.current_z) < self.entry_z:
                continue
            mu_minus_s0 = -ps.current_z * ps.spread_std
            common = dict(a=a, b=b, beta=ps.beta, half_life=ps.half_life,
                          z=ps.current_z, mu_minus_s0=mu_minus_s0, corr=ps.corr_returns)
            candidates.append(PairSignal(role="a", **common))
            candidates.append(PairSignal(role="b", **common))
        self._signals = {}
        for sig in candidates:
            entity = sig.a if sig.role == "a" else sig.b
            cur = self._signals.get(entity)
            if cur is None or abs(sig.z) > abs(cur.z):
                self._signals[entity] = sig

    # -- universe -----------------------------------------------------------

    def prefilter(self, markets: list[Market]) -> list[Market]:
        # commodity_spot keeps feedable spot-price markets within the calibrated
        # horizon; we additionally keep only entities with an active signal so the
        # runner doesn't group/evaluate the whole metals book to abstain on it.
        if not self._fitted:
            # prefilter may run standalone in tests; ensure signals exist.
            self._fit_pairs()
        kept = []
        for m in super().prefilter(markets):
            spec = self._spec(m.get("question", ""))
            if spec and spec[0] in self._signals:
                kept.append(m)
        return kept

    # -- pricing ------------------------------------------------------------

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            spec = self._spec(m.get("question", ""))
            if spec is None:
                continue
            entity, factor, annual_vol = spec
            sig = self._signals.get(entity)
            if sig is None:
                continue  # no active dislocation on this leg -> abstain
            parsed = parse_threshold(m.get("question", ""))
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            spot = o["value"] * factor  # feed value in the market's quoted unit
            T = years_to_close(m)
            sigma = max(annual_vol * math.sqrt(T), self.min_vol)
            drift = sig.drift(T, self.reversion_capture)
            # P(spot_at_close > threshold) under lognormal diffusion WITH the
            # mean-reversion drift folded into the log-price.
            surv = 1.0 - norm_cdf((math.log(threshold / spot) - drift) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {
                "entity": entity, "spot": spot, "threshold": threshold,
                "direction": direction, "annual_vol": annual_vol, "T": T,
                "sigma": sigma, "surv": surv, "p": p, "drift": drift,
                "obs_ts": o.get("ts"), "sig": sig,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        sig: PairSignal = d["sig"]
        ts = d.get("obs_ts") or "latest"
        rich, cheap = (sig.a, sig.b) if sig.z >= 0 else (sig.b, sig.a)
        return (
            f"- Pair **{sig.a} ↔ {sig.b}**: z = **{sig.z:+.2f}** "
            f"(|z| ≥ {self.entry_z:.1f}), half-life {sig.half_life:.0f}d, "
            f"ρ = {sig.corr:+.2f}. {rich} rich vs {cheap} → convergence expected.\n"
            f"- {d['entity']} spot anchor: **${d['spot']:,.2f}** (feed @ {ts})\n"
            f"- Threshold: **${d['threshold']:,.2f}** ({d['direction']}); "
            f"{(d['threshold'] / d['spot'] - 1):+.1%} vs spot\n"
            f"- Reversion drift on {d['entity']}: **{d['drift']:+.3f}** log "
            f"(capture {self.reversion_capture:.2f}, T={d['T']:.2f}y); "
            f"σ_eff={d['sigma']:.3f}\n"
            f"- P(spot_at_close > threshold) = {d['surv']:.3f} "
            f"→ P({d['direction']}) = **{d['p']:.3f}**"
        )
