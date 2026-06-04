"""Monte-Carlo stochastic-diffusion bot: fat-tailed terminal-price distribution.

A drop-in upgrade to the closed-form lognormal `commodity_spot`: it trades the EXACT
same resolvable spot-price markets (it subclasses CommoditySpotStrategy and reuses its
strict unit/currency matcher, feed->market factor, prefilter, and correlation_key
verbatim) but replaces the analytic pricing with a Monte-Carlo simulation of the
terminal price.

Process: a **block bootstrap** of historical daily log-returns. To price a strike at
horizon T we resample blocks of consecutive real daily returns (block ~10 trading
days, so volatility clustering / spikes survive), compound over n_days = round(T*252),
exponentiate, and read P(exceed) = fraction of simulated terminal prices above the
threshold. Returns are DEMEANED so the process is zero-drift (matching commodity_spot's
validated assumption) — the only difference from the lognormal is the TAIL SHAPE.

Why: the lognormal has thin (Gaussian-in-log) tails. Real commodity returns have fat
tails / jumps (silver, the 2024 cocoa run). The bootstrap captures the *empirical* tail,
so the edge over the lognormal is concentrated in the far-from-spot strikes that are
commodity_spot's stated edge source. For interior strikes the two price ~identically —
so this is a tail refinement, and it ships behind a backtest gate (see config).

Calibration uses yfinance history via `research.data_fetch` (the `research` extra). If
that's unavailable or a series is too short, the affected commodity falls back to the
parent lognormal — never a silent abstain.
"""

from __future__ import annotations

import logging
import math
import zlib
from typing import Any

from ._model import years_to_close
from .base import Market
from .commodity_spot import _SPECS, CommoditySpotStrategy
from .ladder import parse_threshold

logger = logging.getLogger(__name__)

# commodity_spot's entities -> yfinance tickers (subset of research DEFAULT_UNIVERSE).
_ENTITIES = [s[0] for s in _SPECS]


class DiffusionMcStrategy(CommoditySpotStrategy):
    name = "diffusion_mc"
    description = (
        "Monte-Carlo stochastic-diffusion pricer for commodity spot-price markets. "
        "Same strict matcher/markets as commodity_spot, but prices each strike off a "
        "block-bootstrap simulation of historical daily returns (zero-drift, demeaned) "
        "instead of a closed-form lognormal — capturing the fat tails/jumps the "
        "lognormal misses, where the edge lives (far-from-spot strikes). Falls back to "
        "the lognormal when history is missing."
    )

    def __init__(self, period: str = "3y", n_sims: int = 20000, block_len: int = 10,
                 min_returns: int = 250, drift_mode: str = "zero", drift_cap: float = 0.15,
                 process: str = "student_t", jitter: float = 0.4, **params: Any):
        super().__init__(**params)  # vols/min_vol/max_horizon_years flow through params
        self.period = period
        self.n_sims = int(n_sims)
        self.block_len = int(block_len)
        self.min_returns = int(min_returns)
        # Terminal-distribution process:
        #  "bootstrap"  — block-bootstrap of historical returns. Best body calibration,
        #                 but CANNOT extrapolate beyond observed moves -> catastrophic on
        #                 unprecedented tail events (assigns ~0 to a strike that hits).
        #  "student_t"  — Student-t innovations (df fit per commodity). Continuous fat
        #                 tails that EXTRAPOLATE -> keeps most of the body edge AND fixes
        #                 the bootstrap's tail miss. (Diagnostic: best overall.) DEFAULT.
        #  "hybrid"     — bootstrap + small Gaussian jitter so sums can exceed historical
        #                 extremes (a cheaper tail patch).
        self.process = process
        self.jitter = float(jitter)
        self._tparams: dict[str, tuple[float, float]] = {}  # entity -> (df, daily_scale)
        # Drift handling. "zero" (demeaned, the validated commodity_spot assumption) vs
        # "hist" (keep the calibration-window mean drift). Diagnostics showed the
        # bootstrap is BETTER-calibrated than the lognormal but loses raw PnL because
        # both are zero-drift on directional "exceed X" markets — drift is the lever
        # that targets PnL. drift_cap bounds the annualized drift so a strong trailing
        # trend can't run the fair value away.
        self.drift_mode = drift_mode
        self.drift_cap = float(drift_cap)
        # {entity: demeaned daily log-returns}; {entity: mean daily log-return}.
        self._returns: dict[str, Any] = {}
        self._drift: dict[str, float] = {}

    # --- calibration ---------------------------------------------------------

    def bind(self, observations: Any) -> None:
        super().bind(observations)  # refresh the spot-price handle every call (backtest re-binds)
        if not self._returns:       # calibrate ONCE — backtest.py calls bind() every step
            self._calibrate()

    def set_returns(self, entity: str, returns: Any, drift: float = 0.0) -> None:
        """Test/eval seam: inject a demeaned daily-log-return array (+ optional mean
        daily drift) without network."""
        self._returns[entity] = returns
        self._drift[entity] = float(drift)

    def _calibrate(self) -> None:
        try:
            import numpy as np

            from ..research.data_fetch import DEFAULT_UNIVERSE, fetch_yf_history
        except ImportError as e:  # research extra absent -> everything falls back to lognormal
            logger.warning("diffusion_mc: calibration deps missing (%s) — using lognormal fallback", e)
            return
        for entity in _ENTITIES:
            ticker = DEFAULT_UNIVERSE.get(entity)
            if not ticker:
                continue
            try:
                df = fetch_yf_history(ticker, period=self.period)
                close = df["Close"].astype(float).to_numpy()
            except Exception as e:  # noqa: BLE001 - one bad series must not abort calibration
                logger.warning("diffusion_mc: %s history failed (%s)", entity, e)
                continue
            close = close[np.isfinite(close) & (close > 0)]
            if len(close) < self.min_returns + 1:
                logger.info("diffusion_mc: %s only %d points (<%d) — lognormal fallback",
                            entity, len(close), self.min_returns)
                continue
            logret = np.diff(np.log(close))
            logret = logret[np.isfinite(logret)]
            if len(logret) < self.min_returns:
                continue
            self._returns[entity] = logret - logret.mean()  # demean -> zero drift
            self._drift[entity] = float(logret.mean())       # raw mean (used by drift_mode="hist")
            implied_vol = float(logret.std() * math.sqrt(252))
            logger.info("diffusion_mc: calibrated %s (%d returns, implied vol %.2f/yr)",
                        entity, len(logret), implied_vol)

    # --- simulation + pricing ------------------------------------------------

    def group(self, markets: list[Market]) -> list[list[Market]]:
        """Bucket strikes by (commodity, close-month) so the terminal distribution is
        simulated ONCE per ladder and every strike reads off the same sorted sample
        (free cross-strike monotonicity)."""
        groups: dict[tuple[str, int], list[Market]] = {}
        for m in markets:
            spec = self._spec(m.get("question", ""))
            if spec is None:
                continue
            key = (spec[0], round(years_to_close(m) * 12))
            groups.setdefault(key, []).append(m)
        return list(groups.values())

    def _tparams_for(self, entity: str):
        """(df, daily_scale) of a Student-t fit to the entity's demeaned returns, cached.
        scale is set so the t's std equals the realized daily vol."""
        if entity in self._tparams:
            return self._tparams[entity]
        import numpy as np
        rets = self._returns[entity]
        dvol = float(np.std(rets))
        df = 5.0
        try:
            from scipy import stats
            df = float(stats.t.fit(rets)[0])
        except Exception:  # noqa: BLE001 - scipy missing/odd fit -> sane default df
            pass
        df = min(max(df, 3.0), 15.0)  # df>2 for finite var; clamp noise
        scale = dvol / math.sqrt(df / (df - 2.0))
        self._tparams[entity] = (df, scale)
        return df, scale

    def _simulate_terminal(self, entity: str, spot: float, T: float):
        """N simulated terminal prices, or None if uncalibrated."""
        import numpy as np

        rets = self._returns.get(entity)
        if rets is None or len(rets) < self.min_returns:
            return None
        n_days = max(1, round(T * 252))
        # Deterministic seed per (entity, horizon, process) — stable across runs.
        seed = zlib.crc32(f"{entity}:{n_days}:{self.process}".encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        if self.process == "student_t":
            df, scale = self._tparams_for(entity)
            term_logret = (rng.standard_t(df, size=(self.n_sims, n_days)) * scale).sum(axis=1)
        else:
            bl = min(self.block_len, len(rets))
            n_blocks = math.ceil(n_days / bl)
            starts = rng.integers(0, len(rets) - bl + 1, size=(self.n_sims, n_blocks))
            idx = starts[:, :, None] + np.arange(bl)[None, None, :]  # (N, n_blocks, bl)
            term_logret = rets[idx].reshape(self.n_sims, n_blocks * bl)[:, :n_days].sum(axis=1)
            if self.process == "hybrid":  # add Gaussian jitter so sums can exceed observed extremes
                dvol = float(np.std(rets))
                term_logret = term_logret + rng.normal(0.0, self.jitter * dvol, size=(self.n_sims, n_days)).sum(axis=1)

        if self.drift_mode == "hist":
            cap_daily = self.drift_cap / 252.0
            d = max(min(self._drift.get(entity, 0.0), cap_daily), -cap_daily)
            term_logret = term_logret + d * n_days
        return spot * np.exp(term_logret)

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None or not group:
            return {}
        spec = self._spec(group[0].get("question", ""))
        if spec is None:
            return {}
        entity, factor, _vol = spec
        o = self._obs.latest_observation(entity)
        if not o or o.get("value") is None or o["value"] <= 0:
            return {}
        spot = o["value"] * factor
        T = years_to_close(group[0])
        term = self._simulate_terminal(entity, spot, T)
        if term is None:
            return super().estimate(group)  # lognormal fallback for uncalibrated commodities

        import numpy as np

        out: dict[str, float] = {}
        for m in group:
            parsed = parse_threshold(m.get("question", ""))
            if parsed is None:
                continue
            threshold, direction = parsed
            if threshold <= 0:
                continue
            surv = float(np.mean(term > threshold))  # P(terminal > threshold)
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            # lognormal reference at the same vol, to show the tail delta in the comment
            sigma = max(_vol * math.sqrt(T), self.min_vol)
            from ._model import norm_cdf
            ln_surv = 1.0 - norm_cdf(math.log(threshold / spot) / sigma)
            ln_p = ln_surv if direction == "exceeds" else 1.0 - ln_surv
            self._explanations[m["id"]] = {
                "entity": entity, "spot": spot, "threshold": threshold, "direction": direction,
                "T": T, "n_sims": self.n_sims, "surv": surv, "p": p, "lognormal_p": ln_p,
                "obs_ts": o.get("ts"),
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d or "n_sims" not in d:
            return super().explain(market_id)  # fallback path used the lognormal
        return (
            f"- {d['entity']} spot anchor: **${d['spot']:,.2f}** (feed @ {d.get('obs_ts') or 'latest'})\n"
            f"- Threshold: **${d['threshold']:,.2f}** ({d['direction']}); "
            f"{(d['threshold'] / d['spot'] - 1):+.1%} vs spot, T={d['T']:.2f}y\n"
            f"- Monte-Carlo ({d['n_sims']:,} sims, block-bootstrap fat tails): "
            f"P({d['direction']}) = **{d['p']:.3f}**\n"
            f"- vs lognormal {d['lognormal_p']:.3f} → tail delta **{d['p'] - d['lognormal_p']:+.3f}**"
        )
