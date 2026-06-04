"""Monte-Carlo stochastic-diffusion bot: fat-tailed terminal-price distribution.

A drop-in upgrade to the closed-form lognormal `commodity_spot`: it trades the EXACT
same resolvable spot-price markets (it subclasses CommoditySpotStrategy and reuses its
strict unit/currency matcher, feed->market factor, prefilter, and correlation_key
verbatim) but replaces the analytic pricing with a Monte-Carlo simulation of the
terminal price.

Process (default "ksb"): a **kernel-smoothed block bootstrap** of historical daily
log-returns. To price a strike at horizon T we resample blocks of consecutive real
daily returns (block ~10 trading days, so volatility clustering / spikes survive),
convolve each day with a variance-preserving Student-t kernel (Silverman bandwidth) so
the simulation can extrapolate beyond any historically observed move, compound over
n_days = round(T*252), exponentiate, and read P(exceed) = fraction of simulated terminal
prices above the threshold. Returns are DEMEANED so the process is zero-drift (matching
commodity_spot's validated assumption) — the difference from the lognormal is the
empirical SHAPE (skew + fat tails + vol clustering), not the level.

Why it beats the lognormal: the lognormal has thin (Gaussian-in-log) tails AND a
symmetric body. Real commodity returns are skewed with fat tails / jumps. The multi-fold
walk-forward bench (scripts/diffusion_bench.py) shows that *under the framework's
realistic per-market stake cap* the empirical pricer beats the lognormal on Brier, PnL,
Sharpe AND worst-fold across all 8 commodities — the cap neutralizes the lognormal's
tail over-confidence, then the better body calibration wins. The edge peaks at MEDIUM
horizons (21-63d, where most clone ladders close) and fades to a tie at very short (no
compounding) or very long (CLT) horizons. The kernel smoothing closes the plain
bootstrap's only hole (it cannot price a move larger than any in history).

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
        "kernel-smoothed block-bootstrap simulation of historical daily returns "
        "(zero-drift, demeaned) instead of a closed-form lognormal — capturing the "
        "empirical skew/fat-tails the lognormal misses. Beats the lognormal on "
        "calibration AND capped PnL/Sharpe in multi-fold walk-forward. Falls back to "
        "the lognormal when history is missing."
    )

    def __init__(self, period: str = "10y", n_sims: int = 20000, block_len: int = 10,
                 min_returns: int = 250, drift_mode: str = "zero", drift_cap: float = 0.15,
                 process: str = "ksb", jitter: float = 0.4, df_kernel: float = 4.0,
                 **params: Any):
        super().__init__(**params)  # vols/min_vol/max_horizon_years flow through params
        self.period = period
        self.n_sims = int(n_sims)
        self.block_len = int(block_len)
        self.min_returns = int(min_returns)
        # Terminal-distribution process (chosen by the multi-fold walk-forward bench,
        # scripts/diffusion_bench.py — see header). KEY FINDING: under the framework's
        # realistic per-market stake CAP, the empirical-body processes beat the lognormal
        # on EVERY axis (Brier, PnL, Sharpe, worst-fold) across 8 commodities x 5 folds —
        # the cap neutralizes the lognormal's tail OVER-confidence, then better body
        # calibration wins. Edge peaks at MEDIUM horizons (21-63d), where most clone
        # ladders close, and fades to a tie at <=5d (no compounding) and >=126d (CLT).
        #  "ksb"        — Kernel-Smoothed Bootstrap: block-bootstrap the empirical returns,
        #                 then convolve each day with a variance-preserving Student-t kernel
        #                 (Silverman bandwidth). Matches the plain bootstrap's best-in-class
        #                 PnL/Brier EXACTLY *and* can extrapolate beyond observed moves, so
        #                 it has no catastrophic-tail hole. DEFAULT (best + tail-safe).
        #  "bootstrap"  — plain block bootstrap. Ties ksb on the body but CANNOT extrapolate
        #                 beyond observed moves (assigns ~0 to an unprecedented strike).
        #  "student_t"  — Student-t innovations (df fit per commodity). Most aggressive
        #                 extrapolator; slightly worse body calibration than ksb/bootstrap.
        #  "hybrid"     — bootstrap + Gaussian jitter (a cruder, thin-tailed tail patch).
        self.process = process
        self.jitter = float(jitter)
        self.df_kernel = float(df_kernel)
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
            # Block bootstrap of empirical daily returns (the body) -> (n_sims, n_days).
            bl = min(self.block_len, len(rets))
            n_blocks = math.ceil(n_days / bl)
            starts = rng.integers(0, len(rets) - bl + 1, size=(self.n_sims, n_blocks))
            idx = starts[:, :, None] + np.arange(bl)[None, None, :]  # (N, n_blocks, bl)
            daily = rets[idx].reshape(self.n_sims, n_blocks * bl)[:, :n_days]
            if self.process == "ksb":
                # Convolve each day with a variance-preserving Student-t kernel at Silverman
                # bandwidth: lets the sum exceed any observed move (extrapolation) while the
                # per-day vol is preserved EXACTLY, so the body calibration is untouched.
                s = float(np.std(rets))
                h = 0.9 * len(rets) ** (-0.2)
                dfk = max(self.df_kernel, 2.5)
                z = rng.standard_t(dfk, size=daily.shape) * math.sqrt((dfk - 2.0) / dfk)
                daily = (daily + h * s * z) / math.sqrt(1.0 + h * h)
            elif self.process == "hybrid":  # cruder Gaussian jitter so sums can exceed observed extremes
                dvol = float(np.std(rets))
                daily = daily + rng.normal(0.0, self.jitter * dvol, size=daily.shape)
            term_logret = daily.sum(axis=1)

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
            f"- Monte-Carlo ({d['n_sims']:,} sims, kernel-smoothed bootstrap): "
            f"P({d['direction']}) = **{d['p']:.3f}**\n"
            f"- vs lognormal {d['lognormal_p']:.3f} → tail delta **{d['p'] - d['lognormal_p']:+.3f}**"
        )
