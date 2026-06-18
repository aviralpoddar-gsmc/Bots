"""f_P(S_T) — our forward distribution of an equity's terminal price.

Construction (per the approved plan), as a Monte-Carlo terminal-price SAMPLE so the
selection engine can price any structure off one sorted draw (exactly how
`diffusion_mc.group` prices a whole ladder off one sample):

    ln(S_T / S_0) = drift*T  +  beta_c * r_comm  +  beta_m * r_mkt  +  idio

  - r_comm: commodity terminal log-return, drawn from the parent `diffusion_mc`
    kernel-smoothed block bootstrap (fat tails / skew / vol-clustering) — REUSED via
    its `set_returns` test-seam so we are not coupled to the clone markets. Demeaned
    (zero-drift); the SHAPE is the point.
  - beta_c, beta_m, idio: from the two-factor regression (`research.beta`). Weak
    betas are shrunk and idio is inflated by beta uncertainty; if the beta is
    statistically ~0 we abstain (return None) rather than fabricate a view.
  - drift: chosen by `mode`. "drift_neutral" centers f_P risk-neutrally
    (E[S_T] = S_0 e^{(r-q)T}), so any edge vs the chain is pure VOL/SHAPE (fatter
    tails / different total vol than the implied surface) — the honest v1 edge with
    no fundamentals. "directional" adds a fundamentals/LLM view drift `mu_view`
    (zero in v1; the tal/LLM channels are off seams).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ...strategies.diffusion_mc import DiffusionMcStrategy
from ..research.beta import BetaFit, fit_beta
from ..research.universe import commodity_ticker

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


@dataclass
class Forecast:
    ticker: str
    s0: float
    T: float
    r: float
    q: float
    terminal: np.ndarray      # MC sample of S_T (sorted ascending)
    sigma_fp: float           # annualized total vol implied by the sample
    mode: str
    beta: BetaFit
    n_sims: int

    def prob_above(self, level: float) -> float:
        return float(np.mean(self.terminal > level))


def _commodity_terminal_returns(commodity: str, T: float, *, n_sims: int, period: str,
                                process: str, as_of=None) -> np.ndarray | None:
    """Demeaned commodity terminal log-returns from the diffusion_mc bootstrap.

    `as_of` slices the calibration history to <= that date (no lookahead)."""
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history

    ticker = commodity_ticker(commodity)
    if ticker is None:
        return None
    try:
        df = fetch_yf_history(ticker, period=period)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        close = df["Close"].astype(float).to_numpy()
    except Exception as e:  # noqa: BLE001
        logger.warning("forecast: commodity %s history failed (%s)", commodity, e)
        return None
    close = close[np.isfinite(close) & (close > 0)]
    if len(close) < 260:
        return None
    logret = np.diff(np.log(close))
    demeaned = logret - logret.mean()
    diff = DiffusionMcStrategy(n_sims=n_sims, process=process)
    diff.set_returns(commodity, demeaned)             # bypass clone-market calibration
    term = diff._simulate_terminal(commodity, spot=1.0, T=T)  # terminal multipliers
    if term is None:
        return None
    return np.log(np.asarray(term, dtype=float))      # back to log-returns


def build_forecast(*, ticker: str, commodity: str, market: str, s0: float, T: float,
                   r: float, q: float = 0.0, mode: str = "drift_neutral",
                   mu_view: float = 0.0, n_sims: int = 20000, period: str = "10y",
                   process: str = "ksb", shrink: float = 0.3, idio_floor: float = 0.05,
                   beta_lookback_days: int = 504, beta: BetaFit | None = None,
                   rng: np.random.Generator | None = None, as_of=None,
                   drift_shrink: float = 0.5, drift_cap: float = 0.30) -> Forecast | None:
    """Build f_P. Returns None when the commodity beta is too weak to justify a view.

    `as_of` (a date) enforces no-lookahead across every calibration input — used by the
    walk-forward backtest. When None (live trading) all available history is used."""
    if T <= 0 or s0 <= 0:
        return None
    beta = beta or fit_beta(ticker, commodity, market, lookback_days=beta_lookback_days,
                            shrink=shrink, idio_floor=idio_floor, period="5y", as_of=as_of)
    if beta is None:
        return None
    if beta.weak:
        logger.info("forecast: %s/%s beta too weak (raw=%.3f se=%.3f r2=%.3f) — abstain",
                    ticker, commodity, beta.beta_c_raw, beta.beta_c_se, beta.r2)
        return None

    rng = rng or np.random.default_rng(abs(hash((ticker, round(T, 4), mode))) % (2**32))

    r_comm = _commodity_terminal_returns(commodity, T, n_sims=n_sims, period=period,
                                         process=process, as_of=as_of)
    if r_comm is None:
        return None
    n = len(r_comm)

    # Market factor: zero-drift normal at the market's realized vol over the horizon.
    sigma_m = _realized_vol(market, period="2y", as_of=as_of)
    r_mkt = rng.normal(0.0, sigma_m * math.sqrt(T), size=n)
    # Idiosyncratic: zero-drift normal at the regression's (inflated) idio vol.
    idio = rng.normal(0.0, beta.sigma_idio * math.sqrt(T), size=n)

    systematic = beta.beta_c * r_comm + beta.beta_m * r_mkt + idio
    var_sample = float(np.var(systematic))            # total log-variance over horizon

    if mode == "directional":
        # Real-world drift: a shrunk, capped estimate of the equity's expected return.
        # The drift-neutral backtest showed f_P is systematically biased LOW (PIT mean
        # 0.63) because these names trended up — a no-lookahead trailing-drift estimate
        # de-biases the center. Shrunk (momentum is fragile) and capped (a trend can't
        # run the fair value away). mu_view overrides it when a real signal is supplied.
        mu_ann = mu_view if mu_view else _trailing_drift(ticker, as_of=as_of, period=period)
        mu_ann = max(-drift_cap, min(drift_cap, drift_shrink * mu_ann))
        drift = mu_ann * T - 0.5 * var_sample
    else:  # drift_neutral: martingale center, edge is pure vol/shape
        drift = (r - q) * T - 0.5 * var_sample

    log_st = drift + systematic
    terminal = np.sort(s0 * np.exp(log_st))
    sigma_fp = math.sqrt(max(var_sample, 1e-12) / T)
    return Forecast(ticker=ticker, s0=s0, T=T, r=r, q=q, terminal=terminal,
                    sigma_fp=sigma_fp, mode=mode, beta=beta, n_sims=n)


def _trailing_drift(ticker: str, *, as_of=None, lookback_days: int = 252,
                    period: str = "5y", default: float = 0.0) -> float:
    """Annualized trailing mean log-return up to `as_of` (no lookahead). The raw
    drift estimate fed to the directional forecast; build_forecast shrinks + caps it."""
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history
    try:
        df = fetch_yf_history(ticker, period=period)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        close = df["Close"].astype(float).to_numpy()
        close = close[np.isfinite(close) & (close > 0)]
        if len(close) < 60:
            return default
        logret = np.diff(np.log(close))[-lookback_days:]
        return float(np.mean(logret) * _TRADING_DAYS)
    except Exception:  # noqa: BLE001
        return default


def _realized_vol(ticker: str, *, period: str = "2y", default: float = 0.18, as_of=None) -> float:
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history
    try:
        df = fetch_yf_history(ticker, period=period)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        close = df["Close"].astype(float).to_numpy()
        close = close[np.isfinite(close) & (close > 0)]
        if len(close) < 60:
            return default
        logret = np.diff(np.log(close))
        return float(np.std(logret) * math.sqrt(_TRADING_DAYS))
    except Exception:  # noqa: BLE001
        return default
