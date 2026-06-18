"""Commodity/market beta: how a commodity move propagates to the equity.

    r_equity = alpha + beta_c * r_commodity + beta_m * r_market + eps,  eps ~ N(0, sigma_idio^2)

A two-factor OLS on overlapping daily log returns (same `np.linalg.lstsq` pattern as
`research.pairs._ols_hedge`, extended to two regressors). The forecast carries the
commodity Monte-Carlo sample *through* this regression per-draw to build the equity
terminal distribution f_P.

Two instability guards from the plan (commodity->equity beta is the crux and the
least stable part of the thesis):
  - **Shrink** beta_c toward 0 by `shrink` (a weak/noisy beta should not move f_P much).
  - **Inflate** sigma_idio by the beta standard error (Var(beta) uncertainty feeds the
    forecast spread), and expose `weak` so the forecaster can abstain.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ...research.data_fetch import fetch_yf_history
from .universe import commodity_ticker

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


@dataclass
class BetaFit:
    equity: str
    commodity: str
    alpha: float           # daily intercept
    beta_c: float          # commodity beta (shrunk)
    beta_m: float          # market beta
    beta_c_raw: float      # pre-shrink commodity beta
    beta_c_se: float       # std error of beta_c (instability)
    sigma_idio: float      # annualized idiosyncratic vol (inflated by beta uncertainty)
    r2: float
    n_obs: int

    @property
    def weak(self) -> bool:
        """True when the commodity beta is statistically indistinguishable from 0
        (|t| < 1.5) or the regression explains almost nothing."""
        if self.beta_c_se <= 0:
            return True
        return abs(self.beta_c_raw) / self.beta_c_se < 1.5 or self.r2 < 0.02


def _log_returns(close: np.ndarray) -> np.ndarray:
    close = close[np.isfinite(close) & (close > 0)]
    return np.diff(np.log(close))


def fit_beta(equity: str, commodity: str, market: str, *, lookback_days: int = 504,
             shrink: float = 0.3, idio_floor: float = 0.05, period: str = "5y",
             as_of=None) -> BetaFit | None:
    """Fit the two-factor regression. Returns None if data is too thin to align.

    `as_of` (a date) enforces NO LOOKAHEAD: only returns up to and including that date
    are used, so a backtest forecasting t+h never peeks at data after t."""
    comm_ticker = commodity_ticker(commodity)
    if comm_ticker is None:
        logger.warning("fit_beta: no commodity ticker for %s", commodity)
        return None
    try:
        import pandas as pd
        eq = fetch_yf_history(equity, period=period)["Close"].astype(float)
        cm = fetch_yf_history(comm_ticker, period=period)["Close"].astype(float)
        mk = fetch_yf_history(market, period=period)["Close"].astype(float)
    except Exception as e:  # noqa: BLE001
        logger.warning("fit_beta: history fetch failed (%s)", e)
        return None
    panel = pd.concat({"eq": eq, "cm": cm, "mk": mk}, axis=1).dropna()
    if as_of is not None:
        panel = panel[panel.index <= pd.Timestamp(as_of)]
    panel = panel.tail(lookback_days + 1)
    if len(panel) < 60:
        logger.info("fit_beta: only %d aligned points for %s/%s", len(panel), equity, commodity)
        return None
    r_eq = _log_returns(panel["eq"].to_numpy())
    r_cm = _log_returns(panel["cm"].to_numpy())
    r_mk = _log_returns(panel["mk"].to_numpy())
    n = min(len(r_eq), len(r_cm), len(r_mk))
    r_eq, r_cm, r_mk = r_eq[-n:], r_cm[-n:], r_mk[-n:]

    X = np.column_stack([np.ones(n), r_cm, r_mk])
    coef, *_ = np.linalg.lstsq(X, r_eq, rcond=None)
    alpha, beta_c_raw, beta_m = (float(c) for c in coef)
    resid = r_eq - X @ coef
    dof = max(n - 3, 1)
    sigma2 = float(resid @ resid) / dof
    # Coefficient covariance = sigma^2 (X'X)^-1; SE of beta_c is sqrt of its diagonal.
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        beta_c_se = float(math.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        beta_c_se = float("inf")
    ss_tot = float(((r_eq - r_eq.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0

    beta_c = beta_c_raw * (1.0 - shrink)  # shrink toward 0
    # Annualized idiosyncratic vol, inflated by beta uncertainty (Var(beta_c)*Var(r_cm)).
    idio_daily = math.sqrt(sigma2 + beta_c_se ** 2 * float(np.var(r_cm)))
    sigma_idio = max(idio_daily * math.sqrt(_TRADING_DAYS), idio_floor)
    return BetaFit(equity=equity, commodity=commodity, alpha=alpha, beta_c=beta_c,
                   beta_m=beta_m, beta_c_raw=beta_c_raw, beta_c_se=beta_c_se,
                   sigma_idio=sigma_idio, r2=r2, n_obs=n)
