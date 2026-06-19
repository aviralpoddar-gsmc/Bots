"""Directional signal for company bets — commodity time-series momentum (TSMOM).

The live P&L taught the lesson cleanly: the bot lost by being systematically SHORT a
trending-up commodity complex (bearish spreads on gold miners that kept rising). The fix
is a real directional view that trades WITH the driver's trend rather than defaulting
bearish.

TSMOM (12-month return, skipping the most recent month to avoid short-term reversal) is
one of the most robust documented factors in commodities. We compute it on the COMMODITY
(the exogenous driver), then propagate to the equity via the screened beta:

    mu_view_equity = beta_c · annualized_commodity_trend   (shrunk + capped)

A positive mu_view tilts f_P up → the selector naturally prefers a bull-call spread; a
negative one → a bear-put spread. So the SAME spread machinery expresses the direction;
only the sign/strength of the view is new. No-lookahead: only returns up to `as_of` are
used. The signal is VALIDATED through the same walk-forward gate before it trades — TSMOM
can be regime-dependent, so the gate is the arbiter, not the theory.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..research.universe import commodity_ticker

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def commodity_momentum(commodity: str, *, as_of=None, lookback_days: int = 252,
                       skip_days: int = 21, period: str = "5y") -> float:
    """Annualized trailing trend of the commodity over [as_of-lookback, as_of-skip].

    Skipping the most recent ~month is the standard TSMOM construction (the last month
    tends to mean-revert). Returns an annualized log-return; 0.0 if data is too thin.
    """
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history
    ticker = commodity_ticker(commodity)
    if ticker is None:
        return 0.0
    try:
        df = fetch_yf_history(ticker, period=period)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        close = df["Close"].astype(float).dropna()
    except Exception:  # noqa: BLE001
        return 0.0
    if len(close) < lookback_days + skip_days + 5:
        return 0.0
    window = close.iloc[-(lookback_days + skip_days): (-skip_days if skip_days else None)]
    if len(window) < 30 or window.iloc[0] <= 0:
        return 0.0
    total_log = math.log(float(window.iloc[-1]) / float(window.iloc[0]))
    years = len(window) / _TRADING_DAYS
    return total_log / years if years > 0 else 0.0


def commodity_ann_vol(commodity: str, *, as_of=None, lookback_days: int = 252,
                      period: str = "5y") -> float:
    """Annualized realized vol of the commodity's daily log-returns up to `as_of`.

    Used to gauge trend *quality*: a 10%/yr trend means something different in a 12%-vol
    commodity than in a 40%-vol one. 0.0 if data is too thin.
    """
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history
    ticker = commodity_ticker(commodity)
    if ticker is None:
        return 0.0
    try:
        df = fetch_yf_history(ticker, period=period)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        close = df["Close"].astype(float).dropna()
    except Exception:  # noqa: BLE001
        return 0.0
    if len(close) < 40:
        return 0.0
    rets = np.log(close / close.shift(1)).dropna().to_numpy()[-lookback_days:]
    if len(rets) < 30:
        return 0.0
    return float(np.std(rets, ddof=1) * math.sqrt(_TRADING_DAYS))


def momentum_drift(*, commodity: str, beta_c: float, as_of=None, lookback_days: int = 252,
                   shrink: float = 0.7, drift_cap: float = 0.35,
                   lookbacks: tuple[int, ...] | None = None,
                   min_strength: float = 0.0) -> tuple[float, float]:
    """(mu_view, conviction): equity drift = beta_c · commodity TSMOM, shrunk + capped.

    conviction is |mu_view| / drift_cap in [0,1] — fed to sizing so stronger trends get
    larger (still capped) positions. (0, 0) when the trend is negligible.

    Two ex-ante robustness enhancements (off by default → callers opt in via config):

    * `lookbacks` — blend the trend across several horizons (e.g. 3/6/12-month) instead
      of a single 12-month window. Averaging horizons is the standard defence against a
      single-lookback's luck-of-the-endpoint fragility.
    * `min_strength` — a **trend-quality regime filter**. Strength = |trend| / annual vol
      (an annualized trend-Sharpe). Below the threshold the trend is choppy/weak and we
      ABSTAIN (0, 0) rather than bet noise — momentum's documented failure mode is
      whipsaw in range-bound regimes, which this skips.
    """
    if lookbacks:
        vals = [commodity_momentum(commodity, as_of=as_of, lookback_days=L) for L in lookbacks]
        vals = [v for v in vals if abs(v) > 1e-9]
        mom = sum(vals) / len(vals) if vals else 0.0
    else:
        mom = commodity_momentum(commodity, as_of=as_of, lookback_days=lookback_days)
    if abs(mom) < 1e-6 or beta_c == 0:
        return 0.0, 0.0
    if min_strength > 0.0:
        vol = commodity_ann_vol(commodity, as_of=as_of)
        if vol > 0.0 and abs(mom) / vol < min_strength:
            return 0.0, 0.0   # weak/choppy trend — abstain
    mu = shrink * beta_c * mom
    mu = max(-drift_cap, min(drift_cap, mu))
    conviction = min(abs(mu) / drift_cap, 1.0)
    return mu, conviction
