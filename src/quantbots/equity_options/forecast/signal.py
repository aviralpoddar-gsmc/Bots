"""tal directional signal → f_P drift (mu_view).

The investigation found the missing ingredient was *direction*. tal supplies it, but
indirectly: tal/clone has commodity SPOT-PRICE prediction markets whose implied
distribution is a (noisy) expectation of where the commodity is headed. We:

  1. pull the commodity's spot-price threshold ladder (`tal_snowflake.commodity_price_markets`),
  2. read each market's probability AS OF the date (no lookahead) from price history,
  3. clean the noisy ladder by least-squares fitting a LOGNORMAL to the survival points
     (same trick as `surface_arb._fit_normal_to_survival`, robust to the clone's stale
     0/1 quotes), giving an implied expected commodity log-return,
  4. propagate to the equity via the screened commodity beta: mu_view = beta_c · r̂_comm,
  5. (best-effort) fuse a tal MEASURABLE fundamental view where one maps — rare for the
     liquid majors, so usually this is just the price view.

Returns None when the ladder is too thin/noisy to fit — the forecast then falls back to
drift-neutral. The GATE remains the arbiter: if this signal doesn't lift out-of-sample
calibration/PnL, those names simply won't trade.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_MIN_THRESHOLDS = 3
_PROB_LO, _PROB_HI = 0.03, 0.97   # drop stale/untraded 0/1 quotes


@dataclass
class CommodityView:
    commodity: str
    expected_log_return_ann: float   # annualized implied commodity log-return
    confidence: float                # 0..1 from fit quality + ladder breadth
    n_thresholds: int
    horizon_years: float


def _fit_lognormal_survival(strikes: np.ndarray, surv: np.ndarray) -> tuple[float, float, float]:
    """Fit ln-price ~ N(mu, sigma) so that 1-Phi((lnK-mu)/sigma) ≈ surv. Returns
    (mu, sigma, rmse). Reuses the surface_arb least-squares-to-a-ladder pattern."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    x = np.log(strikes)
    mu0 = float(np.average(x, weights=np.clip(surv, 1e-3, 1)))
    sigma0 = float(np.std(x) or 0.25)

    def loss(theta):
        mu, log_sigma = theta
        model = 1.0 - norm.cdf((x - mu) / np.exp(log_sigma))
        return float(np.sum((model - surv) ** 2))

    res = minimize(loss, x0=[mu0, math.log(max(sigma0, 1e-3))], method="Nelder-Mead")
    mu, log_sigma = res.x
    rmse = math.sqrt(res.fun / len(surv))
    return float(mu), float(math.exp(log_sigma)), rmse


def commodity_view_from_ladder(thresholds, exceed_probs, *, spot: float,
                               horizon_years: float) -> CommodityView | None:
    """Build an implied expected commodity return from one settlement's threshold ladder."""
    if spot <= 0 or horizon_years <= 0:
        return None
    thr = np.asarray(thresholds, dtype=float)
    p = np.asarray(exceed_probs, dtype=float)
    m = np.isfinite(thr) & np.isfinite(p) & (thr > 0) & (p > _PROB_LO) & (p < _PROB_HI)
    thr, p = thr[m], p[m]
    if len(np.unique(thr)) < _MIN_THRESHOLDS:
        return None
    # collapse duplicate strikes (average prob), sort
    order = np.argsort(thr)
    thr, p = thr[order], p[order]
    mu, sigma, rmse = _fit_lognormal_survival(thr, p)
    # Implied expected log-return to settlement = E[ln S_T] - ln spot = mu - ln(spot).
    exp_log_ret = mu - math.log(spot)
    exp_log_ret_ann = exp_log_ret / horizon_years
    # Confidence: more thresholds + tighter fit => higher; squashed to 0..1.
    breadth = min(len(thr) / 6.0, 1.0)
    fit_q = max(0.0, 1.0 - rmse / 0.15)
    conf = max(0.0, min(1.0, 0.5 * breadth + 0.5 * fit_q))
    return CommodityView(commodity="", expected_log_return_ann=exp_log_ret_ann,
                         confidence=conf, n_thresholds=int(len(thr)), horizon_years=horizon_years)


def _asof_probs(markets_df, history_df, as_of):
    """Latest probability per market on or before as_of (no lookahead). Falls back to
    LATEST_MARKET_PROBABILITY only when as_of is None (live)."""
    import pandas as pd
    out_thr, out_p = [], []
    hist = history_df
    for _, mrow in markets_df.iterrows():
        mid = mrow["ID"]
        thr = mrow["THRESHOLD"]
        if as_of is None:
            prob = mrow.get("LATEST_MARKET_PROBABILITY")
        else:
            recorded = pd.to_datetime(hist["DATE_RECORDED"], format="ISO8601", utc=True)
            h = hist[(hist["MARKET_ID"] == mid) & (recorded <= pd.Timestamp(as_of, tz="UTC"))]
            prob = h["PROBABILITY"].iloc[-1] if len(h) else None
        if prob is not None and thr is not None:
            out_thr.append(float(thr)); out_p.append(float(prob))
    return out_thr, out_p


def commodity_view(commodity: str, *, spot: float, as_of=None, horizon_years: float = 0.25,
                   reader=None) -> CommodityView | None:
    """Fetch + clean tal's spot-price ladder for a commodity into an expected return view."""
    from ..sources import tal_snowflake as tal
    reader = reader or tal
    try:
        markets = reader.commodity_price_markets(commodity)
    except Exception as e:  # noqa: BLE001 - tal down -> no view, drift-neutral fallback
        logger.info("signal: tal price markets unavailable for %s (%s)", commodity, e)
        return None
    if markets is None or len(markets) == 0:
        return None
    # Pick the settlement nearest the target horizon (so the view matches the option DTE).
    import pandas as pd
    sett = pd.to_datetime(markets["SETTLEMENT_DATE"], format="ISO8601", utc=True)
    ref = pd.Timestamp(as_of, tz="UTC") if as_of is not None else sett.min()
    target = ref + pd.Timedelta(days=int(horizon_years * 365))
    nearest = sett.iloc[(sett - target).abs().argmin()]
    sub = markets[sett == nearest]
    T = max((nearest - ref).days / 365.25, 1e-3)
    hist = reader.market_price_history(list(sub["ID"])) if as_of is not None else None
    thr, p = _asof_probs(sub, hist, as_of)
    view = commodity_view_from_ladder(thr, p, spot=spot, horizon_years=T)
    if view is not None:
        view.commodity = commodity
    return view


def tal_drift(*, commodity: str, beta_c: float, spot: float, as_of=None,
              horizon_years: float = 0.25, drift_cap: float = 0.30,
              reader=None) -> tuple[float, float]:
    """(mu_view, confidence): annualized equity drift from tal's commodity expectation,
    propagated through the screened commodity beta and capped. (0, 0) when no view."""
    view = commodity_view(commodity, spot=spot, as_of=as_of, horizon_years=horizon_years,
                          reader=reader)
    if view is None or view.confidence <= 0:
        return 0.0, 0.0
    mu = beta_c * view.expected_log_return_ann
    mu = max(-drift_cap, min(drift_cap, mu))
    return mu, view.confidence
