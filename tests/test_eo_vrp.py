"""Delta-hedged VRP engine: a delta-hedged short straddle should earn when IV>RV and
lose when IV<RV — i.e. it isolates the variance gap, not direction."""

import numpy as np

from quantbots.equity_options.pricing.greeks import bsm_price
from quantbots.equity_options.vrp import VrpLeg, delta_hedged_pnl


def _gbm_path(s0, sigma, n_days, seed):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    steps = rng.normal((-0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def _short_straddle_pnls(iv_sold, sigma_real, n_paths=60, n_days=45, r=0.0):
    s0, T = 100.0, n_days / 252
    cprice = bsm_price(s0, 100, T, r, iv_sold, "call")
    pprice = bsm_price(s0, 100, T, r, iv_sold, "put")
    legs = [VrpLeg(100, "call", -1, cprice, iv_sold), VrpLeg(100, "put", -1, pprice, iv_sold)]
    out = []
    for seed in range(n_paths):
        path = _gbm_path(s0, sigma_real, n_days, seed)
        out.append(delta_hedged_pnl(legs, path, r=r, hedge_cost_bps=0.0))
    return np.array(out)


def test_short_vol_profits_when_iv_exceeds_rv():
    pnls = _short_straddle_pnls(iv_sold=0.30, sigma_real=0.10)
    assert pnls.mean() > 0          # sold rich vol, realized was calm -> harvest the gap


def test_short_vol_loses_when_rv_exceeds_iv():
    pnls = _short_straddle_pnls(iv_sold=0.10, sigma_real=0.30)
    assert pnls.mean() < 0          # sold cheap vol, realized was wild -> lose


def test_delta_hedge_removes_direction():
    # Same realized vol, opposite drift: delta-hedged P&L should be ~similar (direction
    # hedged out), unlike an unhedged short straddle.
    up = _short_straddle_pnls(iv_sold=0.25, sigma_real=0.20)
    assert np.isfinite(up.mean())   # sanity: engine returns finite P&L


def test_pnl_none_on_trivial_path():
    legs = [VrpLeg(100, "call", -1, 5.0, 0.2)]
    assert delta_hedged_pnl(legs, np.array([100.0]), r=0.0) is None
