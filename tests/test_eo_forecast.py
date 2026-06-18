"""f_P construction: the diffusion_mc reuse seam + drift-neutral centering.

Exercises the REAL parent diffusion bootstrap (via set_returns) by feeding a
synthetic price series through `research.data_fetch.fetch_yf_history` — no network.
"""

import math

import numpy as np
import pandas as pd
import pytest

from quantbots.equity_options.forecast import underlying as fu
from quantbots.equity_options.research.beta import BetaFit


def _synthetic_history(n=800, sigma_daily=0.02, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma_daily, n)
    close = 50.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=idx)


def _beta():
    return BetaFit(equity="ZZ", commodity="COPPER", alpha=0.0, beta_c=1.2, beta_m=0.8,
                   beta_c_raw=1.2, beta_c_se=0.1, sigma_idio=0.2, r2=0.5, n_obs=700)


def test_drift_neutral_forecast_is_martingale(monkeypatch):
    from quantbots.research import data_fetch
    monkeypatch.setattr(data_fetch, "fetch_yf_history", lambda *a, **k: _synthetic_history())

    s0, T, r = 100.0, 0.25, 0.02
    fc = fu.build_forecast(ticker="ZZ", commodity="COPPER", market="SPY", s0=s0, T=T,
                           r=r, q=0.0, mode="drift_neutral", n_sims=20000, beta=_beta())
    assert fc is not None
    # Drift-neutral => E[S_T] ~= S0 * e^{(r-q)T} (martingale), within MC error.
    assert float(np.mean(fc.terminal)) == pytest.approx(s0 * math.exp(r * T), rel=0.03)
    assert fc.sigma_fp > 0
    assert fc.terminal[0] < fc.terminal[-1]   # sorted ascending


def test_trailing_drift_no_lookahead(monkeypatch):
    # A series that is flat then ramps up: drift as-of the flat era must be ~0, not the ramp.
    n = 600
    flat = np.full(300, 50.0)
    ramp = 50.0 * np.exp(np.linspace(0, 0.5, 300))
    close = np.concatenate([flat, ramp])
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": close}, index=idx)
    from quantbots.research import data_fetch
    monkeypatch.setattr(data_fetch, "fetch_yf_history", lambda *a, **k: df)
    early = fu._trailing_drift("X", as_of=idx[280])     # still in the flat era
    late = fu._trailing_drift("X", as_of=idx[-1])       # after the ramp
    assert abs(early) < 0.05
    assert late > early                                  # only the late window sees the trend


def test_directional_shifts_center_up(monkeypatch):
    from quantbots.research import data_fetch
    monkeypatch.setattr(data_fetch, "fetch_yf_history", lambda *a, **k: _synthetic_history())
    monkeypatch.setattr(fu, "_trailing_drift", lambda *a, **k: 0.20)   # +20%/yr trend
    kw = dict(ticker="ZZ", commodity="COPPER", market="SPY", s0=100.0, T=0.25, r=0.02,
              n_sims=20000, beta=_beta())
    neutral = fu.build_forecast(mode="drift_neutral", **kw)
    direct = fu.build_forecast(mode="directional", **kw)
    assert float(np.mean(direct.terminal)) > float(np.mean(neutral.terminal))


def test_weak_beta_abstains():
    weak = BetaFit(equity="ZZ", commodity="COPPER", alpha=0.0, beta_c=0.0, beta_m=0.0,
                   beta_c_raw=0.05, beta_c_se=0.2, sigma_idio=0.2, r2=0.001, n_obs=700)
    assert weak.weak
    fc = fu.build_forecast(ticker="ZZ", commodity="COPPER", market="SPY", s0=100.0,
                           T=0.25, r=0.02, beta=weak)
    assert fc is None
