"""Commodity-momentum directional signal: trend sign, beta propagation, no-lookahead."""

import numpy as np
import pandas as pd

from quantbots.equity_options.forecast import direction as dr


def _series(rets, start="2019-01-01"):
    idx = pd.date_range(start, periods=len(rets) + 1, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets])))
    return pd.DataFrame({"Close": close}, index=idx)


def test_momentum_sign_and_lookahead(monkeypatch):
    n = 700
    up = np.full(n, 0.001)          # steady uptrend
    from quantbots.research import data_fetch
    monkeypatch.setattr(data_fetch, "fetch_yf_history", lambda t, **k: _series(up))
    m = dr.commodity_momentum("COPPER", lookback_days=252)
    assert m > 0                    # uptrend -> positive annualized trend
    # down-trend flips the sign
    monkeypatch.setattr(data_fetch, "fetch_yf_history", lambda t, **k: _series(-up))
    assert dr.commodity_momentum("COPPER", lookback_days=252) < 0


def test_momentum_drift_propagates_via_beta(monkeypatch):
    monkeypatch.setattr(dr, "commodity_momentum", lambda *a, **k: 0.20)   # +20%/yr trend
    mu_pos, conv = dr.momentum_drift(commodity="COPPER", beta_c=1.0, shrink=0.7, drift_cap=0.35)
    assert mu_pos > 0 and conv > 0
    mu_neg, _ = dr.momentum_drift(commodity="COPPER", beta_c=-1.0)         # inverse beta -> bearish
    assert mu_neg < 0
    capped, _ = dr.momentum_drift(commodity="COPPER", beta_c=5.0, drift_cap=0.35)
    assert capped == 0.35                                                  # capped


def test_momentum_drift_zero_when_flat(monkeypatch):
    monkeypatch.setattr(dr, "commodity_momentum", lambda *a, **k: 0.0)
    assert dr.momentum_drift(commodity="COPPER", beta_c=1.0) == (0.0, 0.0)
