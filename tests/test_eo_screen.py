"""Lead-lag universe screen: recovers a known synthetic lag; stability filter works."""

import numpy as np
import pandas as pd

from quantbots.equity_options.research import screen as sc


def _series(rets, start="2019-01-01"):
    idx = pd.date_range(start, periods=len(rets) + 1, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets])))
    return pd.DataFrame({"Close": close}, index=idx)


def test_screen_recovers_lagged_predictor(monkeypatch):
    rng = np.random.default_rng(0)
    n = 900
    comm = rng.normal(0, 0.02, n)
    mkt = rng.normal(0, 0.01, n)
    eq = np.empty(n)
    eq[:2] = rng.normal(0, 0.01, 2)
    eq[2:] = 0.8 * comm[:-2] + rng.normal(0, 0.005, n - 2)  # commodity LEADS equity by 2d

    from quantbots.research.data_fetch import DEFAULT_UNIVERSE
    hg = DEFAULT_UNIVERSE["COPPER"]
    data = {"TESTEQ": _series(eq), hg: _series(comm), "SPY": _series(mkt)}
    monkeypatch.setattr(sc, "fetch_yf_history", lambda t, **k: data[t])

    res = sc.screen_equity("TESTEQ", commodities=["COPPER"], market="SPY", max_lag=5)
    assert res is not None
    assert res.commodity == "COPPER"
    assert res.lag == 2                       # recovers the true lead
    assert res.beta > 0 and abs(res.tstat) > 2 and res.stable
    assert res.passes


def test_screen_rejects_noise(monkeypatch):
    rng = np.random.default_rng(1)
    n = 900
    data = {"TESTEQ": _series(rng.normal(0, 0.01, n)),
            "HG=F": _series(rng.normal(0, 0.02, n)),
            "SPY": _series(rng.normal(0, 0.01, n))}
    from quantbots.research.data_fetch import DEFAULT_UNIVERSE
    data[DEFAULT_UNIVERSE["COPPER"]] = data.pop("HG=F")
    monkeypatch.setattr(sc, "fetch_yf_history", lambda t, **k: data[t])
    res = sc.screen_equity("TESTEQ", commodities=["COPPER"], market="SPY", max_lag=5)
    # Pure noise: should not clear the |t|>=2 + stability bar.
    assert res is None or not res.passes
