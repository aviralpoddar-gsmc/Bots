"""tal directional signal: ladder→expected-return fit, and beta propagation/cap."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from quantbots.equity_options.forecast import signal as sig


def _ladder(median: float, sigma: float, strikes):
    """exceed-probs for a lognormal with the given median price."""
    mu = math.log(median)
    return [float(1.0 - norm.cdf((math.log(k) - mu) / sigma)) for k in strikes]


def test_ladder_recovers_bullish_view():
    spot = 100.0
    strikes = [80, 90, 100, 110, 120, 130]
    probs = _ladder(110.0, 0.20, strikes)          # implied median 110 > spot 100
    v = sig.commodity_view_from_ladder(strikes, probs, spot=spot, horizon_years=1.0)
    assert v is not None
    assert v.expected_log_return_ann == pytest.approx(math.log(110 / 100), abs=0.03)
    assert v.n_thresholds >= 4 and v.confidence > 0


def test_ladder_bearish_view():
    strikes = [80, 90, 100, 110, 120]
    probs = _ladder(90.0, 0.20, strikes)           # implied median below spot
    v = sig.commodity_view_from_ladder(strikes, probs, spot=100.0, horizon_years=1.0)
    assert v is not None and v.expected_log_return_ann < 0


def test_ladder_too_thin_is_none():
    # After dropping stale 0/1 quotes, < 3 thresholds remain.
    assert sig.commodity_view_from_ladder([100, 110], [0.5, 0.4], spot=100, horizon_years=1.0) is None
    assert sig.commodity_view_from_ladder([80, 90, 100], [1.0, 1.0, 1.0], spot=100,
                                          horizon_years=1.0) is None   # all stale


def test_tal_drift_propagates_and_caps(monkeypatch):
    view = sig.CommodityView(commodity="COPPER", expected_log_return_ann=0.20,
                             confidence=0.8, n_thresholds=5, horizon_years=0.25)
    monkeypatch.setattr(sig, "commodity_view", lambda *a, **k: view)
    mu, conf = sig.tal_drift(commodity="COPPER", beta_c=1.5, spot=50.0, drift_cap=0.30)
    assert mu == pytest.approx(0.30)               # 1.5*0.20=0.30 at the cap
    assert conf == 0.8
    mu_neg, _ = sig.tal_drift(commodity="COPPER", beta_c=-2.0, spot=50.0, drift_cap=0.30)
    assert mu_neg == pytest.approx(-0.30)          # negative beta -> negative, capped


def test_tal_drift_no_view(monkeypatch):
    monkeypatch.setattr(sig, "commodity_view", lambda *a, **k: None)
    assert sig.tal_drift(commodity="X", beta_c=1.0, spot=10.0) == (0.0, 0.0)
