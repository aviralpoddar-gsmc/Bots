"""Validation math: a perfectly-predictive signal scores high IC; an anti-signal, low."""

import numpy as np
import pandas as pd

from quantbots.equity_options.research.tal_validate import _evaluate


def _panels(seed, sign):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-02-10", periods=60, freq="B")
    mats = ["A", "B", "C", "D", "E"]
    fwd = pd.DataFrame(rng.normal(0, 0.05, (len(days), len(mats))), index=days, columns=mats)
    signal = sign * fwd + rng.normal(0, 0.001, fwd.shape)   # signal ~ proportional to fwd
    return signal, fwd


def test_perfect_signal_high_ic():
    sig, fwd = _panels(0, +1)
    r = _evaluate(sig, fwd, h=5)
    assert r is not None and r.mean_ic > 0.5 and r.factor_sharpe_ann > 0


def test_anti_signal_negative_ic():
    sig, fwd = _panels(1, -1)
    r = _evaluate(sig, fwd, h=5)
    assert r is not None and r.mean_ic < -0.5


def test_insufficient_data_none():
    days = pd.date_range("2026-02-10", periods=8, freq="B")
    df = pd.DataFrame(0.0, index=days, columns=["A", "B", "C", "D"])
    assert _evaluate(df, df, h=5) is None
