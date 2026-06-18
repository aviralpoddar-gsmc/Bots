"""Backtest gate: pure helpers, no-lookahead slicing, and a mocked orchestrator run."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantbots.equity_options import backtest as bt
from quantbots.equity_options.config import (
    DEFAULT_RISK_LIMITS,
    EquityOptionsConfig,
    Underlying,
)
from quantbots.equity_options.forecast.underlying import Forecast
from quantbots.equity_options.pricing.greeks import bsm_price
from quantbots.equity_options.research.beta import BetaFit


# --- pure helpers ------------------------------------------------------------

def test_third_friday_known():
    assert bt.third_friday(2024, 1) == date(2024, 1, 19)
    assert bt.third_friday(2026, 9) == date(2026, 9, 18)


def test_target_expiry_is_a_third_friday_near_horizon():
    e = bt.target_expiry(date(2024, 4, 1), 90)
    assert e == bt.third_friday(e.year, e.month)
    assert abs((e - date(2024, 6, 30)).days) <= 31


def test_strike_grid_spans_band():
    g = bt.strike_grid(100.0)
    assert min(g) <= 60 and max(g) >= 140
    assert all(b > a for a, b in zip(g, g[1:]))   # strictly increasing


def test_implied_prob_above_monotonic():
    probs = [bt.implied_prob_above(100, k, 0.25, 0.02, 0.3) for k in (80, 100, 120)]
    assert probs[0] > probs[1] > probs[2]
    assert bt.implied_prob_above(100, 10, 0.25, 0.02, 0.3) > 0.99   # deep ITM call


def test_structure_pnl():
    # Long call strike 100, paid 5: terminal 120 -> (20-5)*100 = 1500.
    assert bt._structure_pnl([{"strike": 100, "kind": "call", "qty": 1}], 5.0, 120) == 1500
    # Bear put spread long 100 / short 90, paid 4: terminal 85 -> (10-4)*100 = 600.
    legs = [{"strike": 100, "kind": "put", "qty": 1}, {"strike": 90, "kind": "put", "qty": -1}]
    assert bt._structure_pnl(legs, 4.0, 85) == 600


# --- no-lookahead ------------------------------------------------------------

def test_close_asof_does_not_peek(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": list(range(10))}, index=idx)  # 0..9 by date
    from quantbots.research import data_fetch
    monkeypatch.setattr(data_fetch, "fetch_yf_history", lambda *a, **k: df)
    # As of day index 4 (2024-01-05) the close must be 4, never a later value.
    assert bt.close_asof("X", "2024-01-05") == 4.0


# --- gate logic --------------------------------------------------------------

def test_gate_pass_and_fail():
    good = bt.BacktestResult(underlying="X")
    # 12 trades: forecast nails outcomes (skill high), implied uninformative, PnL steady.
    good.outcomes = [1, 0] * 6
    good.forecast_probs = [0.9, 0.1] * 6
    good.implied_probs = [0.5] * 12
    good.pnls = [100, 120, 90, 110, 130, 105, 95, 115, 100, 120, 90, 110]
    good.folds = 12
    assert good.gate()[0]   # strong skill + Sharpe + >=12 trades

    bad = bt.BacktestResult(underlying="Y")
    bad.outcomes = [1, 0] * 6
    bad.forecast_probs = [0.1, 0.9] * 6                        # anti-correlated -> neg skill
    bad.implied_probs = [0.5] * 12
    bad.pnls = [-100, -120, 50, -90, -110, -130, 40, -80, -100, 30, -90, -120]
    bad.folds = 12
    assert not bad.gate()[0]

    assert bt.BacktestResult(underlying="Z").gate()[0] is False   # no data


def test_gate_marginal_pass_is_fail():
    """A barely-positive result (the original >0 trap) must NOT pass the strict gate."""
    r = bt.BacktestResult(underlying="X")
    r.outcomes = [1, 0] * 8
    r.forecast_probs = [0.55, 0.45] * 8     # tiny skill
    r.implied_probs = [0.5] * 16
    r.pnls = [10, -8, 9, -7, 12, -9, 11, -8, 10, -7, 9, -8, 11, -9, 10, -7]  # ~noise Sharpe
    passed, reason = r.gate()
    assert not passed and "need >=" in reason


def test_gate_needs_minimum_trades():
    r = bt.BacktestResult(underlying="X")
    r.outcomes = [1, 0]
    r.forecast_probs = [0.9, 0.1]
    r.implied_probs = [0.5, 0.5]
    r.pnls = [100, 100]
    assert r.gate()[0] is False   # < 12 trades


def test_gate_persistence_and_freshness(monkeypatch, tmp_path):
    gate_file = tmp_path / "gate.json"
    monkeypatch.setattr(bt, "GATE_FILE", gate_file)
    bt.save_gate_results({"FCX": {"passed": True}, "NEM": {"passed": False}})
    assert bt.passing_tickers(max_age_days=14) == {"FCX"}
    # A zero-day freshness window makes any saved result stale -> nothing passes.
    assert bt.passing_tickers(max_age_days=0) == set()


# --- mocked orchestrator -----------------------------------------------------

class _FakeChain:
    def __init__(self, r=0.02, iv=0.30):
        self.r, self.iv = r, iv

    def historical_chain(self, underlying, *, as_of, expiry, spot, r, half_spread=0.04):
        T = (expiry - as_of).days / 365.25
        rows = []
        for k in (40, 44, 48, 52, 56):
            for kind in ("call", "put"):
                mid = bsm_price(spot, k, T, self.r, self.iv, kind)
                if mid <= 0.05:
                    continue
                rows.append({"symbol": f"{underlying}{kind[0]}{k}", "underlying": underlying,
                             "strike": float(k), "kind": kind, "expiry": expiry,
                             "dte": (expiry - as_of).days, "bid": mid * .98, "ask": mid * 1.02,
                             "mid": mid, "iv": self.iv, "open_interest": None,
                             "delta": None, "gamma": None, "vega": None, "theta": None})
        return rows


def test_run_backtest_mocked(monkeypatch):
    cfg = EquityOptionsConfig(
        underlyings=[Underlying("FCX", "COPPER", "SPY")],
        risk_limits=dict(DEFAULT_RISK_LIMITS), structures=["long_put", "bear_put_spread"],
        forecast={})

    monkeypatch.setattr(bt, "close_asof", lambda t, iso: 48.0)  # spot == realized == 48

    def _fake_forecast(**kw):
        rng = np.random.default_rng(0)
        T = kw["T"]
        var = 0.4 ** 2 * T
        term = np.sort(48.0 * np.exp(0.02 * T - 0.5 * var + rng.normal(0, var ** 0.5, 30000)))
        beta = BetaFit("FCX", "COPPER", 0, 1, 1, 1, 0.1, 0.2, 0.4, 500)
        return Forecast("FCX", 48.0, T, 0.02, 0.0, term, 0.4, "drift_neutral", beta, len(term))

    from quantbots.equity_options.forecast import underlying as fu
    from quantbots.equity_options.sources import underlying as und_src
    monkeypatch.setattr(fu, "build_forecast", _fake_forecast)
    monkeypatch.setattr(und_src, "risk_free_rate", lambda **k: 0.02)

    res = bt.run_backtest(cfg, "FCX", as_of_dates=[date(2024, 4, 1), date(2024, 5, 1)],
                          horizon_days=90, chain_client=_FakeChain())
    assert res.folds >= 1
    assert res.n_trades >= 1
    assert len(res.outcomes) > 0 and len(res.crps) >= 1
    assert isinstance(res.gate(), tuple)
