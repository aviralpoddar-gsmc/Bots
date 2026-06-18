"""Edge / selection / sizing / portfolio on a synthetic chain (no network)."""

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from quantbots.equity_options.config import DEFAULT_RISK_LIMITS
from quantbots.equity_options.edge import Leg, evaluate, risk_profile
from quantbots.equity_options.forecast.underlying import Forecast
from quantbots.equity_options.occ import build_occ
from quantbots.equity_options.portfolio import allocate
from quantbots.equity_options.pricing.greeks import bsm_price
from quantbots.equity_options.research.beta import BetaFit
from quantbots.equity_options.selection import select
from quantbots.equity_options.sizing import size_contracts

SPOT = 100.0
R = 0.02
IV_MARKET = 0.20
DTE = 45
T = DTE / 365.25


def _beta():
    return BetaFit(equity="ZZ", commodity="COPPER", alpha=0.0, beta_c=1.0, beta_m=1.0,
                   beta_c_raw=1.0, beta_c_se=0.1, sigma_idio=0.2, r2=0.4, n_obs=500)


def _forecast(vol: float) -> Forecast:
    rng = np.random.default_rng(0)
    var = vol ** 2 * T
    log_st = (R) * T - 0.5 * var + rng.normal(0.0, math.sqrt(var), size=40000)
    terminal = np.sort(SPOT * np.exp(log_st))
    return Forecast(ticker="ZZ", s0=SPOT, T=T, r=R, q=0.0, terminal=terminal,
                    sigma_fp=vol, mode="drift_neutral", beta=_beta(), n_sims=len(terminal))


def _chain(iv=IV_MARKET):
    expiry = (datetime.now(UTC).date() + timedelta(days=DTE))
    rows = []
    for k in (70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130):
        for kind in ("call", "put"):
            mid = bsm_price(SPOT, k, T, R, iv, kind)
            if mid <= 0.05:
                continue
            rows.append({
                "symbol": build_occ("ZZ", expiry, kind, k), "underlying": "ZZ",
                "strike": float(k), "kind": kind, "expiry": expiry, "dte": DTE,
                "bid": mid * 0.99, "ask": mid * 1.01, "mid": mid, "iv": iv,
                "open_interest": 1000,
                "delta": None, "gamma": None, "vega": None, "theta": None,
            })
    return rows


def test_risk_profile_credit_vertical():
    # Bull put spread: sell 95 put @3, buy 90 put @1.5 -> credit 1.5, width 5.
    legs = [Leg(95, "put", -1, 3.0), Leg(90, "put", +1, 1.5)]
    res = evaluate(np.array([80.0, 92.0, 100.0]), legs, r=0.0, T=0.1)
    assert res.is_credit and res.cost == pytest.approx(-1.5)
    mp, ml = risk_profile(legs, res.cost)
    assert mp == pytest.approx(1.5)      # keep the credit if both puts expire OTM
    assert ml == pytest.approx(3.5)      # width - credit


def test_credit_structures_selected_when_iv_is_rich():
    # Market IV 0.45, but our forecast vol is only 0.18 -> options look EXPENSIVE, so
    # SELLING premium (credit) is favorable. Expect credit candidates with positive edge.
    cands = select("ZZ", _chain(iv=0.45), lambda T: _forecast(0.18), r=R, q=0.0,
                   limits=DEFAULT_RISK_LIMITS,
                   structures=["bull_put_spread", "bear_call_spread", "iron_condor"])
    assert cands, "expected credit candidates when implied vol >> our forecast vol"
    top = cands[0]
    assert top.edge.is_credit and top.edge.edge > 0
    assert top.premium > 0 and top.cost_per_share < 0     # max-loss sizing; credit cost
    # defined-risk: max loss is bounded by the spread width * 100
    assert top.premium <= 30 * 100


def test_vertical_payoff_and_cost():
    # Bull call spread 95/105: long 95 call, short 105 call. Max payoff = width = 10.
    terminal = np.array([90.0, 100.0, 120.0])
    legs = [Leg(95, "call", +1, 6.0), Leg(105, "call", -1, 2.0)]
    res = evaluate(terminal, legs, r=0.0, T=0.1)
    assert res.cost == pytest.approx(4.0)            # 6 - 2 debit
    # payoffs: at 90 ->0, at 100 ->5, at 120 ->10 (capped at width); mean = 5
    assert res.ev_payoff == pytest.approx(5.0)


def test_higher_forecast_vol_creates_positive_edge():
    # Our vol (0.40) >> market IV (0.20): options are cheap, so buying earns edge.
    cands = select("ZZ", _chain(), lambda T: _forecast(0.40), r=R, q=0.0,
                   limits=DEFAULT_RISK_LIMITS, structures=["long_call", "long_put",
                                                           "bull_call_spread"])
    assert cands, "expected tradable candidates when fp vol >> implied vol"
    assert cands[0].edge.edge > 0
    assert cands[0].edge.score > 0


def test_no_edge_when_vols_match():
    # Our vol == market IV and drift-neutral: edge should be ~0, nothing passes the hurdle.
    cands = select("ZZ", _chain(), lambda T: _forecast(0.20), r=R, q=0.0,
                   limits=DEFAULT_RISK_LIMITS, structures=["long_call", "long_put"])
    assert all(c.edge.edge <= DEFAULT_RISK_LIMITS["edge_hurdle"] * c.cost_per_share or
               c.edge.edge < 0.5 for c in cands)


def test_sizing_and_allocation_caps():
    cands = select("ZZ", _chain(), lambda T: _forecast(0.45), r=R, q=0.0,
                   limits=DEFAULT_RISK_LIMITS, structures=["long_call", "long_put"])
    assert cands
    n = size_contracts(cands[0], bankroll=100_000.0, limits=DEFAULT_RISK_LIMITS)
    assert n >= 1
    # premium per contract must respect the per-trade cap.
    assert n * cands[0].premium <= DEFAULT_RISK_LIMITS["max_premium_per_trade"] + cands[0].premium
    allocs = allocate(cands, bankroll=100_000.0, limits=DEFAULT_RISK_LIMITS,
                      commodity_of={"ZZ": "COPPER"})
    assert len(allocs) <= 1                           # one structure per underlying
    total = sum(a.premium_total for a in allocs)
    assert total <= DEFAULT_RISK_LIMITS["max_total_premium"]
