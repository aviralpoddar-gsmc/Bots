"""BSM pricing / greeks / IV inversion."""

import math

import pytest

from quantbots.equity_options.pricing.greeks import (
    bsm_price,
    greeks,
    implied_vol,
)


def test_atm_call_known_value():
    # S=K=100, t=1, r=0, sigma=0.2 -> Black-Scholes call ~= 7.9656.
    c = bsm_price(100, 100, 1.0, 0.0, 0.2, "call")
    assert c == pytest.approx(7.9656, abs=1e-3)


def test_put_call_parity():
    s, k, t, r, q, sigma = 100, 110, 0.75, 0.03, 0.01, 0.25
    c = bsm_price(s, k, t, r, sigma, "call", q)
    p = bsm_price(s, k, t, r, sigma, "put", q)
    lhs = c - p
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_iv_round_trip():
    for sigma in (0.1, 0.2, 0.5, 0.8):
        price = bsm_price(100, 105, 0.5, 0.02, sigma, "call")
        iv = implied_vol(price, 100, 105, 0.5, 0.02, "call")
        assert iv == pytest.approx(sigma, abs=1e-4)


def test_iv_none_below_intrinsic():
    # A price below discounted intrinsic has no real IV.
    assert implied_vol(0.01, 200, 100, 0.5, 0.0, "call") is None


def test_greek_signs():
    g_call = greeks(100, 100, 0.5, 0.02, 0.2, "call")
    g_put = greeks(100, 100, 0.5, 0.02, 0.2, "put")
    assert 0 < g_call["delta"] < 1
    assert -1 < g_put["delta"] < 0
    assert g_call["gamma"] > 0 and g_call["vega"] > 0
    assert g_call["theta"] < 0  # long option decays


def test_expiry_intrinsic():
    assert bsm_price(120, 100, 0.0, 0.0, 0.2, "call") == pytest.approx(20.0)
    assert bsm_price(80, 100, 0.0, 0.0, 0.2, "call") == pytest.approx(0.0)
