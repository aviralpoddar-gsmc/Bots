"""Tests for the single-source, price-anchored bots + processing signals."""

import time

import pytest

from quantbots.processing import signals as sig
from quantbots.sources import cftc
from quantbots.strategies import get_strategy


class Obs:
    def __init__(self, v):
        self.v = v

    def latest_observation(self, entity, source=None):
        return self.v.get(entity)


def mkt(q, yrs=0.7, mid=None):
    return {"id": mid or q, "question": q, "probability": 0.5, "isResolved": False,
            "closeTime": time.time() * 1000 + yrs * 365.25 * 86400 * 1000, "totalLiquidity": 200}


COTTON = "Will ICE Cotton No. 2 front-month futures exceed 70 cents/lb on Mar 31, 2027?"
COCOA = "Will ICE cocoa (NY) nearest-futures price exceed 4500 USD/t on Dec 31, 2027?"


# --- base: matching + meaningful gate ---------------------------------------

def test_excludes_basis_and_requires_price():
    s = get_strategy("cftc_positioning")
    s.bind(Obs({"CME_COTTON": {"value": 76.7}, "SIG_COTTON_CFTC": {"value": 3.0, "payload": {}}}))
    kept = s.prefilter([mkt(COTTON),
                        mkt("Will the cotton calendar spread exceed 2 cents/lb?"),
                        mkt("Will US cotton abandonment exceed 40%?")])  # no price word
    assert [m["question"] for m in kept] == [COTTON]


def test_abstains_without_signal():
    s = get_strategy("cftc_positioning")
    s.bind(Obs({"CME_COTTON": {"value": 76.7}}))  # no SIG_*
    assert s.estimate([mkt(COTTON, mid="x")]) == {}


def test_abstains_when_signal_weak():
    s = get_strategy("cftc_positioning", min_z=1.0)
    s.bind(Obs({"CME_COTTON": {"value": 76.7}, "SIG_COTTON_CFTC": {"value": 0.4, "payload": {}}}))
    assert s.estimate([mkt(COTTON, mid="x")]) == {}


def test_min_drift_gate():
    # tiny drift below min_drift -> abstain even though a signal exists
    s = get_strategy("cftc_positioning", k=0.0001, min_z=0.0, min_drift=0.01)
    s.bind(Obs({"CME_COTTON": {"value": 76.7}, "SIG_COTTON_CFTC": {"value": 1.0, "payload": {}}}))
    assert s.estimate([mkt(COTTON, mid="x")]) == {}


# --- cftc: direction (fade the crowd) ---------------------------------------

def test_cftc_fades_crowded_long():
    s = get_strategy("cftc_positioning")
    base = {"CME_COTTON": {"value": 76.7}}
    s.bind(Obs({**base, "SIG_COTTON_CFTC": {"value": 2.5, "payload": {"netpct": 0.3}}}))
    p_long = s.estimate([mkt(COTTON, mid="a")])["a"]
    s.bind(Obs({**base, "SIG_COTTON_CFTC": {"value": -2.5, "payload": {"netpct": -0.3}}}))
    p_short = s.estimate([mkt(COTTON, mid="b")])["b"]
    # crowded long -> bearish -> lower P(exceed) than crowded short
    assert p_long < p_short


def test_cftc_covers_cotton_and_cocoa():
    s = get_strategy("cftc_positioning")
    s.bind(Obs({"CME_COCOA": {"value": 3894.0}, "SIG_COCOA_CFTC": {"value": 2.0, "payload": {}}}))
    assert s.estimate([mkt(COCOA, mid="c")])  # cocoa priced


# --- fas: drifts to fundamental fair value ----------------------------------

def test_fas_drifts_toward_fair():
    # lower fundamental fair value -> more bearish -> lower P(exceed). Robust to the
    # drift cap; asserts the direction/monotonicity of the FAS drift.
    s = get_strategy("fas_fundamental", reversion_rate=0.5)
    s.bind(Obs({"CME_COTTON": {"value": 76.7}, "SIG_COTTON_FAS": {"value": 60.0, "payload": {}}}))
    p_low = s.estimate([mkt(COTTON, mid="a")])["a"]
    s.bind(Obs({"CME_COTTON": {"value": 76.7}, "SIG_COTTON_FAS": {"value": 95.0, "payload": {}}}))
    p_high = s.estimate([mkt(COTTON, mid="b")])["b"]
    assert p_low < p_high


# --- processing helpers -----------------------------------------------------

def test_obs_payload_parses_json_string():
    # the store returns payload as a JSON string (TEXT column) — must be parsed
    from quantbots.strategies._signal_base import obs_payload
    assert obs_payload({"payload": '{"sur": 0.44}'}) == {"sur": 0.44}
    assert obs_payload({"payload": None}) == {}
    assert obs_payload(None) == {}


def test_signal_drift_handles_string_payload():
    s = get_strategy("cftc_positioning")
    s.bind(Obs({"CME_COTTON": {"value": 76.7},
                "SIG_COTTON_CFTC": {"value": 2.0, "payload": '{"netpct": 0.25}'}}))  # JSON string
    assert s.estimate([mkt(COTTON, mid="x")])  # must not raise


def test_z_helper():
    z, mean, std = sig._z([1, 2, 3, 4, 5], 5)
    assert mean == 3 and z > 0


def test_cftc_net_pct_parsing():
    row = {"open_interest_all": "1000", "m_money_positions_long_all": "300",
           "m_money_positions_short_all": "100", "report_date_as_yyyy_mm_dd": "2026-05-26T00:00:00"}
    date, net, netpct, oi = cftc._net_pct(row)
    assert net == 200 and netpct == pytest.approx(0.2) and date == "2026-05-26"


def test_cftc_net_pct_rejects_zero_oi():
    assert cftc._net_pct({"open_interest_all": "0", "m_money_positions_long_all": "1",
                          "m_money_positions_short_all": "0",
                          "report_date_as_yyyy_mm_dd": "2026-05-26T00:00:00"}) is None
