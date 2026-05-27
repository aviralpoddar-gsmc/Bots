"""Tests for commodity_spot: the strict matcher/unit-guard (what keeps the bot out
of confidently-wrong bets) and the lognormal direction logic."""

import time

import pytest

from quantbots.strategies.commodity_spot import CommoditySpotStrategy


class Obs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


def _spec(q):
    sp = CommoditySpotStrategy()._spec(q)
    return sp[0] if sp else None


@pytest.mark.parametrize("question,entity", [
    ("Will Gold spot price exceed $5,181/ozt on May 31, 2026?", "GOLD"),
    ("Will LBMA silver spot price exceed 100 USD per troy oz on May 31?", "SILVER"),
    ("Will Platinum spot price (LBMA AM fix) exceed 700 USD per troy oz on May 31?", "PLATINUM"),
    ("Will Palladium spot price exceed 800 USD per troy oz on June 30?", "PALLADIUM"),
    ("Will the copper spot price exceed $12,900 USD/MT on Feb 28?", "COPPER"),
    ("Will Brent crude oil spot price exceed $130/barrel on June 30?", "BRENT_OIL"),
    ("Will WTI crude oil spot price exceed $70/barrel on Dec 31?", "WTI_OIL"),
    ("Will US RBOB gasoline spot price exceed $4/gallon on Dec 31?", "GASOLINE"),
])
def test_matches_genuine_spot_price_markets(question, entity):
    assert _spec(question) == entity


@pytest.mark.parametrize("question", [
    # Operational metrics that merely mention the commodity — not its price.
    "Will global gold dental alloy demand exceed 75 tonnes for the year ending 2026?",
    "Will Equinox Gold AISC exceed 1800 USD/oz for the year ending 2026?",
    "Will copper production exceed 2000 kt on June 30, 2026?",
    # Unit / currency / compound traps.
    "Will Palladium spot price exceed 800 koz Pd on June 30?",        # koz = volume
    "Will LME nickel spot price exceed 13000 CNY per metric ton on July 31?",  # yuan
    "Will zinc sulfate spot price exceed $800/t on June 30, 2027?",   # chemical, not metal
    "Will copper sulfate price exceed $5000/t on June 30?",           # chemical
    "Will natural gas spot price exceed 40 EUR/MWh on June 30?",      # European gas, wrong benchmark
])
def test_rejects_wrong_metric_unit_or_currency(question):
    assert _spec(question) is None


def _market(question, close_years=0.5):
    return {
        "id": question[:24], "question": question, "probability": 0.5,
        "closeTime": time.time() * 1000 + close_years * 365.25 * 24 * 3600 * 1000,
        "totalLiquidity": 100, "isResolved": False,
    }


def test_direction_and_units_silver_cents_to_dollars():
    # Feed si.f is cents/oz (7515 -> $75.15/oz). Threshold $100/oz is above spot.
    s = CommoditySpotStrategy()
    s.bind(Obs({"SILVER": 7515.3}))
    m = _market("Will LBMA silver spot price exceed 100 USD per troy oz on Dec 31?")
    p = s.estimate([m])[m["id"]]
    assert 0.0 < p < 0.5  # spot below threshold -> exceeding is less likely than even


def test_copper_mt_conversion_direction():
    # Feed hg.f cents/lb (636.73 -> ~$14,037/MT). Threshold $12,900/MT is below spot.
    s = CommoditySpotStrategy()
    s.bind(Obs({"COPPER": 636.73}))
    m = _market("Will the copper spot price exceed $12,900 USD/MT on Dec 31?")
    p = s.estimate([m])[m["id"]]
    assert p > 0.5  # spot already above threshold -> likely to still exceed


def test_horizon_cap_excludes_far_dated():
    s = CommoditySpotStrategy(max_horizon_years=1.25)
    near = _market("Will Gold spot price exceed $5,181/ozt soon?", close_years=0.5)
    far = _market("Will Gold spot price exceed $5,181/ozt much later in 2030?", close_years=4.0)
    near["id"], far["id"] = "near", "far"
    kept = s.prefilter([near, far])
    ids = {m["id"] for m in kept}
    assert "near" in ids and "far" not in ids


def test_correlation_key_groups_by_commodity():
    s = CommoditySpotStrategy()
    k1 = s.correlation_key(_market("Will Gold spot price exceed $5,181/ozt on Dec 31?"))
    k2 = s.correlation_key(_market("Will Gold spot price exceed $6,000/ozt on Jun 30?"))
    assert k1 == k2 == "GOLD"
