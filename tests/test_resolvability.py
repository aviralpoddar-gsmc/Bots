"""Tests for the cancellation-aware resolvability scorer — the ordering must match
the observed decided-rates (price >> production/demand; LBMA precious ~certain)."""

from quantbots.resolvability import resolvability_score as r


def test_operational_metrics_score_near_zero():
    # Production / demand / capacity almost never resolve (0-1% observed).
    assert r("Will Glencore cobalt production exceed 40 kt on June 30, 2026?") < 0.05
    assert r("Will global Zn galvanizing demand exceed 9000kt for the year?") < 0.05
    assert r("Will China lithium refinery capacity utilization exceed 80%?") < 0.05


def test_price_scores_above_operational():
    assert r("Will copper spot price exceed $12,900 USD/MT on June 30?") > 0.15
    assert r("Will copper spot price exceed $12,900 USD/MT?") > \
           r("Will copper production exceed 2000 kt?")


def test_lbma_precious_metals_score_near_certain():
    # LBMA-sourced precious metals resolved 100% in the data.
    assert r("Will Gold spot price (LBMA AM fix) exceed $5,181/ozt on May 31?") > 0.9
    assert r("Will Platinum spot price exceed 700 USD per troy oz?") > 0.8


def test_strong_exchange_benchmark_boosts():
    assert r("Will LME tin spot price exceed 54500 USD per tonne on April 30?") > 0.3
    # ...above an unspecified niche spot price.
    assert r("Will LME tin spot price exceed 54500 USD/t?") > \
           r("Will spodumene concentrate spot price, China exceed $800/t?")


def test_spread_and_inventory_between_price_and_operational():
    spread = r("Will LME Zn cash-3M spread exceed 150 USD/t on June 30?")
    prod = r("Will Vale nickel production exceed 105 kt?")
    price = r("Will silver spot price exceed 60 USD/ozt?")
    assert prod < spread < price


def test_score_bounds():
    assert 0.01 <= r("anything at all") <= 0.99
    assert 0.01 <= r("") <= 0.99
