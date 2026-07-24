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


def test_proprietary_specialty_price_discounted_below_generic_price():
    # Proprietary China-domestic chemical/minor-metal prices (no public free
    # settlement) resolve less than a generic price market but more than inventory.
    zoc = r("Will China ZOC zirconium oxychloride price exceed 4000 USD/t on Sept 30?")
    sponge = r("Will China ex-works zirconium sponge spot price exceed 22 USD/kg on June 30?")
    generic_price = r("Will copper spot price exceed $12,900 USD/MT on June 30?")
    inventory = r("Will global zircon inventory exceed 60 days of supply on June 30?")
    assert inventory < zoc < generic_price
    assert inventory < sponge < generic_price


def test_proprietary_discount_yields_to_exchange_benchmark():
    # If an exchange settles it, the proprietary discount is overridden.
    assert r("Will LME-settled tin sponge spot price exceed 22 USD/kg on June 30?") > 0.3


def test_public_zircon_legs_keep_full_price_score():
    # Iluka's published quarterly price and zircon sand CIF China are the resolvable
    # legs — they must NOT be caught by the proprietary discount.
    assert r("Will Iluka's zircon price for the quarter ending December 2026 exceed 1550 USD/t?") >= 0.20
    assert r("Will zircon sand price CIF China exceed 1800 USD/t on December 31, 2026?") >= 0.20


def test_score_bounds():
    assert 0.01 <= r("anything at all") <= 0.99
    assert 0.01 <= r("") <= 0.99
