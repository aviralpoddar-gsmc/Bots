"""Tests for term_structure: date ordinals, grouping by metric+threshold,
informative-anchored time smoothing, and abstention guards."""

import time

from quantbots.strategies.term_structure import (
    TermStructureStrategy,
    date_ordinal,
)


def _market(question, prob, volume=0, mid=None):
    return {
        "id": mid or question,
        "question": question,
        "probability": prob,
        "volume": volume,
        "totalLiquidity": 100,
        "closeTime": time.time() * 1000 + 365 * 24 * 3600 * 1000,
        "isResolved": False,
    }


# ---- date ordinals ----

def test_date_ordinal_orders_chronologically():
    a = date_ordinal("jun 30 2027")
    b = date_ordinal("sep 30 2027")
    c = date_ordinal("mar 31 2028")
    assert a < b < c


def test_date_ordinal_handles_quarter_and_bare_year():
    assert date_ordinal("q3 2026") is not None
    assert date_ordinal("2028") is not None
    assert date_ordinal("garbage") is None


# ---- grouping ----

def test_groups_by_metric_threshold_across_dates():
    s = TermStructureStrategy()
    ms = [
        _market("Will X production exceed 100 kt on June 30, 2027?", 0.9, mid="a"),
        _market("Will X production exceed 100 kt on September 30, 2027?", 0.5, mid="b"),
        _market("Will X production exceed 250 kt on June 30, 2027?", 0.4, mid="c"),  # diff threshold
    ]
    groups = s.group(ms)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]  # the two 100kt dates group; 250kt is its own curve


# ---- smoothing ----

def test_stale_default_pulled_toward_traded_neighbors():
    # Three dates of one metric+threshold; the middle is an untraded 0.50 default
    # bracketed by traded ~0.9 dates -> it should be pulled up toward ~0.9.
    s = TermStructureStrategy(bandwidth=6.0)
    group = [
        _market("Will X exceed 100 on June 30, 2027?", 0.90, volume=50, mid="jun"),
        _market("Will X exceed 100 on September 30, 2027?", 0.50, volume=0, mid="sep"),
        _market("Will X exceed 100 on December 31, 2027?", 0.90, volume=50, mid="dec"),
    ]
    out = s.estimate(group)
    assert out["sep"] > 0.70  # pulled well above its stale 0.50


def test_traded_anchors_are_not_traded():
    # We trust traded dates and don't bet against them — only stale dates are filled.
    s = TermStructureStrategy(bandwidth=6.0)
    group = [
        _market("Will X exceed 100 on June 30, 2027?", 0.90, volume=50, mid="jun"),
        _market("Will X exceed 100 on September 30, 2027?", 0.50, volume=0, mid="sep"),
        _market("Will X exceed 100 on December 31, 2027?", 0.90, volume=50, mid="dec"),
    ]
    out = s.estimate(group)
    assert "jun" not in out and "dec" not in out  # anchors untouched
    assert "sep" in out


def test_far_isolated_stale_date_stays_near_prior():
    # A stale date far (>>bandwidth) from the only anchors shrinks toward 0.5,
    # so we don't confidently extrapolate the curve.
    s = TermStructureStrategy(bandwidth=6.0, prior_strength=1.0)
    group = [
        _market("Will X exceed 100 on June 30, 2027?", 0.95, volume=50, mid="a"),
        _market("Will X exceed 100 on September 30, 2027?", 0.95, volume=50, mid="b"),
        _market("Will X exceed 100 on December 31, 2034?", 0.50, volume=0, mid="far"),
    ]
    out = s.estimate(group)
    assert abs(out["far"] - 0.5) < 0.1  # barely moved -> no confident far bet


def test_abstains_without_enough_anchors():
    # All untraded 0.50 -> no informative anchors -> abstain.
    s = TermStructureStrategy(min_anchors=2)
    group = [
        _market("Will X exceed 100 on June 30, 2027?", 0.50, mid="a"),
        _market("Will X exceed 100 on September 30, 2027?", 0.50, mid="b"),
        _market("Will X exceed 100 on December 31, 2027?", 0.50, mid="c"),
    ]
    assert s.estimate(group) == {}


def test_abstains_with_too_few_dates():
    s = TermStructureStrategy(min_dates=3)
    group = [
        _market("Will X exceed 100 on June 30, 2027?", 0.90, volume=50, mid="a"),
        _market("Will X exceed 100 on December 31, 2027?", 0.50, mid="b"),
    ]
    assert s.estimate(group) == {}


def test_pinned_strike_anchors_but_not_traded():
    s = TermStructureStrategy(skip_extreme=0.02)
    group = [
        _market("Will X exceed 100 on June 30, 2027?", 0.99, volume=50, mid="pinned"),
        _market("Will X exceed 100 on September 30, 2027?", 0.50, volume=0, mid="stale"),
        _market("Will X exceed 100 on December 31, 2027?", 0.95, volume=50, mid="dec"),
    ]
    out = s.estimate(group)
    assert "pinned" not in out          # 0.99 not traded
    assert out["stale"] > 0.70          # but it pulled the stale date up


def test_correlation_key_groups_one_curve():
    s = TermStructureStrategy()
    k1 = s.correlation_key(_market("Will X exceed 100 on June 30, 2027?", 0.9))
    k2 = s.correlation_key(_market("Will X exceed 100 on September 30, 2027?", 0.5))
    k3 = s.correlation_key(_market("Will X exceed 250 on June 30, 2027?", 0.9))
    assert k1 == k2 != k3
