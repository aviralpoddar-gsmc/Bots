"""Tests for ladder_arb: date-aware grouping (the collapse-bug regression),
monotonicity enforcement, informative weighting, and abstention guards."""

import time

from quantbots.strategies.ladder_arb import (
    LadderArbStrategy,
    isotonic_decreasing,
    ladder_key,
)


def _market(question, prob, volume=0, close_years=0.5, mid=None):
    return {
        "id": mid or question,
        "question": question,
        "probability": prob,
        "volume": volume,
        "closeTime": time.time() * 1000 + close_years * 365.25 * 24 * 3600 * 1000,
        "totalLiquidity": 100,
        "isResolved": False,
    }


# ---- isotonic kernel ----

def test_isotonic_already_decreasing_unchanged():
    assert isotonic_decreasing([0.9, 0.5, 0.1], [1, 1, 1]) == [0.9, 0.5, 0.1]


def test_isotonic_pools_violation_to_weighted_mean():
    # [0.69, 0.92, 0.69] violates at the middle -> pool first two to their mean.
    out = isotonic_decreasing([0.69, 0.92, 0.69], [1, 1, 1])
    assert out[0] == out[1]  # pooled
    assert abs(out[0] - 0.805) < 1e-9
    assert all(out[i] >= out[i + 1] - 1e-9 for i in range(len(out) - 1))


def test_isotonic_weight_pulls_toward_heavy_point():
    # A heavy 0.30 and a light 0.90 (violation) pool closer to 0.30.
    out = isotonic_decreasing([0.30, 0.90], [9, 1])
    assert out[0] == out[1]
    assert out[0] < 0.45


# ---- grouping / the date-collapse bug fix ----

def test_ladder_key_separates_resolution_dates():
    a = ladder_key("Will X production exceed 100 kt on June 30, 2027?")
    b = ladder_key("Will X production exceed 100 kt on June 30, 2028?")
    assert a[0] == b[0]        # same metric
    assert a[1] != b[1]        # different date -> different group


def test_ladder_key_collapses_strikes_same_date():
    a = ladder_key("Will X production exceed 100 kt on June 30, 2027?")
    b = ladder_key("Will X production exceed 250 kt on June 30, 2027?")
    assert a == b              # only the threshold differs -> one ladder


def test_group_partitions_by_metric_and_date():
    s = LadderArbStrategy()
    ms = [
        _market("Will X exceed 100 kt on June 30, 2027?", 0.6, mid="a"),
        _market("Will X exceed 250 kt on June 30, 2027?", 0.4, mid="b"),
        _market("Will X exceed 100 kt on June 30, 2028?", 0.6, mid="c"),
    ]
    groups = s.group(ms)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]     # 2027 ladder has 2 strikes, 2028 has 1


# ---- estimate ----

def test_estimate_enforces_monotonic_survival():
    # Survival rising with threshold (0.4 -> 0.6 -> 0.5) is impossible; output must
    # be non-increasing in threshold.
    s = LadderArbStrategy()
    group = [
        _market("Will P exceed 62 on June 30, 2027?", 0.40, volume=10, mid="t62"),
        _market("Will P exceed 70 on June 30, 2027?", 0.60, volume=10, mid="t70"),
        _market("Will P exceed 82 on June 30, 2027?", 0.50, volume=10, mid="t82"),
    ]
    out = s.estimate(group)
    assert out["t62"] >= out["t70"] >= out["t82"]


def test_estimate_corrects_overpriced_high_strike():
    # P(>100)=0.92 above P(>60)=0.69 -> the >100 estimate must drop below market.
    s = LadderArbStrategy()
    group = [
        _market("Will S exceed 60 kt on June 30, 2026?", 0.69, volume=50, mid="t60"),
        _market("Will S exceed 100 kt on June 30, 2026?", 0.92, volume=50, mid="t100"),
        _market("Will S exceed 150 kt on June 30, 2026?", 0.69, volume=50, mid="t150"),
    ]
    out = s.estimate(group)
    assert out["t100"] < 0.92


def test_below_direction_handled():
    # "below X" markets: survival = 1 - prob. A coherent below-ladder is
    # non-decreasing in prob, so estimates should stay sane and in range.
    s = LadderArbStrategy()
    group = [
        _market("Will P be below 62 on June 30, 2027?", 0.30, volume=10, mid="b62"),
        _market("Will P be below 70 on June 30, 2027?", 0.20, volume=10, mid="b70"),
        _market("Will P be below 82 on June 30, 2027?", 0.45, volume=10, mid="b82"),
    ]
    out = s.estimate(group)
    assert all(0.01 <= v <= 0.99 for v in out.values())


def test_flat_ladder_abstains():
    s = LadderArbStrategy()
    group = [
        _market("Will X exceed 10 on June 30, 2027?", 0.5, mid="a"),
        _market("Will X exceed 20 on June 30, 2027?", 0.5, mid="b"),
        _market("Will X exceed 30 on June 30, 2027?", 0.5, mid="c"),
    ]
    assert s.estimate(group) == {}


def test_too_few_strikes_abstains():
    s = LadderArbStrategy()
    group = [
        _market("Will X exceed 10 on June 30, 2027?", 0.7, mid="a"),
        _market("Will X exceed 20 on June 30, 2027?", 0.3, mid="b"),
    ]
    assert s.estimate(group) == {}


def test_informative_weighting_does_not_let_defaults_drag_curve():
    # One informative strike (volume) at 0.30; two untraded 0.50 defaults. The
    # 0.50s violate monotonicity above the informative point, so they get pulled
    # DOWN toward it rather than dragging it up.
    s = LadderArbStrategy()
    group = [
        _market("Will X exceed 10 on June 30, 2027?", 0.30, volume=500, mid="t10"),
        _market("Will X exceed 20 on June 30, 2027?", 0.50, volume=0, mid="t20"),
        _market("Will X exceed 30 on June 30, 2027?", 0.50, volume=0, mid="t30"),
    ]
    out = s.estimate(group)
    # The informative low strike should barely move; the high defaults drop below 0.5.
    assert out["t10"] > 0.30 - 0.05 and out["t10"] < 0.45
    assert out["t20"] < 0.50 and out["t30"] < 0.50


def test_skip_extreme_anchors_but_does_not_trade_pinned_strikes():
    # A strike pinned at 0.00 anchors the fit but is not itself traded; the other
    # strikes are still corrected toward coherence.
    s = LadderArbStrategy(skip_extreme=0.02)
    group = [
        _market("Will I exceed 2 on June 30, 2026?", 0.00, volume=2961, mid="t2"),
        _market("Will I exceed 3 on June 30, 2026?", 0.88, volume=1364, mid="t3"),
        _market("Will I exceed 4 on June 30, 2026?", 0.40, volume=1482, mid="t4"),
    ]
    out = s.estimate(group)
    assert "t2" not in out          # pinned at 0.00 -> not traded
    assert "t3" in out and "t4" in out
    assert out["t3"] >= out["t4"]   # but the fit it anchored is still monotonic


def test_correlation_key_groups_by_ladder():
    s = LadderArbStrategy()
    k1 = s.correlation_key(_market("Will X exceed 10 on June 30, 2027?", 0.6))
    k2 = s.correlation_key(_market("Will X exceed 99 on June 30, 2027?", 0.4))
    k3 = s.correlation_key(_market("Will X exceed 10 on June 30, 2028?", 0.6))
    assert k1 == k2 != k3
