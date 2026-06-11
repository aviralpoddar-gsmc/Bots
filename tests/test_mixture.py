"""Tests for the pure mixture math behind the mercury_ensemble strategy:
per-sample probability, posterior-predictive mean, aleatoric/epistemic
decomposition, direction agreement, and disagreement-shrinkage."""

import math

from quantbots.strategies._mixture import (
    direction_agreement,
    fit_normal,
    mixture,
    prob_from_normal,
    shrink,
)


PCT = {"p10": 60.0, "p25": 65.0, "p50": 70.0, "p75": 75.0, "p90": 80.0}


def test_fit_normal_mu_is_median():
    mu, _ = fit_normal(PCT, spread_mult=1.0)
    assert mu == 70.0


def test_fit_normal_spread_mult_scales_sigma_linearly():
    _, s1 = fit_normal(PCT, spread_mult=1.0)
    _, s2 = fit_normal(PCT, spread_mult=2.0)
    assert abs(s2 - 2.0 * s1) < 1e-12


def test_fit_normal_wider_percentiles_give_larger_sigma():
    wide = {"p10": 40.0, "p25": 55.0, "p50": 70.0, "p75": 85.0, "p90": 100.0}
    _, narrow_s = fit_normal(PCT, spread_mult=1.0)
    _, wide_s = fit_normal(wide, spread_mult=1.0)
    assert wide_s > narrow_s


def test_prob_from_normal_exceeds_at_median_is_half():
    # strike == mu -> P(exceed) == 0.5
    assert abs(prob_from_normal(100.0, "exceeds", mu=100.0, sigma=10.0) - 0.5) < 1e-9


def test_prob_from_normal_below_inverts_exceeds():
    p_ex = prob_from_normal(110.0, "exceeds", mu=100.0, sigma=10.0)
    p_be = prob_from_normal(110.0, "below", mu=100.0, sigma=10.0)
    assert abs((p_ex + p_be) - 1.0) < 1e-9


def test_mixture_mean_is_average_of_samples():
    m = mixture([0.2, 0.4, 0.6])
    assert abs(m.mean - 0.4) < 1e-12


def test_epistemic_is_zero_when_samples_agree():
    m = mixture([0.5, 0.5, 0.5])
    assert m.epistemic == 0.0


def test_epistemic_positive_when_samples_disagree():
    assert mixture([0.1, 0.9]).epistemic > 0.0


def test_total_variance_identity_holds():
    # Law of total variance for a Bernoulli mixture: aleatoric + epistemic == p̄(1-p̄)
    m = mixture([0.1, 0.5, 0.9, 0.7])
    assert abs((m.aleatoric + m.epistemic) - m.mean * (1.0 - m.mean)) < 1e-12


def test_aleatoric_is_nonnegative_for_extreme_disagreement():
    m = mixture([0.0, 1.0])  # maximal disagreement
    assert m.aleatoric >= 0.0


def test_agreement_full_when_all_samples_same_side():
    # market at 0.5, every sample bullish -> all agree with the mean's side
    assert direction_agreement([0.6, 0.7, 0.8], current_prob=0.5, mean=0.7) == 1.0


def test_agreement_counts_minority_against():
    # 3 of 4 above the market price -> 0.75 agreement
    a = direction_agreement([0.7, 0.7, 0.7, 0.3], current_prob=0.5, mean=0.6)
    assert abs(a - 0.75) < 1e-12


def test_shrink_returns_market_price_under_high_epistemic():
    r = shrink(mean=0.8, current_prob=0.5, epistemic=1.0, tau=0.04)
    assert r.confidence == 0.0
    assert r.estimate == 0.5


def test_shrink_returns_mean_under_zero_epistemic():
    r = shrink(mean=0.8, current_prob=0.5, epistemic=0.0, tau=0.04)
    assert r.confidence == 1.0
    assert abs(r.estimate - 0.8) < 1e-12


def test_shrink_is_partial_between_market_and_mean():
    r = shrink(mean=0.8, current_prob=0.5, epistemic=0.02, tau=0.04)
    assert 0.5 < r.estimate < 0.8
    assert math.isclose(r.confidence, 0.5)
