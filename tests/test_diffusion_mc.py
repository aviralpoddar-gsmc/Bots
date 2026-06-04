"""Tests for diffusion_mc: it inherits commodity_spot's unit guard, prices off a
simulated terminal distribution with FATTER tails than the lognormal, keeps ladders
monotone/coherent, and falls back to the lognormal when a commodity is uncalibrated.
Synthetic returns are injected via set_returns() so tests need no network."""

import time

import numpy as np
import pytest

from quantbots.strategies.commodity_spot import CommoditySpotStrategy
from quantbots.strategies.diffusion_mc import DiffusionMcStrategy


class Obs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v, "ts": "2026-06-03"} if v is not None else None


def _market(question, close_years=0.08):
    return {
        "id": question[:30], "question": question, "probability": 0.5,
        "closeTime": time.time() * 1000 + close_years * 365.25 * 24 * 3600 * 1000,
        "totalLiquidity": 100, "isResolved": False,
    }


_GOLD_Q = "Will Gold spot price exceed ${T}/ozt on Dec 31, 2026?"


def _fat_returns(n=5000, df=3, scale=0.006, seed=0):
    """Fat-tailed (Student-t) daily log-returns, demeaned."""
    r = np.random.default_rng(seed).standard_t(df, n) * scale
    return r - r.mean()


# 1. Inherits the strict unit/currency guard from commodity_spot.
@pytest.mark.parametrize("q", [
    "Will Gold spot price exceed $5,181/ozt on May 31, 2026?",
    "Will the copper spot price exceed $12,900 USD/MT on Feb 28?",
    "Will global gold dental alloy demand exceed 75 tonnes for 2026?",      # trap
    "Will LME nickel spot price exceed 13000 CNY per metric ton on July?",  # trap
    "Will zinc sulfate spot price exceed $800/t on June 30, 2027?",         # trap
])
def test_inherits_unit_guard(q):
    assert DiffusionMcStrategy()._spec(q) == CommoditySpotStrategy()._spec(q)


# 2. CORE: fat tails put MORE probability in the FAR tail than the matched lognormal.
#    (Fat-tailed dists are thinner in the shoulders, fatter only beyond ~2.3σ — so we
#    test beyond the lognormal's 99th pct, at a 1-day horizon where the effect is clean.
#    min_vol floor disabled so the two price at the same sigma. The edge shrinks toward
#    longer horizons as returns compound to ~normal — that's what the backtest gate checks.)
def test_fat_tail_beats_lognormal_in_far_tail():
    rets = _fat_returns()                       # Student-t(3) daily returns
    daily_std = float(rets.std())
    spot = 2000.0
    T = 1.0 / 252.0                             # one trading day -> single fat-tailed draw
    implied_vol = daily_std * np.sqrt(252)
    sigma_T = implied_vol * np.sqrt(T)          # == daily_std
    strike = spot * np.exp(2.326 * sigma_T)     # the lognormal's 99th pct: P(exceed)=0.01
    m = _market(_GOLD_Q.replace("{T}", f"{strike:,.0f}"), close_years=T)

    s_log = CommoditySpotStrategy(vols={"GOLD": implied_vol}, min_vol=1e-6)
    s_log.bind(Obs({"GOLD": spot}))
    p_log = s_log.estimate([m])[m["id"]]

    s_diff = DiffusionMcStrategy(vols={"GOLD": implied_vol}, min_vol=1e-6,
                                 n_sims=80000, block_len=1)
    s_diff.set_returns("GOLD", rets)
    s_diff.bind(Obs({"GOLD": spot}))            # _returns non-empty -> no network calibration
    p_diff = s_diff.estimate([m])[m["id"]]

    assert abs(p_log - 0.01) < 0.003            # lognormal ≈ 1% beyond its own 99th pct
    assert p_diff > 0.012                        # fat tails -> meaningfully MORE than 1%


# 3 & 4. Monotone ladder + direction symmetry, all off one simulated sample.
def test_monotone_and_direction():
    s = DiffusionMcStrategy(vols={"GOLD": 0.16}, n_sims=40000, block_len=1)
    s.set_returns("GOLD", _fat_returns())
    s.bind(Obs({"GOLD": 2000.0}))
    strikes = [1800, 1950, 2000, 2100, 2300]
    group = [_market(_GOLD_Q.replace("{T}", f"{k:,}")) for k in strikes]
    est = s.estimate(group)
    ps = [est[m["id"]] for m in group]
    assert all(ps[i] >= ps[i + 1] - 1e-9 for i in range(len(ps) - 1))  # non-increasing

    below = _market("Will Gold spot price below $2,100/ozt on Dec 31, 2026?")
    exceed = _market("Will Gold spot price exceed $2,100/ozt on Dec 31, 2026?")
    e2 = s.estimate([below, exceed])
    assert abs(e2[below["id"]] + e2[exceed["id"]] - 1.0) < 0.02


# 5. Absurd strike clamps to 0.01, not 0.0.
def test_clamp():
    s = DiffusionMcStrategy(vols={"GOLD": 0.16}, n_sims=20000, block_len=1)
    s.set_returns("GOLD", _fat_returns())
    s.bind(Obs({"GOLD": 2000.0}))
    m = _market(_GOLD_Q.replace("{T}", "9,000,000"))
    assert s.estimate([m])[m["id"]] == 0.01


# 6. group() buckets by (commodity, close-month); different commodities split.
def test_grouping_by_entity_and_month():
    s = DiffusionMcStrategy()
    g1 = _market(_GOLD_Q.replace("{T}", "2000"), close_years=0.1)
    g2 = _market(_GOLD_Q.replace("{T}", "2500"), close_years=0.1)
    cu = _market("Will the copper spot price exceed $12,900 USD/MT on Dec 31, 2026?", close_years=0.1)
    groups = s.group([g1, g2, cu])
    assert len(groups) == 2
    assert any(len(grp) == 2 for grp in groups) and any(len(grp) == 1 for grp in groups)


# 7. Uncalibrated commodity -> lognormal fallback (no crash, no abstain).
def test_fallback_to_lognormal_when_uncalibrated():
    s = DiffusionMcStrategy(vols={"GOLD": 0.16})
    s.set_returns("SILVER", _fat_returns())  # non-empty -> bind skips network calibration
    s.bind(Obs({"GOLD": 2000.0}))            # GOLD has NO returns -> must fall back
    m = _market(_GOLD_Q.replace("{T}", "2100"))
    p = s.estimate([m]).get(m["id"])
    assert p is not None and 0.01 <= p <= 0.99


# 7b. THE FIX: student-t extrapolates beyond the historical return range; the bootstrap
#     cannot (it only resamples observed moves), so it under-prices unprecedented tails.
def test_student_t_extrapolates_beyond_bootstrap():
    # Bounded history (|move| <= 0.02). Over a 1-day horizon the bootstrap terminal can
    # never exceed the largest observed move, so it puts ZERO mass beyond it; student-t
    # has continuous support and extrapolates. (This is the bootstrap flaw the fix addresses
    # — it shows in the raw distribution; the 0.01 clamp would hide it on a point estimate.)
    rng = np.random.default_rng(1)
    bounded = rng.uniform(-0.02, 0.02, 4000)
    bounded = bounded - bounded.mean()
    spot, T = 2000.0, 1.0 / 252.0
    strike = spot * np.exp(0.03)  # beyond the ~0.02 historical max single move

    boot = DiffusionMcStrategy(process="bootstrap", n_sims=40000)
    boot.set_returns("GOLD", bounded)
    t_boot = boot._simulate_terminal("GOLD", spot, T)

    tdist = DiffusionMcStrategy(process="student_t", n_sims=40000)
    tdist.set_returns("GOLD", bounded)
    t_t = tdist._simulate_terminal("GOLD", spot, T)

    assert np.mean(t_boot > strike) == 0.0   # bootstrap cannot reach beyond observed moves
    assert np.mean(t_t > strike) > 0.0       # student-t extrapolates into the unobserved tail


# 7c. THE DEPLOYED MODEL: the kernel-smoothed bootstrap (ksb, the default) keeps the
#     plain bootstrap's empirical body but, via the variance-preserving Student-t kernel,
#     ALSO extrapolates past the observed range — so it has neither the bootstrap's
#     catastrophic-tail hole nor a parametric body. (The multi-fold bench shows it ties
#     the bootstrap on PnL/Brier and beats the lognormal on both, while staying tail-safe.)
def test_ksb_is_default_and_extrapolates():
    assert DiffusionMcStrategy().process == "ksb"   # deployed default
    rng = np.random.default_rng(1)
    bounded = rng.uniform(-0.02, 0.02, 4000)
    bounded = bounded - bounded.mean()
    spot, T = 2000.0, 1.0 / 252.0
    strike = spot * np.exp(0.03)  # beyond the ~0.02 historical max single move

    ksb = DiffusionMcStrategy(process="ksb", n_sims=80000)
    ksb.set_returns("GOLD", bounded)
    t_ksb = ksb._simulate_terminal("GOLD", spot, T)
    assert np.mean(t_ksb > strike) > 0.0          # extrapolates beyond the observed range


# 8. Fixed seed -> identical estimates across calls (no order churn).
def test_seed_stability():
    rets = _fat_returns()
    def run():
        s = DiffusionMcStrategy(vols={"GOLD": 0.16}, n_sims=20000, block_len=10)
        s.set_returns("GOLD", rets)
        s.bind(Obs({"GOLD": 2000.0}))
        m = _market(_GOLD_Q.replace("{T}", "2150"))
        return s.estimate([m])[m["id"]]
    assert run() == run()
