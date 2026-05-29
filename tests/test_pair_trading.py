"""Tests for pair_trading: the mean-reversion drift it layers on top of
commodity_spot's lognormal, the rich/cheap direction, abstention without a
signal, and the per-entity signal selection from a panel. No network: signals
are injected directly or built from a fake pair_stats + fake panel."""

import math
import time
from types import SimpleNamespace

import pytest

from quantbots.strategies.commodity_spot import CommoditySpotStrategy
from quantbots.strategies.pair_trading import PairSignal, PairTradingStrategy


class Obs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


def _market(question, close_years=0.25):
    return {
        "id": question[:32], "question": question, "probability": 0.5,
        "closeTime": time.time() * 1000 + close_years * 365.25 * 24 * 3600 * 1000,
        "totalLiquidity": 100, "isResolved": False,
    }


GOLD_Q = "Will Gold spot price exceed $2,000/ozt on a near date?"


def _strategy_with_signal(sig: PairSignal, gold_spot=2000.0):
    """A pair_trading strategy with fitting bypassed and one injected signal."""
    s = PairTradingStrategy(reversion_capture=0.5)
    s._fitted = True                      # skip the network fit
    entity = sig.a if sig.role == "a" else sig.b
    s._signals = {entity: sig}
    s._obs = Obs({"GOLD": gold_spot})
    return s


# --- drift math --------------------------------------------------------------

def test_drift_sign_and_horizon_scaling():
    # z>0 => leg a rich => negative drift (expected to fall). Half-life 40d.
    sig = PairSignal(a="GOLD", b="SILVER", role="a", beta=0.7, half_life=40.0,
                     z=2.0, mu_minus_s0=-2.0 * 0.06, corr=0.77)
    short = sig.drift(years=0.05, capture=0.5)
    long = sig.drift(years=0.5, capture=0.5)
    assert short < 0 and long < 0          # rich leg drifts down
    assert abs(long) > abs(short)          # more reversion realised over a longer horizon
    # cheap leg of the same dislocation drifts up, scaled by 1/beta
    cheap = PairSignal(a="GOLD", b="SILVER", role="b", beta=0.7, half_life=40.0,
                       z=2.0, mu_minus_s0=-2.0 * 0.06, corr=0.77)
    assert cheap.drift(years=0.5, capture=0.5) > 0


def test_capture_damps_drift():
    sig = PairSignal(a="GOLD", b="SILVER", role="a", beta=0.7, half_life=40.0,
                     z=2.0, mu_minus_s0=-0.12, corr=0.77)
    assert abs(sig.drift(0.5, capture=0.25)) < abs(sig.drift(0.5, capture=1.0))


# --- estimate direction vs the zero-drift baseline ---------------------------

def test_rich_leg_prices_below_zero_drift_baseline():
    # Gold is the rich leg (role a, z>0). Threshold == spot, so commodity_spot
    # (zero drift) prices ~0.50; pair_trading must price the 'exceeds' strike LOWER.
    sig = PairSignal(a="GOLD", b="SILVER", role="a", beta=0.7, half_life=40.0,
                     z=2.0, mu_minus_s0=-2.0 * 0.06, corr=0.77)
    s = _strategy_with_signal(sig, gold_spot=2000.0)
    m = _market(GOLD_Q)  # threshold $2,000 == spot
    p_pair = s.estimate([m])[m["id"]]

    base = CommoditySpotStrategy()
    base.bind(Obs({"GOLD": 2000.0}))
    p_base = base.estimate([m])[m["id"]]

    assert abs(p_base - 0.5) < 0.02        # zero drift at the money
    assert p_pair < p_base - 0.05          # drift pushes the rich leg's exceed-prob down


def test_cheap_leg_prices_above_zero_drift_baseline():
    # Gold is the cheap leg (role b, z>0 => partner a rich => gold expected to rise).
    sig = PairSignal(a="SILVER", b="GOLD", role="b", beta=0.7, half_life=40.0,
                     z=2.0, mu_minus_s0=-2.0 * 0.06, corr=0.77)
    s = _strategy_with_signal(sig, gold_spot=2000.0)
    m = _market(GOLD_Q)
    p_pair = s.estimate([m])[m["id"]]
    assert p_pair > 0.55


def test_abstains_without_active_signal():
    s = PairTradingStrategy()
    s._fitted = True
    s._signals = {}                        # nothing dislocated
    s._obs = Obs({"GOLD": 2000.0})
    m = _market(GOLD_Q)
    assert s.estimate([m]) == {}


def test_explanation_mentions_pair_and_z():
    sig = PairSignal(a="GOLD", b="SILVER", role="a", beta=0.7, half_life=40.0,
                     z=2.0, mu_minus_s0=-0.12, corr=0.77)
    s = _strategy_with_signal(sig)
    m = _market(GOLD_Q)
    s.estimate([m])
    text = s.explain(m["id"])
    assert text and "GOLD ↔ SILVER" in text and "z =" in text and "drift" in text


# --- signal selection from a panel -------------------------------------------

def _fake_pair_stats_factory(table):
    """table: {(a,b): dict-of-stats}. Returns a pair_stats(panel, a, b) stand-in."""
    def pair_stats(panel, a, b):
        d = table.get((a, b))
        if d is None:
            return None
        return SimpleNamespace(a=a, b=b, **d)
    return pair_stats


class _FakePanel:
    def __init__(self, columns):
        self.columns = list(columns)
    empty = False


def test_set_signals_filters_and_picks_most_dislocated():
    s = PairTradingStrategy(
        pairs=[["GOLD", "SILVER"], ["GOLD", "PLATINUM"], ["WTI_OIL", "BRENT_OIL"]],
        entry_z=1.5, max_half_life_days=90, min_abs_corr=0.4,
    )
    table = {
        # GOLD appears in two pairs; the PLATINUM pair is more dislocated -> wins.
        ("GOLD", "SILVER"):    dict(half_life=59.0, corr_returns=0.77, current_z=1.6,
                                    spread_std=0.05, beta=0.71),
        ("GOLD", "PLATINUM"):  dict(half_life=60.0, corr_returns=0.61, current_z=-2.2,
                                    spread_std=0.07, beta=0.90),
        # Below entry_z -> dropped entirely.
        ("WTI_OIL", "BRENT_OIL"): dict(half_life=5.0, corr_returns=0.93, current_z=0.5,
                                       spread_std=0.02, beta=1.0),
    }
    panel = _FakePanel(["GOLD", "SILVER", "PLATINUM", "WTI_OIL", "BRENT_OIL"])
    s.set_signals_from_panel(panel, _fake_pair_stats_factory(table))

    assert "WTI_OIL" not in s._signals and "BRENT_OIL" not in s._signals  # |z|<entry_z
    # GOLD kept from the GOLD/PLATINUM pair (|z|=2.2 beats 1.6)
    assert s._signals["GOLD"].b == "PLATINUM" and s._signals["GOLD"].role == "a"
    assert math.isclose(abs(s._signals["GOLD"].z), 2.2)
    # SILVER (cheap-side leg of GOLD/SILVER) is present as role b
    assert s._signals["SILVER"].role == "b"


def test_set_signals_drops_low_correlation_and_slow_pairs():
    s = PairTradingStrategy(pairs=[["GOLD", "SILVER"], ["SILVER", "COPPER"]],
                            entry_z=1.0, max_half_life_days=90, min_abs_corr=0.4)
    table = {
        ("GOLD", "SILVER"):  dict(half_life=200.0, corr_returns=0.8, current_z=3.0,
                                  spread_std=0.05, beta=0.7),   # too slow
        ("SILVER", "COPPER"): dict(half_life=22.0, corr_returns=0.2, current_z=3.0,
                                   spread_std=0.05, beta=2.5),  # too uncorrelated
    }
    panel = _FakePanel(["GOLD", "SILVER", "COPPER"])
    s.set_signals_from_panel(panel, _fake_pair_stats_factory(table))
    assert s._signals == {}


def test_correlation_key_inherited_groups_by_commodity():
    s = PairTradingStrategy()
    k1 = s.correlation_key(_market("Will Gold spot price exceed $5,181/ozt on Dec 31?"))
    k2 = s.correlation_key(_market("Will Gold spot price exceed $6,000/ozt on Jun 30?"))
    assert k1 == k2 == "GOLD"
