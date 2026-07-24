"""Tests for the conditional ("B if A") coherence-arb strategy + resolvability."""

from __future__ import annotations

from quantbots.resolvability import resolvability_score
from quantbots.strategies.conditional_arb import (
    ConditionalArbStrategy,
    parse_conditional,
)

PRED_Q = "LME zinc SHG spot price at least 2800 USD/t on Jun 30 2026"
QTY_Q = "LME zinc SHG spot price at least 3400 USD/t on Jun 30 2026"
COND_Q = f"IF [{PRED_Q}] = YES: {QTY_Q}"


def _m(mid, question, prob, volume=10.0):
    return {"id": mid, "question": question, "probability": prob, "volume": volume}


def test_parse_conditional_roundtrip():
    assert parse_conditional(COND_Q) == (PRED_Q, QTY_Q)
    assert parse_conditional("not a conditional") is None
    # A predicate containing brackets still parses (greedy to the last ` = YES: `).
    weird = "IF [zinc [LME] >= 2800] = YES: zinc >= 3400"
    assert parse_conditional(weird) == ("zinc [LME] >= 2800", "zinc >= 3400")


def test_group_links_legs_to_conditional():
    s = ConditionalArbStrategy()
    markets = [_m("c", COND_Q, 0.30), _m("a", PRED_Q, 0.69), _m("b", QTY_Q, 0.30)]
    groups = s.group(markets)
    assert len(groups) == 1
    assert {m["id"] for m in groups[0]} == {"a", "b", "c"}


def test_nested_exact_constraint_c_equals_b_over_a():
    # B = ">=3400" implies A = ">=2800": fair P(B|A) = P(B)/P(A) = 0.30/0.69.
    s = ConditionalArbStrategy()
    group = [_m("c", COND_Q, 0.30), _m("a", PRED_Q, 0.69), _m("b", QTY_Q, 0.30)]
    out = s.estimate(group)
    assert set(out) == {"c"}  # only the conditional is traded
    assert out["c"] == round(0.30 / 0.69, 12) or abs(out["c"] - 0.30 / 0.69) < 1e-9


def test_nested_abstains_when_legs_survival_inverted():
    # b > a is itself a ladder violation — defer to ladder_arb, don't compound it.
    s = ConditionalArbStrategy()
    group = [_m("c", COND_Q, 0.50), _m("a", PRED_Q, 0.40), _m("b", QTY_Q, 0.60)]
    assert s.estimate(group) == {}


def test_abstains_when_already_coherent():
    # Conditional already at b/a within dev_band → nothing to do.
    s = ConditionalArbStrategy(dev_band=0.03)
    fair = 0.30 / 0.69
    group = [_m("c", COND_Q, round(fair, 4)), _m("a", PRED_Q, 0.69), _m("b", QTY_Q, 0.30)]
    assert s.estimate(group) == {}


def test_abstains_on_uninformative_legs():
    # Both legs untouched at 0.50 default with no volume → no information to trade.
    s = ConditionalArbStrategy()
    group = [
        _m("c", COND_Q, 0.30, volume=0),
        _m("a", PRED_Q, 0.50, volume=0),
        _m("b", QTY_Q, 0.50, volume=0),
    ]
    assert s.estimate(group) == {}


def test_abstains_when_predicate_near_impossible():
    s = ConditionalArbStrategy(min_predicate_prob=0.05)
    group = [_m("c", COND_Q, 0.30), _m("a", PRED_Q, 0.02), _m("b", QTY_Q, 0.01)]
    assert s.estimate(group) == {}


def test_non_nested_frechet_band_projection():
    # Independent measurables: only the Fréchet band binds. a=0.5,b=0.5 →
    # band [0, 1]; c stays inside → abstain. Push c outside via tighter legs.
    pred = "Will the FOMC cut rates in June 2026?"
    qty = "Will COMEX copper settle above 5 USD/lb on Jun 30 2026?"
    cond = f"IF [{pred}] = YES: {qty}"
    s = ConditionalArbStrategy()
    # a=0.40, b=0.90 → hi = min(1, 0.9/0.4)=1.0, lo=max(0,(0.4+0.9-1)/0.4)=0.75.
    # c quoted at 0.50 < lo → project up to 0.75.
    group = [_m("c", cond, 0.50), _m("a", pred, 0.40), _m("b", qty, 0.90)]
    out = s.estimate(group)
    assert set(out) == {"c"}
    assert abs(out["c"] - 0.75) < 1e-9


def test_resolvability_is_product_of_legs():
    # Both legs are LME price markets (~0.35 each) → product ~0.12, discounted but
    # tradeable; far below either leg alone (correctly reflecting double cancel risk).
    leg = resolvability_score(PRED_Q)
    cond = resolvability_score(COND_Q)
    assert abs(cond - leg * leg) < 1e-9
    assert cond < leg
    assert cond > 0.04  # clears a typical min_resolvability gate


def test_resolvability_trigger_predicate_kills_conditional():
    # A production-trigger predicate (~never resolves) drags the conditional below
    # any sane min_resolvability, so the allocator won't fund it.
    pred = "Will Glencore zinc production exceed 1,000 kt in 2026?"
    qty = "LME zinc SHG spot price at least 3400 USD/t on Jun 30 2026"
    cond = f"IF [{pred}] = YES: {qty}"
    assert resolvability_score(cond) < 0.02
