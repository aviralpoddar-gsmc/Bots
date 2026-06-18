"""Tests for semantic_arb: the LLM-free machinery — token blocking, constraint
projection, and the estimate() pipeline with relations injected (no live model).

The LLM call itself is stubbed: we replace `_extract_relations` so the tests
exercise the projection / violation-gating / explanation logic deterministically.
"""

import time

import pytest

from quantbots.strategies.semantic_arb import (
    SemanticArbStrategy,
    block_markets,
    jaccard,
    project_constraints,
)


def _market(question, prob, volume=0, mid=None, close_years=0.5):
    return {
        "id": mid or question,
        "question": question,
        "probability": prob,
        "volume": volume,
        "closeTime": time.time() * 1000 + close_years * 365.25 * 24 * 3600 * 1000,
        "totalLiquidity": 100,
        "isResolved": False,
        "outcomeType": "BINARY",
    }


# ---- jaccard ----

def test_jaccard_basic():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), {"a"}) == 0.0


# ---- blocking ----

def test_block_groups_same_entity_and_date():
    # Two cerium-price phrasings, same year, one moved off 0.50 -> one cluster.
    ms = [
        _market("Will US cerium metal price exceed 3.5 USD/kg in 2027?", 0.40, volume=10, mid="a"),
        _market("Will the US cerium price stay above 3.5 USD/kg during 2027?", 0.50, mid="b"),
    ]
    clusters = block_markets(ms)
    assert len(clusters) == 1
    assert {m["id"] for m in clusters[0]} == {"a", "b"}


def test_block_separates_different_years():
    ms = [
        _market("Will cerium price exceed 3.5 USD/kg in 2027?", 0.40, volume=10, mid="a"),
        _market("Will cerium price exceed 3.5 USD/kg in 2028?", 0.40, volume=10, mid="b"),
    ]
    assert block_markets(ms) == [] or all(len(c) == 1 for c in block_markets(ms))


def test_block_drops_all_default_clusters():
    # Both untouched at 0.50 -> nothing anchors a correction -> dropped.
    ms = [
        _market("Will scandium balance exceed 5 t in 2027?", 0.50, mid="a"),
        _market("Will the scandium market balance top 5 t during 2027?", 0.50, mid="b"),
    ]
    assert block_markets(ms) == []


def test_block_caps_cluster_size():
    ms = [_market(f"Will neon price exceed {k} USD in 2027?", 0.40, volume=5, mid=str(k))
          for k in range(20)]
    clusters = block_markets(ms, max_cluster=12)
    assert clusters and all(len(c) <= 12 for c in clusters)


# ---- projection kernel ----

def test_project_equivalent_pools_to_weighted_mean():
    # a (heavy, traded) at 0.30, b (light, default) at 0.50 -> pull together,
    # closer to the heavy leg.
    out = project_constraints(
        {"a": 0.30, "b": 0.50}, {"a": 5.0, "b": 1.0},
        [{"a": "a", "b": "b", "type": "equivalent"}],
        correction_strength=1.0,
    )
    assert out["a"] == pytest.approx(out["b"], abs=1e-3)
    assert out["a"] == pytest.approx((5 * 0.30 + 1 * 0.50) / 6, abs=1e-3)


def test_project_implies_only_acts_on_violation():
    # P(a) <= P(b) already holds (0.3 <= 0.6) -> untouched.
    rel = [{"a": "a", "b": "b", "type": "implies"}]
    out = project_constraints({"a": 0.3, "b": 0.6}, {}, rel, correction_strength=1.0)
    assert out["a"] == pytest.approx(0.3) and out["b"] == pytest.approx(0.6)
    # Violated (0.7 > 0.4) -> pooled so a <= b.
    out2 = project_constraints({"a": 0.7, "b": 0.4}, {}, rel, correction_strength=1.0)
    assert out2["a"] <= out2["b"] + 1e-9


def test_project_negation_sums_to_one():
    out = project_constraints(
        {"a": 0.70, "b": 0.70}, {}, [{"a": "a", "b": "b", "type": "negation"}],
        correction_strength=1.0,
    )
    assert out["a"] + out["b"] == pytest.approx(1.0, abs=1e-6)


def test_project_exclusive_caps_sum_at_one():
    out = project_constraints(
        {"a": 0.7, "b": 0.6}, {}, [{"a": "a", "b": "b", "type": "exclusive"}],
        correction_strength=1.0,
    )
    assert out["a"] + out["b"] <= 1.0 + 1e-6


def test_project_correction_strength_partial_move():
    full = project_constraints({"a": 0.30, "b": 0.50}, {},
                               [{"a": "a", "b": "b", "type": "equivalent"}],
                               correction_strength=1.0)
    half = project_constraints({"a": 0.30, "b": 0.50}, {},
                               [{"a": "a", "b": "b", "type": "equivalent"}],
                               correction_strength=0.5)
    # Half-move lands between the original price and the full-coherence point.
    assert 0.30 < half["a"] < full["a"] + 1e-9


# ---- estimate pipeline (relations injected) ----

def _strategy(**kw):
    defaults = dict(min_confidence=0.8, dev_band=0.03,
                    correction_strength=1.0, skip_same_ladder=False)
    defaults.update(kw)
    return SemanticArbStrategy(**defaults)


def test_estimate_trades_equivalent_gap(monkeypatch):
    s = _strategy()
    group = [
        _market("Will cerium price exceed 3.5 USD/kg in 2027?", 0.30, volume=10, mid="a"),
        _market("Will cerium stay above 3.5 USD/kg through 2027?", 0.50, mid="b"),
    ]
    monkeypatch.setattr(s, "_extract_relations",
                        lambda qs: [{"a": 0, "b": 1, "type": "equivalent",
                                     "confidence": 0.95, "why": "same threshold/period"}])
    out = s.estimate(group)
    # Both legs move toward the weighted consensus (heavy 'a' pulls it down).
    assert set(out) == {"a", "b"}
    assert out["a"] == pytest.approx(out["b"], abs=1e-3)
    assert s.explain("a") and "equivalent" in s.explain("a")


def test_estimate_abstains_when_coherent(monkeypatch):
    s = _strategy()
    group = [
        _market("Will cerium price exceed 3.5 USD/kg in 2027?", 0.40, volume=10, mid="a"),
        _market("Will cerium stay above 3.5 USD/kg through 2027?", 0.41, volume=10, mid="b"),
    ]
    monkeypatch.setattr(s, "_extract_relations",
                        lambda qs: [{"a": 0, "b": 1, "type": "equivalent",
                                     "confidence": 0.95, "why": "x"}])
    # Prices already agree within dev_band -> no trade.
    assert s.estimate(group) == {}


def test_estimate_drops_low_confidence(monkeypatch):
    s = _strategy()
    group = [_market("q a", 0.30, volume=10, mid="a"), _market("q b", 0.60, mid="b")]
    monkeypatch.setattr(s, "_extract_relations",
                        lambda qs: [{"a": 0, "b": 1, "type": "equivalent",
                                     "confidence": 0.5, "why": "unsure"}])
    assert s.estimate(group) == {}


def test_estimate_skips_same_ladder(monkeypatch):
    # Same measurable_key (only the threshold differs) -> ladder_arb's turf.
    s = _strategy(skip_same_ladder=True)
    group = [
        _market("Will cerium price exceed 3 USD/kg in 2027?", 0.30, volume=10, mid="a"),
        _market("Will cerium price exceed 5 USD/kg in 2027?", 0.60, volume=10, mid="b"),
    ]
    monkeypatch.setattr(s, "_extract_relations",
                        lambda qs: [{"a": 0, "b": 1, "type": "implies",
                                     "confidence": 0.95, "why": "x"}])
    assert s.estimate(group) == {}


def test_extract_relations_voting(monkeypatch):
    # 2 samples, agreement 0.5 -> need >=1; a relation seen once survives but a
    # garbled-index one is dropped at parse time.
    s = SemanticArbStrategy(n_samples=2, agreement=1.0)
    calls = iter([
        '{"relations":[{"a":1,"b":2,"type":"equivalent","confidence":0.9,"why":"x"}]}',
        '{"relations":[{"a":1,"b":2,"type":"equivalent","confidence":0.8,"why":"y"}]}',
    ])

    class _FakeLLM:
        def json_completion(self, system, user, temperature=0.0):
            return next(calls)

    s._llm = _FakeLLM()
    rels = s._extract_relations(["q1", "q2"])
    # Found in both samples (agreement 1.0 satisfied), mean confidence averaged.
    assert len(rels) == 1
    assert rels[0]["type"] == "equivalent"
    assert rels[0]["confidence"] == pytest.approx(0.85)
