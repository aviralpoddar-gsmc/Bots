from __future__ import annotations

from constraint_scanner.detectors import (
    detect_answer_sum_violations,
    detect_duplicate_titles,
    detect_numeric_cdf_monotonicity,
    detect_pseudo_numeric_bounds,
    normalize_title,
)
from constraint_scanner.models import Answer, FullMarket, LiteMarket


def _numeric(id_, midpoints_probs, sum_to_one=False, outcome="MULTI_NUMERIC"):
    """Build an independent numeric market from (midpoint, probability) pairs."""
    return FullMarket.model_validate(
        {
            "id": id_,
            "question": "BTC price at year-end?",
            "outcomeType": outcome,
            "mechanism": "cpmm-multi-1",
            "shouldAnswersSumToOne": sum_to_one,
            "unit": "USD",
            "answers": [
                {"id": str(i), "text": f"~{mid}", "midpoint": mid, "probability": p}
                for i, (mid, p) in enumerate(midpoints_probs)
            ],
        }
    )


def _binary(id_: str, q: str, p: float) -> LiteMarket:
    return LiteMarket.model_validate(
        {"id": id_, "question": q, "outcomeType": "BINARY", "probability": p}
    )


def test_normalize_title():
    assert normalize_title("Will BTC > $100,000?") == "will btc 100 000"
    assert normalize_title("Will BTC > $100,000!?  ") == "will btc 100 000"


def test_duplicate_titles_flags_only_when_spread():
    ms = [
        _binary("a", "Will BTC exceed $100k by 2027?", 0.62),
        _binary("b", "will btc exceed 100k by 2027", 0.71),
        _binary("c", "Different question entirely", 0.40),
    ]
    vs = detect_duplicate_titles(ms)
    assert len(vs) == 1
    assert set(vs[0].market_ids) == {"a", "b"}
    assert vs[0].detail["spread"] > 0.05


def test_duplicate_titles_ignores_small_spread():
    ms = [
        _binary("a", "Same question", 0.50),
        _binary("b", "Same question", 0.51),
    ]
    assert detect_duplicate_titles(ms) == []


def test_answer_sum_violation_detected():
    market = FullMarket.model_validate(
        {
            "id": "x",
            "question": "Who wins?",
            "outcomeType": "MULTIPLE_CHOICE",
            "mechanism": "cpmm-multi-1",
            "shouldAnswersSumToOne": True,
            "answers": [
                {"id": "1", "text": "A", "probability": 0.40},
                {"id": "2", "text": "B", "probability": 0.40},
                {"id": "3", "text": "C", "probability": 0.30},
            ],
        }
    )
    vs = detect_answer_sum_violations([market])
    assert len(vs) == 1
    assert vs[0].detail["direction"] == "over"
    assert abs(vs[0].detail["sum_probability"] - 1.10) < 1e-6


def test_answer_sum_ignores_independent_multichoice():
    market = FullMarket.model_validate(
        {
            "id": "y",
            "question": "Independent answers",
            "outcomeType": "MULTIPLE_CHOICE",
            "shouldAnswersSumToOne": False,
            "answers": [
                {"id": "1", "probability": 0.5},
                {"id": "2", "probability": 0.5},
                {"id": "3", "probability": 0.5},
            ],
        }
    )
    assert detect_answer_sum_violations([market]) == []


def test_pseudo_numeric_value_probability_match():
    good = LiteMarket.model_validate(
        {
            "id": "p1",
            "question": "BTC price by year-end",
            "outcomeType": "PSEUDO_NUMERIC",
            "min": 0,
            "max": 200000,
            "probability": 0.5,
            "value": 100000,
            "isLogScale": False,
        }
    )
    assert detect_pseudo_numeric_bounds([good]) == []


def test_pseudo_numeric_value_probability_mismatch():
    bad = LiteMarket.model_validate(
        {
            "id": "p2",
            "question": "BTC price",
            "outcomeType": "PSEUDO_NUMERIC",
            "min": 0,
            "max": 100,
            "probability": 0.5,
            "value": 80,
            "isLogScale": False,
        }
    )
    vs = detect_pseudo_numeric_bounds([bad])
    assert len(vs) == 1
    assert vs[0].detail["reason"] == "value_probability_mismatch"


def test_api_answer_shape_parses():
    # The public API renames prob -> probability and uses pool: {YES, NO}.
    a = Answer.model_validate(
        {
            "id": "1",
            "text": "100k-150k",
            "index": 3,
            "probability": 0.42,
            "pool": {"YES": 120.0, "NO": 80.0},
            "midpoint": 125000,
            "isOther": False,
        }
    )
    assert a.probability == 0.42
    assert a.midpoint == 125000
    assert a.pool == {"YES": 120.0, "NO": 80.0}
    assert a.index == 3


def test_answer_sum_covers_numeric_sum_to_one():
    # NUMBER buckets are a partition that must sum to 1.
    market = _numeric(
        "n1", [(10, 0.5), (20, 0.5), (30, 0.3)], sum_to_one=True, outcome="NUMBER"
    )
    vs = detect_answer_sum_violations([market])
    assert len(vs) == 1
    assert vs[0].detail["direction"] == "over"


def test_answer_sum_ignores_other_answer():
    market = FullMarket.model_validate(
        {
            "id": "z",
            "question": "Who wins?",
            "outcomeType": "MULTIPLE_CHOICE",
            "shouldAnswersSumToOne": True,
            "answers": [
                {"id": "1", "text": "A", "probability": 0.6},
                {"id": "2", "text": "B", "probability": 0.4},
                {"id": "3", "text": "Other", "probability": 0.3, "isOther": True},
            ],
        }
    )
    # 0.6 + 0.4 = 1.0 once Other is excluded.
    assert detect_answer_sum_violations([market]) == []


def test_cdf_monotonicity_clean_when_decreasing():
    # Survival probabilities fall as the bucket midpoint rises — valid.
    market = _numeric("c1", [(10, 0.9), (20, 0.5), (30, 0.2)])
    assert detect_numeric_cdf_monotonicity([market]) == []


def test_cdf_monotonicity_flags_inversion():
    # Higher bucket (20) priced above the lower one (10): impossible.
    market = _numeric("c2", [(10, 0.5), (20, 0.8), (30, 0.2)])
    vs = detect_numeric_cdf_monotonicity([market])
    assert len(vs) == 1
    assert abs(vs[0].severity - 0.3) < 1e-9
    assert vs[0].detail["best_arb"] == {"buy_yes": "~10", "buy_no": "~20"}
    assert len(vs[0].detail["adjacent_inversions"]) == 1


def test_cdf_monotonicity_sorts_unordered_answers():
    # Same data as the clean case, but answers arrive out of midpoint order.
    market = _numeric("c3", [(30, 0.2), (10, 0.9), (20, 0.5)])
    assert detect_numeric_cdf_monotonicity([market]) == []


def test_cdf_monotonicity_accumulated_drift():
    # Each adjacent gap (0.009) is below tolerance, but the 10->30 span (0.018)
    # exceeds it — the running-min logic catches the best non-adjacent arb.
    market = _numeric("c4", [(10, 0.500), (20, 0.509), (30, 0.518)])
    vs = detect_numeric_cdf_monotonicity([market])
    assert len(vs) == 1
    assert abs(vs[0].severity - 0.018) < 1e-9
    assert vs[0].detail["adjacent_inversions"] == []


def test_cdf_monotonicity_skips_sum_to_one():
    # A sum-to-one numeric market is a density, not a survival function.
    market = _numeric("c5", [(10, 0.5), (20, 0.8), (30, 0.2)], sum_to_one=True)
    assert detect_numeric_cdf_monotonicity([market]) == []
