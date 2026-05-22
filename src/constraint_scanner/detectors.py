from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .models import FullMarket, LiteMarket

ANSWER_SUM_TOLERANCE = 0.02
DUPLICATE_PROB_GAP = 0.05
PSEUDO_NUMERIC_TOLERANCE = 1e-3
# Minimum survival-probability inversion to flag as a CDF monotonicity arb.
# A gap of g is also the guaranteed edge per Ṁ1 staked, so this is a floor on
# the edge we bother reporting (1 cent), filtering AMM rounding noise.
CDF_MONOTONICITY_TOLERANCE = 0.01

# cpmm-multi-1 markets whose answer probabilities are a partition that must
# sum to 1 (when shouldAnswersSumToOne is not explicitly False).
SUM_TO_ONE_TYPES = {"MULTIPLE_CHOICE", "NUMBER", "MULTI_NUMERIC", "DATE"}
# Numeric markets whose independent answers (shouldAnswersSumToOne == False)
# encode a survival function P(X >= bucket_edge) over ascending midpoints,
# which must be monotonically non-increasing.
CUMULATIVE_NUMERIC_TYPES = {"MULTI_NUMERIC", "DATE"}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


@dataclass
class Violation:
    kind: str
    severity: float
    market_ids: list[str]
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "market_ids": self.market_ids,
            "detail": self.detail,
        }


def normalize_title(title: str) -> str:
    s = title.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def detect_answer_sum_violations(markets: Iterable[FullMarket]) -> list[Violation]:
    """Sum-to-one markets (MC / NUMBER / sum-to-one MULTI_NUMERIC) whose answer
    probabilities don't sum to ~1. Excludes 'Other' answers, which aren't part
    of the partition."""
    out: list[Violation] = []
    for m in markets:
        if m.outcome_type not in SUM_TO_ONE_TYPES:
            continue
        if not m.answers:
            continue
        if m.should_answers_sum_to_one is False:
            continue
        probs = [
            a.probability
            for a in m.answers
            if a.probability is not None and not a.is_other
        ]
        if len(probs) < 2:
            continue
        total = sum(probs)
        deviation = abs(total - 1.0)
        if deviation > ANSWER_SUM_TOLERANCE:
            out.append(
                Violation(
                    kind="answer_sum",
                    severity=deviation,
                    market_ids=[m.id],
                    detail={
                        "question": m.question,
                        "answer_count": len(probs),
                        "sum_probability": round(total, 6),
                        "deviation": round(deviation, 6),
                        "direction": "over" if total > 1.0 else "under",
                    },
                )
            )
    return out


def detect_numeric_cdf_monotonicity(markets: Iterable[FullMarket]) -> list[Violation]:
    """Independent numeric markets (MULTI_NUMERIC / DATE with
    shouldAnswersSumToOne == False) encode a survival function: each answer's
    probability is P(X >= bucket edge), so probabilities must be non-increasing
    as the bucket midpoint rises. An inversion (a higher bucket priced above a
    lower one) is a structural arbitrage: buy YES on the lower bucket and NO on
    the higher one for a guaranteed payout >= 1 at cost (1 - edge). Severity is
    the best single such edge available in the market (max over i<j of pj - pi).
    """
    out: list[Violation] = []
    for m in markets:
        if m.outcome_type not in CUMULATIVE_NUMERIC_TYPES:
            continue
        if m.should_answers_sum_to_one is not False:
            continue
        if not m.answers:
            continue
        pts = sorted(
            (
                (a.midpoint, a.probability, a.text)
                for a in m.answers
                if a.midpoint is not None
                and a.probability is not None
                and not a.is_other
            ),
            key=lambda x: x[0],
        )
        if len(pts) < 2:
            continue

        # Best arb = max over i<j of (p_j - p_i), via running min of survival prob.
        run_min_p, run_min_t = pts[0][1], pts[0][2]
        best_edge, best = 0.0, None
        for _, p, t in pts[1:]:
            if p - run_min_p > best_edge:
                best_edge, best = p - run_min_p, (run_min_t, t)
            if p < run_min_p:
                run_min_p, run_min_t = p, t
        if best_edge <= CDF_MONOTONICITY_TOLERANCE:
            continue

        inversions = [
            {
                "below": {"text": lo_t, "midpoint": lo_mid, "prob": round(lo_p, 4)},
                "above": {"text": hi_t, "midpoint": hi_mid, "prob": round(hi_p, 4)},
                "edge": round(hi_p - lo_p, 4),
            }
            for (lo_mid, lo_p, lo_t), (hi_mid, hi_p, hi_t) in zip(pts, pts[1:])
            if hi_p - lo_p > CDF_MONOTONICITY_TOLERANCE
        ]
        out.append(
            Violation(
                kind="numeric_cdf_monotonicity",
                severity=best_edge,
                market_ids=[m.id],
                detail={
                    "question": m.question,
                    "unit": m.unit,
                    "answer_count": len(pts),
                    "max_edge": round(best_edge, 4),
                    "best_arb": (
                        {"buy_yes": best[0], "buy_no": best[1]} if best else None
                    ),
                    "adjacent_inversions": inversions,
                },
            )
        )
    return out


def detect_duplicate_titles(markets: Iterable[LiteMarket]) -> list[Violation]:
    groups: dict[str, list[LiteMarket]] = defaultdict(list)
    for m in markets:
        if m.outcome_type != "BINARY":
            continue
        if m.probability is None:
            continue
        groups[normalize_title(m.question)].append(m)

    out: list[Violation] = []
    for norm, ms in groups.items():
        if len(ms) < 2:
            continue
        probs = [m.probability for m in ms if m.probability is not None]
        if not probs:
            continue
        spread = max(probs) - min(probs)
        if spread < DUPLICATE_PROB_GAP:
            continue
        out.append(
            Violation(
                kind="duplicate_title",
                severity=spread,
                market_ids=[m.id for m in ms],
                detail={
                    "normalized_title": norm,
                    "count": len(ms),
                    "spread": round(spread, 6),
                    "probabilities": [round(p, 4) for p in probs],
                    "questions": [m.question for m in ms],
                },
            )
        )
    return out


def _expected_pseudo_value(p: float, lo: float, hi: float, log_scale: bool) -> float:
    if log_scale:
        return (hi - lo + 1) ** p + lo - 1
    return lo + (hi - lo) * p


def detect_pseudo_numeric_bounds(markets: Iterable[LiteMarket]) -> list[Violation]:
    out: list[Violation] = []
    for m in markets:
        if m.outcome_type != "PSEUDO_NUMERIC":
            continue
        if m.min is None or m.max is None or m.value is None or m.probability is None:
            continue
        if m.min >= m.max:
            out.append(
                Violation(
                    kind="pseudo_numeric_bounds",
                    severity=1.0,
                    market_ids=[m.id],
                    detail={
                        "question": m.question,
                        "reason": "min_not_less_than_max",
                        "min": m.min,
                        "max": m.max,
                    },
                )
            )
            continue
        if not (0.0 <= m.probability <= 1.0):
            out.append(
                Violation(
                    kind="pseudo_numeric_bounds",
                    severity=abs(m.probability - 0.5),
                    market_ids=[m.id],
                    detail={
                        "question": m.question,
                        "reason": "probability_out_of_range",
                        "probability": m.probability,
                    },
                )
            )
            continue
        if not (m.min <= m.value <= m.max):
            out.append(
                Violation(
                    kind="pseudo_numeric_bounds",
                    severity=1.0,
                    market_ids=[m.id],
                    detail={
                        "question": m.question,
                        "reason": "value_outside_bounds",
                        "value": m.value,
                        "min": m.min,
                        "max": m.max,
                    },
                )
            )
            continue
        expected = _expected_pseudo_value(
            m.probability, m.min, m.max, bool(m.is_log_scale)
        )
        scale = max(abs(m.max - m.min), 1.0)
        rel_err = abs(m.value - expected) / scale
        if rel_err > PSEUDO_NUMERIC_TOLERANCE:
            out.append(
                Violation(
                    kind="pseudo_numeric_bounds",
                    severity=rel_err,
                    market_ids=[m.id],
                    detail={
                        "question": m.question,
                        "reason": "value_probability_mismatch",
                        "value": m.value,
                        "expected": round(expected, 6),
                        "probability": m.probability,
                        "is_log_scale": bool(m.is_log_scale),
                    },
                )
            )
    return out
