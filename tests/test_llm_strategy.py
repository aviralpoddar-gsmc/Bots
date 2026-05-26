"""Tests for the LLM strategy's deterministic parts (CDF + band widening),
with the model call stubbed — no network, no real model."""

import json

from quantbots.strategies.llm import LLMStrategy


class FakeLLM:
    def __init__(self, pct):
        self._pct = pct

    def json_completion(self, system, user, temperature=0.0):
        return json.dumps(self._pct)


def _strat(pct, **kw):
    s = LLMStrategy(**kw)
    s.llm = FakeLLM(pct)  # stub out the real local model
    return s


PCT = {"p10": 60, "p25": 65, "p50": 70, "p75": 75, "p90": 80, "reasoning": "x"}


def test_cdf_reads_high_prob_below_median_strike():
    s = _strat(PCT, spread_mult=1.0)
    m = {"id": "a", "question": "Will X exceed 60 by date?", "threshold": 60, "direction": "exceeds"}
    est = s.estimate([m])
    # strike 60 == p10 -> P(exceed) ~ 0.90
    assert est["a"] > 0.8


def test_cdf_reads_low_prob_above_median_strike():
    s = _strat(PCT, spread_mult=1.0)
    m = {"id": "a", "question": "Will X exceed 80?", "threshold": 80, "direction": "exceeds"}
    est = s.estimate([m])
    assert est["a"] < 0.2


def test_band_widening_pulls_extreme_probs_toward_half():
    m = {"id": "a", "question": "Will X exceed 80?", "threshold": 80, "direction": "exceeds"}
    narrow = _strat(PCT, spread_mult=1.0).estimate([m])["a"]
    widened = _strat(PCT, spread_mult=2.0).estimate([m])["a"]
    # Widening the band makes the same out-of-band strike less extreme (closer to 0.5).
    assert widened > narrow


def test_direction_below_inverts():
    s = _strat(PCT, spread_mult=1.0)
    m = {"id": "a", "question": "Will X be below 60?", "threshold": 60, "direction": "below"}
    est = s.estimate([m])
    assert est["a"] < 0.2  # P(below 60) when 60 is p10


def test_max_groups_caps_calls():
    s = _strat(PCT, max_groups=2)
    subjects = ["gold", "silver", "copper", "cotton", "wheat"]
    markets = [
        {"id": w, "question": f"Will {w} exceed 5?", "threshold": 5, "direction": "exceeds"}
        for w in subjects
    ]
    # Five distinct measurables -> five groups, but the cap limits to 2.
    assert len(s.group(markets)) == 2
