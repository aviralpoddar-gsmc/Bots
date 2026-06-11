"""Behavior tests for the mercury_ensemble strategy with the model stubbed —
no network. The fake maps each sample's temperature to a percentile set, so a
deterministic ensemble of agreeing/disagreeing samples can be constructed."""

import json
import os

os.environ.setdefault("INCEPTION_API_KEY", "test-key")  # client is stubbed below

from quantbots.strategies.mercury_ensemble import MercuryEnsembleStrategy  # noqa: E402

# A confident bullish forecast: strike 60 sits at p10, so P(exceed 60) ~ 0.9.
BULLISH = {"p10": 60, "p25": 65, "p50": 70, "p75": 75, "p90": 80, "reasoning": "x"}
BEARISH = {"p10": 20, "p25": 25, "p50": 30, "p75": 35, "p90": 40, "reasoning": "x"}


class FakeMercury:
    """temp_to_pct: a dict, or a callable temperature -> percentile dict."""

    def __init__(self, temp_to_pct):
        self._f = temp_to_pct

    def json_completion(self, system, user, temperature=0.0):
        pct = self._f(temperature) if callable(self._f) else self._f
        return json.dumps(pct)


def _strat(temp_to_pct, **kw):
    kw.setdefault("conf_cap", 0.99)
    kw.setdefault("spread_mult", 1.0)
    kw.setdefault("n_samples", 4)
    kw.setdefault("direction_agreement_floor", 0.7)
    s = MercuryEnsembleStrategy(**kw)
    s.llm = FakeMercury(temp_to_pct)
    return s


def _market(threshold=60, direction="exceeds", prob=0.5):
    return {"id": "a", "question": f"Will X exceed {threshold}?",
            "threshold": threshold, "direction": direction, "probability": prob}


def test_consensus_yields_confident_estimate():
    s = _strat(BULLISH, epistemic_tau=0.04)  # all samples identical -> epistemic 0
    est = s.estimate([_market(prob=0.5)])
    assert est["a"] > 0.8  # full confidence, P(exceed) ~ 0.9


def test_abstains_when_samples_split_on_direction():
    # low temps bullish, high temps bearish -> ~50% direction agreement < floor
    split = lambda t: BULLISH if t < 0.7 else BEARISH  # noqa: E731
    s = _strat(split, n_samples=4, temperature_lo=0.4, temperature_hi=1.0)
    est = s.estimate([_market(prob=0.5)])
    assert "a" not in est  # abstained


def test_disagreement_shrinks_estimate_toward_market():
    # All bullish in direction but spread in magnitude -> positive epistemic.
    spread = lambda t: {"p10": 50 + 40 * t, "p25": 55 + 40 * t, "p50": 60 + 40 * t,  # noqa: E731
                        "p75": 65 + 40 * t, "p90": 70 + 40 * t, "reasoning": "x"}
    mkt = _market(threshold=70, prob=0.5)  # below all sample medians -> all bullish, spread in prob
    loose = _strat(spread, n_samples=4, epistemic_tau=1.0).estimate([mkt])["a"]
    tight = _strat(spread, n_samples=4, epistemic_tau=0.02).estimate([mkt])["a"]
    # Stronger shrinkage (smaller tau) pulls the estimate closer to the 0.5 market.
    assert 0.5 < tight < loose


def test_abstains_below_quorum():
    # Every sample returns JSON with no percentile keys -> all fail -> no quorum.
    s = _strat({"reasoning": "no numbers"}, n_samples=4, min_quorum=3)
    assert s.estimate([_market()]) == {}


def test_conf_cap_clamps_extreme_consensus():
    tight = {"p10": 99, "p25": 99.5, "p50": 100, "p75": 100.5, "p90": 101, "reasoning": "x"}
    s = _strat(tight, conf_cap=0.75, epistemic_tau=1.0)
    est = s.estimate([_market(threshold=97, prob=0.5)])["a"]
    assert est == 0.75  # P(exceed 97) ~ 1, clamped to the ceiling


def test_explain_reports_uncertainty():
    s = _strat(BULLISH, epistemic_tau=0.04)
    s.estimate([_market(prob=0.5)])
    text = s.explain("a")
    assert text and "epistemic" in text.lower()
