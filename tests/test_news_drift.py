"""Tests for the 007 news bot: the local-LLM headline classifier (coercion + filter),
the news->SIG_<COM>_NEWS aggregation (recency/confidence weighting, staleness filter,
parse caching), and the NewsDriftStrategy drift + abstain gates. No network: the LLM is
injected via classify_fn / a fake llm."""

import time
from datetime import UTC, datetime, timedelta

from quantbots.llm import news_extractor as nx
from quantbots.processing.signals import compute_news_signal
from quantbots.store.db import Store
from quantbots.strategies.news_drift import NewsDriftStrategy


# --- extractor coercion / hard-filter -------------------------------------------
def test_coerce_keeps_valid_signed_record():
    r = nx._coerce({"commodity": "Gold", "direction": 1, "confidence": 0.8,
                    "is_price_event": True, "benchmark": "LBMA"})
    assert r == {"commodity": "gold", "direction": 1, "confidence": 0.8,
                 "is_price_event": True, "benchmark": "LBMA"}


def test_coerce_filters_untradeable_and_clamps():
    # commodity not in the tradeable set -> null signal
    assert nx._coerce({"commodity": "lithium", "direction": 1, "confidence": 0.9,
                       "is_price_event": True})["commodity"] is None
    # is_price_event false -> null signal even if a commodity is named
    assert nx._coerce({"commodity": "gold", "direction": 1, "confidence": 0.9,
                       "is_price_event": False})["is_price_event"] is False
    # direction/confidence clamped
    c = nx._coerce({"commodity": "wti", "direction": 9, "confidence": 5,
                    "is_price_event": True})
    assert c["direction"] == 1 and c["confidence"] == 1.0


def test_classify_handles_llm_failure():
    class Boom:
        def json_completion(self, *a, **k):
            raise RuntimeError("model down")
    assert nx.classify("Oil falls 3%", llm=Boom())["commodity"] is None


def test_classify_parses_fake_llm():
    class Fake:
        def json_completion(self, system, user):
            return '{"commodity":"gold","direction":1,"confidence":0.7,"is_price_event":true,"benchmark":null}'
    r = nx.classify("Gold edges up on rate-cut hopes", llm=Fake())
    assert r["commodity"] == "gold" and r["direction"] == 1 and r["is_price_event"]


# --- aggregation -> SIG_<COM>_NEWS ----------------------------------------------
def _seed(store, items):
    """items: list of (entity, text, hours_ago)."""
    now = datetime.now(UTC)
    rows = []
    for entity, text, hrs in items:
        ts = (now - timedelta(hours=hrs)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({"source": "rss", "entity": entity, "ts": ts, "value": None,
                     "text": text, "payload": {}})
    store.upsert_observations(rows)
    return now


def _fake_classifier(mapping):
    """headline substring -> record."""
    def fn(text):
        for k, rec in mapping.items():
            if k in text:
                return rec
        return dict(nx._NULL)
    return fn


def test_compute_news_signal_sign_and_count(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    now = _seed(store, [
        ("INVESTING_COMMODITIES", "Gold rallies to record on rate cut", 1),
        ("INVESTING_COMMODITIES", "Gold extends gains amid weak dollar", 2),
        ("OILPRICE_MAIN", "Oil slumps as demand fears mount", 1),
    ])
    classify = _fake_classifier({
        "Gold": {"commodity": "gold", "direction": 1, "confidence": 0.8, "is_price_event": True, "benchmark": None},
        "Oil": {"commodity": "wti", "direction": -1, "confidence": 0.7, "is_price_event": True, "benchmark": None},
    })
    out = compute_news_signal(store, feeds=["INVESTING_COMMODITIES", "OILPRICE_MAIN"],
                              classify_fn=classify, now=now)
    sig = {o.entity: o for o in out}
    assert sig["SIG_GOLD_NEWS"].value > 0.5 and sig["SIG_GOLD_NEWS"].payload["n_items"] == 2
    assert sig["SIG_WTI_NEWS"].value < 0 and sig["SIG_WTI_NEWS"].payload["n_items"] == 1


def test_compute_news_signal_drops_stale_and_caches(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    now = _seed(store, [
        ("INVESTING_COMMODITIES", "Copper jumps on supply outage", 2),       # fresh
        ("INVESTING_COMMODITIES", "Copper old news from last week", 240),     # stale (>96h)
    ])
    rec = {"commodity": "copper", "direction": 1, "confidence": 0.9, "is_price_event": True, "benchmark": None}
    calls = {"n": 0}
    def classify(text):
        calls["n"] += 1
        return rec
    out = compute_news_signal(store, feeds=["INVESTING_COMMODITIES"], classify_fn=classify, now=now)
    sig = {o.entity: o for o in out}
    assert sig["SIG_COPPER_NEWS"].payload["n_items"] == 1   # stale item excluded
    assert calls["n"] == 1                                   # only the fresh item classified
    # re-run: the fresh parse is cached -> classifier NOT called again
    compute_news_signal(store, feeds=["INVESTING_COMMODITIES"], classify_fn=classify, now=now)
    assert calls["n"] == 1


def test_compute_news_signal_recency_weight(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    now = _seed(store, [
        ("INVESTING_COMMODITIES", "Silver surges fresh", 1),     # fresh bullish
        ("INVESTING_COMMODITIES", "Silver fell stale", 80),      # old bearish
    ])
    classify = _fake_classifier({
        "surges": {"commodity": "silver", "direction": 1, "confidence": 0.8, "is_price_event": True, "benchmark": None},
        "fell": {"commodity": "silver", "direction": -1, "confidence": 0.8, "is_price_event": True, "benchmark": None},
    })
    out = compute_news_signal(store, feeds=["INVESTING_COMMODITIES"], classify_fn=classify, now=now)
    sig = {o.entity: o for o in out}
    assert sig["SIG_SILVER_NEWS"].value > 0   # fresher bullish dominates the older bearish


# --- strategy drift + abstain gates ---------------------------------------------
class _Obs:
    def __init__(self, vals):
        self.vals = vals  # entity -> dict
    def latest_observation(self, entity, source=None):
        return self.vals.get(entity)


def _market(question, close_years=0.3):
    return {"id": question[:30], "question": question, "probability": 0.5,
            "closeTime": time.time() * 1000 + close_years * 365.25 * 24 * 3600 * 1000,
            "totalLiquidity": 100, "isResolved": False}


_GOLD_Q = "Will Gold spot price exceed $3,500/ozt on Dec 31, 2026?"


def _obs_with_news(raw, n_items):
    import json
    return _Obs({
        "GOLD": {"entity": "GOLD", "value": 3300.0, "ts": "2026-06-09"},
        "SIG_GOLD_NEWS": {"entity": "SIG_GOLD_NEWS", "value": raw,
                          "payload": json.dumps({"n_items": n_items, "n_pos": n_items, "n_neg": 0,
                                                  "raw": raw, "halflife_h": 36, "top_headlines": []})},
    })


def test_news_drift_bullish_raises_exceed_prob():
    s = NewsDriftStrategy(k=0.08, min_conviction=0.25, min_items=2)
    base = NewsDriftStrategy(k=0.0, min_conviction=0.0, min_items=1, min_drift=0.0)  # zero-drift ref
    m = _market(_GOLD_Q)
    s.bind(_obs_with_news(+0.6, 3)); base.bind(_obs_with_news(+0.6, 3))
    p = s.estimate([m]).get(m["id"])
    p0 = base.estimate([m]).get(m["id"])
    assert p is not None and p0 is not None and p > p0  # bullish news lifts P(exceed above-spot strike)


def test_news_drift_abstains_below_gates():
    m = _market(_GOLD_Q)
    # too few items
    s1 = NewsDriftStrategy(min_items=3); s1.bind(_obs_with_news(+0.6, 2))
    assert s1.estimate([m]) == {}
    # conviction too low
    s2 = NewsDriftStrategy(min_conviction=0.5); s2.bind(_obs_with_news(+0.1, 5))
    assert s2.estimate([m]) == {}
    # no signal at all
    s3 = NewsDriftStrategy(); s3.bind(_Obs({"GOLD": {"value": 3300.0}}))
    assert s3.estimate([m]) == {}
