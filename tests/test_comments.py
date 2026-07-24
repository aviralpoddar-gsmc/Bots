"""Comment-society pilot: reader, judge, consensus, store, strategies, cycle.

No network, no LLM — fakes throughout. The cycle tests assert the DRY contract:
a client whose write methods raise proves the cycle never posts or bets.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from quantbots.comments.consensus import market_consensus, platt_extremize
from quantbots.comments.cycle import run_judge_cycle
from quantbots.comments.judge import build_evidence, judge_comment
from quantbots.comments.reader import attached_bet, comment_text, is_reply
from quantbots.store.db import Store
from quantbots.strategies import get_strategy

# --- fixtures ----------------------------------------------------------------

GOLD_MARKET = {
    "id": "mkt_gold_1",
    "question": "Will the price of gold exceed $2500 per troy ounce on 2026-12-31?",
    "probability": 0.55,
    "totalLiquidity": 500,
    "closeTime": 4102444800000,  # 2100 — far future
    "lastCommentTime": 1_800_000_000_000,
}


def tiptap(text: str) -> dict:
    return {"type": "doc", "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": text}]}]}


def make_comment(cid="c1", author="FedPolicyBot", text="Gold is at $1900, far below the strike.",
                 bet=("YES", 250.0), reply_to=None, market_id="mkt_gold_1") -> dict:
    c = {"id": cid, "contractId": market_id, "userUsername": author,
         "content": tiptap(text), "createdTime": 1_800_000_000_000}
    if bet:
        c.update({"betId": f"bet_{cid}", "betOutcome": bet[0], "betAmount": bet[1]})
    if reply_to:
        c["replyToCommentId"] = reply_to
    return c


NOW_ISO = datetime.now(UTC).isoformat()          # fresh -> HIGH confidence allowed
STALE_ISO = (datetime.now(UTC) - timedelta(days=13)).isoformat()


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "test.sqlite") as s:
        s.upsert_observations([{"source": "stooq", "entity": "GOLD",
                                "ts": NOW_ISO, "value": 4400.0}])
        yield s


class FakeLLM:
    """Returns a canned JSON verdict; records the prompts it saw."""

    def __init__(self, response: dict | str | Exception):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def json_completion(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        if isinstance(self.response, Exception):
            raise self.response
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response)


UNSOUND_HIGH = {"verdict": "unsound", "confidence": "high",
                "factual_errors": ["claims $1900 -> feed says $4400/oz"],
                "reply": "Per LBMA/stooq, gold is $4400/oz, not $1900."}


# --- reader ------------------------------------------------------------------

def test_comment_text_tiptap_and_fallbacks():
    assert comment_text(make_comment(text="hello world")) == "hello world"
    nested = {"type": "doc", "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "a"}]},
        {"type": "blockquote", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "b"}]}]}]}
    assert comment_text({"content": nested}).split() == ["a", "b"]
    assert comment_text({"content": "plain string"}) == "plain string"
    assert comment_text({}) == ""


def test_attached_bet_and_reply_detection():
    c = make_comment()
    assert attached_bet(c) == {"bet_id": "bet_c1", "outcome": "YES", "amount": 250.0,
                               "limit_prob": None}
    assert attached_bet(make_comment(bet=None)) is None
    assert is_reply(make_comment(reply_to="parent")) and not is_reply(c)


# --- consensus math ----------------------------------------------------------

def test_platt_extremize_pushes_away_from_half():
    assert platt_extremize(0.5) == pytest.approx(0.5)
    assert 0.80 < platt_extremize(0.7) < 0.82   # σ(√3·logit(0.7)) ≈ 0.813
    assert platt_extremize(0.3) == pytest.approx(1 - platt_extremize(0.7), abs=1e-9)
    assert platt_extremize(0.999999) <= 0.98    # clamped


def test_market_consensus_one_vote_per_user_latest_wins():
    bets = [  # newest-first, as the API returns
        {"userId": "u1", "amount": 50, "probAfter": 0.70},
        {"userId": "u1", "amount": 50, "probAfter": 0.10},  # older u1 bet — ignored
        {"userId": "u2", "amount": 50, "probAfter": 0.60},
        {"userId": "u3", "amount": 50, "probAfter": 0.80},
        {"userId": "u4", "amount": 50, "probAfter": 0.65, "isRedemption": True},
        {"userId": "u5", "amount": 0.5, "probAfter": 0.99},  # dust
    ]
    c = market_consensus(bets, min_forecasters=3)
    assert c["n_forecasters"] == 3
    assert c["p_mean"] == pytest.approx(0.70)
    assert c["p_extreme"] > 0.70                 # extremized outward
    assert market_consensus(bets, min_forecasters=4) is None


# --- judge -------------------------------------------------------------------

def test_build_evidence_converts_feed_and_requires_anchor(store):
    ev = build_evidence(GOLD_MARKET, store)
    assert ev["entity"] == "GOLD" and ev["feed_value"] == 4400.0
    assert ev["feed_unit"] == "USD/oz" and ev["market_prob"] == 0.55
    # no linked entity -> no evidence -> judge must not run
    assert build_evidence({"id": "x", "question": "Will it rain tomorrow?"}, store) is None


def test_build_evidence_prefers_tal_and_skips_feed_factor(store):
    # tal_smm is already in market units and FRESHER-preferred over stooq
    store.upsert_observations([{"source": "tal_smm", "entity": "GOLD",
                                "ts": NOW_ISO, "value": 4109.38}])
    ev = build_evidence(GOLD_MARKET, store)
    assert ev["feed_source"] == "tal_smm" and ev["feed_value"] == 4109.38
    # stooq silver quotes cents/oz -> factor 0.01 applies ONLY to stooq
    silver_mkt = {"id": "mkt_ag", "question": "Will silver exceed 60 USD per troy ounce?",
                  "probability": 0.4, "totalLiquidity": 500}
    store.upsert_observations([{"source": "stooq", "entity": "SILVER",
                                "ts": NOW_ISO, "value": 7461.5}])
    assert build_evidence(silver_mkt, store)["feed_value"] == pytest.approx(74.615)
    store.upsert_observations([{"source": "tal_smm", "entity": "SILVER",
                                "ts": NOW_ISO, "value": 75.2}])
    assert build_evidence(silver_mkt, store)["feed_value"] == pytest.approx(75.2)


def test_build_evidence_rejects_non_price_markets(store):
    """Pilot bug: a 'silver industrial demand exceed 50%' market got a $/oz
    anchor — every verdict on it was judged against irrelevant evidence."""
    demand_mkt = {"id": "mkt_demand",
                  "question": "Will silver industrial demand exceed 50% by 2027?",
                  "probability": 0.7, "totalLiquidity": 500}
    store.upsert_observations([{"source": "tal_smm", "entity": "SILVER",
                                "ts": NOW_ISO, "value": 75.2}])
    assert build_evidence(demand_mkt, store) is None
    for q in ("Will gold production exceed 3000 tonnes in 2026?",
              "Will copper inventories exceed 500000 tonnes?",
              # pilot v2: premium/spread markets pass the naive price+unit check
              # but the outright spot anchor is irrelevant to a spread's level
              "Will SGE gold premium over LBMA exceed 5 USD/oz on September 30?",
              "Will the copper cathode spread exceed 100 USD per tonne?",
              # pilot v3: side-quantity in the entity's unit
              "Will the selenium byproduct credit per tonne of refined copper exceed 30 USD?"):
        assert build_evidence({"id": "x", "question": q, "probability": 0.5}, store) is None
    # the real price market still passes
    assert build_evidence(GOLD_MARKET, store) is not None


def test_build_evidence_threshold_magnitude_guard(store):
    """Wrong-quantity markets that dodge every keyword still fail the
    threshold-vs-feed scale check (feed GOLD=4400: 20 is 0.005x, 50000 is 11x)."""
    for thr in (20, 50_000):
        q = f"Will the gold price exceed {thr} USD per troy ounce in 2026?"
        assert build_evidence({"id": "x", "question": q, "probability": 0.5}, store) is None
    ok = {"id": "y", "question": "Will the gold price exceed 3000 USD per troy ounce?",
          "probability": 0.9, "totalLiquidity": 500}
    assert build_evidence(ok, store) is not None


def test_judge_never_shows_attached_bet_to_llm(store):
    """Pilot v2 bug: shown the author's bet, the LLM cited comment-vs-own-bet
    inconsistency as a 'factual error'. The bet is fade metadata, not evidence."""
    ev = build_evidence(GOLD_MARKET, store)
    llm = FakeLLM(UNSOUND_HIGH)
    v = judge_comment(llm, ev, make_comment())
    _system, user = llm.calls[0]
    assert "attached_bet" not in user and "betOutcome" not in user
    assert v["bet_outcome"] == "YES"   # ...but the verdict row still carries it


def test_stale_feed_caps_confidence_and_kills_actionability(store):
    """Pilot bug: 13-day-old anchor still produced unsound/HIGH. The cap is now
    enforced in code — stale evidence can never make an actionable verdict."""
    store.upsert_observations([{"source": "tal_pcf", "entity": "SILVER",
                                "ts": STALE_ISO, "value": 57.82}])
    silver_mkt = {"id": "mkt_ag2", "question": "Will silver price exceed 50 USD per troy ounce?",
                  "probability": 0.7, "totalLiquidity": 500}
    ev = build_evidence(silver_mkt, store)
    assert ev["feed_age_days"] > 3
    v = judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment(market_id="mkt_ag2"))
    assert v["verdict"] == "unsound" and v["confidence"] == "medium"  # downgraded
    assert v["reply_draft"] is None                                   # not actionable


def test_judge_actionable_verdict_and_reply_gating(store):
    ev = build_evidence(GOLD_MARKET, store)
    v = judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment())
    assert v["verdict"] == "unsound" and v["confidence"] == "high"
    assert v["reply_draft"] and v["bet_outcome"] == "YES"
    # medium confidence -> reply suppressed (confidence gate)
    v2 = judge_comment(FakeLLM({**UNSOUND_HIGH, "confidence": "medium"}), ev, make_comment())
    assert v2["reply_draft"] is None


def test_judge_is_conservative_on_garbage(store):
    ev = build_evidence(GOLD_MARKET, store)
    for bad in (FakeLLM("not json {"), FakeLLM(RuntimeError("ollama down")),
                FakeLLM({"verdict": "sneaky", "confidence": "ultra"})):
        v = judge_comment(bad, ev, make_comment())
        assert v["verdict"] == "noise" and v["confidence"] == "low"
        assert v["reply_draft"] is None


# --- store roundtrip ---------------------------------------------------------

def _record(store, v, judged_by="adversary_metals_1", evidence=None):
    store.record_comment_verdict(judged_by=judged_by, evidence=evidence or {"e": 1}, **v)


def test_verdict_store_dedupe_and_actionable_filter(store):
    ev = build_evidence(GOLD_MARKET, store)
    _record(store, judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment("c1")))
    _record(store, judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment("c1")))  # dupe
    _record(store, judge_comment(  # unsound but LOW confidence -> not actionable
        FakeLLM({**UNSOUND_HIGH, "confidence": "low"}), ev, make_comment("c2")))
    _record(store, judge_comment(  # no attached bet -> nothing to fade
        FakeLLM(UNSOUND_HIGH), ev, make_comment("c3", bet=None)))
    assert store.judged_comment_ids() == {"c1", "c2", "c3"}
    actionable = store.actionable_verdicts()
    assert [a["comment_id"] for a in actionable] == ["c1"]
    assert store.actionable_verdicts(max_age_hours=0.0) == []


# --- strategies --------------------------------------------------------------

def test_comment_fade_shifts_against_bad_money_fighting_the_anchor(store):
    # GOLD_MARKET: "exceed 2500", feed 4400 -> anchor says YES. A bad NO bet
    # FIGHTS the anchor -> fade it (fair goes ABOVE market prob).
    ev = build_evidence(GOLD_MARKET, store)
    _record(store, judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment("c1", bet=("NO", 250.0))), evidence=ev)
    strat = get_strategy("comment_fade", shift_per_verdict=0.05)
    strat.bind(store)
    assert strat.prefilter([GOLD_MARKET, {"id": "other", "probability": 0.5,
                                          "totalLiquidity": 500}]) == [GOLD_MARKET]
    est = strat.estimate([GOLD_MARKET])
    assert est["mkt_gold_1"] == pytest.approx(0.55 + 0.05)
    assert strat.explain("mkt_gold_1")  # reasoning available for the trade comment
    assert strat.correlation_key(GOLD_MARKET) == "GOLD"


def test_comment_fade_never_fades_bets_agreeing_with_anchor(store):
    """Live near-miss 2026-07-08: wrong-REASONING comments backing a YES that our
    own anchor confirms ('gold > 3500' with gold at 4400) must not be faded —
    that would bet against a data-confirmed outcome."""
    ev = build_evidence(GOLD_MARKET, store)   # anchor: feed 4400 > threshold 2500 -> YES
    _record(store, judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment("c1")), evidence=ev)  # YES bet
    strat = get_strategy("comment_fade", shift_per_verdict=0.05)
    strat.bind(store)
    assert strat.estimate([GOLD_MARKET]) == {}


def test_comment_consensus_trades_toward_extremized_crowd(store):
    store.upsert_comment_consensus(market_id="mkt_gold_1", p_mean=0.70,
                                   p_extreme=0.813, n_forecasters=5)
    strat = get_strategy("comment_consensus", min_forecasters=4)
    strat.bind(store)
    assert strat.estimate([GOLD_MARKET])["mkt_gold_1"] == pytest.approx(0.813)
    # too few forecasters -> abstain
    strat2 = get_strategy("comment_consensus", min_forecasters=9)
    strat2.bind(store)
    assert strat2.estimate([GOLD_MARKET]) == {}


# --- live reply posting --------------------------------------------------------

class RecordingClient:
    def __init__(self, fail_on: set[str] | None = None):
        self.posted: list[tuple] = []
        self.fail_on = fail_on or set()

    def post_comment(self, contract_id, markdown, reply_to_comment_id=None):
        if reply_to_comment_id in self.fail_on:
            raise RuntimeError("boom")
        self.posted.append((contract_id, markdown, reply_to_comment_id))
        return {"id": "posted"}


def test_post_pending_replies_caps_marks_and_survives_failure(store):
    from quantbots.comments.cycle import post_pending_replies
    ev = build_evidence(GOLD_MARKET, store)
    for cid in ("c1", "c2", "c3"):
        _record(store, judge_comment(FakeLLM(UNSOUND_HIGH), ev, make_comment(cid)))
    _record(store, judge_comment(  # medium confidence -> no draft -> never replied
        FakeLLM({**UNSOUND_HIGH, "confidence": "medium"}), ev, make_comment("c4")))
    assert {v["comment_id"] for v in store.pending_replies()} == {"c1", "c2", "c3"}

    client = RecordingClient(fail_on={"c2"})
    n = post_pending_replies(client=client, store=store, max_replies=2)
    # cap=2 counts SUCCESSFUL posts: c1 ok, c2 fails (claimed, no slot), c3 ok
    assert n == 2
    assert [p[2] for p in client.posted] == ["c1", "c3"]
    assert client.posted[0][0] == "mkt_gold_1"          # threaded on the market
    # ALL three marked (claim-first = at-most-once; c2 sacrificed to avoid dupes)
    assert store.pending_replies() == []
    # idempotent: re-run never re-posts
    assert post_pending_replies(client=client, store=store, max_replies=5) == 0
    assert [p[2] for p in client.posted] == ["c1", "c3"]


# --- the cycle (judging never posts/bets) ---------------------------------------

class DryClient:
    """Read-only fake: any write is a test failure."""

    def __init__(self, comments, bets):
        self._comments, self._bets = comments, bets

    def get_comments(self, contract_id=None, **kw):
        return self._comments

    def get_bets(self, **kw):
        return self._bets

    def post_comment(self, *a, **kw):  # pragma: no cover - the assertion IS the test
        raise AssertionError("DRY-RUN VIOLATED: cycle posted a comment")

    def place_bet(self, *a, **kw):  # pragma: no cover
        raise AssertionError("DRY-RUN VIOLATED: cycle placed a bet")


def test_cycle_skips_comments_already_addressed_by_the_fleet(store):
    """A comment answered by ANY adversary-fleet bot is settled — never judged
    (and later never replied to) again, even by a different fleet member."""
    store.upsert_markets([GOLD_MARKET])
    comments = [
        make_comment("c1"),   # will be judged
        make_comment("c2", text="Gold already crossed 9000 USD/oz last week, easy YES."),
        # ...but c2 was already answered by an adversary bot:
        make_comment("r1", author="AdversaryMetalsBot", reply_to="c2",
                     text="Per LBMA, gold is 4123 USD/oz — nowhere near 9000.", bet=None),
    ]
    bets = [{"userId": f"u{i}", "amount": 50, "probAfter": 0.6} for i in range(3)]
    llm = FakeLLM(UNSOUND_HIGH)
    res = run_judge_cycle(bot_name="adversary_metals_1", client=DryClient(comments, bets),
                          store=store, llm=llm, entities={"GOLD"})
    assert res.comments_judged == 1
    assert store.judged_comment_ids() == {"c1"}   # c2 skipped: already addressed


def test_judge_cycle_end_to_end_dry(store):
    store.upsert_markets([GOLD_MARKET])
    comments = [
        make_comment("c1"),                                    # judged
        make_comment("c2", author="DiffusionMcBot"),           # own fleet -> skipped
        make_comment("c3", reply_to="c1"),                     # reply -> skipped
        make_comment("c4", text="lol"),                        # too short -> skipped
    ]
    bets = [{"userId": f"u{i}", "amount": 50, "probAfter": p}
            for i, p in enumerate((0.6, 0.7, 0.8))]
    llm = FakeLLM(UNSOUND_HIGH)
    res = run_judge_cycle(bot_name="adversary_metals_1", client=DryClient(comments, bets),
                          store=store, llm=llm, entities={"GOLD"})
    assert res.comments_judged == 1 and res.markets_scanned == 1
    assert res.consensus_written == 1
    assert store.judged_comment_ids() == {"c1"}
    assert len(llm.calls) == 1
    # idempotent: second run judges nothing new
    res2 = run_judge_cycle(bot_name="adversary_metals_1", client=DryClient(comments, bets),
                           store=store, llm=llm, entities={"GOLD"})
    assert res2.comments_judged == 0
