"""Evidence-anchored comment judge (the Bridgewater supervisor rule).

AIA Forecaster's key negative result (arXiv:2511.07678): an LLM that critiques
other forecasters *by reading their reasoning alone* is WORSE than doing nothing —
it overweights outlier opinions. The supervisor only added value when it gathered
its own evidence and overrode at high confidence. So this judge:

  - builds an EVIDENCE PACK from our own ingested feeds (the observations cache:
    stooq/LBMA spot, FRED, NOAA) + the market's parsed threshold,
  - instructs the model that `unsound` requires the comment's factual claims to
    CONTRADICT the supplied evidence numbers — "I disagree with the logic" is at
    most `noise`,
  - and downstream, only HIGH-confidence unsound verdicts are actionable
    (store.actionable_verdicts), mirroring the paper's confidence gate.

LOCAL LLM only (qwen3:32b via llm/client.py) per the project constraint.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from ..strategies.linker import link_market
from .reader import attached_bet, author, comment_text

logger = logging.getLogger(__name__)

# Feed evidence older than this cannot support a HIGH-confidence verdict — the
# pilot showed the prompt-level staleness guidance alone does not bind.
MAX_FEED_AGE_DAYS_FOR_HIGH = 3.0

# Feed-unit -> market-unit conversion, mirroring commodity_spot._SPECS (silver
# quotes cents/oz on stooq; copper cents/lb vs $/MT markets).
_FEED_FACTOR = {"GOLD": 1.0, "SILVER": 0.01, "PLATINUM": 1.0, "PALLADIUM": 1.0,
                "COPPER": 22.0462, "WTI_OIL": 1.0, "BRENT_OIL": 1.0, "GASOLINE": 1.0}
_UNIT = {"GOLD": "USD/oz", "SILVER": "USD/oz", "PLATINUM": "USD/oz",
         "PALLADIUM": "USD/oz", "COPPER": "USD/tonne", "WTI_OIL": "USD/bbl",
         "BRENT_OIL": "USD/bbl", "GASOLINE": "USD/gal"}

_SYSTEM = """You are an auditor of comments on a commodity prediction market.
You are given EVIDENCE (the market question, its parsed threshold, the current
market probability, and our own trusted data-feed value for the underlying) and
one COMMENT (possibly with the bet its author placed alongside it).

Classify the comment:
- "unsound": ONLY if a factual claim in the comment CONTRADICTS the supplied
  evidence numbers (wrong current price, wrong side of the threshold, impossible
  magnitude). Cite each contradicted claim. Never mark unsound because you
  disagree with reasoning, tone, or vibes — only evidence contradictions count.
- "sound": consistent with the evidence and adds information.
- "noise": no checkable factual content (chit-chat, bare position statements),
  OR you cannot verify its claims with the evidence given. When unsure: noise.

HARD RULES (each of these was a real false positive — follow them exactly):
1. The market probability is NEVER evidence of error. A comment quoting a
   different probability than market_prob was simply written earlier; prices
   move. Ignore all probability claims entirely.
2. Do NOT perform unit conversions yourself. If the comment quotes a value in a
   different unit than feed_unit (e.g. USD/kg vs USD/oz), you cannot verify it:
   verdict "noise". Only compare numbers already in the same unit.
3. Only claims about the underlying quantity (the price of the named commodity)
   are checkable here. Claims about other data sources, mines, news events, or
   other commodities are unverifiable with this evidence: "noise".
4. Statements about the COMMENTER'S OWN state (its prior, its last update, how
   recently it evaluated) are self-descriptions, not claims about the world —
   they can never contradict the evidence. Ignore them.

Confidence: "high" only when the contradiction (or consistency) is unambiguous
and the evidence is fresh; otherwise "medium" or "low". If feed_age_days > 3,
cap confidence at "medium".

Respond with JSON only:
{"verdict":"sound|unsound|noise","confidence":"high|medium|low",
 "factual_errors":["claim -> what the evidence says"],
 "reply":"1-3 sentence courteous markdown reply citing the evidence number, ONLY
if verdict is unsound with high confidence, else empty string"}"""

_VALID_VERDICTS = {"sound", "unsound", "noise"}
_VALID_CONF = {"high", "medium", "low"}


# Source preference: tal Snowflake anchors (scripts/ingest_tal_prices.py) store
# values ALREADY in market units; stooq raw quotes need the feed factor.
_SOURCE_ORDER = ("tal_smm", "tal_pcf", "stooq")

# A spot-price anchor is only valid evidence for a spot-PRICE market. The pilot
# judged a "silver industrial demand exceed 50%" market against $/oz — every
# verdict on it was garbage-in. Mirror commodity_spot's unit discipline: the
# question must mention price/USD AND the entity's quote unit, and must NOT be
# about a non-price measurable (share/demand/production/%...).
_NON_PRICE_Q = re.compile(
    r"%|\bpercent|\bshare\b|\bdemand\b|\bproduction\b|\bconsumption\b|"
    r"\binventor(?:y|ies)\b|\bcapacity\b|\boutput\b|\breserves?\b|"
    # Relative-value markets (SGE premium, cathode premium, TC/RC...): the
    # outright spot anchor is irrelevant evidence for a spread's level.
    r"\bpremium\b|\bspread\b|\bbasis\b|\bdifferential\b|\bdiscount\b|"
    r"\bratio\b|\bmargin\b|\btreatment\s+charge\b|\btc/rc\b|"
    # Side-quantities quoted in the entity's unit ("selenium byproduct credit
    # per tonne of refined copper") — the entity is mentioned, the measured
    # quantity is something else.
    r"\bbyproduct\b|\bcredit\b|\bpenalt(?:y|ies)\b|\bfreight\b", re.I)

# A genuine spot-price threshold sits near the entity's price level. Anything
# wildly off-scale (selenium credit $30 vs copper $15,140; SGE premium $5 vs
# gold $4,123) is measuring a DIFFERENT quantity in the same unit — reject in
# code, whatever the wording was.
_THRESHOLD_FEED_RATIO = (0.2, 5.0)
_UNIT_TOKEN = {
    "GOLD": r"(?:troy\s+)?(?:oz|ounce)", "SILVER": r"(?:troy\s+)?(?:oz|ounce)",
    "PLATINUM": r"(?:troy\s+)?(?:oz|ounce)", "PALLADIUM": r"(?:troy\s+)?(?:oz|ounce)",
    "COPPER": r"\btonne\b|\bmetric\s+ton\b|\bmt\b",
    "WTI_OIL": r"\bbarrel\b|\bbbl\b", "BRENT_OIL": r"\bbarrel\b|\bbbl\b",
    "GASOLINE": r"\bgallon\b|\bgal\b",
}
_PRICE_HINT = re.compile(r"\bprice\b|\busd\b|\$", re.I)


def is_spot_price_market(question: str, entity: str) -> bool:
    if _NON_PRICE_Q.search(question):
        return False
    unit = _UNIT_TOKEN.get(entity)
    if unit is None:
        return False
    return bool(_PRICE_HINT.search(question) and re.search(unit, question, re.I))


def _feed_age_days(feed_ts: str | None) -> float | None:
    if not feed_ts:
        return None
    try:
        ts = datetime.fromisoformat(feed_ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (datetime.now(UTC) - ts).total_seconds() / 86400.0
    except ValueError:
        return None


def build_evidence(market: dict, store: Any) -> dict | None:
    """Evidence pack for one market: parsed threshold + our latest feed value.
    Returns None when we have no anchor (no linked entity, a non-price market,
    or no observation) — the judge must not run without evidence (Bridgewater
    rule): judging against irrelevant evidence is worse than not judging."""
    link = link_market(market)
    if link is None or not link.entities:
        return None
    entity = link.entities[0]
    if not is_spot_price_market(market.get("question", ""), entity):
        return None
    obs = None
    for src in _SOURCE_ORDER:
        obs = store.latest_observation(entity, source=src)
        if obs and obs.get("value") is not None:
            break
    if obs is None or obs.get("value") is None:
        obs = store.latest_observation(entity)
    if obs is None or obs.get("value") is None:
        return None
    factor = _FEED_FACTOR.get(entity, 1.0) if obs.get("source") == "stooq" else 1.0
    feed_value = float(obs["value"]) * factor
    # Magnitude sanity: threshold must be on the entity's price scale.
    if link.threshold is not None and feed_value > 0:
        ratio = link.threshold / feed_value
        if not (_THRESHOLD_FEED_RATIO[0] <= ratio <= _THRESHOLD_FEED_RATIO[1]):
            return None
    age = _feed_age_days(obs.get("ts"))
    return {
        "question": market.get("question", ""),
        "market_prob": market.get("probability"),
        "entity": entity,
        "threshold": link.threshold,
        "direction": link.direction,
        "feed_value": round(feed_value, 4),
        "feed_unit": _UNIT.get(entity, "?"),
        "feed_ts": obs.get("ts"),
        "feed_age_days": round(age, 1) if age is not None else None,
        "feed_source": obs.get("source"),
        "as_of": datetime.now(UTC).date().isoformat(),  # lets the judge assess staleness
        "close_time_ms": market.get("closeTime"),
    }


def judge_comment(llm: Any, evidence: dict, comment: dict) -> dict:
    """One LLM verdict for one comment. Conservative on any failure: noise/low
    (a broken judge must never generate actionable fades).

    The author's attached bet is deliberately NOT shown to the LLM — it is fade
    metadata, not evidence. Pilot v2 showed the model will otherwise cite
    comment-vs-own-bet inconsistency as a "factual error" (it isn't one)."""
    bet = attached_bet(comment)
    user = json.dumps({
        "EVIDENCE": evidence,
        "COMMENT": {"author": author(comment), "text": comment_text(comment)[:2000]},
    }, indent=1)
    verdict, confidence, errors, reply = "noise", "low", [], ""
    try:
        raw = llm.json_completion(_SYSTEM, user)
        parsed = json.loads(raw)
        if parsed.get("verdict") in _VALID_VERDICTS:
            verdict = parsed["verdict"]
        if parsed.get("confidence") in _VALID_CONF:
            confidence = parsed["confidence"]
        errors = [str(e) for e in parsed.get("factual_errors") or []][:5]
        reply = str(parsed.get("reply") or "").strip()
    except Exception as e:  # noqa: BLE001 - LLM down/JSON garbage -> abstain on this one
        logger.warning("judge: LLM failed on comment %s: %s", comment.get("id"), e)
    # ENFORCED staleness cap (the prompt-level cap alone did not bind in the
    # pilot): stale evidence can never produce an actionable HIGH verdict.
    age = evidence.get("feed_age_days")
    if confidence == "high" and (age is None or age > MAX_FEED_AGE_DAYS_FOR_HIGH):
        confidence = "medium"
    # The reply is only ever meaningful on the actionable path.
    if not (verdict == "unsound" and confidence == "high"):
        reply = ""
    return {
        "comment_id": comment.get("id"),
        "market_id": comment.get("contractId"),
        "author": author(comment),
        "entity": evidence.get("entity"),
        "verdict": verdict,
        "confidence": confidence,
        "factual_errors": errors,
        "reply_draft": reply or None,
        "bet_id": (bet or {}).get("bet_id"),
        "bet_outcome": (bet or {}).get("outcome"),
        "bet_amount": (bet or {}).get("amount"),
    }
