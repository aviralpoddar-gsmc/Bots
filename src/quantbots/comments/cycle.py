"""The judge cycle: read new comments on covered markets, judge, store verdicts.

DRY-RUN CONTRACT: this module NEVER posts a comment and NEVER places a bet.
Verdicts + drafted replies land in the store for human review; going live is a
separate, explicit step (posting replies / enabling the comment_fade bot).

Loop-safety rails (we are joining a bot society that replies back — tal runs
ConsensusCorrecterBot et al.): replies to replies are never judged, one verdict
per comment ever (PK dedupe), and our own fleet's comments are skipped entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .consensus import market_consensus
from .judge import build_evidence, judge_comment
from .reader import author, comment_text, is_reply

logger = logging.getLogger(__name__)

# The adversarial/consensus fleet itself (grows as market-type bots are cloned).
# Their replies mark a comment as ADDRESSED — no other fleet member re-replies.
ADVERSARY_FLEET = {"AdversaryMetalsBot", "ConsensusBot"}

# This repo's own bot accounts (probe census 2026-07-08). Never judged: the fade
# path would front-run our own fleet, and replying to ourselves invites loops.
OWN_FLEET = {
    "DiffusionMcBot", "LadderArbBot", "Bot007", "PairTradingBot", "EnsembleBot",
    "ConditionalArbBot", "StockpileFactsBot", "MercuryEnsembleBot", "MarketMakerBot",
    "TermStructureBot", "CommoditySpotBot", "AviralPoddar", "MikhailTal",
} | ADVERSARY_FLEET

_MIN_TEXT_CHARS = 30      # below this there is nothing to fact-check
_COMMENTS_PER_MARKET = 25  # newest N per market per cycle


@dataclass
class CycleResult:
    markets_scanned: int = 0
    comments_seen: int = 0
    comments_judged: int = 0
    consensus_written: int = 0
    verdicts: list[dict] = field(default_factory=list)


def covered_markets(store: Any, entities: set[str]) -> list[dict]:
    """ALL open cached spot-price candidates linked to our anchor entities, most
    recently commented first. Question-level filters only (cheap regex); the
    caller applies build_evidence (feed lookup + threshold-scale sanity) and
    stops once it has enough judgeable markets — pre-cutting to top-N here once
    starved a whole cycle to zero when the most-commented markets were all
    premium/demand questions."""
    from ..strategies.linker import link_market
    from .judge import is_spot_price_market
    out = []
    for m in store.load_open_markets():
        link = link_market(m)
        if not (link and link.entities and link.entities[0] in entities):
            continue
        if not is_spot_price_market(m.get("question", ""), link.entities[0]):
            continue
        out.append(m)
    out.sort(key=lambda m: m.get("lastCommentTime") or 0, reverse=True)
    return out


def post_pending_replies(*, client: Any, store: Any, max_replies: int = 7,
                         max_age_hours: float = 48.0) -> int:
    """Post drafted replies for unsound/high verdicts not yet replied to.

    LIVE writes (the one place this package posts): threaded under the offending
    comment, as the judging bot's account. Caps per run (3 runs/day x default 7
    ≈ 20/day). `mark_replied` before any chance of a double-post on retry; a
    failed post logs and moves on (never crashes the cycle)."""
    posted = 0
    for v in store.pending_replies(max_age_hours=max_age_hours):
        if posted >= max_replies:
            break
        store.mark_replied(v["comment_id"])  # claim first: at-most-once beats at-least-once
        try:
            client.post_comment(v["market_id"], v["reply_draft"],
                                reply_to_comment_id=v["comment_id"])
            posted += 1
            logger.info("replied to %s by @%s on %s", v["comment_id"], v["author"],
                        v["market_id"])
        except Exception as e:  # noqa: BLE001
            logger.warning("reply failed for %s: %s", v["comment_id"], e)
    return posted


def run_judge_cycle(*, bot_name: str, client: Any, store: Any, llm: Any,
                    entities: set[str], max_judgments: int = 50,
                    max_markets: int = 40, min_forecasters: int = 3,
                    ignore_authors: set[str] | None = None) -> CycleResult:
    """One dry pass: judge up to `max_judgments` new comments + refresh consensus."""
    ignore = OWN_FLEET | (ignore_authors or set())
    already = store.judged_comment_ids()
    res = CycleResult()

    for market in covered_markets(store, entities):
        if res.comments_judged >= max_judgments or res.markets_scanned >= max_markets:
            break
        mid = market.get("id")
        evidence = build_evidence(market, store)
        if evidence is None:
            continue  # no anchor / wrong quantity -> must not judge (Bridgewater rule)
        res.markets_scanned += 1

        try:
            comments = client.get_comments(contract_id=mid, limit=_COMMENTS_PER_MARKET)
        except Exception as e:  # noqa: BLE001 - one bad market must not kill the cycle
            logger.warning("cycle: get_comments failed for %s: %s", mid, e)
            continue
        res.comments_seen += len(comments)

        # Comments ALREADY ADDRESSED by any adversary-fleet reply are settled —
        # never judge (and later, never reply to) them again. This catches
        # replies made by OTHER fleet members or outside this store; the verdict
        # PK covers everything that went through this pipeline. The live-reply
        # path must additionally check verdict.replied_at IS NULL.
        addressed = {c.get("replyToCommentId") for c in comments
                     if c.get("replyToCommentId") and author(c) in ADVERSARY_FLEET}

        # Consensus is cheap (one bets call) and independent of judging.
        try:
            bets = client.get_bets(contractId=mid, limit=200)
            cons = market_consensus(bets, min_forecasters=min_forecasters)
            if cons:
                store.upsert_comment_consensus(market_id=mid, **cons)
                res.consensus_written += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("cycle: consensus failed for %s: %s", mid, e)

        for c in comments:
            if res.comments_judged >= max_judgments:
                break
            if c.get("id") in already or c.get("id") in addressed:
                continue
            if is_reply(c) or author(c) in ignore:
                continue
            if len(comment_text(c)) < _MIN_TEXT_CHARS:
                continue
            v = judge_comment(llm, evidence, c)
            store.record_comment_verdict(judged_by=bot_name, evidence=evidence, **v)
            res.comments_judged += 1
            res.verdicts.append(v)
    return res
