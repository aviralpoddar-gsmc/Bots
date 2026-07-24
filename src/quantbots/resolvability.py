"""Cancellation-aware resolvability scoring.

The clone resolves a market YES/NO only when its named source actually publishes a
verifiable value; otherwise it CANCELs and refunds bets. Empirically (9,578 resolved
markets) only ~7% resolve YES/NO — and which ones is highly predictable:

  metric type    decided-rate        benchmark        decided-rate
  -----------    ------------        ---------        ------------
  price          22%                 LBMA             100%
  spread/ratio    8%                 LME               39%
  inventory       6%                 "spot price"      31%
  exports/imports 4%                 China domestic    26%
  production    0.6%                 (precious metals via LBMA: ~100%)
  demand          0%

So edge only *realizes* on markets that actually resolve. This module scores a
market's probability of resolving (vs cancelling) from features visible at
trade-time — the metric type and the named benchmark in the question. The runner
multiplies a signal's expected value by this score so every bot concentrates its
capital where the edge actually pays out, instead of in production/demand ladders
that almost never settle.

Scores approximate the observed decided-rates; they are intentionally
question-text-only (no per-market description fetch) so the filter runs over the
whole 62k universe each cycle.
"""

from __future__ import annotations

import re

# Operational metrics that almost never resolve (0–1% decided).
_OPERATIONAL = re.compile(
    r"\b(production|output|produced|demand|capacity|utili[sz]ation|installed|"
    r"market share|share of|funding|coverage|deliveries|throughput|"
    r"book-to-bill|backlog|headcount|penetration)\b", re.I)
# Price / settlement questions (highest base rate, ~22%).
_PRICE = re.compile(r"\b(price|spot|settlement|fixing|futures|basis)\b", re.I)
# Proprietary specialty-chemical / minor-metal prices: named in the question but
# quoted ONLY by paywalled price-reporting agencies (Asian Metal, SMM, Fastmarkets)
# with no free public settlement and irregular/annual public disclosure (often just
# the USGS annual average). A market dated quarterly/monthly on one of these usually
# has no verifiable value at resolution time -> it CANCELs far more than a generic
# price. Scored between inventory (0.06) and price (0.22). Verified on zirconium:
# ZOC / oxychloride / ex-works-China sponge / zirconium chemicals only publish
# publicly once a year via USGS (see [[zirconium-conditionals]]). The exchange and
# precious-metal boosts below still override this for anything exchange-settled.
# Kept deliberately NARROW: the unambiguous proprietary-benchmark signals only.
# Generic chemistry words (oxide/carbonate/silicate) are intentionally excluded —
# they would sweep in the entire rare-earth-oxide complex (~2k markets) whose
# decided-rate we haven't validated. Scope this to the zirconium chemical chain
# (oxychloride/ZOC, ex-works-China sponge) plus explicit proprietary-agency names.
_PROPRIETARY_PX = re.compile(
    r"\b(oxychloride|ZOC|ex[- ]works|sponge|Asian Metal|SMM|Fastmarkets)\b", re.I)
# Trade-flow (customs data — modest).
_TRADE = re.compile(r"\b(export|exports|import|imports|customs|shipments?)\b", re.I)
_SPREAD = re.compile(r"\b(spread|ratio|premium|discount|crack)\b", re.I)
_INVENTORY = re.compile(r"\b(inventory|inventories|stocks?|warrant)\b", re.I)
# Exchange-settled / officially-published benchmarks (resolve reliably).
_STRONG_SRC = re.compile(
    r"\b(LBMA|LME|COMEX|NYMEX|ICE|CME|SHFE|GFEX|CBOT|GACC|"
    r"official settlement|exchange settlement)\b", re.I)
# LBMA precious metals — essentially always resolve.
_PRECIOUS = re.compile(r"\b(gold|silver|platinum|palladium)\b", re.I)

# U.S. strategic-materials FACT markets. These don't resolve off a commodity feed
# but off a government publication, so the price/production heuristics below would
# misclassify them. Handled up front, ordered by how cleanly the source publishes:
#   - USGS Critical Minerals list: Federal Register, statutory cadence -> resolves.
#   - NDS holdings: DLA report exists but specific holdings partly classified -> medium.
#   - buffer-stock procurement: a policy announcement -> low-ish.
#   - "vault procurement ... kg" / "Project Vault": a quantity figure with no clear
#     public reporting line -> production-like, ~never resolves.
# Conditional ("B if A") markets: `IF [<predicate>] = YES: <quantity>`. A
# conditional resolves YES/NO only if the predicate resolves (YES) AND the
# quantity resolves — so its realized resolvability stacks BOTH legs' cancel risk.
_CONDITIONAL = re.compile(r"^IF \[(.+)\] = YES: (.+)$", re.S)
_CRIT_LIST = re.compile(r"\bcritical minerals list\b", re.I)
_NDS_HOLD = re.compile(r"\bnational defense stockpile\b.*\bposition\b", re.I)
_BUFFER = re.compile(r"\bbuffer[- ]stock procurement\b", re.I)
_VAULT_QTY = re.compile(r"\b(vault procurement|project vault)\b", re.I)


def resolvability_score(question: str) -> float:
    """Estimate P(this market resolves YES/NO rather than CANCEL), in [0.01, 0.99],
    from the question text. Calibrated to observed decided-rates."""
    q = (question or "").strip()
    # Conditional markets stack two legs of cancel risk: approximate realized
    # resolvability as the product of the predicate's and quantity's scores. The
    # legs are plain markets (never conditionals), so the recursion terminates.
    cm = _CONDITIONAL.match(q)
    if cm:
        prod = resolvability_score(cm.group(1)) * resolvability_score(cm.group(2))
        return min(max(prod, 0.01), 0.99)
    # Strategic-materials fact/policy markets — resolve off a government publication.
    if _CRIT_LIST.search(q):
        return 0.90
    if _NDS_HOLD.search(q):
        return 0.35
    if _BUFFER.search(q):
        return 0.10
    if _VAULT_QTY.search(q):
        return 0.03
    # Operational metrics dominate: even if a price word appears, these cancel.
    if _OPERATIONAL.search(q):
        base = 0.02
    elif _PRICE.search(q):
        # Proprietary specialty-chemical/minor-metal price with no public free
        # settlement -> discount toward the inventory rate. An exchange/precious
        # benchmark below can still lift it back up if one is named.
        base = 0.10 if _PROPRIETARY_PX.search(q) else 0.22
    elif _INVENTORY.search(q):
        base = 0.06
    elif _SPREAD.search(q):
        base = 0.08
    elif _TRADE.search(q):
        base = 0.05
    else:
        base = 0.04

    score = base
    if base >= 0.06:  # only price-like questions benefit from a strong benchmark
        if _STRONG_SRC.search(q):
            score = max(score, 0.35)  # exchange-settled non-precious: ~27-40% observed
        if _PRECIOUS.search(q) and _PRICE.search(q):
            score = max(score, 0.90)
        if re.search(r"\bLBMA\b", q, re.I):
            score = 0.97
    return min(max(score, 0.01), 0.99)
