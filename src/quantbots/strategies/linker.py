"""Market ↔ source-entity linker (heuristic, deterministic first pass).

The crux of trading on external data: *which observation is relevant to which
market?* This maps a market's question text to one or more source `entity` keys
(e.g. "...Brent crude..." -> BRENT_OIL) and pulls out the threshold/direction via
the ladder parser. It's keyword-based on purpose — transparent and debuggable.
Replace/augment with local-LLM extraction or embeddings later; the
`MarketLink` contract stays the same.

Note: keyword matching is intentionally loose (a question mentioning "crude oil"
may match both WTI_OIL and BRENT_OIL). Strategies that consume links should be
robust to a market linking to several entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .ladder import parse_threshold

Market = dict[str, Any]

# entity key -> keywords whose presence in a question implies relevance.
DEFAULT_ENTITY_KEYWORDS: dict[str, list[str]] = {
    "WTI_OIL": ["wti", "west texas", "crude oil", "crude", "oil price"],
    "BRENT_OIL": ["brent"],
    "GOLD": ["gold", "xau"],
    "SILVER": ["silver", "xag"],
    "COPPER": ["copper"],
    "NATGAS": ["natural gas", "natgas", "henry hub"],
    "US_CPI_YOY": ["cpi", "inflation", "consumer price"],
    "US_GDP_GROWTH": ["gdp", "gross domestic product"],
    "US_UNEMPLOYMENT": ["unemployment", "jobless", "unemployment rate"],
}

# Our sources are price *levels* and macro *rates*. A question that mentions a
# commodity but asks about an operational/structural metric (production, volume,
# reserves, a company's share, ...) is NOT about the price we track, so linking it
# to a price feed produces confidently-wrong estimates. Suppress those links.
# (Deliberately excludes "%", "rate", "supply", "demand" — those are legit for
# the macro-rate entities and broad price questions.)
EXCLUDE_KEYWORDS: list[str] = [
    "production", "output", "volume", "volumes", "inlet", "throughput",
    "capacity", "reserves", "proved", "balance", "share of", "market share",
    "exports", "imports", "wells", "rig", "inventory", "inventories",
]

# Curated catalog: a phrase pattern in the question -> the exact source entity it
# refers to. This is the "deterministic, easy to process" path — each entry is a
# precise series we have a feed for. Add a line per series as feeds are added.
# (Compiled, case-insensitive.)
_CATALOG_RAW: list[tuple[str, str]] = [
    (r"single[- ]family housing starts", "FRED_HOUST1F"),
    (r"(30[- ]?year fixed.*mortgage|mortgage rate.*(freddie|pmms)|freddie mac pmms)", "FRED_MORTGAGE30US"),
]
CATALOG: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.I), entity) for p, entity in _CATALOG_RAW
]

# Stock-price markets: ticker in parens, e.g. "TeraWulf (WULF) stock price ...".
# Auto-generalizes to any ticker -> entity STOCK_<TICKER> (paired with a Stooq
# <ticker>.us feed). No per-ticker curation needed for linking.
_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")


@dataclass
class MarketLink:
    market_id: str
    entities: list[str]  # matched source entity keys
    threshold: float | None
    direction: str  # "exceeds" | "below"
    question: str = ""


def _matches(question_lc: str, keyword: str) -> bool:
    # Word-boundary match so "oil" doesn't fire inside "spoiled", etc.
    return re.search(rf"\b{re.escape(keyword)}\b", question_lc) is not None


def link_market(
    market: Market, entity_keywords: dict[str, list[str]] | None = None
) -> MarketLink | None:
    """Return a MarketLink if the question maps to any known entity, else None."""
    question = market.get("question", "")
    q = question.lower()

    matched: list[str] = []

    # 1. Stock-price markets: parse the ticker directly (deterministic, general).
    if "stock price" in q:
        matched += [f"STOCK_{t}" for t in _TICKER_RE.findall(question)]

    # 2. Curated catalog: precise phrase -> exact series entity.
    matched += [entity for pat, entity in CATALOG if pat.search(question)]

    # 3. Broad commodity keywords — but only for price-like questions (skip if the
    #    question is about an out-of-scope metric: volume/production/reserves/...).
    if not any(kw in q for kw in EXCLUDE_KEYWORDS):
        kws = entity_keywords or DEFAULT_ENTITY_KEYWORDS
        matched += [ent for ent, words in kws.items() if any(_matches(q, w) for w in words)]

    if not matched:
        return None
    matched = list(dict.fromkeys(matched))  # dedup, keep order
    parsed = parse_threshold(question)
    threshold, direction = parsed if parsed else (None, "exceeds")
    return MarketLink(
        market_id=market["id"],
        entities=matched,
        threshold=threshold,
        direction=direction,
        question=question,
    )


def link_markets(
    markets: list[Market], entity_keywords: dict[str, list[str]] | None = None
) -> dict[str, MarketLink]:
    out: dict[str, MarketLink] = {}
    for m in markets:
        link = link_market(m, entity_keywords)
        if link is not None:
            out[m["id"]] = link
    return out
