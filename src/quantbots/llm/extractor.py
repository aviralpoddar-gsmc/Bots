"""Local-LLM structured parser for market questions.

The deterministic parsers (`strategies/ladder.py`, `strategies/linker.py`) are
regex/keyword and brittle: "iron ore lump premium over 62% Fe fines exceed 10
USD/t" parses the threshold as 62 (the first number after a comparison word) not
10; "global gold dental-alloy demand" keyword-matches the gold *price* feed. This
asks a LOCAL model to read the question and return a structured record instead —
a pure language task (extraction + classification), where small local models are
strong, rather than numeric forecasting, where they're weak.

LOCAL COMPUTE ONLY (see llm/client.py). Output is validated JSON; on any failure
the caller should fall back to the deterministic parser, so the LLM is a strict
upgrade, never a single point of failure.

Slow (~5s/question on qwen3:8b), so this is meant to run as an OFFLINE enrichment
pass that caches results — not in the hot trading loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .client import LocalLLM

_METRIC_TYPES = (
    "price", "spread", "ratio", "production", "demand", "inventory",
    "trade", "capacity", "event", "other",
)

_SYSTEM = """You extract structured data from prediction-market questions about \
commodities, mining, and macro. Output ONE JSON object, no prose, with EXACTLY these keys:

- "subject": the underlying quantity in words, with the threshold number and the \
resolution date removed (e.g. "LME copper spot price", "Glencore cobalt production").
- "metric_type": one of price, spread, ratio, production, demand, inventory, trade, \
capacity, event, other. Use "price" only for the price/level of a tradable thing; \
use "production"/"demand"/"capacity" for operational quantities even if a commodity \
is named.
- "threshold": the NUMBER the outcome is compared against — the value right after \
"exceed/above/below/under/reach/at least". NOT a grade/spec like "62% Fe", a \
contract tenor like "3-month", or a year. Return a number (no commas/units) or null.
- "unit": the threshold's unit, e.g. "USD/MT", "USD/ozt", "USD/barrel", "%", "kt", \
"tonnes". null if none.
- "currency": "USD", "CNY", "EUR", "GBP", ... or null.
- "direction": "exceeds" or "below".
- "resolution_date": ISO "YYYY-MM-DD" if a date is present (use month end if only a \
month is given), else null.
- "benchmark": the named price source if any — LBMA, LME, COMEX, NYMEX, ICE, CME, \
SHFE, GFEX, Fastmarkets, SMM, Argus, Bloomberg, GACC — else null.
- "commodity": the canonical commodity if applicable (gold, silver, platinum, \
palladium, copper, nickel, zinc, lithium, cobalt, WTI, Brent, ...), else null.

Examples:
Q: "Will iron ore lump premium over 62% Fe fines exceed 10 USD/t on June 30, 2027?"
{"subject":"iron ore lump premium over 62% Fe fines","metric_type":"spread",\
"threshold":10,"unit":"USD/t","currency":"USD","direction":"exceeds",\
"resolution_date":"2027-06-30","benchmark":null,"commodity":"iron ore"}

Q: "Will global gold dental alloy demand exceed 75 tonnes for the year ending September 2030?"
{"subject":"global gold dental alloy demand","metric_type":"demand","threshold":75,\
"unit":"tonnes","currency":null,"direction":"exceeds","resolution_date":"2030-09-30",\
"benchmark":null,"commodity":"gold"}

Q: "Will Gold spot price (LBMA AM fix) exceed $5,181/ozt on May 31, 2026?"
{"subject":"gold spot price (LBMA AM fix)","metric_type":"price","threshold":5181,\
"unit":"USD/ozt","currency":"USD","direction":"exceeds","resolution_date":"2026-05-31",\
"benchmark":"LBMA","commodity":"gold"}"""


@dataclass
class ParsedMarket:
    subject: str | None = None
    metric_type: str | None = None
    threshold: float | None = None
    unit: str | None = None
    currency: str | None = None
    direction: str = "exceeds"
    resolution_date: str | None = None
    benchmark: str | None = None
    commodity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(raw: dict[str, Any]) -> ParsedMarket:
    """Validate/normalize the model's JSON into a ParsedMarket."""
    thr = raw.get("threshold")
    if isinstance(thr, str):
        m = re.search(r"-?\d[\d,]*\.?\d*", thr)
        thr = float(m.group(0).replace(",", "")) if m else None
    elif isinstance(thr, (int, float)):
        thr = float(thr)
    else:
        thr = None
    mt = (raw.get("metric_type") or "").strip().lower()
    if mt not in _METRIC_TYPES:
        mt = "other"
    direction = "below" if str(raw.get("direction", "")).lower().startswith("bel") else "exceeds"
    return ParsedMarket(
        subject=(raw.get("subject") or None),
        metric_type=mt,
        threshold=thr,
        unit=(raw.get("unit") or None),
        currency=(raw.get("currency") or None),
        direction=direction,
        resolution_date=(raw.get("resolution_date") or None),
        benchmark=(raw.get("benchmark") or None),
        commodity=(str(raw.get("commodity")).lower() if raw.get("commodity") else None),
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response (tolerates stray prose
    or <think> tokens some local models emit)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None


def extract(question: str, llm: LocalLLM | None = None) -> ParsedMarket | None:
    """LLM-parse one question. Returns None on failure (caller should fall back to
    the deterministic parser)."""
    llm = llm or LocalLLM()
    try:
        raw = _extract_json(llm.json_completion(_SYSTEM, f"Q: {question!r}"))
    except Exception:  # noqa: BLE001 - network/model error -> let caller fall back
        return None
    return _coerce(raw) if raw else None
