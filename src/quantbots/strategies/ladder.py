"""Heuristic parsing of threshold-ladder markets (stdlib only).

Standard Manifold market payloads don't carry a structured "threshold" or
"measurable" — they're embedded in the question text. This module extracts them
with regexes so `surface_arb` has something to fit. It is intentionally simple
and best-effort: replace it with a proper parser (or upstream-tagged fields) for
production use. Kept dependency-free so it can be unit-tested without the `quant`
extra.
"""

from __future__ import annotations

import re
from typing import Any

Market = dict[str, Any]

# "$62", "62.5", "1,200", "-0.5" — first numeric token after a comparison word.
_NUM = r"(-?\s*\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?)"
_EXCEEDS = re.compile(rf"(?:exceed|above|over|greater than|more than|at least|>=|>)\s*{_NUM}", re.I)
_BELOW = re.compile(rf"(?:below|under|less than|at most|<=|<)\s*{_NUM}", re.I)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _to_float(token: str) -> float:
    return float(token.replace(",", "").replace("$", "").replace(" ", ""))


def parse_threshold(question: str) -> tuple[float, str] | None:
    """Return (threshold, direction) where direction is 'exceeds' or 'below'."""
    m = _EXCEEDS.search(question)
    if m:
        return _to_float(m.group(1)), "exceeds"
    m = _BELOW.search(question)
    if m:
        return _to_float(m.group(1)), "below"
    return None


def attach_ladder_fields(market: Market) -> Market:
    """Return a shallow copy with `threshold`/`direction` filled in if parseable."""
    parsed = parse_threshold(market.get("question", ""))
    if parsed is None:
        return market
    threshold, direction = parsed
    return {**market, "threshold": threshold, "direction": direction}


def measurable_key(market: Market) -> str:
    """Group key for the underlying quantity: the question with the threshold
    number stripped out, so all strikes of one measurable collapse together."""
    if market.get("measurable"):
        return str(market["measurable"])
    q = market.get("question", "")
    q = re.sub(_NUM, " ", q)  # drop the strike value
    q = _PUNCT.sub(" ", q.lower())
    return _WS.sub(" ", q).strip()
