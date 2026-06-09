"""Local-LLM news-headline classifier — the digestion engine for the 007 bot.

Turns a free-text commodity headline into a SIGNED, per-commodity directional record:

    {"commodity": "gold"|null, "direction": -1|0|+1, "confidence": 0..1,
     "is_price_event": bool, "benchmark": str|null}

WHY AN LLM (not regex): the 2026-06-09 source-research workflow showed two traps a
regex cannot handle on these wires — substring false-matches ("tin" inside "scrutiny")
and conflicting directions in one headline ("Gold up amid lower oil") — plus the need
to distinguish Henry Hub natural gas from European TTF (only Henry Hub is tradeable on
the clone). Classification/extraction is a LANGUAGE task where small LOCAL models are
strong; numeric forecasting (where they're weak) is deliberately NOT asked of the model.

The model's ONLY job is to read one headline and decide: is this a price-relevant event
for a clone-tradeable commodity, and if so, which commodity and which direction. The
*magnitude* of the eventual trade is bounded mechanically downstream (a small drift),
never by the model's own (unreliable) confidence alone.

LOCAL COMPUTE ONLY (see llm/client.py — Ollama). On any parse/model failure, classify()
returns a null record (treated as "no signal"), so the LLM is never a single point of
failure and the bot simply abstains on that headline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .client import LocalLLM

# Canonical commodity -> the live price ANCHOR entity (must exist in the Stooq feed).
# Coffee is intentionally ABSENT: the clone has no coffee price feed, so coffee news
# has no anchor to drift and is discarded.
COMMODITY_TO_ENTITY: dict[str, str] = {
    "gold": "GOLD", "silver": "SILVER", "platinum": "PLATINUM", "palladium": "PALLADIUM",
    "copper": "COPPER", "wti": "WTI_OIL", "brent": "BRENT_OIL", "gasoline": "GASOLINE",
    "natgas": "NATGAS", "cotton": "CME_COTTON", "cocoa": "CME_COCOA",
    "corn": "CME_CORN", "sugar": "CME_SUGAR", "wheat": "CME_WHEAT",
}

_CANON = sorted(COMMODITY_TO_ENTITY)

_SYSTEM = """You classify ONE commodity-news headline for a trading bot. Output ONE JSON \
object, no prose, EXACTLY these keys:

- "commodity": the SINGLE tradeable commodity the headline is primarily about, as one of: \
gold, silver, platinum, palladium, copper, wti, brent, gasoline, natgas, cotton, cocoa, \
corn, sugar, wheat. Use null if the headline is NOT primarily about one of these, or is \
about an untradeable thing (a company, equity, ETF, coal, LNG shipping, ESG/finance, a \
person, sport, generic macro with no single commodity).
- "direction": +1 if the headline implies the commodity's PRICE should RISE, -1 if it \
implies the price should FALL, 0 if unclear/neutral. Judge the price of the NAMED \
commodity only (e.g. "Gold up amid lower oil" -> for gold +1, NOT about oil).
- "confidence": 0.0-1.0, how clearly the headline states a price-moving direction for that \
commodity (a clear "Oil falls 3% on demand fears" ~0.8; a vague "oil in focus" ~0.2).
- "is_price_event": true ONLY if the headline is about the commodity's PRICE/supply/demand/ \
inventory in a way that should move price. false for corporate news, ratings, ETFs, \
opinion with no catalyst, pure geopolitics with no commodity link.
- "benchmark": a named benchmark if present (LBMA, COMEX, NYMEX, ICE, CME, LME, "Henry Hub", \
"TTF") else null.

CRITICAL RULES:
- "natural gas" / "European gas" / "TTF" = European benchmark, NOT tradeable here -> set \
commodity=null UNLESS it clearly means US Henry Hub natural gas (then commodity="natgas").
- Map "crude"/"oil" to "wti" unless the headline specifically says Brent (-> "brent").
- If the headline names NO commodity from the allowed list, commodity=null, is_price_event=false.
- Never invent a commodity from a substring (e.g. "scrutiny" is NOT tin).

Examples:
H: "Oil falls as investors await clarity after Iran-Israel halt attacks"
{"commodity":"wti","direction":-1,"confidence":0.7,"is_price_event":true,"benchmark":null}
H: "Gold edges up amid lower oil prices, ongoing U.S. rate hike expectations"
{"commodity":"gold","direction":1,"confidence":0.6,"is_price_event":true,"benchmark":null}
H: "European natural gas edges lower after Israel, Iran halt attacks"
{"commodity":null,"direction":0,"confidence":0.0,"is_price_event":false,"benchmark":"TTF"}
H: "Morgan Stanley Sees Asian LNG Prices Soaring to 3.5-Year High"
{"commodity":null,"direction":0,"confidence":0.0,"is_price_event":false,"benchmark":null}
H: "Poland, China lead renewed central bank gold buying"
{"commodity":"gold","direction":1,"confidence":0.6,"is_price_event":true,"benchmark":null}
H: "Technip-Airbus JV to build sustainable aviation fuel plant"
{"commodity":null,"direction":0,"confidence":0.0,"is_price_event":false,"benchmark":null}"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """First JSON object in a model response (tolerates stray prose / <think> tokens)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(text, str):
        return None
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


def _coerce(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate/normalize the model JSON into the canonical record (fail -> null signal)."""
    com = raw.get("commodity")
    com = str(com).strip().lower() if com else None
    if com not in COMMODITY_TO_ENTITY:
        com = None
    d = raw.get("direction")
    try:
        direction = max(-1, min(1, int(d)))
    except (TypeError, ValueError):
        direction = 0
    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    is_price = bool(raw.get("is_price_event")) and com is not None
    if not is_price or com is None:
        # Not a usable signal -> normalize to a clean null record.
        return {"commodity": None, "direction": 0, "confidence": 0.0,
                "is_price_event": False, "benchmark": raw.get("benchmark") or None}
    return {"commodity": com, "direction": direction, "confidence": conf,
            "is_price_event": True, "benchmark": (raw.get("benchmark") or None)}


_NULL = {"commodity": None, "direction": 0, "confidence": 0.0, "is_price_event": False, "benchmark": None}


def classify(headline: str, llm: LocalLLM | None = None) -> dict[str, Any]:
    """Classify one headline into a signed per-commodity record. Returns a null record
    (no signal) on any model/parse failure — the caller then abstains on this item."""
    if not headline or not headline.strip():
        return dict(_NULL)
    llm = llm or LocalLLM()
    try:
        raw = _extract_json(llm.json_completion(_SYSTEM, f"H: {headline.strip()!r}"))
    except Exception:  # noqa: BLE001 - network/model error -> null signal, caller abstains
        return dict(_NULL)
    return _coerce(raw) if raw else dict(_NULL)
