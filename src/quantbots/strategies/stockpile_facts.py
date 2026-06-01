"""Reference-lookup bot for U.S. strategic-materials *fact* markets.

Two families on the clone are not price forecasts at all — they are binary facts
published by the U.S. government, yet the clone seeds them at a coin-flip:

  1. "Will {Metal} be on the U.S. Critical Minerals list as of December 31, Y?"
     Source: the USGS *List of Critical Minerals*, published in the Federal
     Register on a statutory cadence (2018 → 2022 (50 minerals) → 2025; the Energy
     Act of 2020 requires review at least every 3 years). Whether a metal is on
     the list is a published, verifiable fact, and the list is extremely sticky —
     geopolitically central minerals (antimony, gallium, graphite, niobium, the
     rare earths) do not get dropped. → resolvable AND near-deterministic.

  2. "Will the U.S. National Defense Stockpile hold a {Metal} position by Dec 31, Y?"
     Source: the DLA Strategic Materials inventory (Operations Report to Congress).
     An unclassified inventory list exists, but *specific* holdings are partly
     classified, so per-metal resolution is messier and only some metals have an
     unambiguous public record. We therefore price ONLY the metals whose held
     status is documented in public GAO/CRS reporting and abstain on the rest —
     the same strict, "abstain when unsure" stance that keeps `commodity_spot`
     out of confidently-wrong bets.

This bot does NOT touch the vault-procurement kg ladders (that is the structural
`stockpile_grid_arb`) or the buffer-stock markets (that is `stockpile_coherence`).
Ownership is clean so the bots don't fight over the same strikes.

The reference data is a curated, cited table baked into this module (like
`commodity_spot`'s unit specs) rather than a network feed — these are slow-moving,
human-curated facts, and the NDS inventory cannot be cleanly scraped anyway.
Refresh the tables when USGS publishes a new list or DLA a new report.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Market, Strategy

# --- USGS 2022 Final List of Critical Minerals (50) --------------------------
# Federal Register 87 FR 10381 (2022-02-24). Normalised to lowercase singular.
# https://www.federalregister.gov/documents/2022/02/24/2022-04027/2022-final-list-of-critical-minerals
_CRITICAL_LIST_2022: frozenset[str] = frozenset({
    "aluminum", "antimony", "arsenic", "barite", "beryllium", "bismuth", "cerium",
    "cesium", "chromium", "cobalt", "dysprosium", "erbium", "europium", "fluorspar",
    "gadolinium", "gallium", "germanium", "graphite", "hafnium", "holmium", "indium",
    "iridium", "lanthanum", "lithium", "lutetium", "magnesium", "manganese",
    "neodymium", "nickel", "niobium", "palladium", "platinum", "praseodymium",
    "rhodium", "rubidium", "ruthenium", "samarium", "scandium", "tantalum",
    "tellurium", "terbium", "thulium", "tin", "titanium", "tungsten", "vanadium",
    "ytterbium", "yttrium", "zinc", "zirconium",
})
# Group names the clone uses that resolve to on-list constituents.
_GROUP_ON_LIST: frozenset[str] = frozenset({
    "rare earths",            # the individual REEs (cerium, lanthanum, ...) are all listed
    "platinum-group metals",  # palladium, platinum, rhodium, iridium, ruthenium all listed
})

# --- NDS held positions (curated, cited, conservative) -----------------------
# True  = public record clearly shows a held position (or active, funded acquisition
#         that puts a position in place by the 2027/2028 horizons).
# We deliberately do NOT assert False for metals merely lacking public evidence
# (absence of evidence != absence of holding), so unknown metals -> abstain.
# Sources: GAO-24-106959 (NDS, Sep 2024) — inventory "includes copper, nickel,
# lithium, antimony, as well as 16 rare earth elements"; CRS R47833 — germanium
# actively held/recovered ("3,000 kg of 99.999% pure germanium ingots", FY2022).
# The One Big Beautiful Act ($2B NDS Transaction Fund) + 2025 DoD intent to procure
# ~$1B of stockpile materials means the held set is expanding, not shrinking.
_NDS_HELD: dict[str, bool] = {
    "germanium": True,
    "copper": True,
    "nickel": True,
    "lithium": True,
    "antimony": True,
    "rare earths": True,
}

# Conviction levels. These are facts, not diffusions — but allow for a list
# re-review dropping a mineral, an NDS divestiture, or a classified record being
# wrong, so never go fully to 0/1.
_P_ON_LIST = 0.93
_P_OFF_LIST = 0.07
_P_HELD = 0.85
_P_NOT_HELD = 0.12

_LIST_RE = re.compile(r"\bwill\s+(.+?)\s+be on the u\.?s\.? critical minerals list\b", re.I)
_NDS_RE = re.compile(r"national defense stockpile hold a (.+?) position", re.I)


def _normalize_metal(raw: str) -> str:
    """Map a market's metal phrase to the canonical key used in the tables."""
    m = raw.strip().lower()
    m = re.sub(r"\s*\(natural\)", "", m)        # "graphite (natural)" -> "graphite"
    m = re.sub(r"\bmetal\b", "", m).strip()     # "magnesium metal" -> "magnesium"
    m = re.sub(r"\s+", " ", m)
    return m


class StockpileFactsStrategy(Strategy):
    name = "stockpile_facts"
    description = (
        "Reference-lookup bot for U.S. strategic-materials fact markets. Prices "
        "'on the Critical Minerals list' against the USGS 2022 list (near-"
        "deterministic, the list is sticky) and 'NDS holds a position' against "
        "documented DLA/GAO holdings — betting hard where the market sits at a "
        "coin-flip but the published fact is known, abstaining where the public "
        "record is ambiguous. Does not touch vault kg ladders or buffer markets."
    )

    def _classify(self, question: str) -> tuple[str, str, float] | None:
        """Return (family, metal, fair_prob) or None to abstain."""
        m = _LIST_RE.search(question)
        if m:
            metal = _normalize_metal(m.group(1))
            on = metal in _CRITICAL_LIST_2022 or metal in _GROUP_ON_LIST
            return "critical_list", metal, (_P_ON_LIST if on else _P_OFF_LIST)
        m = _NDS_RE.search(question)
        if m:
            metal = _normalize_metal(m.group(1))
            held = _NDS_HELD.get(metal)
            if held is None:
                return None  # ambiguous public record -> abstain
            return "nds_hold", metal, (_P_HELD if held else _P_NOT_HELD)
        return None

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [m for m in super().prefilter(markets)
                if self._classify(m.get("question", "")) is not None]

    def correlation_key(self, market: Market) -> str:
        c = self._classify(market.get("question", ""))
        return f"{c[0]}:{c[1]}" if c else str(market.get("id"))

    def estimate(self, group: list[Market]) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in group:
            c = self._classify(m.get("question", ""))
            if c is None:
                continue
            family, metal, p = c
            p = min(max(p, 0.01), 0.99)
            out[m["id"]] = p
            self._explanations[m["id"]] = {"family": family, "metal": metal, "p": p}
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        if d["family"] == "critical_list":
            on = d["metal"] in _CRITICAL_LIST_2022 or d["metal"] in _GROUP_ON_LIST
            return (
                f"- Fact lookup: **{d['metal']}** {'IS' if on else 'is NOT'} on the "
                f"USGS 2022 Critical Minerals list (50 minerals; statutory review, sticky).\n"
                f"- The list rarely drops geopolitically central minerals → "
                f"P(on list) = **{d['p']:.2f}**"
            )
        held = _NDS_HELD.get(d["metal"])
        return (
            f"- Fact lookup: the U.S. National Defense Stockpile "
            f"{'HOLDS' if held else 'does not hold'} a **{d['metal']}** position "
            f"(GAO-24-106959 / CRS R47833; held set expanding under the $2B NDS fund).\n"
            f"- P(holds position) = **{d['p']:.2f}**"
        )
