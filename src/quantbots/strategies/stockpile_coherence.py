"""Time-coherence arb on the cumulative "by-date" strategic-materials markets.

For a single metal the clone runs several cumulative event markets across a set of
resolution dates:

  - "Will {Metal} be included in Project Vault procurement by {date}?"
  - "Will the U.S. begin strategic buffer-stock procurement of {Metal} by {date}?"

Both are **cumulative**: once the event has happened by an earlier date it has
happened by every later date, so P(by t) MUST be non-decreasing in t. We fit the
nearest non-decreasing curve across dates (weighted PAVA, reused from
`stockpile_grid_arb`/`ladder_arb`) and trade strikes that sit off it. This is the
hard-monotonicity counterpart to `term_structure` (which only *smooths* the time
axis); here monotonicity is provable from the "by {date}" phrasing, so we enforce
it as an inequality.

Design note — what this deliberately does NOT do: an earlier version also imposed
a cross-template ceiling ("procurement ⟹ the NDS holds a position"). That was
dropped: "Project Vault" is a clone-specific program, not the NDS, so the
implication across programs isn't sound, and in practice it propagated the seeded
NDS-hold mis-price onto the (better-informed) inclusion markets. The robust,
defensible signal here is purely within-template time monotonicity.

Ownership is disjoint from its siblings (no bot trades the same strike):
  - `stockpile_facts`    owns Critical-Minerals-list + NDS-hold (documented metals).
  - `stockpile_grid_arb` owns the vault "exceed X kg" 2-D grid.
  - this bot           owns the Project-Vault-inclusion + buffer-stock date ladders.

These resolve off a policy announcement / quantity line, so resolvability is low
and many will CANCEL (refund) — a cancel-safe coherence play the runner sizes down
automatically.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Market, Strategy
from .stockpile_facts import _normalize_metal
from .stockpile_grid_arb import _expiry_key, isotonic_increasing

_INCLUSION = re.compile(r"will\s+(.+?)\s+be included in project vault procurement by", re.I)
_BUFFER = re.compile(r"begin strategic buffer[- ]stock procurement of\s+(.+?)\s+by", re.I)


class StockpileCoherenceStrategy(Strategy):
    name = "stockpile_coherence"
    description = (
        "Time-coherence arb on the cumulative 'by-date' strategic-materials "
        "markets (Project-Vault inclusion and buffer-stock procurement). A "
        "cumulative event can't get less likely by a later date, so survival must "
        "be non-decreasing in time; fits the nearest non-decreasing curve via "
        "weighted isotonic regression (PAVA) and trades off-curve dates toward it."
    )

    def __init__(self, dev_band: float = 0.03, informative_weight: float = 5.0,
                 min_dates: int = 3, skip_extreme: float = 0.02, **params: Any):
        super().__init__(dev_band=dev_band, informative_weight=informative_weight,
                         min_dates=min_dates, skip_extreme=skip_extreme, **params)
        self.dev_band = dev_band
        self.informative_weight = informative_weight
        self.min_dates = min_dates
        self.skip_extreme = skip_extreme

    def _classify(self, q: str) -> tuple[str, str] | None:
        """(template, metal) for the two cumulative by-date templates we trade."""
        m = _INCLUSION.search(q)
        if m:
            return "inclusion", _normalize_metal(m.group(1))
        m = _BUFFER.search(q)
        if m:
            return "buffer", _normalize_metal(m.group(1))
        return None

    def prefilter(self, markets: list[Market]) -> list[Market]:
        out = []
        for m in super().prefilter(markets):
            if (self._classify(m.get("question", ""))
                    and _expiry_key(m.get("question", ""))
                    and m.get("probability") is not None):
                out.append(m)
        return out

    def group(self, markets: list[Market]) -> list[list[Market]]:
        # One group per (metal, template) — each is its own time ladder.
        groups: dict[tuple[str, str], list[Market]] = {}
        for m in markets:
            c = self._classify(m.get("question", ""))
            if c:
                groups.setdefault((c[1], c[0]), []).append(m)
        return list(groups.values())

    def correlation_key(self, market: Market) -> str:
        c = self._classify(market.get("question", ""))
        return f"coh:{c[1]}:{c[0]}" if c else str(market.get("id"))

    def _weight(self, market: Market) -> float:
        prob = market.get("probability", 0.5) or 0.5
        informative = (market.get("volume") or 0) > 0 or abs(prob - 0.5) > self.dev_band
        return self.informative_weight if informative else 1.0

    def estimate(self, group: list[Market]) -> dict[str, float]:
        dated = sorted(((m, _expiry_key(m["question"])) for m in group
                        if self._classify(m.get("question", "")) and _expiry_key(m["question"])),
                       key=lambda t: t[1])
        if len(dated) < self.min_dates:
            return {}
        probs = [float(m["probability"]) for m, _e in dated]
        if max(probs) - min(probs) < 1e-9:
            return {}  # flat ladder — no structure to enforce
        weights = [self._weight(m) for m, _e in dated]
        fitted = isotonic_increasing(probs, weights)

        snapshot = [{"expiry": e[0], "market": p, "fit": f}
                    for (_m, e), p, f in zip(dated, probs, fitted)]
        out: dict[str, float] = {}
        for (m, ek), market_p, fit in zip(dated, probs, fitted):
            if m["probability"] <= self.skip_extreme or m["probability"] >= 1 - self.skip_extreme:
                continue
            p = min(max(fit, 0.01), 0.99)
            out[m["id"]] = p
            c = self._classify(m["question"])
            self._explanations[m["id"]] = {
                "template": c[0], "metal": c[1], "expiry": ek[0],
                "market": market_p, "fit": p, "n_dates": len(dated), "snapshot": snapshot,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        rows = "\n".join(
            f"  - by {r['expiry']}: market={r['market']:.2f}, fit={r['fit']:.2f}"
            + (" ← this" if r["expiry"] == d["expiry"] else "")
            for r in d["snapshot"])
        return (
            f"- Time ladder: **{d['metal']}** / {d['template']} | "
            f"{d['n_dates']} dates (cumulative, must be non-decreasing in time)\n"
            f"- Isotonic (PAVA, non-decreasing) fit across resolution dates\n"
            f"- This date (**by {d['expiry']}**): market={d['market']:.2f} → "
            f"fit=**{d['fit']:.3f}**\n"
            f"- Ladder:\n{rows}"
        )
