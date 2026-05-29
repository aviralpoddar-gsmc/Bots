"""Structural arbitrage on the vault-procurement kg ladders — in TWO dimensions.

The clone lists, for each metal, a grid of markets:

    "Will U.S. vault procurement of {Metal} exceed {strike} kg by December 31, {year}?"

across several strikes (3.15e6, 4.2e6, 5.25e6 kg, ...) AND several expiries (2027,
2028, ...). The survival probabilities P(cumulative procurement > strike by year)
obey two hard internal constraints, with NO external data and NO distributional
assumption:

  1. **Monotone-down in strike** — exceeding a higher bar can't be more likely
     (this is what `ladder_arb` already enforces on a single date).
  2. **Monotone-up in expiry** — vault procurement is *cumulative*, so the amount
     held by 2028 ≥ the amount by 2027; hence P(exceed X by 2028) ≥ P(exceed X by
     2027). `term_structure` only *smooths* the time axis; here it is a hard
     inequality, because the quantity can only grow.

Together these define a 2-D monotone surface. We fit the nearest such surface to
the quoted prices by **cyclic isotonic projection** — alternately running the
weighted PAVA fit (reused from `ladder_arb`) down each strike-line and up each
expiry-line until it converges. Alternating projection onto two convex monotone
cones converges to a point in their intersection, i.e. a surface monotone in both
axes. Strikes off that surface are provably mispriced relative to their grid
neighbours, so we trade them toward the fit.

Resolvability note: "vault procurement ... kg" is a quantity figure with no clean
public reporting line, so `resolvability.py` scores it very low — most of these
will CANCEL (refund). That makes this a *cancel-safe* breadth play: the downside
of a wrong-looking arb is tied-up budget, not loss, and the runner's resolvability
weighting keeps the book small. Pure internal coherence, no feed needed.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Market, Strategy
from .ladder_arb import isotonic_decreasing

# "exceed 3.15e+06 kg" / "exceed 4,200,000 kg" — strike is scientific or plain.
# NOTE: the generic `ladder.parse_threshold` truncates "3.15e+06" to 3.15, so this
# family needs its own scientific-notation-aware parser.
_STRIKE_KG = re.compile(r"exceed\s+([0-9][0-9,]*(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)\s*kg", re.I)
_METAL = re.compile(r"vault procurement of\s+(.+?)\s+exceed", re.I)
_MONTH = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:(\d{1,2})\s*,?\s+)?(\d{4})\b",
    re.I)


def isotonic_increasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted isotonic regression enforcing a NON-DECREASING sequence: run the
    non-increasing PAVA on the reversed series and flip back."""
    rev = isotonic_decreasing(values[::-1], weights[::-1])
    return rev[::-1]


def _parse_strike_kg(question: str) -> float | None:
    m = _STRIKE_KG.search(question)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _expiry_key(question: str) -> tuple[int, int, int] | None:
    """Chronological sort key (year, month, day) for the resolution date."""
    m = _DATE.search(question)
    if not m:
        return None
    month = _MONTH[m.group(1).lower()[:3]]
    day = int(m.group(2)) if m.group(2) else 31
    return int(m.group(3)), month, day


def _metal_key(question: str) -> str | None:
    m = _METAL.search(question)
    return m.group(1).strip().lower() if m else None


class StockpileGridArbStrategy(Strategy):
    name = "stockpile_grid_arb"
    description = (
        "Model-free 2-D structural arbitrage on the vault-procurement kg ladders. "
        "Survival must be monotone-DOWN in strike (bigger bar = harder) and "
        "monotone-UP in expiry (cumulative procurement only grows). Fits the "
        "nearest 2-D monotone surface by cyclic isotonic projection (weighted "
        "PAVA, reused from ladder_arb) and trades off-surface strikes toward it. "
        "Cancel-safe: no external data, pure grid coherence."
    )

    def __init__(self, dev_band: float = 0.03, informative_weight: float = 5.0,
                 min_nodes: int = 4, skip_extreme: float = 0.02, max_iter: int = 25,
                 **params: Any):
        super().__init__(dev_band=dev_band, informative_weight=informative_weight,
                         min_nodes=min_nodes, skip_extreme=skip_extreme,
                         max_iter=max_iter, **params)
        self.dev_band = dev_band
        self.informative_weight = informative_weight
        self.min_nodes = min_nodes
        self.skip_extreme = skip_extreme
        self.max_iter = max_iter

    def _is_vault_kg(self, q: str) -> bool:
        return (_parse_strike_kg(q) is not None and _metal_key(q) is not None
                and _expiry_key(q) is not None)

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [m for m in super().prefilter(markets)
                if self._is_vault_kg(m.get("question", ""))
                and m.get("probability") is not None]

    def group(self, markets: list[Market]) -> list[list[Market]]:
        groups: dict[str, list[Market]] = {}
        for m in markets:
            key = _metal_key(m.get("question", ""))
            if key:
                groups.setdefault(key, []).append(m)
        return list(groups.values())

    def correlation_key(self, market: Market) -> str:
        key = _metal_key(market.get("question", ""))
        return f"vault:{key}" if key else str(market.get("id"))

    def _weight(self, market: Market) -> float:
        prob = market.get("probability", 0.5) or 0.5
        informative = (market.get("volume") or 0) > 0 or abs(prob - 0.5) > self.dev_band
        return self.informative_weight if informative else 1.0

    def _fit_surface(self, nodes: dict[tuple[float, tuple], list[float]]
                     ) -> dict[tuple[float, tuple], float]:
        """nodes: (strike, expiry_key) -> [sum_w*surv, sum_w]. Returns the fitted
        survival per node, monotone-down in strike and monotone-up in expiry."""
        strikes = sorted({s for s, _e in nodes})
        expiries = sorted({e for _s, e in nodes})
        surf = {k: acc[0] / acc[1] for k, acc in nodes.items()}
        wt = {k: acc[1] for k, acc in nodes.items()}
        for _ in range(self.max_iter):
            prev = dict(surf)
            # Project each expiry-column to non-increasing in strike.
            for e in expiries:
                line = [(s, surf[(s, e)], wt[(s, e)]) for s in strikes if (s, e) in surf]
                if len(line) >= 2:
                    fit = isotonic_decreasing([v for _s, v, _w in line], [w for _s, _v, w in line])
                    for (s, _v, _w), f in zip(line, fit):
                        surf[(s, e)] = f
            # Project each strike-row to non-decreasing in expiry (cumulative).
            for s in strikes:
                line = [(e, surf[(s, e)], wt[(s, e)]) for e in expiries if (s, e) in surf]
                if len(line) >= 2:
                    fit = isotonic_increasing([v for _e, v, _w in line], [w for _e, _v, w in line])
                    for (e, _v, _w), f in zip(line, fit):
                        surf[(s, e)] = f
            if max((abs(surf[k] - prev[k]) for k in surf), default=0.0) < 1e-7:
                break
        return surf

    def estimate(self, group: list[Market]) -> dict[str, float]:
        usable = [m for m in group if self._is_vault_kg(m.get("question", ""))]
        if len(usable) < self.min_nodes:
            return {}
        probs = [m["probability"] for m in usable]
        if max(probs) - min(probs) < 1e-9:
            return {}  # flat grid — no structure to enforce

        # Aggregate duplicate (strike, expiry) nodes by weighted-mean survival.
        nodes: dict[tuple[float, tuple], list[float]] = {}
        meta: list[tuple[str, float, tuple, float]] = []  # (id, strike, expiry, market_surv)
        for m in usable:
            strike = _parse_strike_kg(m["question"])
            expiry = _expiry_key(m["question"])
            surv = float(m["probability"])  # all "exceed" -> survival = probability
            w = self._weight(m)
            acc = nodes.setdefault((strike, expiry), [0.0, 0.0])
            acc[0] += surv * w
            acc[1] += w
            meta.append((m["id"], strike, expiry, surv))

        if len({s for s, _e in nodes}) < 2 and len({e for _s, e in nodes}) < 2:
            return {}  # need at least one axis with structure
        surf = self._fit_surface(nodes)

        snapshot = sorted(
            ({"strike": s, "expiry": e, "fit": surf[(s, e)],
              "market": nodes[(s, e)][0] / nodes[(s, e)][1]} for (s, e) in nodes),
            key=lambda r: (r["expiry"], r["strike"]))
        out: dict[str, float] = {}
        for mid, strike, expiry, market_surv in meta:
            m = next(x for x in usable if x["id"] == mid)
            if m["probability"] <= self.skip_extreme or m["probability"] >= 1 - self.skip_extreme:
                continue
            p = min(max(surf[(strike, expiry)], 0.01), 0.99)
            out[mid] = p
            self._explanations[mid] = {
                "metal": _metal_key(m["question"]), "strike": strike,
                "expiry": "-".join(str(x) for x in expiry), "market_surv": market_surv,
                "fit_surv": p, "n_nodes": len(nodes), "snapshot": snapshot,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        near = [r for r in d["snapshot"]
                if abs(r["strike"] - d["strike"]) < 1e-9 or str(r["expiry"]).startswith(d["expiry"][:4])]
        rows = "\n".join(
            f"  - {r['strike']:.3g} kg / {r['expiry'][0] if isinstance(r['expiry'],tuple) else r['expiry']}: "
            f"market={r['market']:.2f}, fit={r['fit']:.2f}"
            for r in near[:6])
        return (
            f"- Vault grid: **{d['metal']}** | {d['n_nodes']} (strike × expiry) nodes\n"
            f"- 2-D monotone fit (↓ strike, ↑ expiry; cyclic isotonic / PAVA)\n"
            f"- This node (**{d['strike']:.3g} kg / {d['expiry']}**): "
            f"market_surv={d['market_surv']:.2f} → fit_surv=**{d['fit_surv']:.3f}**\n"
            f"- Grid neighbours:\n{rows}"
        )
