"""Structural arbitrage: enforce monotonicity within a threshold ladder (no LLM,
no data feed, no distributional assumption — pure internal coherence).

A "measurable" (e.g. "SQM lithium sales for the year ending June 2026") is listed
as a ladder of threshold markets: "exceed 60kt?", "...100kt?", "...150kt?". Their
survival probabilities P(value > threshold) MUST be non-increasing in threshold:
exceeding a higher bar can't be more likely than exceeding a lower one. When the
market violates that — survival rising with threshold — at least one strike is
provably mispriced *relative to the others*, and the market's own prices bound the
fair value. Betting toward coherence is +EV at resolution regardless of the
outcome, using no external information.

This bot is **domain-agnostic**: it works on any numeric ladder (production,
demand, price, spread, ...), so it covers the whole universe, not just markets we
have a data feed for. It is the model-free counterpart to `surface_arb` (which
fits a parametric normal CDF and needs numpy/scipy); here we fit a weighted
isotonic regression (pool-adjacent-violators) in pure stdlib, assuming nothing
about the shape of the distribution — only that survival is monotone.

Two design choices keep it honest:
  - **Date-aware grouping.** A ladder is keyed by (metric, *resolution date*), so
    strikes resolving on different dates are never fit together (the shared
    `ladder.measurable_key` strips dates along with the threshold — a bug we avoid).
  - **Informative weighting.** Untraded strikes sitting at the 0.50 default carry
    little weight, so a few traded/:moved prices set the curve and the flat
    defaults are pulled onto it — not the other way around.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Market, Strategy
from .ladder import parse_threshold

# Resolution date = a month name + optional day + 4-digit year, in any of the
# phrasings the clone uses ("on June 30, 2027", "for June 2026", "for the year
# through September 2030", "as of March 31, 2030", "year ending Sept 2030", ...).
# We match the month/day/year core directly so we don't have to enumerate every
# connective; whatever precedes it is left in the metric (consistent across a
# ladder's strikes, so grouping is unaffected).
_MONTH = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_DATE = re.compile(rf"\b({_MONTH})\.?\s+(?:(\d{{1,2}})\s*,?\s+)?(\d{{4}})\b", re.I)
_QUARTER = re.compile(r"\b(Q[1-4]|[1-4]Q|H[12])\s*'?\s*((?:19|20)\d{2})\b", re.I)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_NUM = re.compile(r"(-?\s*\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?)")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _extract_date(question: str) -> tuple[str, str]:
    """Return (date_str, question-with-date-removed). Tiered: month[-day]-year,
    then quarter-year, then a bare year that isn't the threshold value."""
    d = _DATE.search(question)
    if d:
        month, day, year = d.group(1), d.group(2), d.group(3)
        date = f"{month.lower()[:3]} {day or ''} {year}".replace("  ", " ").strip()
        return date, question[: d.start()] + " " + question[d.end():]
    q = _QUARTER.search(question)
    if q:
        return f"{q.group(1).lower()} {q.group(2)}", question[: q.start()] + " " + question[q.end():]
    # Bare year: pick the last 4-digit year that isn't the parsed threshold.
    from .ladder import parse_threshold
    parsed = parse_threshold(question)
    thr = parsed[0] if parsed else None
    years = [(m.group(1), m.span()) for m in _YEAR.finditer(question)]
    years = [(y, span) for y, span in years if thr is None or float(y) != thr]
    if years:
        y, (s, e) = years[-1]
        return y, question[:s] + " " + question[e:]
    return "<nodate>", question


def ladder_key(question: str) -> tuple[str, str]:
    """(metric_key, date_str) for grouping. The metric is the question with both
    the resolution date and the threshold number stripped, so all strikes of one
    measurable on one date collapse together — and different dates do NOT."""
    date, metric = _extract_date(question)
    metric = _NUM.sub(" ", metric)
    metric = _PUNCT.sub(" ", metric.lower())
    metric = _WS.sub(" ", metric).strip()
    return metric, date


def isotonic_decreasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted isotonic regression enforcing a NON-INCREASING sequence (PAVA).

    `values` are ordered by ascending threshold; returns the least-squares
    non-increasing fit. Each block carries (sum w*v, sum w, count, mean); when a
    new block's mean exceeds the previous block's (a non-increasing violation),
    pool them. O(n).
    """
    blocks: list[list[float]] = []  # [sum_wv, sum_w, count, mean]
    for v, w in zip(values, weights):
        blocks.append([v * w, w, 1, v])
        while len(blocks) >= 2 and blocks[-2][3] < blocks[-1][3] - 1e-12:
            b2 = blocks.pop()
            b1 = blocks.pop()
            sw, ww = b1[0] + b2[0], b1[1] + b2[1]
            blocks.append([sw, ww, b1[2] + b2[2], sw / ww if ww else 0.0])
    out: list[float] = []
    for _sw, _ww, count, mean in blocks:
        out.extend([mean] * int(count))
    return out


class LadderArbStrategy(Strategy):
    name = "ladder_arb"

    def __init__(self, dev_band: float = 0.03, informative_weight: float = 5.0,
                 min_strikes: int = 3, skip_extreme: float = 0.02, **params: Any):
        super().__init__(dev_band=dev_band, informative_weight=informative_weight,
                         min_strikes=min_strikes, skip_extreme=skip_extreme, **params)
        # Treat a strike as informative (heavily weighted) if it has volume or has
        # been moved off the 0.50 default by more than this.
        self.dev_band = dev_band
        self.informative_weight = informative_weight
        self.min_strikes = min_strikes
        # Don't TRADE strikes pinned at the clamp extremes: they may be effectively
        # decided/locked, and betting against a market this confident is where the
        # "market knows something we don't" risk lives. They still ANCHOR the fit.
        self.skip_extreme = skip_extreme

    def prefilter(self, markets: list[Market]) -> list[Market]:
        # super() drops resolved / closed / illiquid. Keep anything with a
        # parseable threshold — the domain is every numeric ladder.
        return [
            m for m in super().prefilter(markets)
            if parse_threshold(m.get("question", "")) is not None
            and m.get("probability") is not None
        ]

    def group(self, markets: list[Market]) -> list[list[Market]]:
        groups: dict[tuple[str, str], list[Market]] = {}
        for m in markets:
            groups.setdefault(ladder_key(m.get("question", "")), []).append(m)
        return list(groups.values())

    def correlation_key(self, market: Market) -> str:
        if parse_threshold(market.get("question", "")) is None:
            return str(market.get("id"))
        metric, date = ladder_key(market.get("question", ""))
        return f"{metric}|{date}"

    def _weight(self, market: Market) -> float:
        prob = market.get("probability", 0.5) or 0.5
        informative = (market.get("volume") or 0) > 0 or abs(prob - 0.5) > self.dev_band
        return self.informative_weight if informative else 1.0

    def estimate(self, group: list[Market]) -> dict[str, float]:
        usable = [m for m in group if parse_threshold(m.get("question", "")) is not None]
        if len(usable) < self.min_strikes:
            return {}
        probs = [m["probability"] for m in usable]
        if max(probs) - min(probs) < 1e-9:
            return {}  # fully flat ladder — no structure to enforce

        # Per-market survival = P(value > threshold), tagged with its threshold/weight.
        rows = []
        for m in usable:
            threshold, direction = parse_threshold(m["question"])  # type: ignore[misc]
            surv = m["probability"] if direction == "exceeds" else 1.0 - m["probability"]
            rows.append((threshold, surv, self._weight(m), m["id"]))

        # Aggregate duplicate thresholds (weighted mean survival) before the fit.
        agg: dict[float, list[float]] = {}
        for threshold, surv, w, _id in rows:
            acc = agg.setdefault(threshold, [0.0, 0.0])
            acc[0] += surv * w
            acc[1] += w
        thresholds = sorted(agg)
        values = [agg[t][0] / agg[t][1] for t in thresholds]
        weights = [agg[t][1] for t in thresholds]

        fitted = isotonic_decreasing(values, weights)
        fair_surv = dict(zip(thresholds, fitted))

        # Snapshot of the ladder for explanations: every (threshold, market_surv,
        # fitted_surv, informative?) entry, plus aggregate counts.
        n_informative = sum(1 for _t, _s, w, _id in rows if w > 1.0)
        ladder_snapshot = [
            {"threshold": t, "market_surv": agg[t][0] / agg[t][1], "fit_surv": fair_surv[t]}
            for t in thresholds
        ]

        out: dict[str, float] = {}
        for threshold, market_surv, w, mid in rows:
            m = next(x for x in usable if x["id"] == mid)
            # Pinned-extreme strikes anchor the fit but are not traded.
            if m["probability"] <= self.skip_extreme or m["probability"] >= 1 - self.skip_extreme:
                continue
            _t, direction = parse_threshold(m["question"])  # type: ignore[misc]
            surv = fair_surv[threshold]
            p = surv if direction == "exceeds" else 1.0 - surv
            p = min(max(p, 0.01), 0.99)
            out[mid] = p
            metric, date = ladder_key(m["question"])
            self._explanations[mid] = {
                "metric": metric, "date": date, "threshold": threshold,
                "direction": direction, "market_surv": market_surv,
                "fit_surv": surv, "p": p, "informative_w": w,
                "n_strikes": len(thresholds), "n_informative": n_informative,
                "ladder": ladder_snapshot,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        # Compact ladder view: up to 5 nearest strikes around this one, showing
        # the market's quoted survival vs. the isotonic-fit value.
        ladder = sorted(d["ladder"], key=lambda r: abs(r["threshold"] - d["threshold"]))[:5]
        ladder.sort(key=lambda r: r["threshold"])
        rows = "\n".join(
            f"  - {r['threshold']:g}: market_surv={r['market_surv']:.2f}, "
            f"isotonic={r['fit_surv']:.2f}" + (" ← this strike" if r["threshold"] == d["threshold"] else "")
            for r in ladder
        )
        return (
            f"- Ladder: **{d['metric']}** | resolves {d['date']} | "
            f"{d['n_strikes']} strikes ({d['n_informative']} informative)\n"
            f"- Isotonic (PAVA, non-increasing) fit across the ladder\n"
            f"- This strike (**{d['threshold']:g}**, '{d['direction']}'): "
            f"market_surv={d['market_surv']:.2f}, fit_surv={d['fit_surv']:.2f} "
            f"→ P({d['direction']}) = **{d['p']:.3f}**\n"
            f"- Nearby strikes:\n{rows}"
        )
