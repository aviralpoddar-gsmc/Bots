"""Term-structure (time) coherence: a metric+threshold priced across many
resolution dates must trace a SMOOTH curve in time — trade the dates that don't.

Bot A (`ladder_arb`) enforces coherence across thresholds on one date (the price
ladder must be monotone). This is the orthogonal axis: hold the threshold fixed and
look across resolution dates. A real quantity — a production rate, an inventory, a
spot price — doesn't lurch to a coin-flip for one quarter and snap back, so the
probability P(value > K) should vary smoothly from one date to the next.

What the clone actually looks like (verified): for a given metric+threshold, a few
dates are traded to some level (say ~0.9) while neighbouring dates sit untouched at
the 0.50 default. Those 0.50s are stale, not informed. So this bot fits a smooth
curve through the dates, weighting the *informative* ones (traded, or moved off
0.50) heavily, and pulls the stale/outlier dates onto it. No external data, no
distributional assumption — just temporal smoothness anchored by the dates the
market has actually priced.

Unlike across-threshold coherence the curve is NOT monotone in time (for a snapshot
of a level it can rise or fall), so we use a kernel smoother rather than isotonic
regression. Guards mirror bot A: need >=2 informative anchors and >=3 dates, and
strikes pinned at the clamp anchor the fit but aren't traded.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .base import Market, Strategy
from .ladder import parse_threshold
from .ladder_arb import ladder_key

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_Qn = re.compile(r"q([1-4])")


def date_ordinal(date_str: str) -> int | None:
    """Map a `ladder_key` date string to a sortable month-ordinal (year*12+month),
    so time-distances between resolution dates are well defined."""
    parts = date_str.split()
    if not parts:
        return None
    if parts[0] in _MONTHS:
        return int(parts[-1]) * 12 + _MONTHS[parts[0]]
    q = _Qn.match(date_str)
    if q:
        return int(parts[-1]) * 12 + int(q.group(1)) * 3
    if re.fullmatch(r"(19|20)\d\d", parts[-1]):
        return int(parts[-1]) * 12 + 6  # bare year -> mid-year
    return None


class TermStructureStrategy(Strategy):
    name = "term_structure"
    description = (
        "Time-axis coherence arbitrage. Holding (metric, threshold) fixed and "
        "varying resolution date, the survival curve must vary smoothly through "
        "time — real quantities don't lurch to 0.50 for one date and snap back. "
        "Fits a Gaussian kernel smoother through informative anchor dates and "
        "pulls stale 0.50-pinned dates onto the implied curve. Orthogonal to "
        "ladder_arb (which works across thresholds on one date)."
    )

    def __init__(self, bandwidth: float = 6.0, dev_band: float = 0.03,
                 informative_weight: float = 5.0, min_dates: int = 3,
                 min_anchors: int = 2, skip_extreme: float = 0.02,
                 prior_strength: float = 1.0, **params: Any):
        super().__init__(bandwidth=bandwidth, dev_band=dev_band,
                         informative_weight=informative_weight, min_dates=min_dates,
                         min_anchors=min_anchors, skip_extreme=skip_extreme,
                         prior_strength=prior_strength, **params)
        self.bandwidth = bandwidth          # Gaussian kernel width, in months
        self.dev_band = dev_band            # |prob-0.5|>this (or any volume) => informative
        self.informative_weight = informative_weight
        self.min_dates = min_dates
        self.min_anchors = min_anchors      # need this many informative dates to fit
        self.skip_extreme = skip_extreme
        # Pseudo-count at 0.5: a stale date with no nearby anchor stays ~0.5 (we
        # don't confidently extrapolate the curve far from any traded date).
        self.prior_strength = prior_strength

    def _ts_key(self, market: Market) -> str | None:
        """(metric, threshold, direction) — one term-structure curve. None if the
        question has no parseable threshold or no datable resolution."""
        parsed = parse_threshold(market.get("question", ""))
        if parsed is None:
            return None
        metric, date = ladder_key(market.get("question", ""))
        if date_ordinal(date) is None:
            return None
        threshold, direction = parsed
        return f"{metric}|t{threshold:g}|{direction}"

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [m for m in super().prefilter(markets)
                if m.get("probability") is not None and self._ts_key(m) is not None]

    def group(self, markets: list[Market]) -> list[list[Market]]:
        groups: dict[str, list[Market]] = {}
        for m in markets:
            groups.setdefault(self._ts_key(m), []).append(m)  # type: ignore[arg-type]
        return list(groups.values())

    def correlation_key(self, market: Market) -> str:
        return self._ts_key(market) or str(market.get("id"))

    def _weight(self, market: Market) -> float:
        prob = market.get("probability", 0.5) or 0.5
        informative = (market.get("volume") or 0) > 0 or abs(prob - 0.5) > self.dev_band
        return self.informative_weight if informative else 1.0

    def estimate(self, group: list[Market]) -> dict[str, float]:
        pts = []  # (ordinal, prob, is_anchor, market)
        for m in group:
            _metric, date = ladder_key(m.get("question", ""))
            o = date_ordinal(date)
            if o is None:
                continue
            pts.append((o, m["probability"], self._weight(m) > 1.0, m))
        if len({o for o, _, _, _ in pts}) < self.min_dates:
            return {}
        anchors = [(o, p) for o, p, is_a, _ in pts if is_a]
        if len(anchors) < self.min_anchors:
            return {}  # not enough traded dates to define the curve

        # Estimate each STALE (non-anchor) date from the TRADED anchors only — never
        # from other stale 0.50s, and never overriding a traded price. A 0.5 prior
        # pseudo-count means a date with no nearby anchor stays ~0.5 (no confident
        # extrapolation), so we only correct dates the term structure actually pins.
        bw2 = 2.0 * self.bandwidth * self.bandwidth
        # Anchor dates (for explanation), as (date_str, prob).
        anchor_labels: list[tuple[str, float]] = []
        for o, p, is_a, am in pts:
            if is_a:
                _met, adate = ladder_key(am.get("question", ""))
                anchor_labels.append((adate, p))
        out: dict[str, float] = {}
        for oi, prob_i, is_anchor, m in pts:
            if is_anchor:
                continue  # trust traded dates; we only fill the stale ones
            if m["probability"] <= self.skip_extreme or m["probability"] >= 1 - self.skip_extreme:
                continue
            num = self.prior_strength * 0.5
            den = self.prior_strength
            contribs: list[tuple[str, float, float]] = []  # (date, prob, weight)
            for (oj, pj), (adate, _) in zip(anchors, anchor_labels):
                k = math.exp(-((oi - oj) ** 2) / bw2)  # Gaussian in time-distance
                num += k * pj
                den += k
                contribs.append((adate, pj, k))
            p_smooth = min(max(num / den, 0.01), 0.99)
            out[m["id"]] = p_smooth
            _metric, this_date = ladder_key(m.get("question", ""))
            ts_key = self._ts_key(m) or ""
            self._explanations[m["id"]] = {
                "ts_key": ts_key, "date": this_date,
                "market_prob": m["probability"], "smoothed": p_smooth,
                "bandwidth": self.bandwidth, "n_anchors": len(anchors),
                "contribs": sorted(contribs, key=lambda c: -c[2])[:5],
                "prior_weight": self.prior_strength, "total_weight": den,
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        # Show the top-weighted anchor contributions: which neighbouring dates
        # actually drove the smoothed value, and how much weight each carried.
        contrib_lines = "\n".join(
            f"  - {ad}: market={ap:.2f}, kernel_weight={w:.2f}"
            for ad, ap, w in d["contribs"]
        )
        return (
            f"- Term-structure curve: **{d['ts_key']}**\n"
            f"- This date ({d['date']}): market priced **{d['market_prob']:.2f}** "
            f"(stale — no trade volume / near 0.50)\n"
            f"- Gaussian kernel smoother: bandwidth={d['bandwidth']:.1f} months, "
            f"{d['n_anchors']} traded anchors on this curve\n"
            f"- Top-weighted anchors:\n{contrib_lines}\n"
            f"- Smoothed estimate from neighbours: **{d['smoothed']:.3f}**"
        )
