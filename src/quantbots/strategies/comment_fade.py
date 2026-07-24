"""Adversarial comment-fade strategy: counter-bet HIGH-confidence unsound comments.

The judge cycle (`quantbots judge-comments`) writes evidence-anchored verdicts to
the store; this strategy turns the actionable ones — unsound + high confidence +
an attached bet — into a fair value tilted AGAINST the bad commenter's position.
The normal runner then applies sizing, portfolio caps, and resolvability
weighting, and posts our reasoning as a comment (the pipeline guarantee), so the
fleet's counter-position is itself explained in-thread.

Only the confidence-gated verdicts trade (Bridgewater supervisor rule): a comment
that merely *reads* badly is never faded — only one whose factual claims
contradicted our feed data.
"""

from __future__ import annotations

import json
from typing import Any

from .base import Market, Strategy
from .linker import link_market


class CommentFadeStrategy(Strategy):
    name = "comment_fade"
    description = (
        "Adversarial fade of provably-wrong market comments. An evidence-anchored "
        "local-LLM judge marks comments whose factual claims contradict our data "
        "feeds; this bot bets against those commenters' attached positions, "
        "shifting fair value away from the bad bet (capped)."
    )

    def __init__(self, **params: Any):
        super().__init__(**params)
        self._store: Any = None
        #: per bad bet fair-value shift, and the cap across multiple bad bets
        self.shift_per_verdict = float(params.get("shift_per_verdict", 0.05))
        self.max_shift = float(params.get("max_shift", 0.12))
        self.max_age_hours = float(params.get("max_age_hours", 48.0))

    def bind(self, observations: Any) -> None:
        self._store = observations  # the Store: verdicts live next to observations

    def prefilter(self, markets: list[Market]) -> list[Market]:
        markets = super().prefilter(markets)
        if self._store is None:
            return []
        actionable = {v["market_id"]
                      for v in self._store.actionable_verdicts(max_age_hours=self.max_age_hours)}
        return [m for m in markets if m.get("id") in actionable]

    @staticmethod
    def _fights_anchor(v: dict) -> bool:
        """Only fade a bet on the WRONG SIDE of our data anchor. A commenter with
        wrong reasoning but a conclusion that agrees with the feed (e.g. YES on
        'gold > 3500' while gold sits at 4123) must NOT be faded — betting against
        a data-confirmed outcome is -EV regardless of how bad their argument was."""
        try:
            ev = json.loads(v.get("evidence") or "{}")
            thr, feed = ev.get("threshold"), ev.get("feed_value")
            if thr is None or feed is None:
                return False
            anchor_yes = (feed > thr) if ev.get("direction") == "exceeds" else (feed < thr)
            return (v["bet_outcome"] == "YES") != anchor_yes
        except (TypeError, ValueError):
            return False

    def estimate(self, group: list[Market]) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._store is None:
            return out
        for m in group:
            mid = str(m.get("id"))
            prob = m.get("probability")
            if prob is None:
                continue
            verdicts = [v for v in self._store.actionable_verdicts(
                            market_id=mid, max_age_hours=self.max_age_hours)
                        if self._fights_anchor(v)]
            if not verdicts:
                continue
            # Net direction of the bad money: fade it. YES-bets push our fair DOWN.
            net = sum((v["bet_amount"] or 0) * (1 if v["bet_outcome"] == "YES" else -1)
                      for v in verdicts)
            if net == 0:
                continue
            shift = min(self.shift_per_verdict * len(verdicts), self.max_shift)
            fair = prob - shift if net > 0 else prob + shift
            fair = min(max(fair, 0.02), 0.98)
            out[mid] = fair
            self._explanations[mid] = {
                "n_unsound": len(verdicts),
                "net_bad_mana": round(net, 1),
                "market_prob": prob,
                "fair": round(fair, 4),
                "errors": [e for v in verdicts
                           for e in json.loads(v.get("factual_errors") or "[]")][:3],
            }
        return out

    def correlation_key(self, market: Market) -> str:
        link = link_market(market)
        if link and link.entities:
            return link.entities[0]
        return str(market.get("id"))

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        lines = [f"Fading {d['n_unsound']} comment(s) whose factual claims contradict "
                 f"our data feeds (net Ṁ{d['net_bad_mana']:+.0f} of misinformed flow)."]
        lines += [f"- {e}" for e in d["errors"]]
        return "\n".join(lines)
