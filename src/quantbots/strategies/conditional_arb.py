"""Model-free coherence arbitrage on conditional ("B if A") markets.

The clone auto-generates a conditional as a BINARY market whose question is the
lossless wrapper

    IF [<predicate question>] = YES: <quantity question>

and whose fair price is P(B | A). It resolves derived, never researched:
predicate NO -> CANCEL (refund); predicate YES -> copy the quantity outcome. It is
*seeded at P(B)* (the independence assumption P(B|A)=P(B)), so a fresh conditional
is mispriced by exactly the A-B correlation.

This strategy never models that correlation. It enforces the internal coherence
of the **triangle** {predicate A, quantity B, conditional C} using only the three
traded prices a=P(A), b=P(B), c=P(B|A). From the law of total probability with
P(B|not A) in [0,1]:

    max(0, (a+b-1)/a)  <=  c  <=  min(1, b/a)

If C trades outside that Fréchet band, one leg is provably mispriced relative to
the others (same spirit as `ladder_arb`), so we trade C onto the nearest edge.

**The nested case is exact.** When the quantity *implies* the predicate (e.g.
A = "zinc >= 2800", B = "zinc >= 3400"), then B∩A = B and the band collapses to an
equality:

    c = b / a

— a single deterministic constraint over three traded legs (the zinc price-band
triangle). This is the strongest, safest signal: both legs are LME-resolvable, the
arb is locked at resolution, and a CANCEL only refunds.

Design choices that keep it honest, mirroring `ladder_arb`:
  - **Trust the legs, correct the conditional.** The two legs are primary
    price-threshold markets (often traded); the conditional is auto-generated and
    seeded at independence, so it is the most likely-stale node. v1 trades only C.
  - **Informative gate.** Abstain unless *both* legs are informative (have volume
    or are off the 0.50 default) — never trade a triangle of untouched defaults.
  - **Separation of concerns.** If the two legs are themselves survival-inverted
    (b > a in the nested case), abstain and let `ladder_arb` fix the legs first.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Market, Strategy
from .ladder import measurable_key, parse_threshold

# The auto-generated conditional question: `IF [<predicate>] = YES: <quantity>`.
# Greedy first group grabs up to the last ` = YES: ` so a predicate containing
# brackets still parses. DOTALL so multi-line leg questions match.
_COND_RE = re.compile(r"^IF \[(?P<pred>.+)\] = YES: (?P<qty>.+)$", re.S)


def parse_conditional(question: str) -> tuple[str, str] | None:
    """Return (predicate_question, quantity_question) or None if not a conditional."""
    m = _COND_RE.match((question or "").strip())
    if not m:
        return None
    return m.group("pred").strip(), m.group("qty").strip()


class ConditionalArbStrategy(Strategy):
    name = "conditional_arb"
    description = (
        "Model-free coherence arbitrage on conditional ('B if A') markets. A "
        "conditional prices P(B|A); together with its predicate P(A) and quantity "
        "P(B) legs it must satisfy the Fréchet band max(0,(a+b-1)/a) <= c <= "
        "min(1,b/a) — and exactly c=b/a when the quantity implies the predicate "
        "(nested price bands). Trades the conditional toward that constraint using "
        "no external data. Capital-neutral downside: a predicate-NO CANCEL refunds."
    )

    def __init__(
        self,
        dev_band: float = 0.03,
        min_predicate_prob: float = 0.05,
        informative_band: float = 0.03,
        **params: Any,
    ):
        super().__init__(
            dev_band=dev_band,
            min_predicate_prob=min_predicate_prob,
            informative_band=informative_band,
            **params,
        )
        # Only trade when the implied correction exceeds this (avoid churning noise).
        self.dev_band = dev_band
        # Below this predicate probability b/a is unstable and the conditional is
        # near-certain to CANCEL (predicate resolves NO) — abstain.
        self.min_predicate_prob = min_predicate_prob
        # A leg counts as informative if it has volume or is off 0.50 by > this.
        self.informative_band = informative_band

    # -- universe ---------------------------------------------------------------

    def _open(self, m: Market, now_ms: float) -> bool:
        if m.get("isResolved"):
            return False
        close = m.get("closeTime")
        return not (close and close <= now_ms)

    def prefilter(self, markets: list[Market]) -> list[Market]:
        # Keep every open conditional plus the markets named as its legs. Legs are
        # matched against the RAW universe (not liquidity-filtered) — we only read
        # their price, we don't trade them, so a thin leg must not break the triangle.
        import time

        now_ms = time.time() * 1000
        conds: list[Market] = []
        leg_qs: set[str] = set()
        for m in markets:
            pc = parse_conditional(m.get("question", ""))
            if pc and self._open(m, now_ms):
                conds.append(m)
                leg_qs.update(pc)
        if not conds:
            return []
        legs = [m for m in markets if m.get("question", "") in leg_qs]
        seen: set[str] = set()
        out: list[Market] = []
        for m in conds + legs:
            i = str(m.get("id"))
            if i in seen:
                continue
            seen.add(i)
            out.append(m)
        return out

    def group(self, markets: list[Market]) -> list[list[Market]]:
        # First market wins on duplicate questions (the clone has ~2.5k exact dup
        # sets; we just need one priced instance of each leg).
        by_q: dict[str, Market] = {}
        for m in markets:
            by_q.setdefault(m.get("question", ""), m)
        groups: list[list[Market]] = []
        for m in markets:
            pc = parse_conditional(m.get("question", ""))
            if not pc:
                continue
            pred, qty = by_q.get(pc[0]), by_q.get(pc[1])
            if pred is None or qty is None:
                continue  # a leg isn't in the universe — can't price the triangle
            groups.append([m, pred, qty])
        return groups

    def correlation_key(self, market: Market) -> str:
        pc = parse_conditional(market.get("question", ""))
        if not pc:
            return str(market.get("id"))
        # All conditionals on the same underlying quantity share one budget.
        return "cond|" + measurable_key({"question": pc[1]})

    # -- coherence --------------------------------------------------------------

    def _informative(self, m: Market) -> bool:
        prob = m.get("probability", 0.5) or 0.5
        return (m.get("volume") or 0) > 0 or abs(prob - 0.5) > self.informative_band

    def _is_nested(self, pred_q: str, qty_q: str) -> bool:
        """True when quantity B implies predicate A (B∩A = B), so c = b/a exactly."""
        pa = parse_threshold(pred_q)
        pb = parse_threshold(qty_q)
        if not pa or not pb:
            return False
        if measurable_key({"question": pred_q}) != measurable_key({"question": qty_q}):
            return False
        (ta, da), (tb, db) = pa, pb
        if da != db:
            return False
        # exceeds: >=tb implies >=ta iff tb>=ta. below: <=tb implies <=ta iff tb<=ta.
        return tb >= ta if da == "exceeds" else tb <= ta

    def estimate(self, group: list[Market]) -> dict[str, float]:
        cond = next((m for m in group if parse_conditional(m.get("question", ""))), None)
        if cond is None:
            return {}
        pc = parse_conditional(cond["question"])
        assert pc is not None  # cond matched above
        pred_q, qty_q = pc
        pred = next((m for m in group if m.get("question") == pred_q), None)
        qty = next((m for m in group if m.get("question") == qty_q), None)
        if pred is None or qty is None:
            return {}

        a = pred.get("probability")
        b = qty.get("probability")
        c = cond.get("probability")
        if a is None or b is None or c is None:
            return {}
        if a < self.min_predicate_prob:
            return {}  # predicate near-impossible: b/a unstable + ~certain CANCEL
        if not (self._informative(pred) and self._informative(qty)):
            return {}  # a triangle of untouched defaults carries no information

        nested = self._is_nested(pred_q, qty_q)
        if nested:
            if b > a + 1e-9:
                return {}  # legs are survival-inverted — let ladder_arb fix them first
            target = b / a
            lo = hi = target
            kind = "nested (c = b/a)"
        else:
            lo = max(0.0, (a + b - 1.0) / a)
            hi = min(1.0, b / a)
            if lo <= c <= hi:
                return {}  # already inside the Fréchet band — coherent
            target = min(max(c, lo), hi)  # project onto the nearest violated edge
            kind = "fréchet band"

        target = min(max(target, 0.01), 0.99)
        if abs(target - c) < self.dev_band:
            return {}

        self._explanations[cond["id"]] = {
            "predicate_q": pred_q,
            "quantity_q": qty_q,
            "a": a,
            "b": b,
            "c": c,
            "lo": lo,
            "hi": hi,
            "target": target,
            "kind": kind,
        }
        return {cond["id"]: target}

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        return (
            f"- Conditional triangle ({d['kind']})\n"
            f"  - Predicate A: P(A) = **{d['a']:.3f}** — {d['predicate_q']}\n"
            f"  - Quantity  B: P(B) = **{d['b']:.3f}** — {d['quantity_q']}\n"
            f"- Coherence requires P(B|A) ∈ [{d['lo']:.3f}, {d['hi']:.3f}]\n"
            f"- Market quotes P(B|A) = **{d['c']:.3f}** → fair value "
            f"**{d['target']:.3f}** (trade toward coherence)"
        )
