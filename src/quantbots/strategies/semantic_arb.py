"""Semantic cross-market arbitrage: trade toward LOGICAL coherence across markets
the structural bots can't link.

`ladder_arb` enforces monotonicity *within one measurable's* threshold ladder
(same metric, same date, strikes ordered). It is blind to relations that span a
wording or entity boundary:

  - **Reworded equivalents.** Two questions that resolve YES under identical
    conditions but are phrased differently (the ~2,485 byte-distinct duplicate
    sets, plus looser paraphrases) — `measurable_key` keeps them apart, so
    `ladder_arb` never pools them even though P(A) MUST equal P(B).
  - **Cross-phrasing entailment.** "X price exceeds $5 in 2027" logically implies
    "X price exceeds $4 in 2027" even when the two are written so differently that
    they land in different ladders — survival must still be ordered.
  - **Negation / mutual exclusion.** "above $5" vs "at or below $5" are exact
    negations (P sums to 1); two incompatible outcomes can't both clear.

This bot asks an LLM to read a *cluster* of plausibly-related markets and assert
only the relations that hold **by the meaning of the questions alone** — true
regardless of any real-world fact (so, like monotonicity, betting toward them is
+EV at resolution without any external information). It then projects the current
market prices onto the feasible region those relations define (iterated convex
projection, the partial-order analogue of `ladder_arb`'s isotonic step) and trades
each leg toward its coherent value.

The edge — and the risk — is the relation extraction. A hallucinated relation
would have us trade toward a false constraint, so the design is deliberately
conservative:

  - Prices are **never shown** to the model: relations are price-independent
    truths, and hiding prices stops the model reverse-engineering an "arb" it
    wants to see.
  - Only relations above `min_confidence` are kept; optional self-consistency
    voting (`n_samples`) keeps only relations a supermajority of samples agree on.
  - We **only trade violations** — if the prices already satisfy the relation,
    there is nothing to correct (exactly `ladder_arb`'s coherent-ladder skip).
  - The correction is a **partial** move (`correction_strength`) toward the
    feasible point, not a slam to the boundary: the market price stays informative.
  - Same-`(metric, date)` pairs are skipped by default — that is `ladder_arb`'s
    turf; this bot only claims the cross-ladder relations it can't see.

Local LLM by default (per the repo's local-only rule). `engine: mercury` is
available but, like every hosted-inference use, only under the sanctioned-exception
review (see docs/mercury-ensemble-calibration.md §0).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import Any

from .base import Market, Strategy
from .ladder import measurable_key
from .ladder_arb import _extract_date

logger = logging.getLogger(__name__)

# Words that carry no entity/quantity signal — stripped before tokenising so the
# block key keys on the distinctive noun (the commodity/company/metric), not the
# boilerplate every clone question shares.
_STOPWORDS = frozenset("""
will the for year ending exceed above below over under than less more least most
on of in to be by as at and or a an value calendar annual average avg mean end
period through during reach hit stay remain between from with this that based
its their global total net per cent percent rate ratio amount level number figure
""".split())

_WORD = re.compile(r"[a-z][a-z0-9.\-]*")

# Relation types the model may assert, and what each implies about probabilities.
#   equivalent : P(a) == P(b)            (same conditions, only wording differs)
#   negation   : P(a) == 1 - P(b)        (a YES exactly when b NO)
#   implies    : P(a) <= P(b)            (a is a strictly stronger claim)
#   exclusive  : P(a) + P(b) <= 1        (cannot both resolve YES)
_RELATION_TYPES = frozenset({"equivalent", "negation", "implies", "exclusive"})

_SYSTEM = (
    "You are a careful logician analysing prediction-market questions. You are given "
    "a numbered list of questions. Identify ONLY pairs that have a strict logical "
    "relationship that must hold by the MEANING of the questions alone — true no "
    "matter what happens in the real world.\n\n"
    "Allowed relation types:\n"
    '- "equivalent": A resolves YES under exactly the same conditions as B (same '
    "measured quantity, same threshold value, same comparison direction, same time "
    "period and entity) — only the wording differs.\n"
    '- "negation": A resolves YES exactly when B resolves NO (e.g. "above X" vs "at '
    'or below X" on the same quantity/period).\n'
    '- "implies": whenever A resolves YES, B must also resolve YES because A is a '
    'strictly stronger claim (e.g. "exceeds 5" implies "exceeds 4" for the same '
    "quantity, direction and period).\n"
    '- "exclusive": A and B cannot both resolve YES.\n\n'
    "Be CONSERVATIVE. Report a relation ONLY if you are highly confident it is "
    "logically necessary. Questions that are merely about the same topic, or "
    "correlated, are NOT related — omit them. A difference in the measured quantity, "
    "the threshold number, the direction (above/below), the time period, or the "
    "entity usually breaks the relation. When unsure, leave it out.\n\n"
    'Return ONLY JSON: {"relations": [{"a": <int>, "b": <int>, "type": "<type>", '
    '"confidence": <0..1>, "why": "<short reason>"}]}. Use the question numbers for '
    "a and b. If there are no logical relations, return an empty list."
)


def _significant_tokens(question: str) -> list[str]:
    """Lowercased content tokens: drop stopwords, pure numbers, and units-ish
    fragments, keeping the distinctive nouns (entity / metric)."""
    out = []
    for tok in _WORD.findall(question.lower()):
        tok = tok.strip(".-")
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if not any(c.isalpha() for c in tok):
            continue
        out.append(tok)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def block_markets(
    markets: list[Market], *, max_cluster: int = 12, dev_band: float = 0.03
) -> list[list[Market]]:
    """Cheap blocking: cluster markets that share a resolution date AND their
    rarest content token (the distinctive entity/metric). O(n) given a corpus-wide
    document-frequency table, so it scales to the whole universe.

    Only clusters with >=2 markets and at least one *informative* leg (volume, or
    a price moved off the 0.50 default) are kept — a cluster of all-0.50 markets
    has nothing to anchor a correction, so there is no arb to find.
    """
    # Corpus document frequency: how many questions each token appears in. The
    # rarest token in a question is its most distinctive handle.
    df: Counter[str] = Counter()
    toks: dict[str, list[str]] = {}
    for m in markets:
        t = _significant_tokens(m.get("question", ""))
        toks[m["id"]] = t
        for tok in set(t):
            df[tok] += 1

    blocks: dict[tuple[str, str], list[Market]] = {}
    for m in markets:
        # Only tokens shared with at least one other question can form a
        # cross-market block — a token unique to this question (df==1) would key
        # it into a block of its own and never meet its paraphrase.
        shared = [w for w in set(toks[m["id"]]) if df[w] >= 2]
        if not shared:
            continue
        date, _ = _extract_date(m.get("question", ""))
        # Rarest shared token; deterministic tie-break (lowest df, then longest, a-z).
        rare = min(shared, key=lambda w: (df[w], -len(w), w))
        blocks.setdefault((date, rare), []).append(m)

    clusters: list[list[Market]] = []
    for members in blocks.values():
        if len(members) < 2:
            continue
        informative = [
            m for m in members
            if (m.get("volume") or 0) > 0 or abs((m.get("probability") or 0.5) - 0.5) > dev_band
        ]
        if not informative:
            continue
        if len(members) > max_cluster:
            # Keep the informative legs (they anchor the fit) plus the most
            # off-default of the rest, up to the cap.
            rest = sorted(
                (m for m in members if m not in informative),
                key=lambda m: abs((m.get("probability") or 0.5) - 0.5),
                reverse=True,
            )
            members = (informative + rest)[:max_cluster]
        clusters.append(members)

    # Most-informative clusters first, so a per-run group cap keeps the best.
    clusters.sort(key=lambda c: sum(1 for m in c if (m.get("volume") or 0) > 0), reverse=True)
    return clusters


def project_constraints(
    probs: dict[str, float],
    weights: dict[str, float],
    relations: list[dict[str, Any]],
    *,
    correction_strength: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-7,
    eps: float = 0.02,
) -> dict[str, float]:
    """Project market probabilities onto the feasible region defined by the
    logical relations, by cyclic projection (POCS) onto one constraint at a time.

    Each relation is a projection onto a convex set, so iterating to a fixed point
    yields a coherent assignment; for an equivalence-only cluster it is exactly the
    informative-weighted consensus (the partial-order analogue of `ladder_arb`'s
    isotonic pooling). `correction_strength` < 1 then keeps the result a partial
    move from the original price toward that coherent point.
    """
    p = dict(probs)
    for _ in range(max_iters):
        max_delta = 0.0
        for r in relations:
            a, b, t = r["a"], r["b"], r["type"]
            if a not in p or b not in p:
                continue
            wa, wb = weights.get(a, 1.0), weights.get(b, 1.0)
            if t == "equivalent":
                m = (wa * p[a] + wb * p[b]) / (wa + wb)
                max_delta = max(max_delta, abs(p[a] - m), abs(p[b] - m))
                p[a] = p[b] = m
            elif t == "negation":
                # Pool p[a] with (1 - p[b]); restore the complement onto b.
                m = (wa * p[a] + wb * (1 - p[b])) / (wa + wb)
                max_delta = max(max_delta, abs(p[a] - m), abs(p[b] - (1 - m)))
                p[a], p[b] = m, 1 - m
            elif t == "implies":  # P(a) <= P(b); only act when violated
                if p[a] > p[b]:
                    m = (wa * p[a] + wb * p[b]) / (wa + wb)
                    max_delta = max(max_delta, abs(p[a] - m), abs(p[b] - m))
                    p[a] = p[b] = m
            elif t == "exclusive":  # P(a) + P(b) <= 1; only act when violated
                s = p[a] + p[b]
                if s > 1.0:
                    d = (s - 1.0) / 2.0
                    max_delta = max(max_delta, d)
                    p[a] -= d
                    p[b] -= d
        for k in p:
            p[k] = min(max(p[k], eps), 1 - eps)
        if max_delta < tol:
            break

    out: dict[str, float] = {}
    for k, coherent in p.items():
        orig = probs[k]
        val = orig + correction_strength * (coherent - orig)
        out[k] = min(max(val, eps), 1 - eps)
    return out


class SemanticArbStrategy(Strategy):
    name = "semantic_arb"
    description = (
        "LLM-discovered logical arbitrage across markets the structural bots "
        "can't link. An LLM reads clusters of related questions and asserts only "
        "the relations true by their meaning alone — equivalence, negation, "
        "implication, mutual exclusion — and the bot trades prices toward that "
        "logical coherence. Catches reworded duplicates and cross-phrasing "
        "entailments that ladder_arb's exact-key monotonicity misses; only "
        "violations are traded, and only partway, so a wrong relation can't "
        "max-bet."
    )

    def __init__(
        self,
        engine: str = "local",
        model: str | None = None,
        min_confidence: float = 0.85,
        dev_band: float = 0.03,
        informative_weight: float = 5.0,
        correction_strength: float = 0.7,
        n_samples: int = 1,
        agreement: float = 0.5,
        max_cluster: int = 12,
        max_groups: int = 20,
        skip_same_ladder: bool = True,
        temperature_lo: float = 0.0,
        temperature_hi: float = 0.6,
        timeout: float = 240.0,
        num_ctx: int = 32768,
        api_key: str | None = None,
        **params: object,
    ):
        super().__init__(
            engine=engine, model=model, min_confidence=min_confidence, dev_band=dev_band,
            informative_weight=informative_weight, correction_strength=correction_strength,
            n_samples=n_samples, agreement=agreement, max_cluster=max_cluster,
            max_groups=max_groups, skip_same_ladder=skip_same_ladder,
            temperature_lo=temperature_lo, temperature_hi=temperature_hi,
            timeout=timeout, num_ctx=num_ctx, **params,
        )
        self.engine = engine
        self.min_confidence = min_confidence
        self.dev_band = dev_band
        self.informative_weight = informative_weight
        self.correction_strength = correction_strength
        self.n_samples = max(1, n_samples)
        self.agreement = agreement
        self.max_cluster = max_cluster
        self.max_groups = max_groups
        self.skip_same_ladder = skip_same_ladder
        self.temperature_lo = temperature_lo
        self.temperature_hi = temperature_hi
        # LLM constructed lazily so importing the strategy never requires a model
        # (tests inject relations directly, and `available()` must not connect).
        self._llm: Any = None
        self._llm_kwargs = dict(model=model, timeout=timeout, num_ctx=num_ctx, api_key=api_key)

    # --- LLM plumbing ----------------------------------------------------

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = self._make_llm()
        return self._llm

    def _make_llm(self) -> Any:
        kw = self._llm_kwargs
        if self.engine == "mercury":
            # Hosted inference — sanctioned-exception territory only.
            from ..llm.mercury import MercuryLLM
            return MercuryLLM(model=kw["model"], api_key=kw["api_key"])
        from ..llm.client import LocalLLM
        return LocalLLM(model=kw["model"], timeout=kw["timeout"], num_ctx=kw["num_ctx"])

    def _temperatures(self) -> list[float]:
        if self.n_samples == 1:
            return [self.temperature_lo]
        step = (self.temperature_hi - self.temperature_lo) / (self.n_samples - 1)
        return [self.temperature_lo + i * step for i in range(self.n_samples)]

    def _ask_relations_once(self, questions: list[str], temperature: float) -> list[dict]:
        prompt = "Questions:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        try:
            raw = self.llm.json_completion(system=_SYSTEM, user=prompt, temperature=temperature)
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001 - one cluster's failure must not crash the run
            logger.warning("semantic_arb: relation extraction failed (%s) — skipping cluster",
                           type(e).__name__)
            return []
        rels = data.get("relations") if isinstance(data, dict) else None
        if not isinstance(rels, list):
            return []
        n = len(questions)
        clean = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            t = r.get("type")
            try:
                a, b = int(r["a"]) - 1, int(r["b"]) - 1
                conf = float(r.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if t not in _RELATION_TYPES or not (0 <= a < n) or not (0 <= b < n) or a == b:
                continue
            clean.append({"a": a, "b": b, "type": t, "confidence": conf, "why": str(r.get("why", ""))})
        return clean

    @staticmethod
    def _canon(rel: dict) -> tuple:
        """Canonical key for voting. Symmetric types are order-independent;
        `implies` is directional."""
        if rel["type"] == "implies":
            return (rel["a"], rel["b"], "implies")
        return (min(rel["a"], rel["b"]), max(rel["a"], rel["b"]), rel["type"])

    def _extract_relations(self, questions: list[str]) -> list[dict]:
        """Extract relations over the cluster's questions (0-based indices), with
        optional self-consistency voting across `n_samples`. Kept relations carry
        the mean confidence of the samples that found them."""
        votes: dict[tuple, list[dict]] = {}
        for temp in self._temperatures():
            for r in self._ask_relations_once(questions, temp):
                votes.setdefault(self._canon(r), []).append(r)
        need = math.ceil(self.agreement * self.n_samples)
        out = []
        for key, found in votes.items():
            if len(found) < need:
                continue
            mean_conf = sum(r["confidence"] for r in found) / len(found)
            rep = dict(found[0])
            rep["confidence"] = mean_conf
            out.append(rep)
        return out

    # --- Strategy seam ---------------------------------------------------

    def prefilter(self, markets: list[Market]) -> list[Market]:
        return [
            m for m in super().prefilter(markets)
            if m.get("probability") is not None and m.get("outcomeType", "BINARY") == "BINARY"
        ]

    def group(self, markets: list[Market]) -> list[list[Market]]:
        clusters = block_markets(markets, max_cluster=self.max_cluster, dev_band=self.dev_band)
        return clusters[: self.max_groups]

    def correlation_key(self, market: Market) -> str:
        # Stable per-market handle (no corpus needed): date + the two longest
        # content tokens. Markets that block together share these, so the
        # allocator caps exposure across a logical cluster.
        date, _ = _extract_date(market.get("question", ""))
        toks = sorted(set(_significant_tokens(market.get("question", ""))), key=len, reverse=True)[:2]
        return f"{date}|{'+'.join(sorted(toks))}" if toks else str(market.get("id"))

    def _weight(self, market: Market) -> float:
        prob = market.get("probability", 0.5) or 0.5
        informative = (market.get("volume") or 0) > 0 or abs(prob - 0.5) > self.dev_band
        return self.informative_weight if informative else 1.0

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if len(group) < 2:
            return {}
        questions = [m.get("question", "") for m in group]
        relations = [
            r for r in self._extract_relations(questions)
            if r["confidence"] >= self.min_confidence
        ]
        if self.skip_same_ladder:
            relations = [
                r for r in relations
                if measurable_key(group[r["a"]]) != measurable_key(group[r["b"]])
            ]
        if not relations:
            return {}

        # Re-key relations from cluster indices to market ids for projection.
        ids = [m["id"] for m in group]
        probs = {m["id"]: float(m["probability"]) for m in group}
        weights = {m["id"]: self._weight(m) for m in group}
        id_rels = [
            {"a": ids[r["a"]], "b": ids[r["b"]], "type": r["type"],
             "confidence": r["confidence"], "why": r["why"]}
            for r in relations
        ]
        coherent = project_constraints(
            probs, weights, id_rels,
            correction_strength=self.correction_strength,
        )

        # Only markets that (a) participate in a relation and (b) move by more than
        # dev_band are worth trading; everything else stays at its market price.
        involved = {r["a"] for r in id_rels} | {r["b"] for r in id_rels}
        out: dict[str, float] = {}
        for mid in involved:
            fair = coherent[mid]
            if abs(fair - probs[mid]) <= self.dev_band:
                continue
            out[mid] = fair
            mrels = [r for r in id_rels if r["a"] == mid or r["b"] == mid]
            self._explanations[mid] = {
                "market_prob": probs[mid],
                "fair": fair,
                "n_cluster": len(group),
                "relations": [
                    {
                        "type": r["type"],
                        "confidence": r["confidence"],
                        "why": r["why"],
                        "other_q": next(m["question"] for m in group
                                        if m["id"] == (r["b"] if r["a"] == mid else r["a"])),
                        "this_is": "a" if r["a"] == mid else "b",
                    }
                    for r in mrels
                ],
            }
        return out

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        lines = [
            f"- Semantic-coherence arb across a cluster of {d['n_cluster']} related markets",
            f"- Market price **{d['market_prob']:.3f}** → logically-coherent **{d['fair']:.3f}** "
            f"(partial move; correction_strength={self.correction_strength:g})",
            "- Logical relations enforced:",
        ]
        for r in d["relations"]:
            arrow = {"equivalent": "≡", "negation": "= NOT", "implies": "⇒", "exclusive": "⊕"}[r["type"]]
            side = "this" if r["this_is"] == "a" else "other"
            lines.append(
                f"  - **{r['type']}** ({arrow}, conf {r['confidence']:.2f}) vs "
                f"“{r['other_q'][:80]}” — {r['why'][:120]}"
            )
        return "\n".join(lines)
