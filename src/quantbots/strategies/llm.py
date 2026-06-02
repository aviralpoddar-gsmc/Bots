"""Phase 3 reference strategy: local-LLM percentile -> CDF (no per-strike calls).

The trick (ported from TAL's thesis_response): don't ask the model to price each
strike. Ask it *once* per measurable for a percentile distribution of the
underlying quantity, then read every strike off the CDF. One call prices a whole
ladder with cross-strike consistency for free.

We fit a normal to the returned percentiles and use its analytic CDF (smooth
tails, no clamping), with `spread_mult` inflating sigma to correct the
overconfidence local models show (benchmark: ~57-71% p10-p90 coverage vs ideal
~80%). Without this, strikes outside p10-p90 collapse to near-binary 0.10/0.90.

Uses `quantbots.llm.client`, which talks to a LOCAL OpenAI-compatible endpoint
(Ollama / LiteLLM -> local). No hosted inference. Requires the `llm` extra.
"""

from __future__ import annotations

import json
import logging
import re

from ._model import norm_cdf
from ..llm.client import LocalLLM
from .base import Market, Strategy
from .ladder import attach_ladder_fields, measurable_key

logger = logging.getLogger(__name__)

_KEYS = ["p10", "p25", "p50", "p75", "p90"]
# z-scores of the 10/90 and 25/75 percentile pairs of a standard normal.
_Z_10_90 = 2.5631  # p90 - p10 span in sigmas
_Z_25_75 = 1.3490  # p75 - p25 span in sigmas (IQR)

_SYSTEM = (
    "You are a calibrated forecaster. Given a quantity to predict, return ONLY a "
    "JSON object with numeric percentile estimates p10, p25, p50, p75, p90 and a "
    'short "reasoning" string. No prose outside the JSON.'
)


class LLMStrategy(Strategy):
    name = "llm"
    description = (
        "Local-LLM forecaster for the long tail of markets the deterministic bots "
        "can't link. Queries a locally-running model (Ollama / llama.cpp) for a "
        "calibrated probability per market, with confidence caps and CDF-based "
        "spread widening to control hallucination risk. Hosted inference is "
        "blocked until backtested profitability is proven."
    )

    def __init__(
        self,
        model: str | None = None,
        spread_mult: float = 1.5,
        conf_cap: float = 0.80,
        max_groups: int = 20,
        include_terms: list[str] | None = None,
        exclude_terms: list[str] | None = None,
        no_think: bool = False,
        **params: object,
    ):
        super().__init__(model=model, spread_mult=spread_mult, conf_cap=conf_cap,
                         max_groups=max_groups, include_terms=include_terms,
                         exclude_terms=exclude_terms, no_think=no_think, **params)
        self.llm = LocalLLM(model=model)
        # qwen3 "thinking" mode is slow (long CoT before the answer) and the main
        # cause of timeouts. For COSMETIC coverage, calibration is irrelevant, so
        # appending "/no_think" skips it — much faster. Leave off for alpha bots.
        self.no_think = no_think
        # Optional scoping (for coverage bots): only price markets whose question
        # matches include_terms and matches none of exclude_terms. E.g. cover the
        # cocoa/coffee quantity "sea" while excluding price markets other bots own.
        self.include_re = re.compile("|".join(include_terms), re.I) if include_terms else None
        self.exclude_re = re.compile("|".join(exclude_terms), re.I) if exclude_terms else None
        # Local models are overconfident (benchmark: ~57-71% coverage vs ideal ~80%),
        # so their p10-p90 bands are too narrow. Widen the percentile spread by this
        # factor before reading probabilities off the CDF.
        self.spread_mult = spread_mult
        # Confidence cap: never let the bot express more conviction than this. On
        # obscure markets outside the model's knowledge the CDF can hit 0.99/0.01
        # (hallucinated certainty); clamping the final estimate to [1-cap, cap]
        # bounds bet size so a hallucinated 99% can't become a max-size bet.
        self.conf_cap = conf_cap
        # Bound LLM calls per run (each group = one call); the run's budget cap then
        # bounds spend among whatever this produces.
        self.max_groups = max_groups

    def prefilter(self, markets: list[Market]) -> list[Market]:
        markets = super().prefilter(markets)
        out: list[Market] = []
        for m in (attach_ladder_fields(m) for m in markets):
            if m.get("threshold") is None:
                continue
            q = m.get("question", "")
            if self.include_re and not self.include_re.search(q):
                continue
            if self.exclude_re and self.exclude_re.search(q):
                continue
            out.append(m)
        return out

    def group(self, markets: list[Market]) -> list[list[Market]]:
        groups: dict[str, list[Market]] = {}
        for m in markets:
            groups.setdefault(measurable_key(m), []).append(m)
        # Cap the number of LLM calls per run (one call per group).
        return list(groups.values())[: self.max_groups]

    def _ask_percentiles(self, group: list[Market]) -> dict | None:
        subject = measurable_key(group[0])
        prompt = (
            f"Predict the distribution of: {subject}.\n"
            f"Context questions:\n" + "\n".join(f"- {m['question']}" for m in group[:20])
        )
        # Anchor the model to the ladder's SCALE/UNITS so it doesn't answer on the
        # wrong order of magnitude (e.g. returning 0.2 when strikes are 250-650k tonnes).
        strikes = [m["threshold"] for m in group if m.get("threshold") is not None]
        if strikes:
            prompt += (
                f"\nThese questions ask whether the value exceeds thresholds ranging "
                f"from {min(strikes):g} to {max(strikes):g}. Give p10..p90 as numbers "
                f"on that SAME scale and units."
            )
        if self.no_think:
            prompt += "\n/no_think"  # qwen3: skip chain-of-thought for speed
        try:
            raw = self.llm.json_completion(system=_SYSTEM, user=prompt)
        except Exception as e:  # noqa: BLE001 - LLM timeout/error on one ladder must
            # not crash the whole run; abstain on this ladder and move on.
            logger.warning("llm: ladder %r failed (%s) — abstaining", subject[:60], type(e).__name__)
            return None
        try:
            pct = json.loads(raw)
            if all(k in pct for k in _KEYS):
                return pct
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def estimate(self, group: list[Market]) -> dict[str, float]:
        pct = self._ask_percentiles(group)
        if pct is None:
            return {}
        p10, p25, p50, p75, p90 = sorted(float(pct[k]) for k in _KEYS)
        # Fit a normal: mu = median; sigma from both the 10-90 span and the IQR
        # (averaged for robustness), then inflated by spread_mult to counter the
        # model's overconfidence.
        mu = p50
        sigma_raw = 0.5 * ((p90 - p10) / _Z_10_90 + (p75 - p25) / _Z_25_75)
        sigma = max(sigma_raw, 1e-9) * self.spread_mult

        # Scale-sanity backstop: if the model answered on a wildly different order of
        # magnitude than the ladder's strikes (a unit/scale mistake), the CDF would
        # pin every strike to the clamp floor — junk. Abstain instead, leaving the
        # market an honest 0.50 rather than a fake flat ladder.
        strikes = [m["threshold"] for m in group if m.get("threshold") is not None]
        if strikes:
            smid = sorted(strikes)[len(strikes) // 2]
            center = abs(mu) if abs(mu) > 1e-9 else max(abs(p90 - p10), 1e-9)
            ratio = smid / center if center > 0 else float("inf")
            if ratio > 20 or ratio < 0.05:
                logger.warning("llm: scale mismatch on %r (strikes~%.3g vs model~%.3g) — abstaining",
                               measurable_key(group[0])[:60], smid, center)
                return {}

        out: dict[str, float] = {}
        for m in group:
            if m.get("threshold") is None:
                continue
            cdf = norm_cdf((m["threshold"] - mu) / sigma)  # P(quantity <= strike)
            p = 1.0 - cdf if m.get("direction", "exceeds") == "exceeds" else cdf
            # Clamp to the confidence cap so hallucinated certainty can't max-bet.
            out[m["id"]] = min(max(p, 1.0 - self.conf_cap), self.conf_cap)
        return out
