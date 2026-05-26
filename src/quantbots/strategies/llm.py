"""Phase 3 reference strategy: local-LLM percentile -> CDF (no per-strike calls).

The trick (ported from TAL's thesis_response): don't ask the model to price each
strike. Ask it *once* per measurable for a percentile distribution of the
underlying quantity, then read every strike off the interpolated CDF. One call
prices a whole ladder with cross-strike consistency for free.

Uses `quantbots.llm.client`, which talks to a LOCAL OpenAI-compatible endpoint
(Ollama / LiteLLM -> local). No hosted inference. Requires the `llm` extra.
"""

from __future__ import annotations

import json

import numpy as np

from ..llm.client import LocalLLM
from .base import Market, Strategy
from .ladder import attach_ladder_fields, measurable_key

_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
_KEYS = ["p10", "p25", "p50", "p75", "p90"]

_SYSTEM = (
    "You are a calibrated forecaster. Given a quantity to predict, return ONLY a "
    "JSON object with numeric percentile estimates p10, p25, p50, p75, p90 and a "
    'short "reasoning" string. No prose outside the JSON.'
)


class LLMStrategy(Strategy):
    name = "llm"

    def __init__(self, model: str | None = None, **params: object):
        super().__init__(model=model, **params)
        self.llm = LocalLLM(model=model)

    def prefilter(self, markets: list[Market]) -> list[Market]:
        markets = super().prefilter(markets)
        return [m for m in (attach_ladder_fields(m) for m in markets)
                if m.get("threshold") is not None]

    def group(self, markets: list[Market]) -> list[list[Market]]:
        groups: dict[str, list[Market]] = {}
        for m in markets:
            groups.setdefault(measurable_key(m), []).append(m)
        return list(groups.values())

    def _ask_percentiles(self, group: list[Market]) -> dict | None:
        subject = measurable_key(group[0])
        prompt = (
            f"Predict the distribution of: {subject}.\n"
            f"Context questions:\n" + "\n".join(f"- {m['question']}" for m in group[:20])
        )
        raw = self.llm.json_completion(system=_SYSTEM, user=prompt)
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
        values = [float(pct[k]) for k in _KEYS]
        out: dict[str, float] = {}
        for m in group:
            if m.get("threshold") is None:
                continue
            # CDF at the strike, interpolated across the percentile points.
            cdf = float(np.interp(m["threshold"], values, _LEVELS))
            p = 1.0 - cdf if m.get("direction", "exceeds") == "exceeds" else cdf
            out[m["id"]] = float(np.clip(p, 0.01, 0.99))
        return out
