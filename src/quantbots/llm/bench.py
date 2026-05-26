"""Benchmark local LLMs for forecasting, using our data feeds as ground truth.

"Which local model should we trade with?" is an empirical question. This asks
each candidate model for a percentile distribution of quantities we *already know
the true value of* (from the observations cache — mortgage rate, cotton, ONI,
gold, ...), then scores:

- validity:  fraction of items returning parseable, monotonic percentiles
- coverage:  fraction where the true value falls inside the model's p10–p90
             (calibration — well-calibrated ~80%)
- p50 error: median |p50 - actual| / scale (point accuracy)
- latency:   avg seconds per forecast

Fully local, reproducible, and grounded in real numbers. Requires the `llm` extra
and a running local model server (Ollama).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..store.db import Store
from .client import LocalLLM

_KEYS = ["p10", "p25", "p50", "p75", "p90"]

_SYSTEM = (
    "You are a calibrated quantitative forecaster. Return ONLY a JSON object with "
    "numeric percentile estimates p10, p25, p50, p75, p90 (the distribution of the "
    "asked quantity) and a short 'reasoning' string. No prose outside the JSON."
)

# entity in the observations cache -> human description (with explicit units) for
# the LLM. The cached value is the ground truth.
BENCH_ITEMS: dict[str, str] = {
    "FRED_MORTGAGE30US": "the US 30-year fixed mortgage rate in percent (Freddie Mac PMMS weekly average)",
    "FRED_HOUST1F": "US single-family housing starts, seasonally-adjusted annual rate, in thousands of units",
    "CME_COTTON": "the ICE Cotton No. 2 front-month futures price, in US cents per pound",
    "ENSO_ONI": "the Oceanic Nino Index (ONI), a 3-month running-mean sea-surface-temperature anomaly, in degrees Celsius",
    "STOCK_WULF": "the TeraWulf (WULF) stock price, in US dollars",
    "GOLD": "the gold price, in US dollars per troy ounce",
    "WTI_OIL": "the WTI crude oil price, in US dollars per barrel",
}


@dataclass
class ModelScore:
    model: str
    n: int = 0
    valid: int = 0
    covered: int = 0
    errors: list[float] = field(default_factory=list)
    total_latency: float = 0.0

    @property
    def validity(self) -> float:
        return self.valid / self.n if self.n else 0.0

    @property
    def coverage(self) -> float:
        return self.covered / self.valid if self.valid else 0.0

    @property
    def median_error(self) -> float:
        if not self.errors:
            return float("inf")
        s = sorted(self.errors)
        return s[len(s) // 2]

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.n if self.n else 0.0


def _parse_percentiles(raw: str) -> list[float] | None:
    try:
        j = json.loads(raw)
        vals = [float(j[k]) for k in _KEYS]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return sorted(vals)  # tolerate slightly out-of-order percentiles


def benchmark(
    models: list[str],
    asof: str,
    store: Store | None = None,
    llm_factory: Callable[[str], object] | None = None,
) -> list[ModelScore]:
    """Score each model against cached ground-truth values. Returns ModelScores
    sorted best-first (highest coverage, then lowest p50 error)."""
    store = store or Store()
    llm_factory = llm_factory or (lambda m: LocalLLM(model=m))

    truth: dict[str, float] = {}
    for entity in BENCH_ITEMS:
        o = store.latest_observation(entity)
        if o and o.get("value") is not None:
            truth[entity] = o["value"]

    scores: list[ModelScore] = []
    for model in models:
        llm = llm_factory(model)
        sc = ModelScore(model=model)
        for entity, desc in BENCH_ITEMS.items():
            if entity not in truth:
                continue
            sc.n += 1
            user = (
                f"Estimate the probability distribution of {desc}, as of {asof}. "
                "Give percentiles p10, p25, p50, p75, p90 as numbers."
            )
            t0 = time.time()
            try:
                raw = llm.json_completion(system=_SYSTEM, user=user)
            except Exception:  # noqa: BLE001 - a failed call just scores 0 here
                sc.total_latency += time.time() - t0
                continue
            sc.total_latency += time.time() - t0
            vals = _parse_percentiles(raw)
            if vals is None:
                continue
            sc.valid += 1
            actual = truth[entity]
            p10, _p25, p50, _p75, p90 = vals
            if p10 <= actual <= p90:
                sc.covered += 1
            scale = abs(actual) if abs(actual) > 1 else 1.0
            sc.errors.append(abs(p50 - actual) / scale)
        scores.append(sc)

    scores.sort(key=lambda s: (-s.coverage, s.median_error))
    return scores
