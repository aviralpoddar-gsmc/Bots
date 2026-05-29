"""Score our local LLM on ForecastBench's resolved question sets.

ForecastBench publishes a nightly question set and (later) a resolution set with
ground-truth outcomes. We re-use their *exact zero-shot prompt* (Halawi et al.
2024, "Approaching Human-Level Forecasting with Language Models") to query the
local model so the Brier score is directly comparable to their leaderboard.

We focus on **market questions with a freeze value** — the realistic info set
when betting on the Manifold clone: question + background + criteria + the
current market price. Dataset questions have 8 horizons with scalar resolutions
and need a separate harness; skip them here.

Why this exists: before deploying a binary-LLM bot on the clone we want an
honest Brier number on contamination-free questions. ForecastBench's top LLMs
score ~0.136 overall; supers ~0.09; the public ~0.13. If qwen3:32b lands at
0.18+ we shouldn't build the bot.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .client import LocalLLM

_DATA_REPO = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"
# Fallback for environments where raw.githubusercontent.com DNS is hijacked
# (Pi-hole / corp filters) — api.github.com is usually fine.
_API_REPO = (
    "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents"
)
_CACHE_DIR = os.path.expanduser("~/.cache/quantbots/forecastbench")

# Exact prompt from src/helpers/llm_prompts.py in forecastingresearch/forecastbench
# (ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT). Keep this verbatim so our scores
# are apples-to-apples with the public leaderboard.
PROMPT_TEMPLATE = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Market value on {freeze_datetime}:
{freeze_datetime_value}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}
"""

# *0.42*, * 0.42 *, *.42*, *42%* — all things we've seen models emit.
_ASTERISK_NUM = re.compile(r"\*\s*([01]?\.?\d+%?)\s*\*")
_FALLBACK_NUM = re.compile(r"\b([01]?\.\d+|0|1)\b")


def _http_get_json(url: str, timeout: float = 30.0, accept: str | None = None) -> dict:
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _cached_json(raw_url: str, api_url: str, cache_name: str) -> dict:
    """Fetch once, then read from local cache. Falls back from raw → API endpoint
    so DNS hijacking of raw.githubusercontent.com doesn't break things."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, cache_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    try:
        data = _http_get_json(raw_url)
    except (urllib.error.URLError, TimeoutError, OSError):
        data = _http_get_json(api_url, accept="application/vnd.github.v3.raw")
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def fetch_question_set(date: str) -> dict:
    """Pull the LLM question set for a given forecast_due_date (YYYY-MM-DD)."""
    fname = f"{date}-llm.json"
    return _cached_json(
        f"{_DATA_REPO}/datasets/question_sets/{fname}",
        f"{_API_REPO}/datasets/question_sets/{fname}",
        fname,
    )


def fetch_resolution_set(date: str) -> dict:
    fname = f"{date}_resolution_set.json"
    return _cached_json(
        f"{_DATA_REPO}/datasets/resolution_sets/{fname}",
        f"{_API_REPO}/datasets/resolution_sets/{fname}",
        fname,
    )


def parse_forecast(raw: str) -> float | None:
    """Extract a probability from a model response. Returns None if unparseable.

    Tries the canonical `*0.42*` form first, then any decimal in [0, 1] as a
    fallback (some open models ignore the asterisk instruction)."""
    for pat in (_ASTERISK_NUM, _FALLBACK_NUM):
        for m in pat.finditer(raw):
            s = m.group(1).rstrip("%")
            try:
                v = float(s)
            except ValueError:
                continue
            if s.endswith("%") or v > 1.0:  # tolerate "42%" or "42"
                v /= 100.0
            if 0.0 <= v <= 1.0:
                return v
    return None


def render_prompt(q: dict, today: str) -> str:
    """Fill the template from a market-question dict."""
    # Market questions store the close time here; resolution_dates is "N/A".
    resolution_date = q.get("market_info_close_datetime") or q.get("resolution_dates") or "unknown"
    return PROMPT_TEMPLATE.format(
        question=q["question"],
        background=q.get("background") or "(no background provided)",
        resolution_criteria=q.get("resolution_criteria", ""),
        freeze_datetime=q.get("freeze_datetime", ""),
        freeze_datetime_value=q.get("freeze_datetime_value", ""),
        today_date=today,
        resolution_date=resolution_date,
    )


@dataclass
class EvalResult:
    date: str
    model: str
    n: int = 0           # resolved binary market questions attempted
    parsed: int = 0      # forecasts successfully extracted from model output
    brier_sum: float = 0.0
    brier_freeze_sum: float = 0.0    # baseline: just bet the market freeze price
    brier_naive_sum: float = 0.0     # baseline: always 0.5
    latency_sum: float = 0.0
    items: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    @property
    def brier(self) -> float:
        return self.brier_sum / self.parsed if self.parsed else float("nan")

    @property
    def brier_freeze(self) -> float:
        return self.brier_freeze_sum / self.parsed if self.parsed else float("nan")

    @property
    def brier_naive(self) -> float:
        return self.brier_naive_sum / self.parsed if self.parsed else float("nan")

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.n if self.n else 0.0

    @property
    def avg_latency(self) -> float:
        return self.latency_sum / self.n if self.n else 0.0


def evaluate(
    date: str,
    n: int | None = None,
    llm_factory: Callable[[], object] | None = None,
    verbose: bool = False,
) -> EvalResult:
    """Score the local LLM on ForecastBench's resolved binary market questions.

    Args:
        date: forecast_due_date used for the paired question / resolution sets.
        n: cap on questions to evaluate (None = all resolved binary market questions).
        llm_factory: returns an object with `text_completion(user) -> str`.
            Defaults to LocalLLM() (qwen3:32b on local Ollama).
        verbose: per-question logging.
    """
    qset = fetch_question_set(date)
    rset = fetch_resolution_set(date)

    # Index resolutions by id. Market questions have a single (id, resolved_to);
    # dataset questions have one row per horizon — skip those by source filter.
    res_by_id: dict[str, dict] = {}
    for r in rset["resolutions"]:
        if not r.get("resolved"):
            continue
        if r["source"] not in {"infer", "manifold", "metaculus", "polymarket"}:
            continue
        rv = r.get("resolved_to")
        if rv is None or rv not in (0.0, 1.0):
            continue  # only score strict binary outcomes
        res_by_id[r["id"]] = r

    llm = (llm_factory or LocalLLM)()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out = EvalResult(date=date, model=getattr(llm, "model", "unknown"))

    for q in qset["questions"]:
        if q["id"] not in res_by_id:
            continue
        if n is not None and out.n >= n:
            break
        out.n += 1
        truth = res_by_id[q["id"]]["resolved_to"]
        prompt = render_prompt(q, today)
        t0 = time.time()
        try:
            raw = llm.text_completion(prompt)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 - keep iterating across failures
            out.latency_sum += time.time() - t0
            if verbose:
                print(f"  [err] {q['id']}: {e}")
            continue
        out.latency_sum += time.time() - t0

        forecast = parse_forecast(raw)
        try:
            freeze = float(q.get("freeze_datetime_value") or "nan")
        except ValueError:
            freeze = float("nan")
        item = {
            "id": q["id"],
            "question": q["question"][:80],
            "freeze": freeze,
            "forecast": forecast,
            "truth": truth,
        }
        if forecast is None:
            if verbose:
                print(f"  [unparsed] {q['question'][:60]}... raw={raw[:80]!r}")
            out.items.append(item)
            continue
        out.parsed += 1
        out.brier_sum += (forecast - truth) ** 2
        if 0.0 <= freeze <= 1.0:
            out.brier_freeze_sum += (freeze - truth) ** 2
        else:
            out.brier_freeze_sum += (0.5 - truth) ** 2
        out.brier_naive_sum += (0.5 - truth) ** 2
        out.items.append(item)
        if verbose:
            print(
                f"  {q['question'][:55]:55s}  truth={truth:.0f}  "
                f"freeze={freeze:.3f}  pred={forecast:.3f}"
            )

    return out
