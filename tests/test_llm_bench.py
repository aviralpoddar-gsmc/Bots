import json

from quantbots.llm.bench import BENCH_ITEMS, benchmark


class FakeStore:
    """Minimal store: known ground-truth values for two bench entities."""
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        if entity in self.values:
            return {"entity": entity, "value": self.values[entity], "source": "test"}
        return None


class FakeLLM:
    """Centers its percentiles per entity by matching a keyword in the prompt."""
    def __init__(self, centers, valid=True):
        self.centers = centers  # {keyword: center}
        self.valid = valid

    def json_completion(self, system, user, temperature=0.0):
        if not self.valid:
            return "not json"
        c = next((v for k, v in self.centers.items() if k in user.lower()), 1.0)
        return json.dumps({"p10": c * 0.8, "p25": c * 0.9, "p50": c,
                           "p75": c * 1.1, "p90": c * 1.2, "reasoning": "x"})


def test_benchmark_scores_accurate_model_higher():
    store = FakeStore({"FRED_MORTGAGE30US": 6.5, "GOLD": 4500.0})

    # accurate model centers near each truth; off model is far below on both.
    def factory(model):
        if model == "accurate":
            return FakeLLM({"mortgage": 6.5, "gold": 4500.0})
        return FakeLLM({"mortgage": 3.0, "gold": 3000.0})

    scores = benchmark(["accurate", "off"], "May 2026", store=store, llm_factory=factory)
    # Sorted best-first: the accurate model should rank first with full coverage.
    assert scores[0].model == "accurate"
    assert scores[0].coverage == 1.0
    assert scores[0].median_error < scores[1].median_error


def test_benchmark_counts_invalid_json():
    store = FakeStore({"GOLD": 4500.0})
    scores = benchmark(["bad"], "May 2026", store=store, llm_factory=lambda m: FakeLLM({}, valid=False))
    assert scores[0].valid == 0
    assert scores[0].n == 1


def test_benchmark_only_scores_entities_with_truth():
    # Only one of the bench entities has a cached value.
    assert "GOLD" in BENCH_ITEMS
    store = FakeStore({"GOLD": 4500.0})
    scores = benchmark(["m"], "now", store=store, llm_factory=lambda m: FakeLLM({"gold": 4500.0}))
    assert scores[0].n == 1 and scores[0].valid == 1 and scores[0].coverage == 1.0
