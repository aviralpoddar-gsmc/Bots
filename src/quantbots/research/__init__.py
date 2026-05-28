"""Research/analysis tooling for the framework.

Pair-trading research lives here. Pure-analysis modules (no trading, no DB
writes besides a local cache). The driver script `scripts/research_pairs.py`
turns these into a report + charts in `data/research/pairs_YYYYMMDD/`.

Optional `research` extra (numpy/scipy/pandas/matplotlib) — installed via
`uv sync --extra research`.
"""
