"""Equity ↔ metal correlation matrix: which equities are driven by which metal.

Daily-log-return Pearson correlation of each candidate commodity-equity against each
metal futures series (over a recent aligned window). Reuses the curated equity list from
`screen.py` and the metal price feeds from `research.data_fetch.DEFAULT_UNIVERSE`. The
output is one column per metal; `per_metal` slices it into a ranked table per metal.
"""

from __future__ import annotations

import numpy as np

from ...research.data_fetch import DEFAULT_UNIVERSE, fetch_yf_history
from .screen import CANDIDATE_EQUITIES

# The exchange-traded metals we have a clean futures feed for.
METALS = ["GOLD", "SILVER", "PLATINUM", "PALLADIUM", "COPPER"]


def _logret(ticker: str, period: str):
    df = fetch_yf_history(ticker, period=period)
    c = df["Close"].astype(float)
    return np.log(c / c.shift(1)).dropna()


def build_matrix(equities=None, metals=METALS, *, period: str = "3y", lookback_days: int = 504):
    """Return a DataFrame: rows = equities, cols = metals, cells = return correlation.

    Correlations are computed PAIRWISE on each (equity, metal) overlap — a single short
    or gappy series can't collapse the whole matrix (a joint dropna would)."""
    import pandas as pd

    equities = equities or list(CANDIDATE_EQUITIES)
    metal_rets = {}
    for m in metals:
        t = DEFAULT_UNIVERSE.get(m)
        if not t:
            continue
        try:
            metal_rets[m] = _logret(t, period)
        except Exception:  # noqa: BLE001
            pass
    rows: dict[str, dict] = {}
    for e in equities:
        try:
            er = _logret(e, period)
        except Exception:  # noqa: BLE001
            continue
        rows[e] = {}
        for m, mr in metal_rets.items():
            j = pd.concat([er, mr], axis=1).dropna().tail(lookback_days)
            rows[e][m] = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 30 else float("nan")
    return pd.DataFrame.from_dict(rows, orient="index")[list(metal_rets)]


def per_metal(matrix, *, min_corr: float = 0.2) -> dict:
    """{metal: Series(equity -> corr)} sorted desc, filtered to |corr| >= min_corr."""
    out = {}
    for m in matrix.columns:
        s = matrix[m].sort_values(ascending=False)
        out[m] = s[s.abs() >= min_corr]
    return out


def dominant_metal(matrix) -> dict:
    """{equity: (metal, corr)} — the metal each equity is most correlated with."""
    out = {}
    for e in matrix.index:
        row = matrix.loc[e]
        m = row.abs().idxmax()
        out[e] = (m, float(row[m]))
    return out
