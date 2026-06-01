"""Pair-trading analytics over a wide price panel.

Three things this layer does, all from pure numpy / pandas:

1. **Correlation matrix** on log returns (Pearson + Spearman). What moves
   together in returns space.
2. **Cointegration screen** (Engle-Granger lite): for each candidate pair,
   compute the OLS hedge ratio β, build the spread `s = y − β·x`, and measure
   its mean-reversion via the Ornstein-Uhlenbeck half-life. Shorter half-life
   = more reliably mean-reverting = better pair candidate.
3. **Per-pair diagnostics**: ratio time series, rolling mean, ±2σ bands,
   z-score, max-deviation history. Used by the chart module.

We deliberately avoid statsmodels — the OU half-life via OLS on Δs vs s_{t-1}
is well-understood, statsmodels-free, and good enough for a first pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Asset classes used to color the correlation matrix and group pairs by sector.
ASSET_CLASS: dict[str, str] = {
    "GOLD": "Precious", "SILVER": "Precious", "PLATINUM": "Precious", "PALLADIUM": "Precious",
    "COPPER": "Base", "ALB": "Base/EV", "MP": "Critical",
    "WTI_OIL": "Energy", "BRENT_OIL": "Energy", "NATGAS": "Energy",
    "GASOLINE": "Energy", "HEATING_OIL": "Energy",
    "CORN": "Ag", "WHEAT": "Ag", "SOYBEANS": "Ag",
    "COTTON": "Ag", "SUGAR": "Ag", "COCOA": "Ag", "COFFEE": "Ag",
    "KTOS": "Defense", "AVAV": "Defense",
    "SPX": "Equity-Macro", "TLT": "Rates", "BTC": "Crypto",
    "FRED_DGS10": "Rates", "FRED_MORTGAGE30US": "Rates",
    "FRED_DTWEXBGS": "FX", "FRED_VIXCLS": "Vol",
    "ENSO_ONI": "Climate",
}


# -----------------------------------------------------------------------------
# Data prep
# -----------------------------------------------------------------------------

def align_panel(panel: pd.DataFrame, *, lookback_days: int = 750, min_coverage: float = 0.8) -> pd.DataFrame:
    """Restrict to the recent window where coverage is reasonable.

    - forward-fill first to bridge weekend / holiday gaps (BTC trades weekends
      so its dates appear in the union index where commodities don't quote)
    - take the last `lookback_days` rows of that filled panel
    - drop columns whose pre-ffill coverage in that window is below `min_coverage`
    - re-restrict to weekday business days where most-traded series exist
    """
    if panel.empty:
        return panel
    # Drop weekend rows where the major commodity markets are closed — we don't
    # want crypto-only days dominating the index.
    weekdays = panel[panel.index.dayofweek < 5]
    filled = weekdays.ffill().tail(lookback_days)
    # Coverage check uses the FILLED window: a series qualifies if its ffilled
    # value exists on most business days. This catches series that just don't
    # exist back that far (e.g. a newer ETF).
    keep = [c for c in filled.columns if filled[c].notna().mean() >= min_coverage]
    out = filled[keep].dropna()
    return out


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log(panel / panel.shift(1)).dropna(how="all")


# -----------------------------------------------------------------------------
# Correlation
# -----------------------------------------------------------------------------

def correlation_matrix(panel: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    """Correlation of log returns. `method`: pearson | spearman | kendall."""
    rets = log_returns(panel).dropna()
    return rets.corr(method=method)


def top_correlated_pairs(corr: pd.DataFrame, *, n: int = 20, exclude_self: bool = True,
                         min_abs: float = 0.0) -> list[tuple[str, str, float]]:
    """Flatten the upper triangle of `corr` and return the strongest pairs.

    Returns list of (a, b, corr) sorted by |corr| desc.
    """
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if exclude_self and a == b:
                continue
            if pd.isna(v):
                continue
            if abs(v) < min_abs:
                continue
            pairs.append((a, b, float(v)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:n]


# -----------------------------------------------------------------------------
# Cointegration (Engle-Granger lite via OU half-life)
# -----------------------------------------------------------------------------

@dataclass
class PairStats:
    a: str
    b: str
    n_obs: int
    corr_returns: float
    beta: float            # OLS hedge ratio:  a_t ≈ alpha + beta * b_t
    alpha: float
    half_life: float       # OU half-life of the spread, in days (NaN if non-mean-reverting)
    spread_mean: float
    spread_std: float
    current_z: float       # latest spread z-score
    abs_z_max_252: float   # max |z| over the last year — episodes of dislocation


def _ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS of y on x with intercept. Returns (alpha, beta)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _ou_half_life(spread: np.ndarray) -> float:
    """OU half-life via OLS: Δs_t = α + φ * s_{t-1} + ε. Half-life = -ln(2)/ln(1+φ).

    Negative φ means mean-reverting; positive φ means runaway. Returns NaN if
    φ ≥ 0 (no mean reversion) or if there isn't enough data.
    """
    if len(spread) < 30:
        return float("nan")
    s = spread - np.nanmean(spread)
    s_lag = s[:-1]
    ds = np.diff(s)
    if np.std(s_lag) < 1e-12:
        return float("nan")
    _, phi = _ols_hedge(ds, s_lag)
    if phi >= 0:
        return float("nan")
    try:
        hl = -math.log(2) / math.log(1 + phi)
    except (ValueError, ZeroDivisionError):
        return float("nan")
    return float(hl)


def pair_stats(panel: pd.DataFrame, a: str, b: str) -> PairStats | None:
    """Compute hedge ratio, spread, z-score, half-life for one pair on log prices."""
    if a not in panel.columns or b not in panel.columns:
        return None
    sub = panel[[a, b]].dropna()
    if len(sub) < 60:
        return None
    log_a = np.log(sub[a].values)
    log_b = np.log(sub[b].values)
    alpha, beta = _ols_hedge(log_a, log_b)
    spread = log_a - (alpha + beta * log_b)
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread))
    half_life = _ou_half_life(spread)
    z = (spread - spread_mean) / spread_std if spread_std > 0 else np.zeros_like(spread)
    abs_z_max_252 = float(np.max(np.abs(z[-252:]))) if len(z) >= 1 else float("nan")
    rets = log_returns(sub).dropna()
    corr_returns = float(rets[a].corr(rets[b])) if len(rets) > 2 else float("nan")
    return PairStats(
        a=a, b=b, n_obs=len(sub),
        corr_returns=corr_returns,
        beta=beta, alpha=alpha, half_life=half_life,
        spread_mean=spread_mean, spread_std=spread_std,
        current_z=float(z[-1]) if len(z) else float("nan"),
        abs_z_max_252=abs_z_max_252,
    )


def cointegration_shortlist(panel: pd.DataFrame, *, max_half_life_days: float = 90.0,
                            min_corr_returns: float = 0.4,
                            max_pairs: int = 40) -> list[PairStats]:
    """Score every pair and keep ones that (a) have a sensibly positive return
    correlation, (b) have a finite OU half-life within `max_half_life_days`.

    Ranked by half-life ascending (fastest mean-reverters first)."""
    cols = list(panel.columns)
    out: list[PairStats] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            ps = pair_stats(panel, a, b)
            if ps is None:
                continue
            if not (math.isfinite(ps.half_life) and 0 < ps.half_life <= max_half_life_days):
                continue
            if not math.isfinite(ps.corr_returns) or abs(ps.corr_returns) < min_corr_returns:
                continue
            out.append(ps)
    out.sort(key=lambda p: p.half_life)
    return out[:max_pairs]


# -----------------------------------------------------------------------------
# Per-pair diagnostic series for plotting
# -----------------------------------------------------------------------------

def pair_series(panel: pd.DataFrame, a: str, b: str, *, lookback_days: int = 750) -> pd.DataFrame:
    """For plotting: aligned (a, b, ratio, log_spread, z, normalized) series."""
    sub = panel[[a, b]].dropna().tail(lookback_days)
    if len(sub) < 10:
        return pd.DataFrame()
    log_a = np.log(sub[a].values)
    log_b = np.log(sub[b].values)
    alpha, beta = _ols_hedge(log_a, log_b)
    log_spread = log_a - (alpha + beta * log_b)
    mean = log_spread.mean()
    std = log_spread.std()
    z = (log_spread - mean) / std if std > 0 else np.zeros_like(log_spread)
    norm_a = sub[a] / sub[a].iloc[0] * 100
    norm_b = sub[b] / sub[b].iloc[0] * 100
    ratio = sub[a] / sub[b]
    return pd.DataFrame({
        "a": sub[a].values, "b": sub[b].values,
        "norm_a": norm_a.values, "norm_b": norm_b.values,
        "ratio": ratio.values,
        "log_spread": log_spread, "z": z,
    }, index=sub.index)
