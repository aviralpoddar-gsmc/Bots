"""Fuse the directional factors into one equity drift (mu_view) for the forecast.

Combines whatever factors are available — momentum (always), the FRED macro tilt
(real-rate for precious, dollar otherwise), and carry / CFTC-positioning when their
point-in-time CSVs are present — into a single capped annualized drift. Weights are priors
from the research's t-stats (carry strongest), renormalized over the factors actually
available, so the model degrades gracefully: momentum-only today, momentum+carry+positioning
once the CSVs land. Signals are read AS OF a date (no lookahead) for backtests.
"""

from __future__ import annotations

import numpy as np

from . import factors as F
from ..forecast.direction import momentum_drift

# Research-derived priors (∝ reported t-stats); renormalized over available factors.
# `tal` is the commodity price-consensus view (forecast/signal.tal_drift): a peer
# directional factor to momentum, but UNVALIDATED — it only contributes where tal has a
# usable price ladder, and its weight is scaled by the per-fold fit confidence, so a thin
# or noisy ladder barely moves the blend. The gate decides whether it earns its place.
WEIGHTS = {"momentum": 0.30, "tal": 0.30, "carry": 0.30, "positioning": 0.25, "macro": 0.15}

# commodity entity -> futures ticker used in the carry/positioning CSVs
_CSV_TICKER = {"COPPER": "HG", "GOLD": "GC", "SILVER": "SI", "WTI_OIL": "CL",
               "BRENT_OIL": "CO", "NATGAS": "NG"}

_CACHE: dict = {}


def _macro_series(commodity: str):
    key = "real_rate" if commodity in ("GOLD", "SILVER", "PLATINUM", "PALLADIUM") else "dollar"
    if key not in _CACHE:
        _CACHE[key] = F.real_rate_signal() if key == "real_rate" else F.dollar_signal()
    return _CACHE[key]


def _asof(series, as_of) -> float:
    import pandas as pd
    if series is None or len(series) == 0:
        return 0.0
    if as_of is None:
        return float(series.iloc[-1])
    s = series[series.index <= pd.Timestamp(as_of)]
    return float(s.iloc[-1]) if len(s) else 0.0


def fused_drift(*, equity: str, commodity: str, beta_c: float, as_of=None,
                drift_cap: float = 0.35, spot: float | None = None,
                horizon_years: float = 0.25, momentum_lookbacks: tuple[int, ...] | None = None,
                momentum_min_strength: float = 0.0, reader=None) -> tuple[float, dict]:
    """(mu_view, components). Each factor contributes a sign-oriented, capped drift; the
    weighted blend (over available factors) is the equity's directional view.

    `spot` + `horizon_years` enable the tal price-consensus factor (omit spot to skip it,
    e.g. when tal is unavailable). The momentum sub-component honours the same
    `momentum_lookbacks` / `momentum_min_strength` config as standalone momentum so a
    fused-vs-momentum gate comparison is apples-to-apples."""
    comps: dict[str, float] = {}
    wts: dict[str, float] = {}
    bsign = np.sign(beta_c) if beta_c else 1.0

    # momentum (annualized drift already in equity space)
    mu_mom, _ = momentum_drift(commodity=commodity, beta_c=beta_c, as_of=as_of,
                               drift_cap=drift_cap, lookbacks=momentum_lookbacks,
                               min_strength=momentum_min_strength)
    comps["momentum"] = mu_mom
    wts["momentum"] = WEIGHTS["momentum"]

    # tal commodity price-consensus view (already equity-space via beta); weight ∝ confidence
    if spot is not None and spot > 0:
        from ..forecast.signal import tal_drift
        mu_tal, conf = tal_drift(commodity=commodity, beta_c=beta_c, spot=spot, as_of=as_of,
                                 horizon_years=horizon_years, drift_cap=drift_cap, reader=reader)
        if conf > 0.0:
            comps["tal"] = mu_tal
            wts["tal"] = WEIGHTS["tal"] * conf

    # macro regime tilt: z in ~[-2,2] -> capped drift, oriented by beta sign
    mz = _asof(_macro_series(commodity), as_of)
    comps["macro"] = float(np.clip(mz / 2.0, -1, 1) * drift_cap * bsign)
    wts["macro"] = WEIGHTS["macro"]

    # carry / positioning from point-in-time CSVs (commodity-level z) -> drift via beta sign
    tk = _CSV_TICKER.get(commodity)
    if tk:
        carry = F.carry_from_csv()
        if carry is not None and tk in getattr(carry, "columns", []):
            cz = _asof(carry[tk], as_of)
            comps["carry"] = float(np.clip(cz / 2.0, -1, 1) * drift_cap * bsign)
            wts["carry"] = WEIGHTS["carry"]
        pos = F.positioning_from_csv()
        if pos is not None and tk in getattr(pos, "columns", []):
            pz = _asof(pos[tk], as_of)
            comps["positioning"] = float(np.clip(pz / 2.0, -1, 1) * drift_cap * bsign)
            wts["positioning"] = WEIGHTS["positioning"]

    # Drop any non-finite component (e.g. a stale carry CSV with trailing NaNs) BEFORE
    # blending — otherwise one NaN makes the weighted sum NaN and the cap-clamp silently
    # returns ±drift_cap, i.e. garbage masquerading as a max-conviction view.
    bad = [k for k in list(wts) if not np.isfinite(comps.get(k, 0.0))]
    for k in bad:
        wts.pop(k, None)
        comps[k] = float("nan")  # keep visible in the returned components for diagnostics
    # weighted blend over available (finite) factors, renormalized; cap
    tot = sum(wts.values()) or 1.0
    mu = sum(comps[k] * wts[k] for k in wts) / tot
    mu = max(-drift_cap, min(drift_cap, mu)) if np.isfinite(mu) else 0.0
    return mu, comps
