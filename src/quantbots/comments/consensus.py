"""Crowd consensus per market + Platt extremization (AIA Forecaster, §calibration).

The paper's biggest single calibration lever: forecaster ensembles hedge toward
0.5, so push the mean outward with p̂ = σ(α·logit(p̄)), α = √3. Our "forecasters"
are the clone's bot bettors: each user's latest bet on a market implies their
probability (probAfter — where they were willing to leave the price).
"""

from __future__ import annotations

import math
from typing import Any

ALPHA = math.sqrt(3.0)  # Platt coefficient from the paper's ablation
_P_FLOOR, _P_CEIL = 0.02, 0.98


def platt_extremize(p: float, alpha: float = ALPHA) -> float:
    """σ(α·logit(p)), clamped away from 0/1. Identity at p=0.5."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    out = 1.0 / (1.0 + math.exp(-alpha * logit))
    return min(max(out, _P_FLOOR), _P_CEIL)


def market_consensus(bets: list[dict[str, Any]], *, min_forecasters: int = 3,
                     min_amount: float = 1.0) -> dict | None:
    """Pool one market's bets into an extremized consensus probability.

    One vote per user: their LATEST bet's probAfter (bets arrive newest-first
    from the API; we keep the first seen per user). Redemptions, cancels, and
    dust are skipped. Returns {p_mean, p_extreme, n_forecasters} or None when
    fewer than `min_forecasters` distinct bettors are present — a two-bot
    "consensus" is noise, not a crowd.
    """
    seen: set[str] = set()
    probs: list[float] = []
    for b in bets:
        uid = b.get("userId")
        if not uid or uid in seen:
            continue
        if b.get("isRedemption") or b.get("isCancelled"):
            continue
        if (b.get("amount") or 0) < min_amount:
            continue
        p = b.get("probAfter")
        if p is None or not 0 < p < 1:
            continue
        seen.add(uid)
        probs.append(float(p))
    if len(probs) < min_forecasters:
        return None
    p_mean = sum(probs) / len(probs)
    return {"p_mean": p_mean, "p_extreme": platt_extremize(p_mean),
            "n_forecasters": len(probs)}
