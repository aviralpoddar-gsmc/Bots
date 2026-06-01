"""Compute normalized SIG_* signals from each source's history (stdlib only).

Each function pulls the relevant source's history, computes a clean, normalized
signal (z-score / fair value / anomaly), and returns Observations with the raw
numbers in the payload (surfaced in trade comments). `run_all` persists them.

Signals produced:
- ``SIG_COTTON_FAS``   : fundamental fair value (cents/lb) from FAS ex-China SUR,
                         payload {sur, sur_z, sur_mean, elasticity}.
- ``SIG_<COM>_CFTC``   : managed-money net-% z-score (extreme -> reversion),
                         payload {netpct, mean, std, n}.
- ``SIG_COCOA_WX``     : cocoa-belt drought signal = -(precip z), payload {prcp30, z}.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from ..sources import cftc, weather
from ..sources.base import Observation
from ..sources.fas_psd import _download_csv, _world_by_year

logger = logging.getLogger(__name__)

# Calibrated FAS->price relationship (scripts/research_softs.py, 1999-2025).
_FAS_COTTON = {"elasticity": -0.39, "sur_ref": 0.487, "price_ref": 68.4}


def _z(values: list[float], latest: float) -> tuple[float, float, float]:
    """(z-score, mean, std) of `latest` vs the distribution `values`."""
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    z = (latest - mean) / std if std > 0 else 0.0
    return z, mean, std


def compute_fas_cotton(max_age_hours: float = 24) -> list[Observation]:
    try:
        text = _download_csv("cotton", max_age_hours)
    except Exception as exc:
        logger.warning("signals fas_cotton: %s", exc)
        return []
    stocks = _world_by_year(text, "Ending Stocks", {"China"})
    use = _world_by_year(text, "Domestic Use", {"China"})
    sur = {y: stocks[y] / use[y] for y in set(stocks) & set(use) if use[y] > 0}
    if len(sur) < 10:
        return []
    yr = max(sur)
    latest = sur[yr]
    logs = [math.log(v) for v in sur.values()]
    z, logmean, _ = _z(logs, math.log(latest))
    c = _FAS_COTTON
    fair = c["price_ref"] * (latest / c["sur_ref"]) ** c["elasticity"]
    return [Observation(
        source="signal", entity="SIG_COTTON_FAS", ts=f"{yr}-08-01T00:00:00",
        value=fair,
        payload={"sur": latest, "sur_z": z, "sur_mean": math.exp(logmean),
                 "elasticity": c["elasticity"], "marketing_year": yr},
    )]


def compute_cftc(commodities: list[str], window: int = 156) -> list[Observation]:
    """Managed-money net-% z-score over a trailing window (default ~3y of weeks)."""
    out: list[Observation] = []
    for com in commodities:
        try:
            hist = cftc.fetch_history(com)
        except Exception as exc:
            logger.warning("signals cftc %s: %s", com, exc)
            continue
        vals = [v for _, v in hist][-window:]
        if len(vals) < 26:
            continue
        latest = vals[-1]
        z, mean, std = _z(vals, latest)
        date = hist[-1][0]
        out.append(Observation(
            source="signal", entity=f"SIG_{com.upper()}_CFTC", ts=f"{date}T00:00:00",
            value=z, payload={"netpct": latest, "mean": mean, "std": std, "n": len(vals)},
        ))
    return out


def compute_weather_cocoa(start: str | None = None, end: str | None = None) -> list[Observation]:
    """Cocoa-belt (Ivory Coast) drought signal = -(trailing-30d precip z) over ~2y."""
    import datetime as _dt

    if not end or not start:
        today = _dt.date.today()
        end = (today - _dt.timedelta(days=7)).isoformat()
        start = (today - _dt.timedelta(days=730)).isoformat()
    try:
        hist = weather.fetch_history(6.8, -5.3, start, end)  # Ivory Coast cocoa belt
    except Exception as exc:
        logger.warning("signals weather cocoa: %s", exc)
        return []
    if len(hist) < 120:
        return []
    prcp = [p for _, _, p in hist]
    # rolling 30-day sums; z the latest vs the distribution of all rolling sums
    roll = [sum(prcp[i - 30:i]) for i in range(30, len(prcp) + 1)]
    latest = roll[-1]
    z, mean, std = _z(roll, latest)
    return [Observation(
        source="signal", entity="SIG_COCOA_WX", ts=f"{hist[-1][0]}T00:00:00",
        value=-z,  # low precip (drought) -> positive (bullish supply-risk) signal
        payload={"prcp30": latest, "prcp_mean": mean, "precip_z": z},
    )]


def run_all(store: Any, commodities: list[str] | None = None) -> int:
    """Compute all signals and upsert them. Returns count written."""
    commodities = commodities or ["cotton", "cocoa", "coffee"]
    obs: list[Observation] = []
    obs += compute_fas_cotton()
    obs += compute_cftc(commodities)
    obs += compute_weather_cocoa()
    n = store.upsert_observations(obs) if obs else 0
    logger.info("processing: wrote %d signals (%s)", n, ", ".join(o.entity for o in obs))
    return n
