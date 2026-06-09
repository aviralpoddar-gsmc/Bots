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

from ..sources import atl3, cftc, weather
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


def _country_by_year(text: str, attribute: str, country: str) -> dict[int, float]:
    """One country's attribute per marketing year (1000-bale units for cotton)."""
    import csv as _csv
    import io as _io
    out: dict[int, float] = {}
    for row in _csv.DictReader(_io.StringIO(text)):
        if row["Attribute_Description"] != attribute or row["Country_Name"] != country:
            continue
        try:
            out[int(row["Market_Year"])] = float(row["Value"])
        except (ValueError, KeyError):
            continue
    return out


def compute_fas_balance(max_age_hours: float = 24, recent: int = 4) -> list[Observation]:
    """Cotton balance-sheet quantities (MILLION 480-lb bales) per marketing year,
    for the fas_balance bot. World production / mill use (domestic use) / ending
    stocks, plus China imports & Brazil exports. value=latest MY; payload.by_my
    maps marketing_year -> value so the strategy can pick the market's year."""
    try:
        text = _download_csv("cotton", max_age_hours)
    except Exception as exc:
        logger.warning("signals fas_balance: %s", exc)
        return []
    specs = [
        ("SIG_COTTON_WORLD_PRODUCTION", lambda: _world_by_year(text, "Production", set())),
        ("SIG_COTTON_WORLD_MILLUSE", lambda: _world_by_year(text, "Domestic Use", set())),
        ("SIG_COTTON_WORLD_ENDSTOCKS", lambda: _world_by_year(text, "Ending Stocks", set())),
        ("SIG_COTTON_CHINA_IMPORTS", lambda: _country_by_year(text, "Imports", "China")),
        ("SIG_COTTON_BRAZIL_EXPORTS", lambda: _country_by_year(text, "Exports", "Brazil")),
    ]
    out: list[Observation] = []
    for entity, fn in specs:
        series = {y: v / 1000.0 for y, v in fn().items()}  # 1000-bale -> million bales
        if not series:
            continue
        yr = max(series)
        by_my = {str(y): series[y] for y in sorted(series)[-recent:]}
        out.append(Observation(
            source="signal", entity=entity, ts=f"{yr}-08-01T00:00:00", value=series[yr],
            payload={"by_my": by_my, "unit": "million_bales", "marketing_year": yr},
        ))
    return out


def compute_wasde(max_age_hours: float = 24) -> list[Observation]:
    """Revision-tracked world cotton ending stocks (million bales), stamped by the
    USDA report month (Calendar_Year-Month). Each monthly WASDE/circular gets its
    own ts, so the store accumulates the revision history the wasde_event overlay
    diffs to get the month-over-month *surprise*. (The bulk file is a snapshot, so
    history builds up one release at a time.)"""
    import csv as _csv
    import io as _io
    try:
        text = _download_csv("cotton", max_age_hours)
    except Exception as exc:
        logger.warning("signals wasde: %s", exc)
        return []
    es = _world_by_year(text, "Ending Stocks", set())
    if not es:
        return []
    yr = max(es)
    report = None
    for r in _csv.DictReader(_io.StringIO(text)):
        if r["Attribute_Description"] == "Ending Stocks" and r["Market_Year"] == str(yr):
            report = (r["Calendar_Year"], r["Month"])
            break
    ts = f"{report[0]}-{report[1]}-01T00:00:00" if report else f"{yr}-08-01T00:00:00"
    return [Observation(
        source="signal", entity="SIG_COTTON_WASDE", ts=ts, value=es[yr] / 1000.0,
        payload={"report": f"{report[0]}-{report[1]}" if report else None,
                 "marketing_year": yr, "unit": "million_bales"},
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


# US upland cotton production shares (approx, recent NASS — refresh if they drift).
# Used to weight per-state crop condition into a national yield proxy. Weights are
# renormalized over whichever states actually have data, so missing states are safe.
_COTTON_STATE_WEIGHTS = {"TX": 0.42, "GA": 0.16, "AR": 0.09, "MS": 0.08, "NC": 0.06}


def compute_cotton_condition_index(store: Any) -> list[Observation]:
    """Production-weighted US cotton good+excellent % -> SIG_COTTON_COND_IDX.

    Sharper than the national NASS print: a single stressed major state (e.g. a
    Texas drought) moves the index even when the national average looks healthy.
    Reads the per-state NASS_COTTON_COND_GE_<ST> observations the nass source
    ingested; falls back to the national NASS_COTTON_COND_GE when no state data is
    present (so the bot degrades gracefully to its old behaviour, never worse).
    """
    # Collect (state, value, ts, year). NASS condition is seasonal and per-state
    # publication weeks differ, so a state can return LAST season's final reading
    # when this season's isn't out yet — blending it would mix seasons and can flip
    # the sign. So keep only states from the most recent season (max year present).
    cand: list[tuple[str, float, str, int]] = []
    for st in _COTTON_STATE_WEIGHTS:
        o = store.latest_observation(f"NASS_COTTON_COND_GE_{st}")
        if not o or o.get("value") is None:
            continue
        ts = o.get("ts") or ""
        try:
            yr = int(ts[:4])
        except (ValueError, TypeError):
            yr = 0
        cand.append((st, o["value"], ts, yr))
    num = den = 0.0
    parts: dict[str, float] = {}
    latest_ts = ""
    if cand:
        max_year = max(c[3] for c in cand)
        for st, val, ts, yr in cand:
            if yr != max_year:
                continue  # stale (prior-season) reading — exclude
            w = _COTTON_STATE_WEIGHTS[st]
            parts[st] = val
            num += val * w
            den += w
            latest_ts = max(latest_ts, ts)
    if den > 0:
        idx = num / den  # renormalized over present states
        return [Observation(
            source="signal", entity="SIG_COTTON_COND_IDX",
            ts=latest_ts or "1970-01-01T00:00:00", value=idx,
            payload={"by_state": parts, "weights": _COTTON_STATE_WEIGHTS, "n_states": len(parts)},
        )]
    # Fallback: national print, so the index entity always exists if any data does.
    o = store.latest_observation("NASS_COTTON_COND_GE")
    if o and o.get("value") is not None:
        return [Observation(
            source="signal", entity="SIG_COTTON_COND_IDX",
            ts=o.get("ts") or "1970-01-01T00:00:00", value=o["value"],
            payload={"fallback": "national"},
        )]
    return []


def compute_atl3_cocoa(window: int = 120) -> list[Observation]:
    """Z-score the tropical-Atlantic SST anomaly over a trailing window (~10y of
    months) -> SIG_ATL3_COCOA. Warm anomaly -> positive z; the cocoa SIGN is
    applied by the strategy (unvalidated)."""
    try:
        hist = atl3.fetch_history()
    except Exception as exc:
        logger.warning("signals atl3: %s", exc)
        return []
    vals = [v for _, v in hist][-window:]
    if len(vals) < 24:
        return []
    latest = vals[-1]
    z, mean, std = _z(vals, latest)
    return [Observation(
        source="signal", entity="SIG_ATL3_COCOA", ts=hist[-1][0], value=z,
        payload={"anom": latest, "mean": mean, "std": std, "n": len(vals)},
    )]


def compute_cotton_drought() -> list[Observation]:
    """Texas drought severity (USDM DSCI) z-scored vs a SAME-WEEK-OF-YEAR baseline
    -> SIG_COTTON_DROUGHT. DSCI is seasonal (summer droughts), so a raw-level z
    would mark every summer as 'drought' and invert the signal; deseasonalizing by
    week-of-year fixes that. High drought-z → supply stress → bullish cotton."""
    import datetime as _dt
    from collections import defaultdict as _dd

    from ..sources import usdm

    try:
        hist = usdm.fetch_history("48")
    except Exception as exc:
        logger.warning("signals usdm: %s", exc)
        return []
    if len(hist) < 104:  # need ~2y of weeks
        return []
    by_week: dict[int, list[float]] = _dd(list)
    for ts, v in hist:
        wk = _dt.date.fromisoformat(ts[:10]).isocalendar()[1]
        by_week[wk].append(v)
    ts_latest, v_latest = hist[-1]
    wk = _dt.date.fromisoformat(ts_latest[:10]).isocalendar()[1]
    # Baseline = same week across years; widen to ±2 weeks if a single week is thin.
    base = [x for x in by_week.get(wk, []) if x != v_latest]
    if len(base) < 8:
        base = [v for w, vs in by_week.items() if abs(w - wk) <= 2 for v in vs if v != v_latest]
    if len(base) < 8:
        return []
    z, mean, std = _z(base, v_latest)
    return [Observation(
        source="signal", entity="SIG_COTTON_DROUGHT", ts=ts_latest, value=z,
        payload={"dsci": v_latest, "woy": wk, "woy_mean": mean, "woy_std": std, "n_base": len(base)},
    )]


def compute_cocoa_stocks(window: int = 156) -> list[Observation]:
    """Z-score the ICE certified cocoa stock vs its trailing history ->
    SIG_COCOA_STOCK_Z. Low stock (negative z) = tight deliverable supply; the cocoa
    SIGN is applied by the strategy."""
    from ..sources import ice_stocks
    try:
        hist = ice_stocks.fetch_history()
    except Exception as exc:
        logger.warning("signals ice_stocks: %s", exc)
        return []
    vals = [v for _, v in hist][-window:]
    if len(vals) < 24:
        return []
    latest = vals[-1]
    z, mean, std = _z(vals, latest)
    return [Observation(
        source="signal", entity="SIG_COCOA_STOCK_Z", ts=hist[-1][0], value=z,
        payload={"cert_stock": latest, "mean": mean, "std": std, "n": len(vals)},
    )]


NEWS_FEEDS = ["INVESTING_COMMODITIES", "OILPRICE_MAIN", "MINING_DOT_COM", "EIA_TODAY_IN_ENERGY"]


def _parse_ts(ts: str | None):
    """Best-effort parse of an RSS pubDate / ISO ts -> aware UTC datetime, or None.
    Handles RFC-822 (oilprice/mining) and 'YYYY-MM-DD HH:MM:SS' / ISO (investing)."""
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime
    if not ts:
        return None
    try:
        dt = parsedate_to_datetime(ts)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts[:19], fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _ollama_reachable(timeout: float = 3.0) -> bool:
    import urllib.request
    from ..llm.client import DEFAULT_BASE_URL
    url = DEFAULT_BASE_URL.rsplit("/v1", 1)[0] + "/api/tags"
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False


def compute_news_signal(store: Any, feeds: list[str] | None = None, halflife_h: float = 36.0,
                        window_hours: float = 96.0, model: str = "gemma4:latest",
                        classify_fn: Any = None, now: Any = None) -> list[Observation]:
    """Digest recent commodity headlines into SIG_<COM>_NEWS observations: a
    recency-decayed, confidence-weighted MEAN SIGNED direction per commodity, in
    [-1, 1] (the 007 / news_drift signal). Classification is LOCAL-LLM, cached by
    headline hash (source='news_parse') so re-runs skip the model. ts ordering on
    RSS rows is unreliable (mixed RFC-822/ISO), so freshness is filtered by PARSED
    age in Python, not the SQL `since`. `classify_fn` lets tests inject a fake
    classifier (no network); `now` is injectable for deterministic tests."""
    import hashlib
    import json as _json
    from datetime import datetime, timezone

    from ..llm.news_extractor import COMMODITY_TO_ENTITY
    from ..llm.news_extractor import classify as _llm_classify

    feeds = feeds or NEWS_FEEDS
    now = now or datetime.now(timezone.utc)

    if classify_fn is None:
        if not _ollama_reachable():
            logger.info("news signal: local LLM endpoint unreachable — skipping")
            return []
        from ..llm.client import LocalLLM
        llm = LocalLLM(model=model)
        classify_fn = lambda h: _llm_classify(h, llm)  # noqa: E731

    def com_short(entity: str) -> str:
        return entity.replace("CME_", "").replace("_OIL", "")

    agg: dict[str, list] = {}
    to_cache: list[Observation] = []
    for feed in feeds:
        for it in store.load_observations(entity=feed, source="rss", limit=500):
            text = (it.get("text") or "").strip()
            if not text:
                continue
            dt = _parse_ts(it.get("ts"))
            if dt is None:
                continue  # can't establish freshness -> skip (don't risk stale)
            age_h = (now - dt).total_seconds() / 3600.0
            if age_h < 0 or age_h > window_hours:
                continue
            key = f"NP:{hashlib.md5(text.encode()).hexdigest()[:16]}"
            cached = store.latest_observation(entity=key, source="news_parse")
            if cached:
                p = cached.get("payload")
                rec = _json.loads(p) if isinstance(p, str) else (p or {})
            else:
                rec = classify_fn(text)
                to_cache.append(Observation(source="news_parse", entity=key,
                                            ts=it.get("ts") or now.isoformat(),
                                            text=text[:200], payload=rec))
            if not rec or not rec.get("is_price_event") or not rec.get("commodity"):
                continue
            entity = COMMODITY_TO_ENTITY.get(rec["commodity"])
            if not entity:
                continue
            w = math.exp(-age_h / halflife_h) if halflife_h > 0 else 1.0
            agg.setdefault(com_short(entity), []).append(
                (w, float(rec.get("confidence", 0.0)), int(rec.get("direction", 0)), text))
    if to_cache:
        store.upsert_observations(to_cache)

    out: list[Observation] = []
    for com, rows in agg.items():
        wc = sum(w * c for w, c, _, _ in rows)
        if wc <= 0:
            continue
        raw = sum(w * c * d for w, c, d, _ in rows) / wc  # in [-1, 1]
        top = [t for _, _, _, t in sorted(rows, key=lambda r: -(r[0] * r[1]))[:3]]
        out.append(Observation(
            source="signal", entity=f"SIG_{com}_NEWS", ts=now.isoformat(), value=raw,
            payload={"n_items": len(rows), "n_pos": sum(1 for _, _, d, _ in rows if d > 0),
                     "n_neg": sum(1 for _, _, d, _ in rows if d < 0), "raw": raw,
                     "halflife_h": halflife_h, "top_headlines": top}))
    return out


def run_all(store: Any, commodities: list[str] | None = None) -> int:
    """Compute all signals and upsert them. Returns count written."""
    commodities = commodities or ["cotton", "cocoa", "coffee"]
    obs: list[Observation] = []
    obs += compute_fas_cotton()
    obs += compute_fas_balance()
    obs += compute_wasde()
    obs += compute_cftc(commodities)
    obs += compute_weather_cocoa()
    obs += compute_cotton_condition_index(store)
    obs += compute_atl3_cocoa()
    obs += compute_cotton_drought()
    obs += compute_cocoa_stocks()
    try:
        obs += compute_news_signal(store)  # LOCAL-LLM news digestion (the 007 signal)
    except Exception as e:  # noqa: BLE001 - a news/LLM hiccup must not sink the other signals
        logger.warning("news signal failed: %s", e)
    n = store.upsert_observations(obs) if obs else 0
    logger.info("processing: wrote %d signals (%s)", n, ", ".join(o.entity for o in obs))
    return n
