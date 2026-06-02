"""ICE certified cocoa warehouse stocks (keyless) — a deliverable-supply signal.

ICE Futures U.S. publishes a daily cocoa certified-stock report as a legacy .xls
(OLE2/BIFF) at a dated public URL (no key):

    https://www.ice.com/publicdocs/futures_us_reports/cocoa/cocoa_cert_stock_<YYYYMMDD>.xls

Certified stock = cocoa graded and deliverable against the ICE cocoa futures the
clone has price markets on. Falling certified deliverable stock signals physical
tightness → bullish cocoa; rising stock → easing. It is daily and widely watched,
so treat it as a short-horizon CONFIRMER (a capped drift), not slow alpha. It is
orthogonal to the weather / Atlantic-SST cocoa signals.

Layout (verified): row "Total Bags" → certified total (the price-relevant number);
row "GRAND TOTAL:" → all bags in licensed warehouses (certified + uncertified).
We emit ICE_COCOA_CERT_STOCK and ICE_COCOA_WAREHOUSE_TOTAL.

History: ICE keeps dated files back years, so fetch_history() backfills weekly
samples into an on-disk cache (data/research/ice/, gitignored) for the signal
layer to z-score; subsequent runs only fetch new dates.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_BASE = "https://www.ice.com/publicdocs/futures_us_reports/cocoa/cocoa_cert_stock_{}.xls"
_DIR = Path(__file__).resolve().parents[3] / "data" / "research" / "ice"
_CACHE = _DIR / "cocoa_cert.csv"
_HEADERS = {"User-Agent": "quantbots/0.1"}


def _parse(content: bytes) -> tuple[float | None, float | None]:
    """(certified bags, grand-total bags) from the .xls, matched by row label."""
    import xlrd  # lazy: base env may lack it; .venv has xlrd 2.0.2 (reads .xls/BIFF)

    sh = xlrd.open_workbook(file_contents=content).sheet_by_index(0)
    cert = grand = None
    for i in range(sh.nrows):
        label = str(sh.cell_value(i, 0)).strip()
        nums = [
            sh.cell_value(i, j)
            for j in range(sh.ncols)
            if isinstance(sh.cell_value(i, j), (int, float)) and sh.cell_value(i, j) != ""
        ]
        if label == "Total Bags" and len(nums) >= 3 and cert is None:
            cert = float(nums[2])  # DR, NY, Total -> the certified Total
        elif label.startswith("GRAND TOTAL") and nums and grand is None:
            grand = float(nums[0])
    return cert, grand


def _fetch_day(d: dt.date) -> tuple[float | None, float | None] | None:
    """Parsed (cert, grand) for one date, or None if no file (weekend/holiday)."""
    try:
        r = requests.get(_BASE.format(d.strftime("%Y%m%d")), timeout=25, headers=_HEADERS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ice_stocks %s: %s", d, exc)
        return None
    if not r.ok or r.content[:4].hex() != "d0cf11e0":  # not an OLE2/BIFF .xls
        return None
    try:
        cert, _grand = res = _parse(r.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ice_stocks parse %s: %s", d, exc)
        return None
    return res if cert is not None else None


def _load_cache() -> dict[str, tuple[float | None, float | None]]:
    if not _CACHE.exists():
        return {}
    out: dict[str, tuple[float | None, float | None]] = {}
    for row in csv.DictReader(_CACHE.open()):
        try:
            out[row["date"]] = (
                float(row["cert"]) if row["cert"] else None,
                float(row["grand"]) if row["grand"] else None,
            )
        except (KeyError, ValueError):
            continue
    return out


def _save_cache(cache: dict[str, tuple[float | None, float | None]]) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "cert", "grand"])
        for d in sorted(cache):
            cert, grand = cache[d]
            w.writerow([d, cert if cert is not None else "", grand if grand is not None else ""])


def fetch_history(backfill_weeks: int = 156, max_fetch: int = 200) -> list[tuple[str, float]]:
    """Weekly-sampled (ISO ts, certified bags), oldest→newest, on-disk cached.

    Targets a midweek day per week back `backfill_weeks` plus the last few business
    days for recency; fetches only dates missing from the cache (bounded by
    `max_fetch`); caches holiday misses as blank so they aren't re-fetched."""
    cache = _load_cache()
    today = dt.date.today()
    targets: set[dt.date] = set()
    for back in range(1, 8):  # recent business days
        d = today - dt.timedelta(days=back)
        if d.weekday() < 5:
            targets.add(d)
    for w in range(backfill_weeks):  # weekly Wednesdays (reliably a business day)
        wd = today - dt.timedelta(weeks=w)
        targets.add(wd - dt.timedelta(days=(wd.weekday() - 2) % 7))
    todo = sorted((t for t in targets if t.isoformat() not in cache), reverse=True)
    fetched = 0
    for t in todo:
        if fetched >= max_fetch:
            break
        res = _fetch_day(t)
        cache[t.isoformat()] = res if res else (None, None)  # cache misses too (skip re-fetch)
        fetched += 1
    if fetched:
        _save_cache(cache)
    return [(f"{d}T00:00:00", c) for d, (c, _g) in sorted(cache.items()) if c is not None]


class IceStocksSource(Source):
    name = "ice_stocks"

    def fetch(self) -> list[Observation]:
        bw = int(self.params.get("backfill_weeks", 156))
        try:
            hist = fetch_history(backfill_weeks=bw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ice_stocks: %s", exc)
            return []
        if not hist:
            return []
        ts, cert = hist[-1]
        out = [Observation(source=self.name, entity="ICE_COCOA_CERT_STOCK", ts=ts, value=cert)]
        # grand total of the latest cached date, if we have it
        grand = _load_cache().get(ts[:10], (None, None))[1]
        if grand is not None:
            out.append(Observation(source=self.name, entity="ICE_COCOA_WAREHOUSE_TOTAL", ts=ts, value=grand))
        return out
