"""USDA NASS QuickStats — US crop statistics (cotton condition, price received).

NASS QuickStats is a free **keyed** REST API. Get a key (instant, email + ToS) at
https://quickstats.nass.usda.gov/api and set it as ``NASS_API_KEY`` (or pass
``api_key`` in params). Without a key this source logs and returns ``[]`` — the
cotton bot's primary signal is the FAS stocks-to-use drift, so NASS condition is
an optional in-season *nudge*, not a hard dependency.

Endpoint:  GET https://quickstats.nass.usda.gov/api/api_GET/
Query model is "What / Where / When", e.g. for the national cotton crop-condition
good+excellent share we sum the latest week's PCT GOOD and PCT EXCELLENT.

Configure in config/sources.yaml:

    - name: nass
      params:
        # api_key: read from NASS_API_KEY env if omitted
        series:
          - entity: NASS_COTTON_COND_GE     # good+excellent %, summed
            sum_short_desc:
              - "COTTON - CONDITION, MEASURED IN PCT GOOD"
              - "COTTON - CONDITION, MEASURED IN PCT EXCELLENT"
            agg_level_desc: NATIONAL
          - entity: NASS_COTTON_PRICE        # price received, cents/lb
            short_desc: "COTTON, UPLAND - PRICE RECEIVED, MEASURED IN $ / LB"
            agg_level_desc: NATIONAL
"""

from __future__ import annotations

import logging
import os

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_URL = "https://quickstats.nass.usda.gov/api/api_GET/"


def _latest(records: list[dict]) -> dict | None:
    """Most recent record by (year, end_code/reference period)."""
    def key(r: dict) -> tuple:
        return (r.get("year", ""), r.get("end_code", "") or r.get("reference_period_desc", ""))
    usable = [r for r in records if r.get("Value") not in (None, "", "(D)", "(NA)", "(Z)")]
    return max(usable, key=key) if usable else None


def _value(rec: dict) -> float | None:
    try:
        return float(str(rec["Value"]).replace(",", ""))
    except (KeyError, ValueError):
        return None


class NassSource(Source):
    name = "nass"

    def fetch(self) -> list[Observation]:
        key = self.params.get("api_key") or os.environ.get("NASS_API_KEY")
        if not key:
            logger.info("nass: no NASS_API_KEY set — skipping (cotton bot runs without it)")
            return []
        out: list[Observation] = []
        for s in self.params.get("series", []):
            try:
                out.extend(self._fetch_series(key, s))
            except Exception as exc:
                logger.warning("nass %s: %s", s.get("entity"), exc)
        return out

    def _query(self, key: str, short_desc: str, extra: dict) -> list[dict]:
        params = {
            "key": key, "commodity_desc": extra.get("commodity_desc", "COTTON"),
            "short_desc": short_desc, "agg_level_desc": extra.get("agg_level_desc", "NATIONAL"),
            "format": "JSON", "year__GE": str(extra.get("year_ge", 2018)),
        }
        resp = requests.get(_URL, params=params, timeout=45)
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])

    def _fetch_series(self, key: str, s: dict) -> list[Observation]:
        # Summed series (e.g. good + excellent) — one query per short_desc, summed.
        descs = s.get("sum_short_desc") or ([s["short_desc"]] if s.get("short_desc") else [])
        total = 0.0
        ts = None
        got = False
        for d in descs:
            rec = _latest(self._query(key, d, s))
            if rec is None:
                continue
            v = _value(rec)
            if v is None:
                continue
            total += v
            got = True
            ts = f"{rec.get('year')}-{rec.get('end_code') or '12-31'}"
        if not got:
            return []
        return [Observation(
            source=self.name, entity=s["entity"], ts=f"{ts}T00:00:00" if ts else "1970-01-01T00:00:00",
            value=total, payload={"summed_of": descs},
        )]
