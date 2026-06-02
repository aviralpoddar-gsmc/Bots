"""US Drought Monitor — keyless drought severity by state (DSCI).

The USDM (droughtmonitor.unl.edu) publishes a weekly Drought Severity and Coverage
Index (DSCI, 0–500; higher = more/worse drought) per state through a keyless JSON
API:

    GET https://usdmdataservices.unl.edu/api/StateStatistics/GetDSCI
        ?aoi=<state FIPS>&startdate=M/D/YYYY&enddate=M/D/YYYY&statisticsType=1
    header:  Accept: application/json   (WITHOUT it the API returns CSV — load-bearing)

We track Texas (aoi=48), the dominant US cotton state (~40% of the crop), as entity
``USDM_DSCI_TX``. DSCI is strongly SEASONAL (drought peaks in summer), so the signal
layer (processing/signals.compute_cotton_drought) z-scores it against a same-week-of-
year baseline, NOT the raw level — a raw-level z would flag every summer as "drought"
and flip the sign.

Unconventional angle: drought severity LEADS the USDA NASS crop-condition print
(condition is the field-observed consequence of moisture) and is published weekly
with no key.
"""

from __future__ import annotations

import datetime as dt
import logging

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_URL = "https://usdmdataservices.unl.edu/api/StateStatistics/GetDSCI"
# aoi (state FIPS) -> entity suffix. Texas is the cotton heartland.
_DEFAULT_STATES = {"48": "TX"}


def fetch_history(aoi: str = "48", start: str = "1/1/2000") -> list[tuple[str, float]]:
    """Weekly (ISO ts, DSCI) for one state aoi, oldest→newest."""
    today = dt.date.today()
    end = f"{today.month}/{today.day}/{today.year}"  # API wants un-padded M/D/YYYY
    resp = requests.get(
        _URL,
        params={"aoi": aoi, "startdate": start, "enddate": end, "statisticsType": "1"},
        headers={"Accept": "application/json"},
        timeout=45,
    )
    resp.raise_for_status()
    out: list[tuple[str, float]] = []
    for rec in resp.json():
        try:
            md = rec["mapDate"][:10]  # "2000-01-04T00:00:00" -> "2000-01-04"
            dsci = float(rec["dsci"])
        except (KeyError, ValueError, TypeError):
            continue
        out.append((f"{md}T00:00:00", dsci))
    out.sort()
    return out


class UsdmSource(Source):
    name = "usdm"

    def fetch(self) -> list[Observation]:
        states = self.params.get("states") or _DEFAULT_STATES
        out: list[Observation] = []
        for aoi, suffix in states.items():
            try:
                hist = fetch_history(aoi)
            except Exception as exc:  # noqa: BLE001 - one bad state must not abort ingest
                logger.warning("usdm %s: %s", suffix, exc)
                continue
            out += [
                Observation(source=self.name, entity=f"USDM_DSCI_{suffix}", ts=ts, value=v)
                for ts, v in hist[-12:]  # short trailing window; the signal layer re-fetches full history
            ]
        return out
