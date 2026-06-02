"""Atlantic SST anomalies (keyless CPC) — a cocoa climate driver.

CPC publishes monthly Atlantic SST indices with no API key:

    https://www.cpc.ncep.noaa.gov/data/indices/sstoi.atl.indices

Columns:  YR MON NATL ANOM SATL ANOM TROP ANOM
(NATL / SATL / TROP = North / South / Tropical Atlantic SST in °C, each followed
by its anomaly vs climatology.)

We emit the TROP (tropical Atlantic) anomaly as entity ``ATL3_SSTA`` — a keyless
PROXY for the equatorial-Atlantic "Atlantic Niño" (the ATL3 box, 20°W–0°, 3°S–3°N)
that the literature links to Gulf-of-Guinea / West-Africa rainfall and therefore
cocoa supply (West Africa ≈ 70% of world cocoa). CPC does not publish the exact
ATL3 box without a key; the tropical-Atlantic anomaly is the closest free proxy.

⚠️ The cocoa sign is UNVALIDATED — warm Atlantic can mean more Guinea rain (better
crop, bearish) OR excess rain / black-pod disease (bullish). Walk-forward check the
sign against cocoa futures before trading this live. The bot ships enabled:false.
"""

from __future__ import annotations

import logging

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.atl.indices"


def fetch_history() -> list[tuple[str, float]]:
    """Monthly (ISO ts, tropical-Atlantic SST anomaly °C), oldest→newest."""
    resp = requests.get(_URL, timeout=30, headers={"User-Agent": "quantbots/0.1"})
    resp.raise_for_status()
    out: list[tuple[str, float]] = []
    for line in resp.text.strip().splitlines()[1:]:  # skip header
        p = line.split()
        if len(p) != 8:
            continue
        try:
            yr, mon = int(p[0]), int(p[1])
            anom = float(p[7])  # TROP anomaly = last column
        except ValueError:
            continue
        out.append((f"{yr:04d}-{mon:02d}-01T00:00:00", anom))
    return out


class AtlanticSstSource(Source):
    name = "atl3"

    def fetch(self) -> list[Observation]:
        try:
            hist = fetch_history()
        except Exception as exc:  # noqa: BLE001 - one bad source must not abort ingest
            logger.warning("atl3: %s", exc)
            return []
        # Keep a short trailing history (the signal layer z-scores it; the strategy
        # uses the latest). entity shared, ts dedups per month.
        return [
            Observation(
                source=self.name, entity="ATL3_SSTA", ts=ts, value=v,
                payload={"index": "tropical_atlantic_sst_anom", "proxy_for": "ATL3"},
            )
            for ts, v in hist[-12:]
        ]
