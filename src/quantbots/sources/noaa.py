"""NOAA CPC — keyless climate indices. Currently the Oceanic Niño Index (ONI).

The ONI is the official ENSO (El Niño / La Niña) indicator: a 3-month running
mean SST anomaly in degrees Celsius. CPC publishes it as a plain ascii table with
no API key:

    https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

Columns: SEAS YR TOTAL ANOM — ANOM is the ONI value (can be negative: La Niña).
We emit the latest reading as entity `ENSO_ONI`.
"""

from __future__ import annotations

import requests

from .base import Observation, Source

_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# 3-month overlapping season -> representative center month (for a sortable ts).
_SEASON_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


class NoaaSource(Source):
    name = "noaa"

    def fetch(self) -> list[Observation]:
        resp = requests.get(_ONI_URL, timeout=30)
        resp.raise_for_status()
        rows = []
        for line in resp.text.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) != 4:
                continue
            seas, yr, _total, anom = parts
            try:
                value = float(anom)
                year = int(yr)
            except ValueError:
                continue
            month = _SEASON_MONTH.get(seas.upper(), 6)
            rows.append((f"{year:04d}-{month:02d}-01T00:00:00", value, seas))
        if not rows:
            return []
        # Emit the trailing window so we keep a short history; the strategy uses
        # the latest. (entity is shared; ts dedups per season.)
        out = [
            Observation(
                source=self.name,
                entity="ENSO_ONI",
                ts=ts,
                value=value,
                payload={"season": seas},
            )
            for ts, value, seas in rows[-12:]
        ]
        return out
