"""FRED — keyless US macro/economic series via the public CSV export.

`fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` returns a series with no
API key. Each configured series yields its latest non-missing value. Series IDs:
`MORTGAGE30US` (30y fixed mortgage rate %), `HOUST1F` (single-family housing
starts, SAAR thousands), `UNRATE` (unemployment %), `CPIAUCSL` (CPI index), etc.

Configure in config/sources.yaml:

    - name: fred
      params:
        series:
          - { entity: FRED_MORTGAGE30US, id: MORTGAGE30US }
          - { entity: FRED_HOUST1F, id: HOUST1F }
"""

from __future__ import annotations

import csv
import io

import requests

from .base import Observation, Source

_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_DEFAULT = [
    {"entity": "FRED_MORTGAGE30US", "id": "MORTGAGE30US"},
    {"entity": "FRED_HOUST1F", "id": "HOUST1F"},
]


class FredSource(Source):
    name = "fred"

    def fetch(self) -> list[Observation]:
        series = self.params.get("series") or _DEFAULT
        out: list[Observation] = []
        for s in series:
            resp = requests.get(_URL, params={"id": s["id"]}, timeout=30)
            resp.raise_for_status()
            rows = list(csv.reader(io.StringIO(resp.text)))
            if len(rows) < 2:
                continue
            # CSV is [["observation_date", SERIES_ID], [date, value], ...];
            # missing values are ".".
            latest = None
            for date, value in rows[1:]:
                if value and value != ".":
                    latest = (date, value)
            if latest is None:
                continue
            date, value = latest
            try:
                fval = float(value)
            except ValueError:
                continue
            out.append(
                Observation(
                    source=self.name,
                    entity=s["entity"],
                    ts=f"{date}T00:00:00",
                    value=fval,
                    payload={"series_id": s["id"]},
                )
            )
        return out
