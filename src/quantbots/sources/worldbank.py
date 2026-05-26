"""World Bank — keyless macro / economic indicators.

The World Bank Open Data API needs no key. Each configured indicator pulls its
most-recent value for a country. Indicator codes: `FP.CPI.TOTL.ZG` (inflation %),
`NY.GDP.MKTP.KD.ZG` (GDP growth %), `SL.UEM.TOTL.ZS` (unemployment %), etc.

Configure in config/sources.yaml:

    - name: worldbank
      params:
        indicators:
          - { entity: US_CPI_YOY, country: US, code: FP.CPI.TOTL.ZG }
          - { entity: US_GDP_GROWTH, country: US, code: NY.GDP.MKTP.KD.ZG }
"""

from __future__ import annotations

import requests

from .base import Observation, Source

_BASE = "https://api.worldbank.org/v2"
_DEFAULT = [
    {"entity": "US_CPI_YOY", "country": "US", "code": "FP.CPI.TOTL.ZG"},
    {"entity": "US_GDP_GROWTH", "country": "US", "code": "NY.GDP.MKTP.KD.ZG"},
    {"entity": "US_UNEMPLOYMENT", "country": "US", "code": "SL.UEM.TOTL.ZS"},
]


class WorldBankSource(Source):
    name = "worldbank"

    def fetch(self) -> list[Observation]:
        indicators = self.params.get("indicators") or _DEFAULT
        mrv = int(self.params.get("recent_values", 1))  # most-recent N values
        out: list[Observation] = []
        for ind in indicators:
            url = f"{_BASE}/country/{ind['country']}/indicator/{ind['code']}"
            resp = requests.get(url, params={"format": "json", "mrv": mrv}, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            # Response is [metadata, [datapoints]]; datapoints may be None.
            if not isinstance(body, list) or len(body) < 2 or not body[1]:
                continue
            for dp in body[1]:
                if dp.get("value") is None:
                    continue
                out.append(
                    Observation(
                        source=self.name,
                        entity=ind["entity"],
                        ts=f"{dp['date']}-12-31T00:00:00",  # annual prints
                        value=float(dp["value"]),
                        payload={"code": ind["code"], "country": ind["country"]},
                    )
                )
        return out
