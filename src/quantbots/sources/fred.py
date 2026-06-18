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
import logging

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _parse_history_csv(text: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for row in list(csv.reader(io.StringIO(text)))[1:]:
        if len(row) < 2:
            continue
        date, value = row[0], row[1]
        if value and value != ".":
            try:
                out.append((date, float(value)))
            except ValueError:
                continue
    return out


def _cache_history(series_id: str, series: list[tuple[str, float]]) -> None:
    import pandas as pd

    from ..research.data_fetch import _cache_path

    df = pd.DataFrame({"value": [v for _, v in series]},
                      index=pd.to_datetime([d for d, _ in series]))
    df.index.name = "Date"
    df.to_pickle(_cache_path("fred", series_id))


def _load_cached_history(series_id: str) -> list[tuple[str, float]]:
    import pandas as pd

    from ..research.data_fetch import _cache_path

    path = _cache_path("fred", series_id)
    if not path.exists():
        raise FileNotFoundError(f"no cached FRED history for {series_id} at {path}")
    df = pd.read_pickle(path)
    return [(str(idx.date()), float(v)) for idx, v in df["value"].items()]


def fetch_history(series_id: str) -> list[tuple[str, float]]:
    """Full (date, value) history for a FRED series, missing values dropped.
    Used by the backtester to replay models against real history. On a network
    failure, serves the last good data from the on-disk research cache (loud
    warning) so a FRED outage doesn't kill the backtest."""
    try:
        resp = requests.get(_URL, params={"id": series_id}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("FRED unreachable (%s) — serving CACHED %s history; may be stale",
                       type(e).__name__, series_id)
        return _load_cached_history(series_id)
    series = _parse_history_csv(resp.text)
    _cache_history(series_id, series)
    return series
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
