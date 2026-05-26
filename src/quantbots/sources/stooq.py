"""Stooq — free, keyless daily quotes for commodities, FX, and indices.

Stooq serves a tiny CSV per symbol set, no API key required, which makes it the
simplest first price feed. Symbol examples: `cl.f` (WTI crude), `gc.f` (gold),
`si.f` (silver), `hg.f` (copper), `ng.f` (natgas), `eurusd`, `^spx`.

Configure in config/sources.yaml:

    - name: stooq
      params:
        symbols:
          WTI_OIL: cl.f
          GOLD: gc.f
          COPPER: hg.f
"""

from __future__ import annotations

import csv
import io

import requests

from .base import Observation, Source

_URL = "https://stooq.com/q/l/"
# A reasonable default basket if none configured.
_DEFAULT_SYMBOLS = {
    "WTI_OIL": "cl.f",
    "BRENT_OIL": "cb.f",
    "GOLD": "gc.f",
    "SILVER": "si.f",
    "COPPER": "hg.f",
    "NATGAS": "ng.f",
}


class StooqSource(Source):
    name = "stooq"

    def fetch(self) -> list[Observation]:
        symbols: dict[str, str] = self.params.get("symbols") or _DEFAULT_SYMBOLS
        out: list[Observation] = []
        # Stooq's /q/l/ returns N/D for comma-joined multi-symbol queries, so we
        # fetch one symbol per request (a handful of calls, well within limits).
        for entity, symbol in symbols.items():
            resp = requests.get(
                _URL,
                params={"s": symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
                timeout=30,
            )
            resp.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(resp.text)))
            if not rows:
                continue
            row = rows[0]
            close = row.get("Close")
            if not close or close in ("N/D", "0"):
                continue
            try:
                value = float(close)
            except ValueError:
                continue
            date = row.get("Date") or ""
            time = row.get("Time") or "00:00:00"
            out.append(
                Observation(
                    source=self.name,
                    entity=entity,
                    ts=f"{date}T{time}",
                    value=value,
                    payload={k: row.get(k) for k in ("Open", "High", "Low", "Close", "Volume")},
                )
            )
        return out
