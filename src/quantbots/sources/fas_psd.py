"""USDA FAS PSD — global supply/demand fundamentals from the keyless bulk CSVs.

USDA Foreign Agricultural Service publishes the Production, Supply & Distribution
database as per-commodity ZIP/CSV bulk downloads that need **no API key**:

    https://apps.fas.usda.gov/psdonline/downloads/psd_<commodity>_csv.zip

Each CSV is one row per (commodity, country, marketing year, attribute). This
source aggregates them to the WORLD level and emits the fundamentals our
soft-commodity bots actually use:

- ``PSD_COTTON_FREE_SUR`` — world-**ex-China** stocks-to-use ("free" SUR). World
  SUR is decoupled from price by China's off-market state reserve; ex-China is the
  metric that carries the cotton price signal (see docs/usda-softs-bots.md §2b).
- ``PSD_COTTON_SUR`` — world stocks-to-use (incl. China), for reference.
- ``PSD_COFFEE_CONS`` — world coffee domestic consumption (1000 60-kg bags).
- ``PSD_COFFEE_CONS_GROWTH`` — YoY % change in world coffee consumption, the
  fundamental behind the clone's "global coffee consumption growth rate" markets.

stdlib only (csv/zipfile) — the core ingest path must not require pandas.

Configure in config/sources.yaml:

    - name: fas_psd
      params:
        max_age_hours: 24
        commodities: [cotton, coffee]
"""

from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from pathlib import Path

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_BASE = "https://apps.fas.usda.gov/psdonline/downloads/psd_{c}_csv.zip"
_CACHE = Path(__file__).resolve().parents[3] / "data" / "research" / "psd"

# Per-commodity: the bulk-CSV "use" attribute name (differs across commodities).
_USE_ATTR = {"cotton": "Domestic Use", "coffee": "Domestic Consumption"}


def _download_csv(commodity: str, max_age_hours: float) -> str:
    """Return the bulk CSV text for a commodity, using an on-disk cache."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    csv_path = _CACHE / f"psd_{commodity}.csv"
    fresh = csv_path.exists() and (time.time() - csv_path.stat().st_mtime) / 3600 < max_age_hours
    if not fresh:
        resp = requests.get(_BASE.format(c=commodity), timeout=90)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            data = zf.read(name).decode("utf-8", errors="replace")
        csv_path.write_text(data)
        return data
    return csv_path.read_text()


def _world_by_year(text: str, attribute: str, exclude: set[str]) -> dict[int, float]:
    """Sum an attribute across countries (minus `exclude`) per marketing year."""
    out: dict[int, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row["Attribute_Description"] != attribute:
            continue
        if row["Country_Name"] in exclude:
            continue
        try:
            yr = int(row["Market_Year"])
            val = float(row["Value"])
        except (ValueError, KeyError):
            continue
        out[yr] = out.get(yr, 0.0) + val
    return out


def _latest_sur(text: str, use_attr: str, exclude: set[str]) -> tuple[int, float] | None:
    """(marketing_year, stocks-to-use) for the latest year with both fields > 0."""
    stocks = _world_by_year(text, "Ending Stocks", exclude)
    use = _world_by_year(text, use_attr, exclude)
    yrs = sorted(set(stocks) & set(use), reverse=True)
    for yr in yrs:
        if use[yr] > 0 and stocks[yr] >= 0:
            return yr, stocks[yr] / use[yr]
    return None


class FasPsdSource(Source):
    name = "fas_psd"

    def fetch(self) -> list[Observation]:
        max_age = float(self.params.get("max_age_hours", 24))
        commodities = self.params.get("commodities") or ["cotton", "coffee"]
        out: list[Observation] = []
        for c in commodities:
            try:
                text = _download_csv(c, max_age)
            except Exception as exc:  # network / format — skip this commodity
                logger.warning("fas_psd %s: %s", c, exc)
                continue
            use_attr = _USE_ATTR.get(c, "Domestic Consumption")
            if c == "cotton":
                self._emit_cotton(text, use_attr, out)
            elif c == "coffee":
                self._emit_coffee(text, use_attr, out)
        return out

    def _emit_cotton(self, text: str, use_attr: str, out: list[Observation]) -> None:
        for entity, exclude in (
            ("PSD_COTTON_SUR", set()),
            ("PSD_COTTON_FREE_SUR", {"China"}),
        ):
            res = _latest_sur(text, use_attr, exclude)
            if res is None:
                continue
            yr, sur = res
            out.append(Observation(
                source=self.name, entity=entity, ts=f"{yr}-08-01T00:00:00",
                value=sur, payload={"marketing_year": yr, "metric": "stocks_to_use"},
            ))

    def _emit_coffee(self, text: str, use_attr: str, out: list[Observation]) -> None:
        cons = _world_by_year(text, use_attr, set())
        cons = {y: v for y, v in cons.items() if v > 0}
        if not cons:
            return
        yr = max(cons)
        out.append(Observation(
            source=self.name, entity="PSD_COFFEE_CONS", ts=f"{yr}-10-01T00:00:00",
            value=cons[yr], payload={"marketing_year": yr, "unit": "1000_60kg_bags"},
        ))
        if (yr - 1) in cons and cons[yr - 1] > 0:
            growth = 100.0 * (cons[yr] / cons[yr - 1] - 1.0)
            out.append(Observation(
                source=self.name, entity="PSD_COFFEE_CONS_GROWTH", ts=f"{yr}-10-01T00:00:00",
                value=growth, payload={"marketing_year": yr, "metric": "yoy_pct"},
            ))
