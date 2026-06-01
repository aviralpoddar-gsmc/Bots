"""Weather — growing-region anomalies for coffee & cocoa (keyless open-meteo).

Coffee and cocoa are not US crops, so NASS is silent and the real supply shocks
are weather in a few key regions: Brazil/Vietnam for coffee (frost & drought),
Ivory Coast/Ghana for cocoa (drought/excess rain). open-meteo's archive API is
keyless and gives daily min-temp and precipitation we can turn into anomalies.

Emits, per region, recent values used by the processing layer to build anomalies:
- ``WX_<REGION>_TMIN``    last available daily minimum temperature (°C)
- ``WX_<REGION>_PRCP30``  trailing 30-day precipitation total (mm)

`fetch_history(region)` returns daily (date, tmin, prcp) for anomaly baselines.

Configure in config/sources.yaml:

    - name: weather
      params:
        regions:
          - { key: COCOA_CI, lat: 6.8, lon: -5.3 }    # Ivory Coast cocoa belt
          - { key: COFFEE_BR, lat: -21.0, lon: -45.0 } # Minas Gerais coffee belt
"""

from __future__ import annotations

import logging

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_URL = "https://archive-api.open-meteo.com/v1/archive"
_DEFAULT_REGIONS = [
    {"key": "COCOA_CI", "lat": 6.8, "lon": -5.3},
    {"key": "COFFEE_BR", "lat": -21.0, "lon": -45.0},
]


def _archive(lat: float, lon: float, start: str, end: str) -> dict:
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_min,precipitation_sum", "timezone": "UTC",
    }
    resp = requests.get(_URL, params=params, timeout=45)
    resp.raise_for_status()
    return resp.json().get("daily", {})


def fetch_history(lat: float, lon: float, start: str, end: str) -> list[tuple[str, float, float]]:
    """Daily (date, tmin, precip) over a window."""
    d = _archive(lat, lon, start, end)
    times = d.get("time", [])
    tmin = d.get("temperature_2m_min", [])
    prcp = d.get("precipitation_sum", [])
    out = []
    for i, t in enumerate(times):
        try:
            out.append((t, float(tmin[i]), float(prcp[i])))
        except (ValueError, TypeError, IndexError):
            continue
    return out


class WeatherSource(Source):
    name = "weather"

    def fetch(self) -> list[Observation]:
        import datetime as _dt

        regions = self.params.get("regions") or _DEFAULT_REGIONS
        # open-meteo archive lags ~5 days; pull a 45-day window ending a week back.
        # Dates passed via params (scripts/cron) to keep this deterministic-friendly.
        end = self.params.get("end_date")
        start = self.params.get("start_date")
        if not end or not start:
            # fall back to a fixed recent window relative to ingest time
            today = _dt.date.today()
            end = (today - _dt.timedelta(days=7)).isoformat()
            start = (today - _dt.timedelta(days=52)).isoformat()
        out: list[Observation] = []
        for r in regions:
            try:
                hist = fetch_history(r["lat"], r["lon"], start, end)
            except Exception as exc:
                logger.warning("weather %s: %s", r.get("key"), exc)
                continue
            if not hist:
                continue
            last_date, last_tmin, _ = hist[-1]
            prcp30 = sum(p for _, _, p in hist[-30:])
            ts = f"{last_date}T00:00:00"
            out.append(Observation(self.name, f"WX_{r['key']}_TMIN", ts, value=last_tmin,
                                   payload={"lat": r["lat"], "lon": r["lon"]}))
            out.append(Observation(self.name, f"WX_{r['key']}_PRCP30", ts, value=prcp30,
                                   payload={"window_days": 30}))
        return out
