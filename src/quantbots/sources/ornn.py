"""Ornn — free, keyless daily GPU compute-price index (OCPI).

Ornn publishes the Ornn Compute Price Index, an institutional benchmark for GPU
*rental* prices built from cleared/printed transactions (also carried on the
Bloomberg Terminal and underpinning ICE/Architect compute futures and Kalshi
compute-price markets). Unlike the operational metrics that dominate the clone,
a GPU rental price is a genuine, externally-published *price* with a named
source — exactly the kind of measurable that resolves YES/NO rather than CANCEL.

The public history endpoint is keyless JSON, one GPU SKU per request:

    GET https://api.ornnai.com/api/gpu/<GPU>/index-history
    -> {"success": true, "gpu_type": "H100 SXM",
        "data": [{"timestamp": "2026-03-25T20:00:00.000Z", "index_value": 1.72}, ...]}

`index_value` is the index level in USD per GPU-hour. We emit the *full* returned
history each fetch (the store upserts by source+entity+ts), so a daily ingest
backfills and then extends the series — useful both for a future compute-price
bot and for backtests.

Configure in config/sources.yaml (entity -> Ornn GPU name):

    - name: ornn
      params:
        gpus:
          GPU_H100_SXM: "H100 SXM"
          GPU_B200: "B200"
"""

from __future__ import annotations

from urllib.parse import quote

import requests

from .base import Observation, Source

_BASE = "https://api.ornnai.com"
# Default basket — every SKU the index currently publishes (verified live). Keys
# are canonical entities (GPU_<MODEL>), values the exact name the API expects.
_DEFAULT_GPUS = {
    "GPU_H100_SXM": "H100 SXM",
    "GPU_H200": "H200",
    "GPU_B200": "B200",
    "GPU_A100_SXM4": "A100 SXM4",
    "GPU_RTX_5090": "RTX 5090",
    "GPU_RTX_PRO_6000_WS": "RTX PRO 6000 WS",
}


class OrnnComputeSource(Source):
    name = "ornn"

    def fetch(self) -> list[Observation]:
        gpus: dict[str, str] = self.params.get("gpus") or _DEFAULT_GPUS
        out: list[Observation] = []
        for entity, gpu in gpus.items():
            resp = requests.get(
                f"{_BASE}/api/gpu/{quote(gpu)}/index-history",
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success"):
                continue
            gpu_type = body.get("gpu_type", gpu)
            for pt in body.get("data", []):
                ts, value = pt.get("timestamp"), pt.get("index_value")
                if ts is None or value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                out.append(
                    Observation(
                        source=self.name,
                        entity=entity,
                        ts=ts,
                        value=value,  # index level, USD / GPU-hour
                        payload={"gpu_type": gpu_type, "unit": "usd_per_gpu_hour"},
                    )
                )
        return out
