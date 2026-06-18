"""Shared thin HTTP layer for Alpaca REST (market-data + trading).

Ported in spirit from `manifold/client.py`: a `requests.Session` with the auth
headers baked in, client-side rate limiting, and retry-on-5xx/429. Two Alpaca hosts
use it — the market-data host and the trading host — so the session logic lives once
here and both `sources/options_chain.py` and `execution/alpaca.py` wrap it.

SAFETY: the trading base URL is supplied by the caller, but the only non-live caller
is the paper host (`paper-api.alpaca.markets`). The live host lives solely in
`execution/live.py`, behind the owner-approval gate. This helper never defaults to a
live URL.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

import requests

logger = logging.getLogger(__name__)

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaHTTP:
    RATE_LIMIT = 200          # Alpaca's basic tier is 200 req/min
    WINDOW = 60

    def __init__(self, base_url: str, *, key: str | None = None, secret: str | None = None,
                 timeout: int = 30, max_retries: int = 3):
        key = key or os.environ.get("ALPACA_API_KEY")
        secret = secret or os.environ.get("ALPACA_SECRET_KEY")
        if not (key and secret):
            raise ValueError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(paper keys) or pass key=/secret=."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_on = {429, 500, 502, 503, 504}
        self._req_times: deque[float] = deque(maxlen=self.RATE_LIMIT)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "quantbots-equity-options/0.1",
            "Accept": "application/json",
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        })

    def _rate_limit(self) -> None:
        now = time.time()
        start = now - self.WINDOW
        while self._req_times and self._req_times[0] < start:
            self._req_times.popleft()
        if len(self._req_times) >= self.RATE_LIMIT:
            time.sleep(max(self._req_times[0] - start, 0))
        self._req_times.append(now)

    def request(self, method: str, endpoint: str, *, params: dict | None = None,
                json: dict | None = None) -> Any:
        url = f"{self.base_url}{endpoint}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._rate_limit()
            try:
                resp = self.session.request(method, url, params=params, json=json,
                                            timeout=self.timeout)
            except requests.RequestException as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
            if resp.status_code in self.retry_on and attempt < self.max_retries:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise requests.HTTPError(f"{resp.status_code} {method} {url}: {resp.text}")
            return resp.json() if resp.content else None
        if last_exc:
            raise last_exc
        return None

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, json: dict | None = None) -> Any:
        return self.request("POST", endpoint, json=json)

    def delete(self, endpoint: str) -> Any:
        return self.request("DELETE", endpoint)
