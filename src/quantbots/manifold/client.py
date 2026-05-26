"""Phase 0 — the connection layer. Correctness here is non-negotiable.

Standard Manifold v0 client, hard-wired to the private clone. This is the one
piece ported faithfully from the parent (TAL) repo: auth header shapes, the two
Cloudflare Access headers, the v0 payload/response shapes, and client-side rate
limiting.

SAFETY: the base URL is a constant. There is deliberately no `platform` argument
and no way to point this client at public manifold.markets. Do not add one.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Clone only. Never parametrize this away.
CLONE_BASE_URL = "https://manifold.mikhailtal.dev/api/v0"


class ManifoldClient:
    """Standard Manifold v0 client for the private clone. Clone-only by design."""

    RATE_LIMIT = 500  # requests ...
    WINDOW = 60  # ... per this many seconds, per IP

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cf_client_id: str | None = None,
        cf_client_secret: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        api_key = api_key or os.environ.get("MANIFOLD_CLONE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key. Pass api_key= or set MANIFOLD_CLONE_API_KEY."
            )
        cf_client_id = cf_client_id or os.environ.get("CF_ACCESS_CLIENT_ID")
        cf_client_secret = cf_client_secret or os.environ.get("CF_ACCESS_CLIENT_SECRET")
        if not (cf_client_id and cf_client_secret):
            raise ValueError(
                "Missing Cloudflare Access token. Set CF_ACCESS_CLIENT_ID and "
                "CF_ACCESS_CLIENT_SECRET (the clone is behind CF Access)."
            )

        self.base_url = CLONE_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_on = {429, 500, 502, 503, 504}
        self._req_times: deque[float] = deque(maxlen=self.RATE_LIMIT)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "quantbots/0.1",
                "Accept": "application/json",
                "Authorization": f"Key {api_key}",  # bet/sell auth
                "CF-Access-Client-Id": cf_client_id,  # Cloudflare Access
                "CF-Access-Client-Secret": cf_client_secret,
            }
        )

    # --- internals -------------------------------------------------------

    def _rate_limit(self) -> None:
        now = time.time()
        start = now - self.WINDOW
        while self._req_times and self._req_times[0] < start:
            self._req_times.popleft()
        if len(self._req_times) >= self.RATE_LIMIT:
            sleep_for = self._req_times[0] - start
            logger.debug("rate limit: sleeping %.2fs", sleep_for)
            time.sleep(max(sleep_for, 0))
        self._req_times.append(now)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
    ) -> Any:
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(self.max_retries + 1):
            resp = self.session.request(
                method, url, params=params, json=data, timeout=self.timeout
            )
            if resp.status_code in self.retry_on and attempt < self.max_retries:
                backoff = 2**attempt
                logger.warning(
                    "%s %s -> %s, retry %d/%d in %ds",
                    method,
                    endpoint,
                    resp.status_code,
                    attempt + 1,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp.json()
        raise requests.RequestException(f"max retries exceeded for {method} {endpoint}")

    # --- reads (CF headers always required; key not strictly needed) -----

    def get_me(self) -> dict:
        """Authenticated account: {id, username, balance, ...}. The best first
        call — a 200 proves both the API key and Cloudflare Access work."""
        return self._request("GET", "me")

    def get_market(self, market_id: str) -> dict:
        return self._request("GET", f"market/{market_id}")

    def get_market_by_slug(self, slug: str) -> dict:
        return self._request("GET", f"slug/{slug}")

    def list_markets(self, limit: int = 1000, before: str | None = None) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        return self._request("GET", "markets", params=params)

    def search_markets(self, term: str, limit: int = 100) -> list[dict]:
        return self._request("GET", "search-markets", params={"term": term, "limit": limit})

    def get_positions(self, market_id: str) -> list[dict]:
        return self._request("GET", f"market/{market_id}/positions")

    def get_bets(self, **params: Any) -> list[dict]:
        return self._request("GET", "bets", params=params)

    # --- writes ----------------------------------------------------------

    def place_bet(
        self,
        market_id: str,
        outcome: str,
        amount: float,
        limit_prob: float | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Place a bet. `outcome` is "YES"/"NO", `amount` is integer mana.
        `limit_prob` (0.01–0.99) makes it a limit order. `dry_run=True` validates
        auth + payload without moving mana — the safest possible first write.

        Response keys: betId, probBefore, probAfter, shares, amount.
        """
        data: dict[str, Any] = {
            "contractId": market_id,
            "outcome": outcome,
            "amount": int(amount),
        }
        if limit_prob is not None:
            data["limitProb"] = round(limit_prob, 2)  # 0.01–0.99
        if dry_run:
            data["dryRun"] = True
        return self._request("POST", "bet", data=data)

    def batch_bet(self, bets: list[dict]) -> Any:
        """Up to 50 bets. Each: {contractId, outcome, amount, limitProb?}."""
        return self._request("POST", "batch-bet", data={"bets": bets})

    def sell_shares(
        self,
        market_id: str,
        outcome: str | None = None,
        shares: float | None = None,
    ) -> dict:
        """Sell shares. Omit both args to sell the whole position."""
        data = {
            k: v
            for k, v in {"outcome": outcome, "shares": shares}.items()
            if v is not None
        }
        return self._request("POST", f"market/{market_id}/sell", data=data)

    def batch_sell(self, sells: list[dict]) -> Any:
        """Up to 50 sells. Each: {contractId, outcome?, shares?}."""
        return self._request("POST", "batch-sell", data={"sells": sells})
