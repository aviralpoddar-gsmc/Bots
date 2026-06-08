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
        timeout: int = 60,
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
            try:
                resp = self.session.request(
                    method, url, params=params, json=data, timeout=self.timeout
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt >= self.max_retries:
                    raise
                backoff = 2**attempt
                logger.warning(
                    "%s %s -> %s, retry %d/%d in %ds",
                    method, endpoint, type(e).__name__,
                    attempt + 1, self.max_retries, backoff,
                )
                time.sleep(backoff)
                continue
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

    def get_portfolio(self, user_id: str | None = None) -> dict:
        """Authoritative account economics, computed server-side by the clone.

        Returns: {balance, investmentValue, totalDeposits, loanTotal,
        dailyProfit, ...}. `investmentValue` is Manifold's own mark-to-market of
        every open position at live prices — the number to trust for "Invested"
        and for profit (balance + investmentValue − totalDeposits), rather than
        recomputing it from a possibly-stale local price cache.

        Omit `user_id` to use the authenticated account (one extra /me call).
        """
        if user_id is None:
            user_id = self.get_me()["id"]
        return self._request("GET", "get-user-portfolio", params={"userId": user_id})

    def get_portfolio_history(
        self, user_id: str | None = None, period: str = "allTime"
    ) -> list[dict]:
        """Time series of portfolio snapshots (each with a server-computed
        `profit` field). `period` ∈ {daily, weekly, monthly, allTime}. Used to
        draw the equity curve straight from Manifold instead of replaying the
        local ledger."""
        if user_id is None:
            user_id = self.get_me()["id"]
        return self._request(
            "GET", "get-user-portfolio-history",
            params={"userId": user_id, "period": period},
        )

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

    def get_open_limit_orders(
        self, market_id: str | None = None, user_id: str | None = None
    ) -> list[dict]:
        """List the account's RESTING (unfilled, uncancelled) limit orders.

        Authoritative live view of the maker's outstanding quotes. Each carries:
        id, contractId, outcome, limitProb, orderAmount (total), amount
        (filled-so-far), shares, isFilled, isCancelled, fills, expiresAt,
        createdTime. Includes orders on closed/resolved markets. Defaults to the
        current account when `user_id` is omitted.
        """
        if user_id is None:
            user_id = self.get_me()["id"]
        params: dict[str, Any] = {"userId": user_id, "kinds": "open-limit"}
        if market_id is not None:
            params["contractId"] = market_id
        return self.get_bets(**params)

    # --- writes ----------------------------------------------------------

    def place_bet(
        self,
        market_id: str,
        outcome: str,
        amount: float,
        limit_prob: float | None = None,
        expires_millis_after: int | None = None,
        expires_at: int | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Place a bet. `outcome` is "YES"/"NO", `amount` is integer mana.
        `limit_prob` (0.01–0.99) makes it a limit order. `dry_run=True` validates
        auth + payload without moving mana — the safest possible first write.

        `expires_millis_after` / `expires_at` are integer **milliseconds** and
        only apply to limit orders (no effect without `limit_prob`). They give a
        resting order a TTL: the server expires the unfilled remainder after the
        deadline (precedence `expiresAt ?? now + expiresMillisAfter`; very large
        values are rejected by the server's MAX_EXPIRES_AT ceiling). An expired
        order reads back as `isCancelled=True` — there is no separate `isExpired`
        flag; already-filled shares are kept as a position. This is the maker's
        TTL re-quote primitive (post fresh quotes each cycle, let stale ones
        self-expire).

        Response keys: betId, probBefore, probAfter, shares, amount, orderAmount,
        isFilled, fills.
        """
        data: dict[str, Any] = {
            "contractId": market_id,
            "outcome": outcome,
            "amount": int(amount),
        }
        if limit_prob is not None:
            data["limitProb"] = round(limit_prob, 2)  # 0.01–0.99, whole-percent
        if expires_millis_after is not None:
            data["expiresMillisAfter"] = int(expires_millis_after)
        if expires_at is not None:
            data["expiresAt"] = int(expires_at)
        if dry_run:
            data["dryRun"] = True
        return self._request("POST", "bet", data=data)

    def add_liquidity(self, market_id: str, amount: float, dry_run: bool = False) -> dict:
        """Subsidize a market's CPMM pool with `amount` mana (integer), deepening
        it so trades move price less per mana. Returns the LiquidityProvision.
        The subsidy is returned to the provider at resolution (refunded on CANCEL).

        NOTE: unlike /bet, the clone's add-liquidity endpoint has NO server-side
        dry-run — it rejects an unrecognized `dryRun` key with HTTP 400. So
        dry_run=True is a purely LOCAL no-op preview (no network call): it returns a
        synthetic record and never touches the pool. Callers that want to validate
        against the server must do a real (small) add.
        """
        if dry_run:
            return {"dryRun": True, "contractId": market_id, "amount": int(amount)}
        return self._request("POST", f"market/{market_id}/add-liquidity",
                             data={"amount": int(amount)})

    def batch_bet(self, bets: list[dict]) -> Any:
        """Up to 50 bets. Each: {contractId, outcome, amount,
        limitProb?, expiresMillisAfter?, expiresAt?}. The maker posts both legs
        of a quote in one call, e.g.
        [{contractId, outcome:"YES", amount, limitProb:f-s, expiresMillisAfter:ttl},
         {contractId, outcome:"NO",  amount, limitProb:f+s, expiresMillisAfter:ttl}].
        """
        return self._request("POST", "batch-bet", data={"bets": bets})

    def cancel_bet(self, bet_id: str) -> dict:
        """Cancel the unfilled remainder of an open limit order by its BET id
        (the id returned when the limit order was placed — NOT the contract id).
        Already-filled shares are kept as a position. Returns the LimitBet.

        The maker's anti-toxic-flow / re-quote control: pull a stale or
        run-over quote instead of waiting for its TTL to expire.
        """
        return self._request("POST", f"bet/cancel/{bet_id}")

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

    def post_comment(self, contract_id: str, markdown: str) -> dict:
        """Post a markdown comment on a market. Used for trade-justification
        comments. Callers should wrap in try/except — a comment failure must
        never unwind a real bet."""
        return self._request("POST", "comment", data={
            "contractId": contract_id,
            "markdown": markdown,
        })
