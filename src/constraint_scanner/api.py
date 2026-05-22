from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import httpx

from .models import FullMarket, LiteMarket

DEFAULT_BASE_URL = os.environ.get("MANIFOLD_API_URL", "http://localhost:8088")
API_KEY = os.environ.get("MANIFOLD_API_KEY")
PAGE_SIZE = 1000


def _headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Key {API_KEY}"
    return h


def _normalize_base(base: str) -> str:
    base = base.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    return base


class ManifoldAPI:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        self.base_url = _normalize_base(base_url)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=_headers(),
            timeout=timeout,
        )

    async def __aenter__(self) -> ManifoldAPI:
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def list_markets_page(
        self,
        limit: int = PAGE_SIZE,
        before: str | None = None,
        sort: str = "created-time",
    ) -> list[LiteMarket]:
        params: dict[str, str | int] = {"limit": limit, "sort": sort}
        if before:
            params["before"] = before
        r = await self._client.get("/v0/markets", params=params)
        r.raise_for_status()
        return [LiteMarket.model_validate(m) for m in r.json()]

    async def iter_markets(self, max_markets: int | None = None) -> AsyncIterator[LiteMarket]:
        before: str | None = None
        fetched = 0
        while True:
            page = await self.list_markets_page(before=before)
            if not page:
                return
            for m in page:
                yield m
                fetched += 1
                if max_markets is not None and fetched >= max_markets:
                    return
            if len(page) < PAGE_SIZE:
                return
            before = page[-1].id

    async def get_market(self, market_id: str) -> FullMarket:
        r = await self._client.get(f"/v0/market/{market_id}")
        r.raise_for_status()
        return FullMarket.model_validate(r.json())

    async def get_markets(
        self, market_ids: list[str], concurrency: int = 8
    ) -> list[FullMarket]:
        sem = asyncio.Semaphore(concurrency)

        async def one(mid: str) -> FullMarket | None:
            async with sem:
                try:
                    return await self.get_market(mid)
                except httpx.HTTPError:
                    return None

        results = await asyncio.gather(*(one(mid) for mid in market_ids))
        return [m for m in results if m is not None]
