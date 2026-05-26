"""Phase 2.5 — live price cache over the platform websocket (optional).

For unrealized PnL and reactive bots you want current prices without hammering
`get_market`. Pattern: subscribe -> buffer -> batch-write into the store's
market_cache. A first version can skip this entirely and just poll the markets a
bot holds; add the websocket when you scale.

Requires the `realtime` extra (`websockets`). This is a minimal scaffold: it
connects, subscribes, pings to stay alive (the server drops silent connections
after ~60s), and hands decoded contract updates to a callback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

WS_URL = "wss://manifold.mikhailtal.dev/api/ws"
PING_INTERVAL = 10  # seconds; server drops silent connections after ~60s
TOPICS = ["global/updated-contract", "global/new-comment"]

logger = logging.getLogger(__name__)

ContractUpdate = dict
Handler = Callable[[ContractUpdate], Awaitable[None] | None]


async def stream_contract_updates(
    on_update: Handler,
    *,
    cf_client_id: str,
    cf_client_secret: str,
    topics: list[str] | None = None,
) -> None:
    """Connect, subscribe, and invoke `on_update` for each contract update.

    Cloudflare Access headers are required on the websocket handshake too.
    Reconnects with backoff on drop.
    """
    import websockets  # imported lazily so the extra is optional

    headers = {
        "CF-Access-Client-Id": cf_client_id,
        "CF-Access-Client-Secret": cf_client_secret,
    }
    topics = topics or TOPICS
    backoff = 1
    while True:
        try:
            async with websockets.connect(WS_URL, additional_headers=headers) as ws:
                await ws.send(json.dumps({"type": "subscribe", "topics": topics}))
                backoff = 1
                ping = asyncio.create_task(_pinger(ws))
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "broadcast" and "contract" in (msg.get("data") or {}):
                            res = on_update(msg["data"]["contract"])
                            if asyncio.iscoroutine(res):
                                await res
                finally:
                    ping.cancel()
        except Exception as e:  # noqa: BLE001 - reconnect on any drop
            logger.warning("websocket dropped: %s; reconnecting in %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def _pinger(ws: object) -> None:
    while True:
        await asyncio.sleep(PING_INTERVAL)
        await ws.send(json.dumps({"type": "ping"}))  # type: ignore[attr-defined]
