"""Flask dashboard server.

Serves the React app (built bundle at web/dist) as the root, plus a JSON+SSE
API at /api/*. Reads from the live SQLite store on every request — no caching.
The /api/refresh endpoint triggers a cache refresh against the Manifold clone
so users can recompute mark-to-market without dropping to the CLI.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from ..manifold.client import ManifoldClient
from ..store.db import Store
from . import data as ddata

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
WEB_DIST = REPO_ROOT / "web" / "dist"

# How often the background thread re-pulls live market prices into the cache so
# per-bot marks stay current (the fleet-level headline already comes straight
# from Manifold's portfolio endpoint and needs no cache). A full pull is ~64 API
# calls — trivial against the 500/min budget — so a couple of minutes keeps
# per-bot PnL within a few mana of reality without hammering the API every tick.
PRICE_REFRESH_INTERVAL_S = 120


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def refresh_market_cache() -> int:
    """Pull every market from the clone and upsert it into the price cache.

    Returns the number of markets refreshed. Shared by the manual /api/refresh
    endpoint and the background refresher so there's one code path.
    """
    client = ManifoldClient()
    markets, before = [], None
    while True:
        page = client.list_markets(limit=1000, before=before)
        if not page:
            break
        markets.extend(page)
        before = page[-1]["id"]
        if len(page) < 1000:
            break
    with Store() as store:
        store.upsert_markets(markets)
    return len(markets)


def _start_price_refresher(interval_s: int = PRICE_REFRESH_INTERVAL_S) -> threading.Thread:
    """Spawn a daemon thread that keeps the price cache fresh.

    Refreshes once immediately on boot (so the first page load has live marks),
    then every `interval_s`. Failures are logged and retried next cycle — a
    transient API hiccup must never kill the thread.
    """
    def loop() -> None:
        while True:
            try:
                n = refresh_market_cache()
                logger.info("background price refresh: %d markets", n)
            except Exception as e:  # noqa: BLE001 - keep the thread alive
                logger.warning("background price refresh failed: %s", e)
            time.sleep(interval_s)

    t = threading.Thread(target=loop, name="price-refresher", daemon=True)
    t.start()
    return t


def create_app(*, fetch_balance: bool = True) -> Flask:
    """Build the Flask app. `fetch_balance=False` skips the /me probe (tests)."""
    # Static folder = built web bundle. We register catch-all routes below to
    # support client-side routing (every non-/api path returns index.html).
    app = Flask(__name__, static_folder=None)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #

    def _live_account() -> dict[str, Any]:
        """Probe the clone once for everything the fleet trades on: identity,
        balance, authoritative portfolio economics, and the profit history.

        The whole fleet shares one account, so these numbers ARE the fleet's
        headline numbers — we pull them straight from Manifold rather than
        recomputing from the local ledger/cache.
        """
        empty = {
            "system": {
                "username": None, "balance": None, "totalDeposits": None,
                "latency_ms": None, "status": "UNKNOWN",
            },
            "portfolio": None, "history": [],
        }
        if not fetch_balance:
            return empty
        t0 = time.time()
        try:
            client = ManifoldClient()
            me = client.get_me()
            uid = me["id"]
            portfolio = client.get_portfolio(uid)
            try:
                history = client.get_portfolio_history(uid, period="allTime")
            except Exception as e:  # noqa: BLE001 - history is non-critical
                logger.warning("portfolio history probe failed: %s", e)
                history = []
            return {
                "system": {
                    "username": me.get("username"),
                    "balance": portfolio.get("balance") or me.get("balance") or 0,
                    "totalDeposits": portfolio.get("totalDeposits") or me.get("totalDeposits") or 0,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "status": "LIVE",
                },
                "portfolio": portfolio,
                "history": history,
            }
        except Exception as e:  # noqa: BLE001 - render even with API down
            logger.warning("account probe failed: %s", e)
            out = dict(empty)
            out["system"] = {
                **empty["system"],
                "latency_ms": int((time.time() - t0) * 1000),
                "status": "DEGRADED",
            }
            return out

    def _snapshot_payload() -> dict[str, Any]:
        live = _live_account()
        system, portfolio, history = live["system"], live["portfolio"], live["history"]
        # Equity curve from Manifold's profit history; fall back to the local
        # per-trade ledger replay only if the API didn't return history.
        with Store() as store:
            equity = ddata.portfolio_equity(history) or ddata.equity_curve(store)
            return {
                "overview": ddata.overview(store, portfolio=portfolio),
                "leaderboard": ddata.leaderboard(store),
                "events": ddata.event_feed(store, limit=80),
                "equity": equity,
                "distribution": ddata.strategy_distribution(store),
                "system": system,
                "cache_age_s": ddata.cache_age_seconds(store),
                "ts": _now_iso(),
            }

    # --------------------------------------------------------------------- #
    # API
    # --------------------------------------------------------------------- #

    @app.get("/api/snapshot")
    def api_snapshot() -> Any:
        return jsonify(_snapshot_payload())

    @app.get("/api/stream")
    def api_stream() -> Response:
        """Server-Sent Events: pushes a fresh snapshot every 5 seconds.

        Browsers natively auto-reconnect with the EventSource API, so we don't
        need to handle reconnects server-side. Each frame is a complete snapshot
        (no diffing) — payload is ~30-50KB, well under SSE practical limits.
        """
        def stream():
            # Initial frame immediately so the client paints on connect.
            payload = json.dumps(_snapshot_payload(), default=str)
            yield f"data: {payload}\n\n"
            while True:
                time.sleep(5)
                try:
                    payload = json.dumps(_snapshot_payload(), default=str)
                    yield f"data: {payload}\n\n"
                except GeneratorExit:
                    return
                except Exception as e:  # noqa: BLE001
                    logger.warning("stream tick failed: %s", e)
                    # Tell the client; EventSource will reconnect.
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    return

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/bots/<name>")
    def api_bot(name: str) -> Any:
        with Store() as store:
            row = ddata.bot_detail(store, name)
        if row is None:
            abort(404, description=f"bot {name!r} not found in config/bots.yaml")
        return jsonify(row)

    @app.get("/api/feed")
    def api_feed() -> Any:
        limit = max(1, min(int(request.args.get("limit", 200)), 500))
        with Store() as store:
            return jsonify(ddata.event_feed(store, limit=limit))

    @app.get("/api/strategies")
    def api_strategies() -> Any:
        with Store() as store:
            return jsonify(ddata.strategy_index(store))

    @app.get("/api/strategies/<name>")
    def api_strategy(name: str) -> Any:
        with Store() as store:
            row = ddata.strategy_detail(store, name)
        if row is None:
            abort(404, description=f"strategy {name!r} not registered")
        return jsonify(row)

    @app.get("/api/markets")
    def api_markets() -> Any:
        page = int(request.args.get("page", 1))
        size = int(request.args.get("size", 50))
        q = request.args.get("q") or None
        mr_raw = request.args.get("min_resolvability")
        mr = float(mr_raw) if mr_raw else None
        with Store() as store:
            return jsonify(ddata.markets_index(
                store, page=page, size=size, q=q, min_resolvability=mr,
            ))

    @app.post("/api/refresh")
    def api_refresh() -> Any:
        t0 = time.time()
        try:
            n = refresh_market_cache()
            elapsed = time.time() - t0
            logger.info("dashboard refresh: %d markets in %.1fs", n, elapsed)
            return jsonify({"ok": True, "markets": n, "elapsed_s": elapsed})
        except Exception as e:  # noqa: BLE001
            logger.exception("dashboard refresh failed")
            return jsonify({"ok": False, "error": str(e), "elapsed_s": time.time() - t0}), 500

    @app.get("/api/healthz")
    def api_healthz() -> Any:
        return jsonify({"ok": True, "ts": _now_iso()})

    # --------------------------------------------------------------------- #
    # React app: static bundle + SPA fallback
    # --------------------------------------------------------------------- #

    @app.get("/")
    @app.get("/<path:path>")
    def serve_spa(path: str = "") -> Any:
        # Anything under /api/* is handled above; if we land here it 404s.
        if path.startswith("api/"):
            abort(404)
        # Try the requested file (assets/, favicon, etc).
        if path:
            target = WEB_DIST / path
            if target.is_file():
                return send_from_directory(WEB_DIST, path)
        index = WEB_DIST / "index.html"
        if not index.exists():
            # Build hasn't been run. Tell the user what to do, in HTML, plainly.
            return Response(
                "<!doctype html><html><head>"
                "<meta charset='utf-8'><title>quantbots dashboard — build required</title>"
                "<style>body{background:#08090c;color:#e8ecef;font-family:ui-monospace,Menlo,monospace;"
                "padding:48px;line-height:1.6;}h1{color:#00d9ff;font-size:18px;}code{background:#1b1f28;"
                "padding:2px 6px;border-radius:4px;color:#00d9ff;}</style></head>"
                "<body><h1>dashboard bundle not built</h1>"
                "<p>The React app hasn't been built yet. Run:</p>"
                "<pre>cd web &amp;&amp; bun install &amp;&amp; bun run build</pre>"
                "<p>Then refresh this page. API endpoints under <code>/api/*</code> are live.</p>"
                "</body></html>",
                status=503,
                content_type="text/html",
            )
        return send_from_directory(WEB_DIST, "index.html")

    # Backwards-compat: old GET /refresh used to bounce back to /. Keep the
    # write-side semantics (refresh) but go through /api/refresh.
    @app.get("/refresh")
    def legacy_refresh() -> Any:
        return api_refresh()

    # Keep per-bot marks fresh in the background. Guarded by fetch_balance so
    # the test app (fetch_balance=False) never spawns a thread or hits the API.
    if fetch_balance and not app.config.get("_price_refresher_started"):
        _start_price_refresher()
        app.config["_price_refresher_started"] = True

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Block; run the dashboard server until interrupted."""
    app = create_app()
    # threaded=True so SSE doesn't block other requests on the same process.
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
