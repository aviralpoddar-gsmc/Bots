"""Flask dashboard server. Single page, polled every 60s.

Reads from the live SQLite store on every request — no caching. The `/refresh`
endpoint triggers a `quantbots refresh` cache update (slow, ~30s) so users can
recompute mark-to-market without dropping to the CLI.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, url_for

from ..config import load_bots
from ..manifold.client import ManifoldClient
from ..store.db import Store
from . import data as ddata

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent


def create_app(*, fetch_balance: bool = True) -> Flask:
    """Build the Flask app. `fetch_balance=False` skips the /me probe (tests)."""
    app = Flask(__name__, template_folder=str(HERE / "templates"))

    def _system_status() -> dict[str, Any]:
        if not fetch_balance:
            return {"username": None, "balance": None, "totalDeposits": None,
                    "latency_ms": None, "status": "UNKNOWN"}
        t0 = time.time()
        try:
            me = ManifoldClient().get_me()
            return {
                "username": me.get("username"),
                "balance": me.get("balance") or 0,
                "totalDeposits": me.get("totalDeposits") or 0,
                "latency_ms": int((time.time() - t0) * 1000),
                "status": "LIVE",
            }
        except Exception as e:  # noqa: BLE001 - render even with API down
            logger.warning("system status probe failed: %s", e)
            return {"username": None, "balance": None, "totalDeposits": None,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "status": "DEGRADED"}

    @app.route("/")
    def index() -> str:
        system = _system_status()
        with Store() as store:
            account = {"balance": system["balance"], "totalDeposits": system["totalDeposits"]}
            lb = ddata.leaderboard(store)
            ctx = {
                "overview": ddata.overview(store, account=account),
                "leaderboard": lb,
                "bots": [b for b in (ddata.bot_detail(store, r["name"]) for r in lb) if b is not None],
                "equity": ddata.equity_curve(store),
                "distribution": ddata.strategy_distribution(store),
                "events": ddata.event_feed(store, limit=60),
                "system": system,
                "cache_age_s": ddata.cache_age_seconds(store),
                "now": datetime.now().strftime("%H:%M:%S"),
                "today": datetime.now().strftime("%a %b %d, %Y"),
            }
        return render_template("index.html", **ctx)

    @app.route("/refresh", methods=["POST", "GET"])
    def refresh_cache() -> Any:
        """Run a cache refresh against the Manifold API, then bounce back to /."""
        t0 = time.time()
        try:
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
            logger.info("dashboard refresh: %d markets in %.1fs", len(markets), time.time() - t0)
        except Exception as e:  # noqa: BLE001 - report but don't crash
            logger.exception("dashboard refresh failed")
            return jsonify({"ok": False, "error": str(e), "elapsed_s": time.time() - t0}), 500
        return redirect(url_for("index"))

    @app.route("/api/snapshot")
    def api_snapshot() -> Any:
        system = _system_status()
        with Store() as store:
            account = {"balance": system["balance"], "totalDeposits": system["totalDeposits"]}
            return jsonify({
                "overview": ddata.overview(store, account=account),
                "leaderboard": ddata.leaderboard(store),
                "events": ddata.event_feed(store, limit=80),
                "cache_age_s": ddata.cache_age_seconds(store),
            })

    @app.route("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    app.jinja_env.filters["humanize_age"] = ddata.humanize_age
    app.jinja_env.filters["short_clock"] = ddata.short_clock
    app.jinja_env.globals["humanize_age"] = ddata.humanize_age
    app.jinja_env.globals["short_clock"] = ddata.short_clock
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Block; run the dashboard server until interrupted."""
    app = create_app()
    app.run(host=host, port=port, debug=debug, use_reloader=False)
