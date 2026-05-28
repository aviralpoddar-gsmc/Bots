"""Flask dashboard server. Single page, polled every 60s.

Reads from the live SQLite store on every request — no caching. The route
handlers stay tiny; everything substantive is in `data.py` so the templates only
see plain dicts. A `/healthz` endpoint and a `/api/snapshot` JSON dump are
provided for live-monitoring tooling that doesn't want to scrape HTML.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

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
            return {"username": None, "balance": None, "latency_ms": None, "status": "UNKNOWN"}
        t0 = time.time()
        try:
            me = ManifoldClient().get_me()
            return {
                "username": me.get("username"),
                "balance": me.get("balance") or 0,
                "latency_ms": int((time.time() - t0) * 1000),
                "status": "LIVE",
            }
        except Exception as e:  # noqa: BLE001 - the dashboard must render even if the API is down
            logger.warning("system status probe failed: %s", e)
            return {
                "username": None, "balance": None,
                "latency_ms": int((time.time() - t0) * 1000),
                "status": "DEGRADED",
            }

    @app.route("/")
    def index() -> str:
        with Store() as store:
            lb = ddata.leaderboard(store)
            ctx = {
                "overview": ddata.overview(store),
                "leaderboard": lb,
                "bots": [b for b in (ddata.bot_detail(store, r["name"]) for r in lb) if b is not None],
                "equity": ddata.equity_curve(store),
                "distribution": ddata.strategy_distribution(store),
                "events": ddata.event_feed(store, limit=60),
                "system": _system_status(),
                "now": datetime.now().strftime("%H:%M:%S"),
                "today": datetime.now().strftime("%a %b %d, %Y"),
            }
        return render_template("index.html", **ctx)

    @app.route("/api/snapshot")
    def api_snapshot() -> Any:
        with Store() as store:
            return jsonify({
                "overview": ddata.overview(store),
                "leaderboard": ddata.leaderboard(store),
                "events": ddata.event_feed(store, limit=80),
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
