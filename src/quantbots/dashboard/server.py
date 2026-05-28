"""Flask dashboard server. Single page, polled every 60s.

Reads from the live SQLite store on every request — no caching. The route
handlers stay tiny; everything substantive is in `data.py` so the templates only
see plain dicts. A `/healthz` endpoint and a `/api/snapshot` JSON dump are
provided for live-monitoring tooling that doesn't want to scrape HTML.
"""

from __future__ import annotations

import logging
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
    """Build the Flask app. `fetch_balance=False` skips the /me call (useful for
    tests / when API creds aren't loaded)."""
    app = Flask(__name__, template_folder=str(HERE / "templates"))

    def _account() -> dict[str, Any] | None:
        if not fetch_balance:
            return None
        try:
            me = ManifoldClient().get_me()
            return {"username": me.get("username"), "balance": me.get("balance") or 0}
        except Exception as e:  # noqa: BLE001 - dashboard must render even if API is down
            logger.warning("could not fetch account: %s", e)
            return None

    @app.route("/")
    def index() -> str:
        with Store() as store:
            ctx = {
                "overview": ddata.overview(store),
                "leaderboard": ddata.leaderboard(store),
                "bots": [
                    b for b in (ddata.bot_detail(store, r["name"])
                                for r in ddata.leaderboard(store))
                    if b is not None
                ],
                "activity": ddata.activity_feed(store, limit=30),
                "account": _account(),
                "refreshed_at": datetime.now().strftime("%H:%M:%S"),
                "humanize_age": ddata.humanize_age,
            }
        return render_template("index.html", **ctx)

    @app.route("/api/snapshot")
    def api_snapshot() -> Any:
        with Store() as store:
            return jsonify({
                "overview": ddata.overview(store),
                "leaderboard": ddata.leaderboard(store),
                "activity": ddata.activity_feed(store, limit=50),
            })

    @app.route("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    # Templates need `humanize_age` everywhere — register as a Jinja filter too.
    app.jinja_env.filters["humanize_age"] = ddata.humanize_age
    app.jinja_env.globals["humanize_age"] = ddata.humanize_age
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Block; run the dashboard server until interrupted."""
    app = create_app()
    app.run(host=host, port=port, debug=debug, use_reloader=False)
