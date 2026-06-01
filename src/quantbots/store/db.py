"""SQLite store: connection, schema bootstrap, and the bot / market_cache rows.

Trade-ledger writes and position aggregation live in `trades.py`; PnL formulas
live in `pnl.py`. The `Store` object ties them together so callers have one
handle.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import pnl as pnl_mod
from . import trades as trades_mod

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = Path(os.environ.get("QUANTBOTS_DB", _REPO_ROOT / "data" / "quantbots.sqlite"))
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA_PATH.read_text())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- bots ------------------------------------------------------------

    def upsert_bot(self, name: str, strategy: str, config: dict | None = None,
                   enabled: bool = True) -> int:
        cfg = json.dumps(config or {})
        self.conn.execute(
            """
            INSERT INTO bot (name, enabled, strategy, config)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                enabled=excluded.enabled,
                strategy=excluded.strategy,
                config=excluded.config
            """,
            (name, int(enabled), strategy, cfg),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT bot_id FROM bot WHERE name=?", (name,)).fetchone()
        return int(row["bot_id"])

    def get_bot(self, name: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM bot WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    # --- trade ledger (delegates to trades.py) ---------------------------

    def record_trade(self, **kwargs: Any) -> int:
        return trades_mod.record_trade(self.conn, **kwargs)

    def trades_for_bot(self, bot_id: int) -> list[dict]:
        return trades_mod.trades_for_bot(self.conn, bot_id)

    def open_positions(self, bot_id: int) -> dict[str, dict]:
        """{market_id: position-summary} for OPEN positions, derived from the ledger."""
        return trades_mod.open_positions(self.conn, bot_id)

    def open_position_legs(self, bot_id: int) -> dict[tuple[str, str], dict]:
        """{(market_id, direction): summary} — every OPEN leg distinctly (the maker
        holds two-sided positions that open_positions would collapse)."""
        return trades_mod.open_position_legs(self.conn, bot_id)

    # --- market cache ----------------------------------------------------

    def upsert_markets(self, markets: list[dict]) -> int:
        rows = []
        for m in markets:
            rows.append(
                (
                    m["id"],
                    m.get("question"),
                    m.get("probability"),
                    m.get("totalLiquidity"),
                    int(bool(m.get("isResolved"))),
                    m.get("resolution"),
                    m.get("closeTime"),
                    m.get("lastUpdatedTime"),
                    json.dumps(m),
                    _now(),
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO market_cache (market_id, question, probability, total_liquidity,
                is_resolved, resolution, close_time, last_updated_time, raw_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                question=excluded.question,
                probability=excluded.probability,
                total_liquidity=excluded.total_liquidity,
                is_resolved=excluded.is_resolved,
                resolution=excluded.resolution,
                close_time=excluded.close_time,
                last_updated_time=excluded.last_updated_time,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def load_open_markets(self) -> list[dict]:
        """Cached, unresolved markets as raw Manifold dicts."""
        rows = self.conn.execute(
            "SELECT raw_json FROM market_cache WHERE is_resolved = 0"
        ).fetchall()
        return [json.loads(r["raw_json"]) for r in rows]

    def get_cached_market(self, market_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT raw_json FROM market_cache WHERE market_id=?", (market_id,)
        ).fetchone()
        return json.loads(row["raw_json"]) if row else None

    # --- observations (ingested external data) ---------------------------

    def upsert_observations(self, observations: list) -> int:
        """Insert/replace observations. Accepts Observation dataclasses or dicts."""
        rows = []
        for o in observations:
            d = o.as_row() if hasattr(o, "as_row") else o
            rows.append(
                (
                    d["source"], d["entity"], d["ts"], d.get("value"),
                    d.get("text"), json.dumps(d.get("payload") or {}), _now(),
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO observations (source, entity, ts, value, text, payload, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, entity, ts) DO UPDATE SET
                value=excluded.value, text=excluded.text,
                payload=excluded.payload, ingested_at=excluded.ingested_at
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def load_observations(self, entity: str | None = None, source: str | None = None,
                          since: str | None = None, limit: int = 1000) -> list[dict]:
        q = "SELECT * FROM observations WHERE 1=1"
        params: list[Any] = []
        if entity:
            q += " AND entity = ?"; params.append(entity)
        if source:
            q += " AND source = ?"; params.append(source)
        if since:
            q += " AND ts >= ?"; params.append(since)
        q += " ORDER BY ts DESC LIMIT ?"; params.append(limit)
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def latest_observation(self, entity: str, source: str | None = None) -> dict | None:
        rows = self.load_observations(entity=entity, source=source, limit=1)
        return rows[0] if rows else None

    def known_entities(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT entity FROM observations").fetchall()
        return [r["entity"] for r in rows]

    # --- pnl (delegates to pnl.py) ---------------------------------------

    def current_prob(self, market_id: str) -> float | None:
        row = self.conn.execute(
            "SELECT probability FROM market_cache WHERE market_id=?", (market_id,)
        ).fetchone()
        return row["probability"] if row else None

    def bot_pnl(self, bot_id: int) -> dict:
        """Realized + unrealized PnL summary for a bot, using cached prices."""
        trades = self.trades_for_bot(bot_id)
        return pnl_mod.bot_pnl(trades, self.current_prob)

    def write_snapshot(self, bot_id: int, snapshot_date: str | None = None) -> dict:
        summary = self.bot_pnl(bot_id)
        snapshot_date = snapshot_date or datetime.now(UTC).date().isoformat()
        self.conn.execute(
            """
            INSERT INTO pnl_snapshot (bot_id, snapshot_date, realized_pnl, unrealized_pnl,
                pnl, total_invested, open_positions, closed_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bot_id, snapshot_date) DO UPDATE SET
                realized_pnl=excluded.realized_pnl,
                unrealized_pnl=excluded.unrealized_pnl,
                pnl=excluded.pnl,
                total_invested=excluded.total_invested,
                open_positions=excluded.open_positions,
                closed_positions=excluded.closed_positions
            """,
            (
                bot_id,
                snapshot_date,
                summary["realized_pnl"],
                summary["unrealized_pnl"],
                summary["pnl"],
                summary["total_invested"],
                summary["open_positions"],
                summary["closed_positions"],
            ),
        )
        self.conn.commit()
        return summary

    def leaderboard(self, snapshot_date: str | None = None) -> list[dict]:
        if snapshot_date:
            rows = self.conn.execute(
                """
                SELECT b.name, s.* FROM pnl_snapshot s JOIN bot b USING (bot_id)
                WHERE s.snapshot_date = ? ORDER BY s.pnl DESC
                """,
                (snapshot_date,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT b.name, s.* FROM pnl_snapshot s JOIN bot b USING (bot_id)
                WHERE s.snapshot_date = (SELECT MAX(snapshot_date) FROM pnl_snapshot)
                ORDER BY s.pnl DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
