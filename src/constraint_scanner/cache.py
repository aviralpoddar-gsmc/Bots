from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import FullMarket, LiteMarket

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "markets.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    outcome_type TEXT NOT NULL,
    mechanism TEXT,
    probability REAL,
    last_updated_time INTEGER,
    is_resolved INTEGER NOT NULL DEFAULT 0,
    lite_json TEXT NOT NULL,
    full_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_markets_outcome ON markets(outcome_type);
CREATE INDEX IF NOT EXISTS idx_markets_resolved ON markets(is_resolved);
"""


class MarketCache:
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MarketCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def upsert_lite(self, markets: Iterable[LiteMarket]) -> int:
        rows = [
            (
                m.id,
                m.question,
                m.outcome_type,
                m.mechanism,
                m.probability,
                m.last_updated_time,
                int(m.is_resolved),
                m.model_dump_json(by_alias=True),
            )
            for m in markets
        ]
        if not rows:
            return 0
        self._conn.executemany(
            """
            INSERT INTO markets (id, question, outcome_type, mechanism, probability,
                                 last_updated_time, is_resolved, lite_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                question=excluded.question,
                outcome_type=excluded.outcome_type,
                mechanism=excluded.mechanism,
                probability=excluded.probability,
                last_updated_time=excluded.last_updated_time,
                is_resolved=excluded.is_resolved,
                lite_json=excluded.lite_json
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    def upsert_full(self, markets: Iterable[FullMarket]) -> int:
        rows = [(m.model_dump_json(by_alias=True), m.id) for m in markets]
        if not rows:
            return 0
        self._conn.executemany("UPDATE markets SET full_json=? WHERE id=?", rows)
        self._conn.commit()
        return len(rows)

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM markets")
        return cur.fetchone()[0]

    def iter_lite(self, include_resolved: bool = False) -> Iterable[LiteMarket]:
        q = "SELECT lite_json FROM markets"
        if not include_resolved:
            q += " WHERE is_resolved = 0"
        for (blob,) in self._conn.execute(q):
            yield LiteMarket.model_validate_json(blob)

    def iter_full(self, outcome_type: str | None = None) -> Iterable[FullMarket]:
        q = "SELECT full_json FROM markets WHERE full_json IS NOT NULL AND is_resolved = 0"
        params: tuple = ()
        if outcome_type:
            q += " AND outcome_type = ?"
            params = (outcome_type,)
        for (blob,) in self._conn.execute(q, params):
            yield FullMarket.model_validate_json(blob)

    def ids_needing_full(self, outcome_type: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT id FROM markets WHERE outcome_type = ? AND is_resolved = 0 AND full_json IS NULL",
            (outcome_type,),
        )
        return [row[0] for row in cur.fetchall()]

    def ids_for_outcome(self, outcome_type: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT id FROM markets WHERE outcome_type = ? AND is_resolved = 0",
            (outcome_type,),
        )
        return [row[0] for row in cur.fetchall()]
