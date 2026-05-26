"""Append-only trade-ledger writes and position aggregation.

The ledger is the source of truth. A "position" is the aggregate of all trade
rows sharing (bot_id, market_id, direction) — it is computed here, never stored.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

ENTRY = "ENTRY"
EXIT_TYPES = ("EXIT", "PARTIAL_EXIT", "RESOLUTION_CLOSE")


def record_trade(
    conn: sqlite3.Connection,
    *,
    bot_id: int,
    market_id: str,
    trade_type: str,
    direction: str,
    amount: float,
    shares: float,
    platform_bet_id: str | None = None,
    price_before: float | None = None,
    price_after: float | None = None,
    llm_estimate: float | None = None,
    reasoning: str | None = None,
    date_executed: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO trade (bot_id, market_id, platform_bet_id, trade_type, direction,
            amount, shares, price_before, price_after, llm_estimate, reasoning, date_executed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bot_id,
            market_id,
            platform_bet_id,
            trade_type,
            direction,
            amount,
            shares,
            price_before,
            price_after,
            llm_estimate,
            reasoning,
            date_executed or datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def trades_for_bot(conn: sqlite3.Connection, bot_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM trade WHERE bot_id=? ORDER BY trade_id", (bot_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def group_positions(trades: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group trade rows by (market_id, direction)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in trades:
        groups[(t["market_id"], t["direction"])].append(t)
    return groups


def summarize_position(position_trades: list[dict]) -> dict[str, Any]:
    """Derive a position summary from its trade rows."""
    direction = position_trades[0]["direction"]
    market_id = position_trades[0]["market_id"]
    entry_amount = sum(t["amount"] for t in position_trades if t["trade_type"] == ENTRY)
    exit_amount = sum(t["amount"] for t in position_trades if t["trade_type"] in EXIT_TYPES)
    entry_shares = sum(t["shares"] for t in position_trades if t["trade_type"] == ENTRY)
    exit_shares = sum(t["shares"] for t in position_trades if t["trade_type"] in EXIT_TYPES)
    net_shares = entry_shares - exit_shares
    # A position is closed when no shares remain (handles losing resolutions,
    # where exit proceeds are 0 but the shares are gone).
    status = "CLOSED" if entry_shares > 0 and net_shares <= 1e-9 else "OPEN"
    return {
        "market_id": market_id,
        "direction": direction,
        "entry_amount": entry_amount,
        "exit_amount": exit_amount,
        "net_shares": net_shares,
        "net_amount": entry_amount - exit_amount,
        "status": status,
    }


def open_positions(conn: sqlite3.Connection, bot_id: int) -> dict[str, dict]:
    """{market_id: summary} for OPEN positions only (keyed by market for the runner)."""
    out: dict[str, dict] = {}
    for (_mid, _dir), pos_trades in group_positions(trades_for_bot(conn, bot_id)).items():
        summary = summarize_position(pos_trades)
        if summary["status"] == "OPEN":
            out[summary["market_id"]] = summary
    return out
