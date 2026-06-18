"""SQLite store for the equity_options package — its OWN database file, never the
clone's. Append-only `option_trade` ledger; positions derived by aggregation.
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB = Path(os.environ.get("EQUITY_OPTIONS_DB", _REPO_ROOT / "data" / "equity_options.sqlite"))
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class OptionsStore:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA_PATH.read_text())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> OptionsStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- contract cache --------------------------------------------------

    def upsert_contract(self, row: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO option_contract (symbol, underlying, expiry, strike, kind,
                multiplier, last_mid, last_iv, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_mid=excluded.last_mid, last_iv=excluded.last_iv,
                updated_at=excluded.updated_at
            """,
            (row["symbol"], row["underlying"], str(row["expiry"]), row["strike"],
             row["kind"], row.get("multiplier", 100), row.get("mid"), row.get("iv"), _now()),
        )
        self.conn.commit()

    # --- ledger ----------------------------------------------------------

    def record_leg(self, *, ticket_id: str, underlying: str, structure: str, symbol: str,
                   trade_type: str, side: str, qty: int, fill_price: float | None,
                   amount: float, broker: str, status: str, broker_order_id: str | None = None,
                   multiplier: int = 100, edge: float | None = None,
                   forecast_vol: float | None = None, reasoning: str | None = None,
                   date_executed: str | None = None) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO option_trade (ticket_id, underlying, structure, symbol, trade_type,
                side, qty, fill_price, amount, multiplier, broker, broker_order_id, status,
                edge, forecast_vol, reasoning, date_executed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, underlying, structure, symbol, trade_type, side, qty, fill_price,
             amount, multiplier, broker, broker_order_id, status, edge, forecast_vol,
             reasoning, date_executed or _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def trades(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM option_trade ORDER BY trade_id").fetchall()]

    def reconcile_fills(self, broker_orders: list[dict]) -> int:
        """Sync the ledger to ACTUAL broker fills (broker = source of truth).

        Matches broker orders to ledger rows by ticket_id (= client_order_id) and
        symbol, and updates each leg's status + fill_price + amount from the real
        fill. This closes the gap where entries were recorded at estimated mids at
        submit time. Returns the number of rows updated.
        """
        updated = 0
        for o in broker_orders:
            ticket = o.get("client_order_id")
            if not ticket:
                continue
            order_status = o.get("status")
            legs = o.get("legs") or [o]  # single-leg orders carry fields at top level
            for leg in legs:
                sym = leg.get("symbol")
                if not sym:
                    continue
                fap = leg.get("filled_avg_price")
                fill = float(fap) if fap else None
                row = self.conn.execute(
                    "SELECT trade_id, side, qty FROM option_trade "
                    "WHERE ticket_id=? AND symbol=? AND trade_type IN ('ENTRY','EXIT')",
                    (ticket, sym)).fetchone()
                if row is None:
                    continue
                amount = None
                if fill is not None:
                    sign = -1 if row["side"] == "BUY" else 1
                    amount = sign * row["qty"] * fill * 100
                self.conn.execute(
                    "UPDATE option_trade SET status=?, "
                    "fill_price=COALESCE(?, fill_price), amount=COALESCE(?, amount) "
                    "WHERE trade_id=?",
                    (leg.get("status") or order_status, fill, amount, row["trade_id"]))
                updated += 1
        self.conn.commit()
        return updated

    def open_positions(self) -> dict[str, dict]:
        """{symbol: {net_contracts, net_cash, ...}} for OPEN legs (net qty != 0).

        Signed contracts: BUY adds, SELL subtracts. A position is closed when net is 0.
        """
        agg: dict[str, dict] = defaultdict(lambda: {"net_contracts": 0, "net_cash": 0.0,
                                                     "underlying": None, "multiplier": 100})
        for t in self.trades():
            if t["status"] in ("canceled", "rejected", "intended"):
                continue
            sign = 1 if t["side"] == "BUY" else -1
            a = agg[t["symbol"]]
            a["net_contracts"] += sign * t["qty"]
            a["net_cash"] += t["amount"]
            a["underlying"] = t["underlying"]
            a["multiplier"] = t["multiplier"]
        return {sym: v for sym, v in agg.items() if v["net_contracts"] != 0}

    # --- snapshots -------------------------------------------------------

    def write_greeks_snapshot(self, *, net_delta: float, net_gamma: float, net_vega: float,
                              net_theta: float, positions: int, premium_at_risk: float) -> None:
        self.conn.execute(
            """
            INSERT INTO greeks_snapshot (snapshot_ts, net_delta, net_gamma, net_vega,
                net_theta, positions, premium_at_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_ts) DO NOTHING
            """,
            (_now(), net_delta, net_gamma, net_vega, net_theta, positions, premium_at_risk),
        )
        self.conn.commit()

    def write_pnl_snapshot(self, *, realized: float, unrealized: float, premium_at_risk: float,
                           open_positions: int, closed_positions: int,
                           snapshot_date: str | None = None) -> None:
        snapshot_date = snapshot_date or datetime.now(UTC).date().isoformat()
        self.conn.execute(
            """
            INSERT INTO option_pnl_snapshot (snapshot_date, realized_pnl, unrealized_pnl,
                pnl, premium_at_risk, open_positions, closed_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_date) DO UPDATE SET
                realized_pnl=excluded.realized_pnl, unrealized_pnl=excluded.unrealized_pnl,
                pnl=excluded.pnl, premium_at_risk=excluded.premium_at_risk,
                open_positions=excluded.open_positions, closed_positions=excluded.closed_positions
            """,
            (snapshot_date, realized, unrealized, realized + unrealized, premium_at_risk,
             open_positions, closed_positions),
        )
        self.conn.commit()
