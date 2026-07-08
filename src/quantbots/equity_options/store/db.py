"""SQLite store for the equity_options package — its OWN database file, never the
clone's. Append-only `option_trade` ledger; positions derived by aggregation.
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB = Path(os.environ.get("EQUITY_OPTIONS_DB", _REPO_ROOT / "data" / "equity_options.sqlite"))
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _nonevent(t: dict) -> bool:
    """Rows that represent no economic event: never-accepted submits and
    lapsed-unfilled orders (no cash moved, no position taken). An expired row
    keeps its submit-time fill_price estimate, so zero `amount` (set by
    reconcile_fills) is the discriminator — a partial fill carries real cash."""
    return (t["status"] in ("canceled", "rejected", "intended")
            or (t["status"] == "expired" and t["amount"] == 0.0))


class MassSettleRefused(RuntimeError):
    """settle_absent would zero more legs than the safety cap allows — one
    glitched empty /v2/positions response must not corrupt the whole ledger."""


class OptionsStore:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        # Two writer processes share this DB (daily cycle + intraday monitor).
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
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
                leg_status = leg.get("status") or order_status
                amount = None
                if fill is not None:
                    sign = -1 if row["side"] == "BUY" else 1
                    amount = sign * row["qty"] * fill * 100
                elif leg_status in ("expired", "canceled"):
                    amount = 0.0  # lapsed unfilled: no cash moved — kill the submit-time estimate
                self.conn.execute(
                    "UPDATE option_trade SET status=?, "
                    "fill_price=COALESCE(?, fill_price), amount=COALESCE(?, amount) "
                    "WHERE trade_id=?",
                    (leg_status, fill, amount, row["trade_id"]))
                updated += 1
        self.conn.commit()
        return updated

    def settle_absent(self, *, broker_open_symbols: set[str],
                      open_order_symbols: set[str], max_legs: int = 5,
                      min_age_hours: float = 1.0, force: bool = False) -> int:
        """Book a closing row for ledger-open legs the broker no longer holds.

        Alpaca paper can remove positions without a closing fill (overnight
        re-marks settled the whole book to zero on 2026-07-07). No cash moved, so
        the offsetting row carries amount 0.0 — the entry premium becomes realized
        loss, matching broker equity. Symbols the broker still holds, or that have
        a working order, are left alone. Returns the number of legs settled.

        Two guards (both bypassed by `force`, for a human-confirmed wipeout):
          - legs whose latest ledger activity is younger than `min_age_hours` are
            skipped — a just-filled entry may not show at the broker yet;
          - more than `max_legs` candidates raises MassSettleRefused — a single
            glitched empty /v2/positions response must not zero the whole book.
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=min_age_hours)).isoformat()
        candidates = []
        for sym, pos in self.open_positions().items():
            if sym in broker_open_symbols or sym in open_order_symbols:
                continue
            last = self.conn.execute(
                "SELECT MAX(date_executed) d FROM option_trade WHERE symbol=?",
                (sym,)).fetchone()["d"]
            if not force and last > cutoff:
                continue
            candidates.append((sym, pos))
        if not force and len(candidates) > max_legs:
            raise MassSettleRefused(
                f"refusing to settle {len(candidates)} legs at once (cap {max_legs}) — "
                "verify the broker positions read and re-run `eo reconcile --force-settle`")
        for sym, pos in candidates:
            net = pos["net_contracts"]
            self.record_leg(
                ticket_id=f"settle-{sym}", underlying=pos["underlying"],
                structure="settle", symbol=sym, trade_type="EXPIRY_CLOSE",
                side="SELL" if net > 0 else "BUY", qty=abs(net), fill_price=None,
                amount=0.0, broker="paper", status="settled",
                reasoning="broker-truth settle: leg absent at broker, no closing fill")
        return len(candidates)

    # --- circuit-breaker latch --------------------------------------------

    def trip_halt(self, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO breaker_halt (id, reason, tripped_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET reason=excluded.reason, "
            "tripped_at=excluded.tripped_at", (reason, _now()))
        self.conn.commit()

    def clear_halt(self) -> None:
        self.conn.execute("DELETE FROM breaker_halt")
        self.conn.commit()

    def halt_reason(self) -> str | None:
        row = self.conn.execute("SELECT reason FROM breaker_halt").fetchone()
        return row["reason"] if row else None

    def open_positions(self) -> dict[str, dict]:
        """{symbol: {net_contracts, net_cash, ...}} for OPEN legs (net qty != 0).

        Signed contracts: BUY adds, SELL subtracts. A position is closed when net is 0.
        """
        agg: dict[str, dict] = defaultdict(lambda: {"net_contracts": 0, "net_cash": 0.0,
                                                     "underlying": None, "multiplier": 100})
        for t in self.trades():
            if _nonevent(t):
                continue
            sign = 1 if t["side"] == "BUY" else -1
            a = agg[t["symbol"]]
            a["net_contracts"] += sign * t["qty"]
            a["net_cash"] += t["amount"]
            a["underlying"] = t["underlying"]
            a["multiplier"] = t["multiplier"]
        # Closure comes only from EXIT fills or settle_absent rows — a lapsed
        # unfilled order (status "expired", no fill) is a non-event, never a close.
        return {sym: v for sym, v in agg.items() if v["net_contracts"] != 0}

    def realized_and_open(self, open_symbols: set[str] | None = None) -> dict:
        """Split the filled/expired ledger into REALIZED (closed) vs OPEN cash flows.

        The ledger records every cash flow (`amount`: <0 paid, >0 received). A
        position is realized once it is no longer held; what remains open carries a
        cost basis whose mark-to-market is the caller's `unrealized` (broker truth).

        `open_symbols`: OCC symbols the BROKER still holds (authoritative). When given,
        any ledger leg whose symbol is NOT held counts as realized. When None (no
        broker available), a symbol is inferred open iff its net signed contracts != 0.
        Lapsed-unfilled orders are non-events (zero cash, no position) — see _nonevent.

        Returns {realized, open_cost_basis, open_positions, closed_positions,
        open_symbols} where realized = closed round-trips + expired-worthless losses.
        """
        by_sym: dict[str, dict] = defaultdict(
            lambda: {"net": 0, "cash": 0.0, "tickets": set()})
        for t in self.trades():
            if _nonevent(t):
                continue
            s = by_sym[t["symbol"]]
            s["net"] += (1 if t["side"] == "BUY" else -1) * t["qty"]
            s["cash"] += t["amount"]
            s["tickets"].add(t["ticket_id"])
        if open_symbols is None:
            open_set = {sym for sym, v in by_sym.items() if v["net"] != 0}
        else:
            open_set = {sym for sym in by_sym if sym in open_symbols}
        realized = sum(v["cash"] for sym, v in by_sym.items() if sym not in open_set)
        open_cost = sum(-v["cash"] for sym, v in by_sym.items() if sym in open_set)
        open_tk = {tk for sym, v in by_sym.items() if sym in open_set for tk in v["tickets"]}
        closed_tk = {tk for v in by_sym.values() for tk in v["tickets"]} - open_tk
        return {"realized": realized, "open_cost_basis": open_cost,
                "open_positions": len(open_tk), "closed_positions": len(closed_tk),
                "open_symbols": open_set}

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
