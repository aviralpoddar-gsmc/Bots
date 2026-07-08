"""Ledger accounting for the equity_options store — `realized_and_open`.

Locks in the snapshot PnL fix: realized must capture closed round-trips AND
expired-worthless legs, while still-open positions are excluded from realized and
carried as open cost basis. Regression guard for the bug where snapshots reported
realized=0 and counted expired legs (which never get a closing row) as open.
"""

from __future__ import annotations

from quantbots.equity_options.store.db import OptionsStore


def _leg(store, ticket, symbol, side, amount, status, qty=1):
    store.record_leg(
        ticket_id=ticket, underlying="FCX", structure="bull_call_spread",
        symbol=symbol, trade_type=("EXIT" if status == "filled" and side == "SELL" else "ENTRY"),
        side=side, qty=qty, fill_price=abs(amount) / (qty * 100), amount=amount,
        broker="paper", status=status)


def test_realized_and_open_ledger_inference(tmp_path):
    with OptionsStore(tmp_path / "eo.sqlite") as store:
        # 1) closed round-trip: paid 500, sold for 700 -> realized +200
        _leg(store, "t1", "RT", "BUY", -500.0, "filled")
        _leg(store, "t1", "RT", "SELL", +700.0, "filled")
        # 2) expired worthless while HELD: paid 300, then a broker-truth settle
        #    row (amount 0, as settle_absent writes) -> realized -300
        _leg(store, "t2", "EXP", "BUY", -300.0, "filled")
        store.record_leg(ticket_id="settle-EXP", underlying="FCX", structure="settle",
                         symbol="EXP", trade_type="EXPIRY_CLOSE", side="SELL", qty=1,
                         fill_price=None, amount=0.0, broker="paper", status="settled")
        # 3) still open: paid 400, filled and held -> NOT realized, cost basis 400
        _leg(store, "t3", "OPEN", "BUY", -400.0, "filled")
        # 4) never filled -> ignored entirely
        _leg(store, "t4", "CXL", "BUY", -999.0, "canceled")

        br = store.realized_and_open()  # infer open from the ledger
        assert round(br["realized"], 2) == -100.0          # +200 - 300
        assert round(br["open_cost_basis"], 2) == 400.0
        assert br["open_positions"] == 1                   # only t3
        assert br["closed_positions"] == 3                 # t1 + t2 + its settle row
        assert br["open_symbols"] == {"OPEN"}

        # settled leg must NOT show up as an open position (the original bug)
        assert "EXP" not in store.open_positions()
        # a lapsed-unfilled order (status expired, amount zeroed by reconcile)
        # is a NON-event: no position, no realized cash
        _leg(store, "t5", "LAPSED", "BUY", 0.0, "expired")
        assert "LAPSED" not in store.open_positions()
        assert round(store.realized_and_open()["realized"], 2) == -100.0


def test_realized_and_open_with_broker_truth(tmp_path):
    """When the broker is authoritative, a symbol it no longer holds is realized
    even if the ledger still shows it open (expiry that never got a ledger row)."""
    with OptionsStore(tmp_path / "eo.sqlite") as store:
        _leg(store, "t1", "GONE", "BUY", -250.0, "filled")  # ledger thinks open...
        _leg(store, "t2", "HELD", "BUY", -600.0, "filled")

        # broker only still holds HELD -> GONE is realized (-250)
        br = store.realized_and_open(open_symbols={"HELD"})
        assert round(br["realized"], 2) == -250.0
        assert round(br["open_cost_basis"], 2) == 600.0
        assert br["open_symbols"] == {"HELD"}
        assert br["open_positions"] == 1
        assert br["closed_positions"] == 1
