"""Position reconstruction, exit rules, close orders, and fill reconciliation."""

from datetime import UTC, date, datetime, timedelta

from quantbots.equity_options.config import DEFAULT_MANAGE_RULES
from quantbots.equity_options.manage import build_close_order, exit_decisions
from quantbots.equity_options.occ import build_occ
from quantbots.equity_options.positions import (
    StructureLeg,
    StructurePosition,
    held_underlyings,
    structures_from_broker,
)
from quantbots.equity_options.store.db import OptionsStore

FAR = datetime.now(UTC).date() + timedelta(days=90)
NEAR = datetime.now(UTC).date() + timedelta(days=5)


def _gdx_vertical(expiry=FAR, long_upl=0.0, short_upl=0.0, long_mv=6000.0, short_mv=-2000.0):
    # Bear put spread: long 6x 100 put @16, short 6x 88 put @7.8. net debit = $4,920.
    return StructurePosition(underlying="GDX", expiry=expiry, legs=[
        StructureLeg(build_occ("GDX", expiry, "put", 100), 6, "put", 100, 16.0, long_mv, long_upl),
        StructureLeg(build_occ("GDX", expiry, "put", 88), -6, "put", 88, 7.8, short_mv, short_upl),
    ])


def test_structures_from_broker_groups_legs():
    raw = [
        {"symbol": build_occ("GDX", FAR, "put", 100), "qty": "6", "avg_entry_price": "16",
         "market_value": "6000", "unrealized_pl": "100"},
        {"symbol": build_occ("GDX", FAR, "put", 88), "qty": "-6", "avg_entry_price": "7.8",
         "market_value": "-2000", "unrealized_pl": "50"},
        {"symbol": "AAPL", "qty": "10", "avg_entry_price": "200",  # equity row -> skipped
         "market_value": "2100", "unrealized_pl": "100"},
    ]
    structs = structures_from_broker(raw)
    assert len(structs) == 1
    s = structs[0]
    assert s.is_vertical and s.contracts == 6 and s.width == 12
    assert s.net_cost == 16 * 6 * 100 - 7.8 * 6 * 100   # debit
    assert held_underlyings(structs) == {"GDX"}


def test_max_profit_and_fractions():
    s = _gdx_vertical(long_upl=1500.0)   # net_cost 4920, width 12 -> max_profit 2280
    assert abs(s.max_profit - 2280) < 1e-6
    assert s.profit_fraction() > 0.6
    s2 = _gdx_vertical(long_upl=-3000.0)
    assert s2.loss_fraction() > 0.6


def test_exit_profit_target():
    s = _gdx_vertical(long_upl=1500.0)
    d = exit_decisions([s], rules=DEFAULT_MANAGE_RULES)
    assert d and "profit target" in d[0].reason


def test_exit_stop_loss():
    s = _gdx_vertical(long_upl=-3100.0)
    d = exit_decisions([s], rules=DEFAULT_MANAGE_RULES)
    assert d and "stop loss" in d[0].reason


def test_exit_dte_guard():
    s = _gdx_vertical(expiry=NEAR)
    d = exit_decisions([s], rules=DEFAULT_MANAGE_RULES)
    assert d and "dte" in d[0].reason


def test_exit_none_when_midrange():
    s = _gdx_vertical(long_upl=200.0)   # small profit, far expiry
    assert exit_decisions([s], rules=DEFAULT_MANAGE_RULES) == []


def test_assignment_guard():
    # Short 88 put ITM (spot below 88), within assignment_dte but beyond min_hold_dte.
    expiry = datetime.now(UTC).date() + timedelta(days=18)
    s = _gdx_vertical(expiry=expiry)
    d = exit_decisions([s], rules=DEFAULT_MANAGE_RULES, spots={"GDX": 85.0})
    assert d and "assignment" in d[0].reason


def test_build_close_order_reverses_legs():
    s = _gdx_vertical()
    order = build_close_order(s)
    assert order.qty == 6 and order.structure == "close"
    sides = {l.symbol[-15:]: l.side for l in order.legs}  # by strike/kind suffix
    # The long (qty>0) leg becomes SELL; the short (qty<0) becomes BUY. All closing.
    assert all(l.closing for l in order.legs)
    assert sorted(l.side for l in order.legs) == ["BUY", "SELL"]
    assert order.limit_price > 0


def test_reconcile_fills_updates_ledger(tmp_path):
    db = tmp_path / "eo.sqlite"
    sym = build_occ("GDX", FAR, "put", 100)
    with OptionsStore(db) as store:
        store.record_leg(ticket_id="T1", underlying="GDX", structure="long_put", symbol=sym,
                         trade_type="ENTRY", side="BUY", qty=6, fill_price=16.0,
                         amount=-16.0 * 6 * 100, broker="paper", status="pending_new")
        n = store.reconcile_fills([{
            "client_order_id": "T1", "status": "filled",
            "legs": [{"symbol": sym, "filled_avg_price": "16.6", "status": "filled"}],
        }])
        assert n == 1
        row = store.conn.execute("SELECT status, fill_price, amount FROM option_trade").fetchone()
        assert row["status"] == "filled"
        assert abs(row["fill_price"] - 16.6) < 1e-9
        assert abs(row["amount"] - (-16.6 * 6 * 100)) < 1e-6
