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
    # profit-target applies to a KEPT (momentum-sleeve) position we ride to its target;
    # non-kept green positions wind down at break-even before reaching the target.
    s = _gdx_vertical(long_upl=1500.0)
    d = exit_decisions([s], rules=DEFAULT_MANAGE_RULES, keep={"GDX"})
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
    # small profit, far expiry: NOT closed when breakeven_close is off
    s = _gdx_vertical(long_upl=200.0)
    rules = {**DEFAULT_MANAGE_RULES, "breakeven_close": False}
    assert exit_decisions([s], rules=rules) == []


def test_legacy_winddown_at_breakeven():
    # A green legacy position (not in `keep`) winds down at break-even...
    s = _gdx_vertical(long_upl=50.0)    # uPnL > 0
    d = exit_decisions([s], rules=DEFAULT_MANAGE_RULES, keep=set())
    assert d and "break-even" in d[0].reason
    # ...but a kept (momentum-sleeve) green position is RIDDEN, not closed.
    d2 = exit_decisions([s], rules=DEFAULT_MANAGE_RULES, keep={"GDX"})
    assert d2 == []
    # a losing legacy position is NOT force-closed (held to recover / stop rule)
    s_loss = _gdx_vertical(long_upl=-100.0)
    assert exit_decisions([s_loss], rules=DEFAULT_MANAGE_RULES, keep=set()) == []


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


class _PagedHTTP:
    """Fake AlpacaHTTP returning canned /v2/orders pages, recording the params."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, endpoint, params=None):
        self.calls.append(dict(params or {}))
        return self.pages.pop(0)


def test_list_all_orders_paginates_past_page_cap():
    from quantbots.equity_options.execution.alpaca import AlpacaPaperBroker

    broker = AlpacaPaperBroker(key="k", secret="s")
    page1 = [{"id": "b", "submitted_at": "2026-07-07T14:00:00Z"},
             {"id": "a", "submitted_at": "2026-07-06T14:00:00Z"}]
    page2 = [{"id": "z", "submitted_at": "2026-07-01T14:00:00Z"}]
    broker._http = _PagedHTTP([page1, page2])
    orders = broker.list_all_orders(status="all", page_size=2)
    assert [o["id"] for o in orders] == ["b", "a", "z"]
    assert "until" not in broker._http.calls[0]
    assert broker._http.calls[1]["until"] == "2026-07-06T14:00:00Z"


AGED = "2026-01-01T00:00:00+00:00"  # old enough to clear settle_absent's age guard


def test_settle_absent_books_ghost_legs_closed(tmp_path):
    db = tmp_path / "eo.sqlite"
    long_sym = build_occ("GDX", FAR, "put", 100)
    short_sym = build_occ("GDX", FAR, "put", 88)
    held_sym = build_occ("WPM", FAR, "put", 90)
    with OptionsStore(db) as store:
        store.record_leg(ticket_id="T1", underlying="GDX", structure="bear_put_spread",
                         symbol=long_sym, trade_type="ENTRY", side="BUY", qty=6,
                         fill_price=16.6, amount=-9960.0, broker="paper", status="filled",
                         date_executed=AGED)
        store.record_leg(ticket_id="T1", underlying="GDX", structure="bear_put_spread",
                         symbol=short_sym, trade_type="ENTRY", side="SELL", qty=6,
                         fill_price=7.8, amount=4680.0, broker="paper", status="filled",
                         date_executed=AGED)
        store.record_leg(ticket_id="T2", underlying="WPM", structure="long_put",
                         symbol=held_sym, trade_type="ENTRY", side="BUY", qty=2,
                         fill_price=5.0, amount=-1000.0, broker="paper", status="filled",
                         date_executed=AGED)
        assert set(store.open_positions()) == {long_sym, short_sym, held_sym}
        n = store.settle_absent(broker_open_symbols={held_sym}, open_order_symbols=set())
        assert n == 2
        assert set(store.open_positions()) == {held_sym}  # broker-held leg untouched
        r = store.realized_and_open(open_symbols={held_sym})
        assert abs(r["realized"] - (-9960.0 + 4680.0)) < 1e-6  # full GDX premium realized
        # idempotent: a second pass settles nothing
        assert store.settle_absent(broker_open_symbols={held_sym},
                                   open_order_symbols=set()) == 0


def test_settle_absent_spares_symbols_with_open_orders(tmp_path):
    db = tmp_path / "eo.sqlite"
    sym = build_occ("FNV", FAR, "put", 270)
    with OptionsStore(db) as store:
        store.record_leg(ticket_id="T1", underlying="FNV", structure="long_put",
                         symbol=sym, trade_type="ENTRY", side="BUY", qty=2,
                         fill_price=60.0, amount=-12000.0, broker="paper", status="filled",
                         date_executed=AGED)
        n = store.settle_absent(broker_open_symbols=set(), open_order_symbols={sym})
        assert n == 0 and set(store.open_positions()) == {sym}


def test_settle_absent_skips_recent_legs(tmp_path):
    """A leg entered seconds ago may just not show at the broker yet (fill race) —
    never settle it on one observation."""
    db = tmp_path / "eo.sqlite"
    sym = build_occ("FNV", FAR, "put", 270)
    with OptionsStore(db) as store:
        store.record_leg(ticket_id="T1", underlying="FNV", structure="long_put",
                         symbol=sym, trade_type="ENTRY", side="BUY", qty=2,
                         fill_price=60.0, amount=-12000.0, broker="paper", status="filled")
        n = store.settle_absent(broker_open_symbols=set(), open_order_symbols=set())
        assert n == 0 and set(store.open_positions()) == {sym}


def test_settle_absent_refuses_mass_settlement(tmp_path):
    """One glitched empty positions() response must not zero the whole book."""
    from quantbots.equity_options.store.db import MassSettleRefused

    db = tmp_path / "eo.sqlite"
    syms = [build_occ("GDX", FAR, "put", 100 + i) for i in range(6)]
    with OptionsStore(db) as store:
        for i, sym in enumerate(syms):
            store.record_leg(ticket_id=f"T{i}", underlying="GDX", structure="long_put",
                             symbol=sym, trade_type="ENTRY", side="BUY", qty=1,
                             fill_price=1.0, amount=-100.0, broker="paper", status="filled",
                             date_executed=AGED)
        import pytest
        with pytest.raises(MassSettleRefused):
            store.settle_absent(broker_open_symbols=set(), open_order_symbols=set())
        assert len(store.open_positions()) == 6  # nothing was settled
        n = store.settle_absent(broker_open_symbols=set(), open_order_symbols=set(),
                                force=True)
        assert n == 6 and store.open_positions() == {}


def test_reconcile_fills_zeroes_expired_unfilled(tmp_path):
    """A lapsed day-limit order moved no cash — its estimated amount must not
    survive as a phantom realized loss."""
    db = tmp_path / "eo.sqlite"
    sym = build_occ("AEM", FAR, "put", 160)
    with OptionsStore(db) as store:
        store.record_leg(ticket_id="T1", underlying="AEM", structure="long_put",
                         symbol=sym, trade_type="ENTRY", side="BUY", qty=3,
                         fill_price=9.1, amount=-2730.0, broker="paper", status="pending_new")
        store.reconcile_fills([{
            "client_order_id": "T1", "status": "expired",
            "legs": [{"symbol": sym, "filled_avg_price": None, "status": "expired"}],
        }])
        row = store.conn.execute("SELECT status, amount FROM option_trade").fetchone()
        assert row["status"] == "expired" and row["amount"] == 0.0
        # a non-event: neither an open position nor realized cash
        assert store.open_positions() == {}
        assert store.realized_and_open(open_symbols=set())["realized"] == 0.0


def test_expired_unfilled_does_not_close_a_held_position(tmp_path):
    """A filled position plus a later lapsed re-entry order on the same symbol
    must remain OPEN (the old expired->closed inference wrongly closed it)."""
    db = tmp_path / "eo.sqlite"
    sym = build_occ("GDX", FAR, "put", 90)
    with OptionsStore(db) as store:
        store.record_leg(ticket_id="T1", underlying="GDX", structure="long_put",
                         symbol=sym, trade_type="ENTRY", side="BUY", qty=10,
                         fill_price=15.6, amount=-15600.0, broker="paper", status="filled",
                         date_executed=AGED)
        store.record_leg(ticket_id="T2", underlying="GDX", structure="long_put",
                         symbol=sym, trade_type="ENTRY", side="BUY", qty=5,
                         fill_price=15.0, amount=0.0,  # reconcile zeroed the lapsed order
                         broker="paper", status="expired", date_executed=AGED)
        pos = store.open_positions()
        assert sym in pos and pos[sym]["net_contracts"] == 10


def test_list_all_orders_raises_on_stalled_pagination():
    """If the broker ignores `until`, fail loudly instead of spinning forever in launchd."""
    from quantbots.equity_options.execution.alpaca import AlpacaPaperBroker

    broker = AlpacaPaperBroker(key="k", secret="s")
    page = [{"id": "a", "submitted_at": "2026-07-07T14:00:00Z"},
            {"id": "b", "submitted_at": "2026-07-07T14:00:00Z"}]

    class _StuckHTTP:
        def get(self, endpoint, params=None):
            return list(page)  # same full page forever

    broker._http = _StuckHTTP()
    import pytest
    with pytest.raises(RuntimeError, match="pagination"):
        broker.list_all_orders(status="all", page_size=2)


def test_account_pnl_fails_fast_on_malformed_payload():
    """A malformed /v2/account response must raise, not silently disarm the breaker."""
    from quantbots.equity_options.execution.alpaca import AlpacaPaperBroker

    broker = AlpacaPaperBroker(key="k", secret="s")

    class _BadHTTP:
        def get(self, endpoint, params=None):
            return {"status": "ACTIVE"}  # no equity / last_equity

    broker._http = _BadHTTP()
    import pytest
    with pytest.raises(KeyError):
        broker.account_pnl()


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
