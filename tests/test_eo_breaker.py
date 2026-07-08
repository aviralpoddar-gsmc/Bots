"""Broker-truth circuit breaker: entries must halt on anomalous account state.

Motivated by 2026-07-07: Alpaca paper re-marked the whole book to zero overnight
(-$17k equity cliff, zero fills), the ledger disagreed with the broker, and the
bot re-armed into the crater. Neither condition may ever open new positions again.
"""

from quantbots.equity_options.breaker import entry_halt_reason
from quantbots.equity_options.store.db import OptionsStore


def test_breaker_trips_on_equity_cliff():
    reason = entry_halt_reason(equity=79750.76, last_equity=97037.96,
                               ledger_open_symbols=set(), broker_open_symbols=set(),
                               open_order_symbols=set(), max_drop_pct=0.08)
    assert reason is not None and "equity" in reason


def test_breaker_trips_on_ledger_broker_mismatch():
    reason = entry_halt_reason(equity=100_000.0, last_equity=100_000.0,
                               ledger_open_symbols={"GDX260918P00100000"},
                               broker_open_symbols=set(), open_order_symbols=set(),
                               max_drop_pct=0.08)
    assert reason is not None and "ledger" in reason


def test_breaker_quiet_when_healthy():
    assert entry_halt_reason(equity=99_000.0, last_equity=100_000.0,
                             ledger_open_symbols={"GDX260918P00100000"},
                             broker_open_symbols={"GDX260918P00100000"},
                             open_order_symbols=set(), max_drop_pct=0.08) is None


def test_breaker_ignores_drop_below_threshold():
    assert entry_halt_reason(equity=95_000.0, last_equity=100_000.0,
                             ledger_open_symbols=set(), broker_open_symbols=set(),
                             open_order_symbols=set(), max_drop_pct=0.08) is None


def test_breaker_trips_at_exact_threshold():
    reason = entry_halt_reason(equity=92_000.0, last_equity=100_000.0,
                               ledger_open_symbols=set(), broker_open_symbols=set(),
                               open_order_symbols=set(), max_drop_pct=0.08)
    assert reason is not None and "equity" in reason


def test_breaker_spares_working_order_symbols():
    """A resting unfilled limit order is ledger-open but not a broker position —
    that must NOT read as a ledger/broker mismatch (it false-alarmed every tick)."""
    assert entry_halt_reason(equity=100_000.0, last_equity=100_000.0,
                             ledger_open_symbols={"PAAS260821C00044000"},
                             broker_open_symbols=set(),
                             open_order_symbols={"PAAS260821C00044000"},
                             max_drop_pct=0.08) is None


def test_halt_latch_persists_and_clears(tmp_path):
    db = tmp_path / "eo.sqlite"
    with OptionsStore(db) as store:
        assert store.halt_reason() is None
        store.trip_halt("equity down 17.8% vs prior close")
        assert "17.8%" in store.halt_reason()
        store.trip_halt("second trip overwrites")  # idempotent upsert
        assert store.halt_reason() == "second trip overwrites"
    with OptionsStore(db) as store:  # survives reconnect (the point of the latch)
        assert store.halt_reason() == "second trip overwrites"
        store.clear_halt()
        assert store.halt_reason() is None
