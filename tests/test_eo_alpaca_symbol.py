"""OCC symbol round-trip + Alpaca order payload shaping (no network)."""

from datetime import date

import pytest

from quantbots.equity_options.execution.base import OptionOrder, OrderLeg
from quantbots.equity_options.occ import build_occ, parse_occ


def test_occ_build_known():
    assert build_occ("AAPL", date(2024, 1, 19), "call", 100.0) == "AAPL240119C00100000"
    assert build_occ("SPY", date(2026, 6, 30), "put", 512.5) == "SPY260630P00512500"


def test_occ_round_trip():
    for und, d, kind, strike in [("FCX", date(2026, 9, 18), "call", 42.5),
                                 ("GDX", date(2026, 12, 18), "put", 60.0)]:
        sym = build_occ(und, d, kind, strike)
        occ = parse_occ(sym)
        assert occ.underlying == und and occ.kind == kind
        assert occ.expiry == d and occ.strike == pytest.approx(strike)


def test_alpaca_single_leg_payload(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    from quantbots.equity_options.execution.alpaca import AlpacaPaperBroker

    broker = AlpacaPaperBroker()  # init builds a session only; no network call
    sym = build_occ("FCX", date(2026, 9, 18), "call", 42.5)
    order = OptionOrder(underlying="FCX", structure="long_call",
                        legs=[OrderLeg(sym, "BUY")], qty=2, limit_price=1.25)
    p = broker._payload(order)
    assert p["order_class"] == "simple" and p["symbol"] == sym
    assert p["side"] == "buy" and p["qty"] == "2" and p["type"] == "limit"


def test_alpaca_mleg_payload(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    from quantbots.equity_options.execution.alpaca import AlpacaPaperBroker

    broker = AlpacaPaperBroker()
    lo = build_occ("FCX", date(2026, 9, 18), "call", 40.0)
    hi = build_occ("FCX", date(2026, 9, 18), "call", 45.0)
    order = OptionOrder(underlying="FCX", structure="bull_call_spread",
                        legs=[OrderLeg(lo, "BUY"), OrderLeg(hi, "SELL")], qty=1,
                        limit_price=1.80)
    p = broker._payload(order)
    assert p["order_class"] == "mleg" and len(p["legs"]) == 2
    assert p["legs"][0]["position_intent"] == "buy_to_open"
    assert p["legs"][1]["position_intent"] == "sell_to_open"
