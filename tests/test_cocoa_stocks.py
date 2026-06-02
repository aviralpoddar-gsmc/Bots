"""Tests for the cocoa_stocks bot: ICE .xls parsing (by row label), the
SIG_COCOA_STOCK_Z signal, and the strategy (low stock -> bullish)."""

import xlrd

from quantbots.processing import signals
from quantbots.sources import ice_stocks
from quantbots.strategies.cocoa_stocks import CocoaStocksStrategy


class FakeSheet:
    def __init__(self, rows):
        self.rows = rows
        self.nrows = len(rows)
        self.ncols = max(len(r) for r in rows)

    def cell_value(self, i, j):
        return self.rows[i][j] if j < len(self.rows[i]) else ""


class FakeBook:
    def __init__(self, sheet):
        self._s = sheet

    def sheet_by_index(self, i):
        return self._s


class FakeObs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


_ROWS = [
    ["Date: 6/1/2026"],
    [""],
    [""],
    ["Cocoa Certified Stock by Port & Growth - Bags"],
    [""],
    ["", "DR", "NY", "Total", "", "DR", "NY", "Total"],
    ["Colombia  Group B", 35304.0, 30695.0, 65999.0],
    # ... (other origins omitted) ...
    ["Total Bags", 235428.0, 51311.0, 286739.0, "Total Lots", 1559.0, 349.0, 1908.0],
    ["Total Bags Reported By ICE FUTURES U.S. Licensed Warehouses"],
    ["Port of Delaware River", 2735344.0],
    ["GRAND TOTAL:", 2871963.0],
]


def test_parse_matches_by_label(monkeypatch):
    monkeypatch.setattr(xlrd, "open_workbook", lambda **k: FakeBook(FakeSheet(_ROWS)))
    cert, grand = ice_stocks._parse(b"ignored")
    assert cert == 286739.0   # "Total Bags" Total column (not the 1908 Lots, not 235428 DR)
    assert grand == 2871963.0  # "GRAND TOTAL:" row, not the "Total Bags Reported By..." header


def test_signal_zscores_stock(monkeypatch):
    # A history sitting around 300k with the latest LOW (250k) -> negative z.
    # Needs >=24 points for the signal to compute.
    hist = [(f"{2023 + i // 12}-{i % 12 + 1:02d}-01T00:00:00", 300000.0 + (i % 3) * 5000)
            for i in range(30)]
    hist += [("2026-01-01T00:00:00", 250000.0)]
    monkeypatch.setattr(ice_stocks, "fetch_history", lambda *a, **k: hist)
    o = signals.compute_cocoa_stocks()[0]
    assert o.entity == "SIG_COCOA_STOCK_Z" and o.value < -1.0  # low stock -> negative z


def test_strategy_low_stock_is_bullish():
    s = CocoaStocksStrategy(k=0.03, min_z=0.7, sign=-1.0)
    # Low stock (z<0) -> bullish (mu>0).
    s.bind(FakeObs({"SIG_COCOA_STOCK_Z": -2.0}))
    mu, detail = s.signal_drift(spot=8000.0, price_entity="CME_COCOA", T=0.5)
    assert mu > 0 and abs(mu - 0.06) < 1e-9 and detail["stock_z"] == -2.0
    # Ample stock (z>0) -> bearish.
    s.bind(FakeObs({"SIG_COCOA_STOCK_Z": 1.5}))
    assert s.signal_drift(8000.0, "CME_COCOA", 0.5)[0] < 0
    # Near normal -> abstain.
    s.bind(FakeObs({"SIG_COCOA_STOCK_Z": 0.4}))
    assert s.signal_drift(8000.0, "CME_COCOA", 0.5) is None


def test_registered():
    from quantbots.sources import get_source
    from quantbots.strategies import get_strategy
    assert isinstance(get_strategy("cocoa_stocks"), CocoaStocksStrategy)
    assert get_source("ice_stocks").name == "ice_stocks"
