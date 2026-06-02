"""Tests for the drought_cotton bot: USDM DSCI parser, the deseasonalized
SIG_COTTON_DROUGHT (week-of-year baseline — the bug-prone part), and the strategy."""

import datetime as dt

from quantbots.processing import signals
from quantbots.sources import usdm
from quantbots.strategies.drought_cotton import DroughtCottonStrategy


class FakeObs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


def test_parser_reads_dsci(monkeypatch):
    payload = [
        {"name": "Texas", "mapDate": "2000-01-04T00:00:00", "dsci": 223},
        {"name": "Texas", "mapDate": "2026-05-26T00:00:00", "dsci": 188},
    ]

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    monkeypatch.setattr(usdm.requests, "get", lambda *a, **k: _Resp())
    hist = usdm.fetch_history("48")
    assert hist[0] == ("2000-01-04T00:00:00", 223.0)
    assert hist[-1] == ("2026-05-26T00:00:00", 188.0)


def _seasonal_hist(latest_value):
    """Weekly DSCI 2008→2026 with a summer peak + per-year spread, last point overridden."""
    import math
    out = []
    d = dt.date(2008, 1, 7)
    while d <= dt.date(2026, 5, 26):
        wk = d.isocalendar()[1]
        seasonal = 120 + 120 * math.exp(-((wk - 30) ** 2) / (2 * 8.0 ** 2))  # bell at week ~30
        spread = ((d.year - 2008) % 5 - 2) * 20  # year-to-year variance so std>0
        out.append((f"{d.isoformat()}T00:00:00", seasonal + spread))
        d += dt.timedelta(days=7)
    out[-1] = (out[-1][0], float(latest_value))
    return out


def _woy_mean(hist, ts):
    wk = dt.date.fromisoformat(ts[:10]).isocalendar()[1]
    vals = [v for t, v in hist[:-1] if dt.date.fromisoformat(t[:10]).isocalendar()[1] == wk]
    return sum(vals) / len(vals), wk


def test_signal_deseasonalizes_by_week_of_year(monkeypatch):
    # A value AT its week-of-year norm -> z≈0 (not flagged), even though summer DSCI
    # is high in absolute terms. A value far ABOVE the week norm -> large positive z.
    base_hist = _seasonal_hist(0)
    norm, _ = _woy_mean(base_hist, base_hist[-1][0])

    monkeypatch.setattr(usdm, "fetch_history", lambda *a, **k: _seasonal_hist(norm))
    o = signals.compute_cotton_drought()[0]
    assert o.entity == "SIG_COTTON_DROUGHT" and abs(o.value) < 0.3  # seasonal-normal -> ~no signal

    monkeypatch.setattr(usdm, "fetch_history", lambda *a, **k: _seasonal_hist(norm + 120))
    o2 = signals.compute_cotton_drought()[0]
    assert o2.value > 1.5  # well above week norm -> strong drought z


def test_strategy_sign_and_gate():
    s = DroughtCottonStrategy(k=0.03, min_z=0.7, sign=1.0)
    # Drought (z>0) -> bullish cotton (mu>0).
    s.bind(FakeObs({"SIG_COTTON_DROUGHT": 2.0}))
    mu, detail = s.signal_drift(spot=70.0, price_entity="CME_COTTON", T=0.5)
    assert mu > 0 and abs(mu - 0.06) < 1e-9 and detail["drought_z"] == 2.0
    # Wetter than normal (z<0) -> bearish.
    s.bind(FakeObs({"SIG_COTTON_DROUGHT": -1.5}))
    assert s.signal_drift(70.0, "CME_COTTON", 0.5)[0] < 0
    # Near seasonal normal -> abstain.
    s.bind(FakeObs({"SIG_COTTON_DROUGHT": 0.3}))
    assert s.signal_drift(70.0, "CME_COTTON", 0.5) is None


def test_registered():
    from quantbots.sources import get_source
    from quantbots.strategies import get_strategy
    assert isinstance(get_strategy("drought_cotton"), DroughtCottonStrategy)
    assert get_source("usdm").name == "usdm"
