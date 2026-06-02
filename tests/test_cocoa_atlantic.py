"""Tests for the cocoa Atlantic-Niño bot (Terminal 2): the ATL3 SST source parser,
the SIG_ATL3_COCOA z-score, and the cocoa_atlantic drift (sign + conviction gate)."""

from quantbots.sources import atl3
from quantbots.strategies.cocoa_atlantic import CocoaAtlanticStrategy


class FakeObs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


def test_parser_reads_trop_anomaly(monkeypatch):
    # Real CPC layout: YR MON NATL ANOM SATL ANOM TROP ANOM (TROP anom = last col).
    sample = (
        "YR MON    NATL    ANOM    SATL    ANOM    TROP    ANOM\n"
        "1982   1   25.67   -0.33   25.26   -0.34   27.38   -0.20\n"
        "2026   4   25.95    0.07   27.55    0.64   29.03    0.50\n"
    )

    class _Resp:
        text = sample
        def raise_for_status(self): pass

    monkeypatch.setattr(atl3.requests, "get", lambda *a, **k: _Resp())
    hist = atl3.fetch_history()
    assert hist[0] == ("1982-01-01T00:00:00", -0.20)
    assert hist[-1] == ("2026-04-01T00:00:00", 0.50)  # picks the TROP anomaly, not SATL/NATL


def test_drift_sign_and_gate():
    # Warm anomaly (z>0) with default sign=-1 -> bearish (mu<0).
    s = CocoaAtlanticStrategy(k=0.02, min_z=0.5, sign=-1.0)
    s.bind(FakeObs({"SIG_ATL3_COCOA": 1.5}))
    mu, detail = s.signal_drift(spot=8000.0, price_entity="CME_COCOA", T=0.5)
    assert mu < 0 and abs(mu - (-0.02 * 1.5)) < 1e-9 and detail["atl3_z"] == 1.5

    # Cool anomaly flips the sign.
    s.bind(FakeObs({"SIG_ATL3_COCOA": -1.0}))
    mu2, _ = s.signal_drift(8000.0, "CME_COCOA", 0.5)
    assert mu2 > 0

    # Below the conviction floor -> abstain.
    s.bind(FakeObs({"SIG_ATL3_COCOA": 0.2}))
    assert s.signal_drift(8000.0, "CME_COCOA", 0.5) is None

    # No signal -> abstain.
    s.bind(FakeObs({}))
    assert s.signal_drift(8000.0, "CME_COCOA", 0.5) is None


def test_sign_is_configurable():
    # The unvalidated sign must be flippable from config without code changes.
    warm = {"SIG_ATL3_COCOA": 1.5}
    neg = CocoaAtlanticStrategy(sign=-1.0); neg.bind(FakeObs(warm))
    pos = CocoaAtlanticStrategy(sign=1.0); pos.bind(FakeObs(warm))
    assert neg.signal_drift(8000.0, "CME_COCOA", 0.5)[0] < 0
    assert pos.signal_drift(8000.0, "CME_COCOA", 0.5)[0] > 0


def test_registered():
    from quantbots.strategies import get_strategy
    from quantbots.sources import get_source
    assert isinstance(get_strategy("cocoa_atlantic"), CocoaAtlanticStrategy)
    assert get_source("atl3").name == "atl3"
