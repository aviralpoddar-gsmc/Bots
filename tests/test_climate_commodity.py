from quantbots.sources import available as available_sources
from quantbots.sources import get_source
from quantbots.strategies import available, get_strategy
from quantbots.strategies.commodity_futures import CommodityFuturesStrategy
from quantbots.strategies.enso import EnsoStrategy


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeObs:
    def __init__(self, data):
        self.data = data

    def latest_observation(self, entity, source=None):
        return self.data.get(entity)


# --- registries -----------------------------------------------------------

def test_new_registrations():
    assert "noaa" in available_sources()
    assert {"enso", "commodity_futures"} <= set(available())


# --- NOAA source ----------------------------------------------------------

def test_noaa_parses_latest_oni(monkeypatch):
    txt = "  SEAS  YR   TOTAL   ANOM\n  NDJ 2025  26.50  -0.30\n  DJF 2026  26.80   0.11\n"
    monkeypatch.setattr("quantbots.sources.noaa.requests.get", lambda *a, **k: _Resp(txt))
    obs = get_source("noaa").fetch()
    assert obs[-1].entity == "ENSO_ONI"
    assert obs[-1].value == 0.11
    # Negative anomalies (La Niña) parse too.
    assert any(o.value == -0.30 for o in obs)


# --- ENSO strategy (Gaussian persistence) ---------------------------------

def test_enso_prefilters_to_oni_markets():
    s = EnsoStrategy()
    ms = [
        {"id": "a", "question": "Will the Oceanic Niño Index (ONI) exceed 2 degrees Celsius?", "isResolved": False},
        {"id": "b", "question": "Will gold exceed $3000?", "isResolved": False},
    ]
    assert [m["id"] for m in s.prefilter(ms)] == ["a"]


def test_enso_high_threshold_unlikely():
    s = get_strategy("enso", monthly_vol=0.25)
    s.bind(FakeObs({"ENSO_ONI": {"value": 0.11, "source": "noaa"}}))
    est = s.estimate([{"id": "a", "question": "Will the ONI 3-month mean exceed 2 degrees Celsius on Dec 31?"}])
    assert est["a"] < 0.2  # current 0.11, far below +2


def test_enso_handles_negative_threshold_below():
    s = EnsoStrategy(monthly_vol=0.25)
    s.bind(FakeObs({"ENSO_ONI": {"value": 0.11, "source": "noaa"}}))
    # La Niña question: 'below -0.5' with current 0.11 -> unlikely.
    est = s.estimate([{"id": "a", "question": "Will the ONI be below -0.5 degrees Celsius?"}])
    assert est["a"] < 0.3


def test_enso_abstains_without_data():
    s = EnsoStrategy()
    s.bind(FakeObs({}))
    assert s.estimate([{"id": "a", "question": "Will the ONI exceed 1 degree?"}]) == {}


# --- commodity futures strategy (lognormal) -------------------------------

def test_commodity_links_cotton_only():
    s = CommodityFuturesStrategy()
    ms = [
        {"id": "a", "question": "Will ICE Cotton No. 2 front-month futures exceed 82 cents/lb on March 31, 2027?", "isResolved": False},
        {"id": "b", "question": "Will the US 30-year mortgage rate exceed 7%?", "isResolved": False},
    ]
    assert [m["id"] for m in s.prefilter(ms)] == ["a"]


def test_commodity_estimate_lognormal():
    s = CommodityFuturesStrategy(annual_vol=0.3)
    s.bind(FakeObs({"CME_COTTON": {"value": 77.0, "source": "stooq"}}))
    est = s.estimate([{"id": "a", "question": "Will ICE Cotton No. 2 front-month futures exceed 50 cents/lb on March 31, 2027?"}])
    assert est["a"] > 0.8  # current 77 well above a 50 strike


def test_commodity_abstains_without_data():
    s = CommodityFuturesStrategy()
    s.bind(FakeObs({}))
    assert s.estimate([{"id": "a", "question": "Will ICE Cotton No. 2 futures exceed 80 cents/lb?"}]) == {}
