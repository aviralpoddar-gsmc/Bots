from quantbots.strategies import get_strategy
from quantbots.strategies.ensemble import EnsembleStrategy
from quantbots.strategies.linker import link_market, link_markets


# --- linker ---------------------------------------------------------------

def test_link_matches_commodity_and_threshold():
    m = {"id": "x", "question": "Will Brent crude exceed $90 by year end?"}
    link = link_market(m)
    assert link is not None
    assert "BRENT_OIL" in link.entities
    assert link.threshold == 90.0 and link.direction == "exceeds"


def test_link_matches_macro():
    link = link_market({"id": "y", "question": "Will US CPI inflation be below 3%?"})
    assert link is not None and "US_CPI_YOY" in link.entities
    assert link.direction == "below"


def test_link_returns_none_when_unmatched():
    assert link_market({"id": "z", "question": "Who wins the 2028 election?"}) is None


def test_link_suppressed_for_operational_metrics():
    # Mentions a commodity but asks about a non-price metric -> no link.
    assert link_market({"id": "z", "question": "Will CNOOC natural gas share of production exceed 30%?"}) is None
    assert link_market({"id": "z", "question": "Will Targa natural gas inlet volumes exceed 7500?"}) is None
    assert link_market({"id": "z", "question": "Will global Yttrium balance exceed 0 t REO?"}) is None


def test_word_boundary_avoids_false_positive():
    # "oil" must not fire inside "spoiled".
    assert link_market({"id": "z", "question": "Will the milk be spoiled by June?"}) is None


def test_link_markets_filters_to_matched():
    ms = [
        {"id": "a", "question": "Will gold exceed $3000?"},
        {"id": "b", "question": "Who wins the election?"},
    ]
    links = link_markets(ms)
    assert set(links) == {"a"}


# --- ensemble -------------------------------------------------------------

class FakeObs:
    def __init__(self, data):
        self.data = data  # {entity: {"value":..,"source":..}}

    def latest_observation(self, entity, source=None):
        return self.data.get(entity)


def test_ensemble_abstains_without_binding():
    s = EnsembleStrategy()
    assert s.estimate([{"id": "a", "question": "Will gold exceed $3000?"}]) == {}


def test_ensemble_high_prob_when_price_far_above_threshold():
    s = get_strategy("ensemble", annual_vol=0.3)
    s.bind(FakeObs({"GOLD": {"value": 4500.0, "source": "stooq"}}))
    est = s.estimate([{"id": "a", "question": "Will gold exceed $3000 by year end?"}])
    # Current 4500 well above a $3000 'exceeds' threshold -> high probability.
    assert est["a"] > 0.9


def test_ensemble_low_prob_when_price_far_below_threshold():
    s = EnsembleStrategy(annual_vol=0.3)
    s.bind(FakeObs({"GOLD": {"value": 2000.0, "source": "stooq"}}))
    est = s.estimate([{"id": "a", "question": "Will gold exceed $3000 by year end?"}])
    assert est["a"] < 0.1


def test_ensemble_direction_below_inverts():
    s = EnsembleStrategy(annual_vol=0.3)
    s.bind(FakeObs({"WTI_OIL": {"value": 50.0, "source": "stooq"}}))
    est = s.estimate([{"id": "a", "question": "Will WTI crude be below $90?"}])
    # Price 50, 'below 90' -> very likely.
    assert est["a"] > 0.9


def test_ensemble_links_stock_ticker():
    s = EnsembleStrategy(annual_vol=0.3)
    s.bind(FakeObs({"STOCK_WULF": {"value": 25.0, "source": "stooq"}}))
    est = s.estimate([{"id": "a", "question": "Will TeraWulf (WULF) stock price exceed 10 USD on May 31, 2027?"}])
    assert est["a"] > 0.7  # 25 already above a $10 strike


def test_effective_sigma_grows_with_horizon():
    import time
    s = EnsembleStrategy(annual_vol=0.5, min_vol=0.01)
    yr = 365.25 * 24 * 3600 * 1000
    near = s._effective_sigma({"closeTime": (time.time()) * 1000 + 0.25 * yr})
    far = s._effective_sigma({"closeTime": (time.time()) * 1000 + 4 * yr})
    assert near < far


def test_ensemble_abstains_when_no_observation():
    s = EnsembleStrategy()
    s.bind(FakeObs({}))  # linked entity exists but no data cached
    assert s.estimate([{"id": "a", "question": "Will gold exceed $3000?"}]) == {}


def test_ensemble_skips_scale_mismatched_link():
    # NATGAS price ~3.5 wrongly linked to a natgas *volume* market (threshold 7500).
    s = EnsembleStrategy(max_ratio=20.0)
    s.bind(FakeObs({"NATGAS": {"value": 3.5, "source": "stooq"}}))
    m = [{"id": "a", "question": "Will Targa natural gas inlet volumes exceed 7500?"}]
    assert s.estimate(m) == {}


def test_signal_prob_threshold_zero_is_high():
    s = EnsembleStrategy()
    p = s._signal_prob(value=10.0, threshold=0.0, direction="exceeds", sigma=0.3)
    assert p is not None and p > 0.9


def test_link_stock_and_catalog():
    # Stock ticker -> STOCK_<TICKER>
    lk = link_market({"id": "a", "question": "Will TeraWulf (WULF) stock price exceed 40 USD on May 31, 2027?"})
    assert lk is not None and lk.entities == ["STOCK_WULF"] and lk.threshold == 40.0
    # Curated catalog -> FRED series
    lk2 = link_market({"id": "b", "question": "Will US single-family housing starts SAAR exceed 800 thousand units?"})
    assert lk2 is not None and "FRED_HOUST1F" in lk2.entities
    lk3 = link_market({"id": "c", "question": "Will the US 30-year fixed mortgage rate (Freddie Mac PMMS) exceed 7%?"})
    assert lk3 is not None and "FRED_MORTGAGE30US" in lk3.entities
