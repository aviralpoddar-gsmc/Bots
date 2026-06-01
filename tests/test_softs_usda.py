"""Tests for the USDA soft-commodity bots (cotton / cocoa / coffee) and the
FAS PSD source parsing. Network-free: strategies use a fake observation handle,
the source is tested via its pure parsing helpers on a synthetic CSV."""

import time

import pytest

from quantbots.sources import fas_psd
from quantbots.strategies import get_strategy


class Obs:
    def __init__(self, values):
        self.values = values

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


def mkt(q, yrs=0.7, mid=None):
    return {
        "id": mid or q, "question": q, "probability": 0.5, "isResolved": False,
        "closeTime": time.time() * 1000 + yrs * 365.25 * 86400 * 1000,
        "totalLiquidity": 200,
    }


# --- cotton -----------------------------------------------------------------

COTTON_Q = "Will ICE Cotton No. 2 front-month futures exceed 70 cents/lb on March 31, 2027?"


def test_cotton_prefilter_keeps_price_drops_operational():
    s = get_strategy("cotton_fundamental")
    kept = s.prefilter([
        mkt(COTTON_Q),
        mkt("Will US Southwest cotton abandonment for the crop year ending October 2029 exceed 40%?"),
        mkt("Will cocoa futures exceed 4000 on Dec 31, 2027?"),
        {**mkt(COTTON_Q), "isResolved": True},
        mkt(COTTON_Q, yrs=3.0),  # beyond max_horizon_years
    ])
    assert [m["question"] for m in kept] == [COTTON_Q]


def test_cotton_drift_is_bounded():
    s = get_strategy("cotton_fundamental", drift_cap=0.05)
    # absurdly tight stocks -> huge raw fundamental pull, must clamp to the cap
    s.bind(Obs({"CME_COTTON": 76.7, "PSD_COTTON_FREE_SUR": 0.01}))
    s.estimate(s.prefilter([mkt(COTTON_Q)]))
    mu = next(iter(s._explanations.values()))["mu"]
    assert abs(mu) <= 0.05 + 1e-9


def test_cotton_direction_symmetry():
    s = get_strategy("cotton_fundamental")
    s.bind(Obs({"CME_COTTON": 76.7, "PSD_COTTON_FREE_SUR": 0.45}))
    up = s.estimate([mkt(COTTON_Q, mid="up")])["up"]
    below_q = COTTON_Q.replace("exceed", "be below")
    dn = s.estimate([mkt(below_q, mid="dn")])["dn"]
    assert up + dn == pytest.approx(1.0, abs=0.02)


def test_cotton_rejects_basis_and_spread_markets():
    """Regression: basis/spread markets must NOT be priced off the outright futures."""
    s = get_strategy("cotton_fundamental")
    traps = [
        "Will the Cotlook A minus ICE Cotton No.2 basis exceed 18 cents/lb on June 30, 2027?",
        "Will the cotton calendar spread exceed 2 cents/lb on Dec 31, 2027?",
        "Will the cotton A-index premium over ICE exceed 10 cents/lb on June 30?",
    ]
    assert s.prefilter([mkt(q) for q in traps]) == []


def test_cotton_no_sur_means_zero_drift():
    s = get_strategy("cotton_fundamental")
    s.bind(Obs({"CME_COTTON": 76.7}))  # no PSD obs
    s.estimate([mkt(COTTON_Q, mid="x")])
    assert s._explanations["x"]["mu"] == 0.0


# --- cocoa ------------------------------------------------------------------

COCOA_Q = "Will ICE cocoa (NY) nearest-futures price exceed 3894 USD/t on December 31, 2027?"


def test_cocoa_matches_only_cocoa_price():
    s = get_strategy("cocoa_fundamental")
    kept = s.prefilter([mkt(COCOA_Q), mkt(COTTON_Q),
                        mkt("Will ECOM cocoa volume for September 2029 exceed 500k tonnes?")])
    assert [m["question"] for m in kept] == [COCOA_Q]


def test_cocoa_at_the_money_is_half():
    s = get_strategy("cocoa_fundamental")
    s.bind(Obs({"CME_COCOA": 3894.0}))  # threshold == spot -> ~0.5
    p = s.estimate([mkt(COCOA_Q, mid="c")])["c"]
    assert p == pytest.approx(0.5, abs=0.02)


# --- coffee -----------------------------------------------------------------

def coffee_q(thr):
    return f"Will global coffee consumption growth rate for the year ending June 2029 exceed {thr}%?"


def test_coffee_matches_only_consumption_growth():
    s = get_strategy("coffee_consumption")
    kept = s.prefilter([
        mkt(coffee_q(5)),
        mkt("Will global specialty coffee market size for the year ending June 2029 exceed 165 billion USD?"),
        mkt("Will ECOM coffee volume for September 2027 exceed 12M bags?"),
    ])
    assert [m["question"] for m in kept] == [coffee_q(5)]


def test_coffee_threshold_monotonic_and_calibrated():
    s = get_strategy("coffee_consumption", mean_growth=1.43, sigma_growth=2.59)
    p_low = s.estimate([mkt(coffee_q(1), mid="lo")])["lo"]
    p_high = s.estimate([mkt(coffee_q(5), mid="hi")])["hi"]
    assert p_low > p_high                 # easier threshold -> higher prob
    assert p_high < 0.15                  # 5% is ~1.4 sigma above mean
    assert p_low == pytest.approx(0.566, abs=0.02)


# --- FAS PSD source parsing -------------------------------------------------

_CSV = (
    "Commodity_Code,Commodity_Description,Country_Code,Country_Name,Market_Year,"
    "Calendar_Year,Month,Attribute_ID,Attribute_Description,Unit_ID,Unit_Description,Value\n"
    # year 2024
    "x,Cotton,US,United States,2024,2024,08,020,Ending Stocks,27,Bales,100\n"
    "x,Cotton,US,United States,2024,2024,08,099,Domestic Use,27,Bales,200\n"
    "x,Cotton,CH,China,2024,2024,08,020,Ending Stocks,27,Bales,300\n"
    "x,Cotton,CH,China,2024,2024,08,099,Domestic Use,27,Bales,100\n"
)


def test_fas_psd_world_and_exchina_sur():
    # world: stocks=(100+300)/use(200+100)=400/300=1.333
    yr, sur = fas_psd._latest_sur(_CSV, "Domestic Use", set())
    assert yr == 2024 and sur == pytest.approx(400 / 300)
    # ex-China: 100/200 = 0.5
    yr2, sur2 = fas_psd._latest_sur(_CSV, "Domestic Use", {"China"})
    assert yr2 == 2024 and sur2 == pytest.approx(0.5)


def test_fas_psd_coffee_growth():
    csv = (
        "Commodity_Code,Commodity_Description,Country_Code,Country_Name,Market_Year,"
        "Calendar_Year,Month,Attribute_ID,Attribute_Description,Unit_ID,Unit_Description,Value\n"
        "x,Coffee,BR,Brazil,2023,2023,10,099,Domestic Consumption,1,Bags,100\n"
        "x,Coffee,BR,Brazil,2024,2024,10,099,Domestic Consumption,1,Bags,105\n"
    )
    src = fas_psd.FasPsdSource()
    out = []
    src._emit_coffee(csv, "Domestic Consumption", out)
    by = {o.entity: o.value for o in out}
    assert by["PSD_COFFEE_CONS"] == 105
    assert by["PSD_COFFEE_CONS_GROWTH"] == pytest.approx(5.0)
