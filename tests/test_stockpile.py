"""Tests for the three strategic-materials bots:
  - stockpile_facts:     USGS-list / NDS fact lookup (high conviction, abstain when unsure)
  - stockpile_grid_arb:  2-D monotone surface on the vault kg ladders
  - stockpile_coherence: time-monotonicity + cross-template ceiling
plus the resolvability scoring extension for these families."""

import time

from quantbots.resolvability import resolvability_score as R
from quantbots.strategies.stockpile_coherence import StockpileCoherenceStrategy
from quantbots.strategies.stockpile_facts import StockpileFactsStrategy
from quantbots.strategies.stockpile_grid_arb import (
    StockpileGridArbStrategy, _parse_strike_kg, isotonic_increasing)


def _m(question, prob=0.5, mid=None):
    return {
        "id": mid or question[:40], "question": question, "probability": prob,
        "closeTime": time.time() * 1000 + 365 * 24 * 3600 * 1000,
        "totalLiquidity": 100, "isResolved": False,
    }


# ============================ resolvability ===============================

def test_resolvability_for_stockpile_families():
    assert R("Will Gallium be on the U.S. Critical Minerals list as of December 31, 2027?") == 0.90
    assert R("Will the U.S. National Defense Stockpile hold a Cobalt position by December 31, 2028?") == 0.35
    assert R("Will the U.S. begin strategic buffer-stock procurement of Niobium by December 31, 2027?") == 0.10
    assert R("Will U.S. vault procurement of Tungsten exceed 3.15e+06 kg by December 31, 2027?") == 0.03
    # didn't break the existing price/operational ordering
    assert R("Will copper spot price exceed $12,900 USD/MT?") > R("Will copper production exceed 2000 kt?")


# ============================ stockpile_facts ==============================

def test_facts_critical_list_lookup():
    s = StockpileFactsStrategy()
    on = _m("Will Gallium be on the U.S. Critical Minerals list as of December 31, 2027?")
    off = _m("Will Unobtanium be on the U.S. Critical Minerals list as of December 31, 2027?")
    assert s.estimate([on])[on["id"]] > 0.9      # on the USGS 2022 list
    assert s.estimate([off])[off["id"]] < 0.1    # not on the list

def test_facts_group_names_resolve_on_list():
    s = StockpileFactsStrategy()
    for name in ["Rare Earths", "Platinum-Group Metals", "Graphite (Natural)"]:
        q = _m(f"Will {name} be on the U.S. Critical Minerals list as of December 31, 2028?")
        assert s.estimate([q])[q["id"]] > 0.9

def test_facts_nds_held_vs_unknown():
    s = StockpileFactsStrategy()
    held = _m("Will the U.S. National Defense Stockpile hold a Germanium position by December 31, 2027?")
    unknown = _m("Will the U.S. National Defense Stockpile hold a Zirconium position by December 31, 2027?")
    assert s.estimate([held])[held["id"]] > 0.7   # documented holding
    assert unknown["id"] not in s.estimate([unknown])  # ambiguous record -> abstain

def test_facts_prefilter_and_corr_key():
    s = StockpileFactsStrategy()
    keep = _m("Will Gallium be on the U.S. Critical Minerals list as of December 31, 2027?")
    drop = _m("Will U.S. vault procurement of Tungsten exceed 3.15e+06 kg by December 31, 2027?")
    kept = {m["id"] for m in s.prefilter([keep, drop])}
    assert keep["id"] in kept and drop["id"] not in kept
    assert s.correlation_key(keep).startswith("critical_list:")


# ============================ stockpile_grid_arb ===========================

def test_strike_parser_handles_scientific_notation():
    assert _parse_strike_kg("exceed 3.15e+06 kg") == 3150000.0
    assert _parse_strike_kg("exceed 4,200,000 kg") == 4200000.0

def test_isotonic_increasing_basic():
    out = isotonic_increasing([0.5, 0.7, 0.6], [1, 1, 1])
    assert out[0] <= out[1] <= out[2]              # non-decreasing
    assert abs(out[1] - out[2]) < 1e-9             # the 0.7/0.6 violation pooled

def _vault(metal, strike, year, prob):
    q = f"Will U.S. vault procurement of {metal} exceed {strike} kg by December 31, {year}?"
    return _m(q, prob, mid=f"{metal}-{strike}-{year}")

def test_grid_enforces_both_monotonicities():
    s = StockpileGridArbStrategy(skip_extreme=0.0, min_nodes=4)
    grp = [
        _vault("Tungsten", "3e+06", 2027, 0.55),
        _vault("Tungsten", "4e+06", 2027, 0.60),   # strike violation (4e6 > 3e6)
        _vault("Tungsten", "5e+06", 2027, 0.40),
        _vault("Tungsten", "3e+06", 2028, 0.50),   # time violation (2028 < 2027 at 3e6)
        _vault("Tungsten", "4e+06", 2028, 0.50),
        _vault("Tungsten", "5e+06", 2028, 0.45),
    ]
    est = s.estimate(grp)
    f = {(mid.split("-")[1], mid.split("-")[2]): p for mid, p in est.items()}
    # monotone-down in strike within 2027
    assert f[("3e+06", "2027")] >= f[("4e+06", "2027")] >= f[("5e+06", "2027")] - 1e-9
    # monotone-up in expiry at the 3e6 strike (cumulative)
    assert f[("3e+06", "2028")] >= f[("3e+06", "2027")] - 1e-9

def test_grid_abstains_on_flat_grid():
    s = StockpileGridArbStrategy(min_nodes=4)
    grp = [_vault("Zinc", st, yr, 0.5) for st in ("3e+06", "4e+06") for yr in (2027, 2028)]
    assert s.estimate(grp) == {}


# ============================ stockpile_coherence =========================

def _incl(metal, month, day, year, prob):
    q = f"Will {metal} be included in Project Vault procurement by {month} {day}, {year}?"
    return _m(q, prob, mid=f"{metal}-incl-{month}{year}")

def test_coherence_time_monotonicity():
    s = StockpileCoherenceStrategy(skip_extreme=0.0, min_dates=3)
    grp = [
        _incl("Gallium", "September", 30, 2026, 0.50),
        _incl("Gallium", "December", 31, 2026, 0.70),
        _incl("Gallium", "June", 30, 2027, 0.60),   # time violation (Jun < Dec)
    ]
    est = s.estimate(grp)
    ordered = [est[f"Gallium-incl-{m}{y}"] for m, y in
               [("September", 2026), ("December", 2026), ("June", 2027)]]
    assert ordered[0] <= ordered[1] <= ordered[2] + 1e-9   # non-decreasing in time
    assert ordered[2] > 0.60                                # the late date pulled up
    assert ordered[1] < 0.70                                # the violating date pulled down

def test_coherence_no_cross_template_ceiling():
    # An NDS-hold market in the same metal must NOT cap the (better-informed)
    # inclusion ladder — that bug is gone. NDS markets aren't even classified here.
    s = StockpileCoherenceStrategy(skip_extreme=0.0, min_dates=3)
    grp = [
        _incl("Lithium", "September", 30, 2026, 0.82),
        _incl("Lithium", "December", 31, 2026, 0.85),
        _incl("Lithium", "June", 30, 2027, 0.88),
        _m("Will the U.S. National Defense Stockpile hold a Lithium position by December 31, 2027?",
           0.45, mid="li-nds"),
    ]
    est = s.estimate(grp)
    assert "li-nds" not in est                              # NDS not traded by this bot
    assert all(p > 0.8 for k, p in est.items() if "incl" in k)  # not dragged to 0.45

def test_coherence_min_dates_abstains():
    s = StockpileCoherenceStrategy(skip_extreme=0.0, min_dates=3)
    grp = [
        _m("Will the U.S. begin strategic buffer-stock procurement of Niobium by December 31, 2027?", 0.45, mid="nb-b1"),
        _m("Will the U.S. begin strategic buffer-stock procurement of Niobium by December 31, 2028?", 0.40, mid="nb-b2"),
    ]
    assert s.estimate(grp) == {}   # only 2 dates < min_dates
