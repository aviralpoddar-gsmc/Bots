from quantbots.backtest import BacktestResult, _reliability, _simulate, backtest
from quantbots.sizing import DEFAULT_LIMITS
from quantbots.strategies import get_strategy


def test_simulate_profits_when_all_correct():
    # est on the right side of every outcome -> all bets win.
    pairs = [(0.9, 1), (0.85, 1), (0.1, 0), (0.15, 0)]
    staked, profit, wins, bets = _simulate(pairs, DEFAULT_LIMITS, liquidity=100)
    assert bets == 4 and wins == 4
    assert profit > 0 and profit == staked  # every bet won its stake


def test_simulate_loses_when_all_wrong():
    pairs = [(0.9, 0), (0.85, 0), (0.1, 1)]
    staked, profit, wins, bets = _simulate(pairs, DEFAULT_LIMITS, liquidity=100)
    assert wins == 0 and profit == -staked


def test_brier_skill_score():
    r = BacktestResult(n=2, brier=0.125, baseline_brier=0.25)
    assert abs(r.skill - 0.5) < 1e-9  # half the error of the baseline


def test_reliability_buckets_group_predictions():
    pairs = [(0.05, 0), (0.07, 0), (0.95, 1), (0.92, 1)]
    rel = _reliability(pairs, buckets=10)
    # Low bucket ~0 outcomes, high bucket ~1 outcomes.
    assert rel[0][1] == 0.0
    assert rel[-1][1] == 1.0


def test_backtest_end_to_end_on_synthetic_series():
    # A rising series: future > current most of the time -> 'exceed current' tends true.
    series = [(f"2020-{i:03d}", 100.0 + i) for i in range(60)]
    strat = get_strategy("ensemble", annual_vol=0.3)
    r = backtest(
        strat, "FRED_MORTGAGE30US",
        "Will the US 30-year fixed mortgage rate (Freddie Mac PMMS) exceed {T}%?",
        series, horizon_steps=12, horizon_years=0.25,
        threshold_fracs=(0.95, 1.0, 1.05),
    )
    assert r.n > 0
    assert 0.0 <= r.brier <= 1.0
    assert r.bets >= 0
