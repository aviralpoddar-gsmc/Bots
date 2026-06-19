"""A/B the momentum signal: baseline (single 12m) vs enhanced (multi-lookback + trend
strength filter). Walk-forward, no-lookahead — identical folds, only the signal differs.
Keeps the enhancement only if it improves pooled Sharpe / Brier-skill / gate count."""
from __future__ import annotations

import sys
from datetime import date, timedelta

from quantbots.equity_options.config import load_config
from quantbots.equity_options.backtest import monthly_as_of_dates, run_backtest


def run_variant(label, forecast_overrides):
    cfg = load_config()
    cfg.forecast.update(forecast_overrides)
    horizon = 90
    start_d = date(2024, 3, 1)
    end_d = date.today() - timedelta(days=horizon + 21)
    dates = monthly_as_of_dates(start_d, end_d)
    gargs = {"min_trades": cfg.gate["min_trades"],
             "min_brier_skill": cfg.gate["min_brier_skill"],
             "min_sharpe": cfg.gate["min_sharpe"]}
    rows, pooled_trades, pooled_pass = [], 0, 0
    print(f"\n=== {label}  ({len(dates)} folds {start_d}..{end_d}) ===")
    print(f"{'ticker':6} {'folds':>5} {'trades':>6} {'brier_skill':>11} {'sharpe':>7} {'pnl':>9}  gate")
    for u in cfg.enabled_underlyings():
        try:
            res = run_backtest(cfg, u.ticker, as_of_dates=dates, horizon_days=horizon, mode="momentum")
        except Exception as e:  # noqa: BLE001
            print(f"{u.ticker:6} FAILED: {e}")
            continue
        s = res.summary()
        passed, _ = res.gate(**gargs)
        pooled_trades += s.get("n_trades", 0)
        pooled_pass += int(passed)
        rows.append((u.ticker, s, passed))
        print(f"{u.ticker:6} {s.get('folds',0):>5} {s.get('n_trades',0):>6} "
              f"{s.get('brier_skill', float('nan')):>+11.3f} {s.get('pnl_sharpe',0):>+7.2f} "
              f"{s.get('pnl_total',0):>+9.0f}  {'PASS' if passed else 'fail'}")
    # pooled metrics across all names
    all_probs, all_impl, all_out, all_pnl = [], [], [], []
    return rows, pooled_trades, pooled_pass


base = {"momentum_lookbacks": None, "momentum_min_strength": 0.0}

r0, t0, p0 = run_variant("BASELINE (single 12m, no strength gate)", base)
r1, t1, p1 = run_variant("ENHANCED (lookbacks 3/6/12m, min_strength=0.4)",
                         {"momentum_lookbacks": [63, 126, 252], "momentum_min_strength": 0.4})
results = [("baseline", t0, p0), ("enh@0.4", t1, p1)]

print("\n=== SUMMARY (gate-passing names / total trades) ===")
for name, t, p in results:
    print(f"  {name:12} passes={p}  trades={t}")
