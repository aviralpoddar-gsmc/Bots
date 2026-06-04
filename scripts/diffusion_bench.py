#!/usr/bin/env python
"""Rigorous multi-fold walk-forward benchmark for the diffusion pricer vs the
closed-form lognormal, on the resolvable commodity spot-price markets.

WHY THIS EXISTS
---------------
The first-pass gate compared diffusion vs lognormal with BOTH models betting from a
0.50 prior. That mechanically rewards over-confidence (whoever is more extreme and
right bets bigger), so the thin-tailed lognormal "won" raw PnL even though it is the
worse probability model. That tells us about the *market structure*, not the model.

This benchmark separates the questions that actually decide deployment:

  1. CALIBRATION   — Brier on the held-out test fold (lower = better probabilities).
  2. STANDALONE PnL — each model as the SOLE pricer, betting from the 0.50 prior that
                      the clone's untraded markets actually sit at, sized by edge and
                      *capped* (the real framework caps per-market stake). The cap is
                      the lever the first gate ignored: uncapped, tail over-confidence
                      pays; capped, it can't be pressed, so body calibration decides.
  3. GAP EDGE      — diffusion betting against a lognormal-priced market: Σ(d-m)(o-m).
                      Positive = the diffusion's deviation from the consensus is
                      informative (it corrects the lognormal where the lognormal errs).
  4. RISK          — per-fold dispersion: worst-fold PnL and Sharpe (mean/std across
                      folds). The diffusion's thesis is tail-robustness, so we care
                      whether it loses less in the bad fold, not just the mean.

All walk-forward (expanding window, fit on train, score on disjoint test) so there is
NO look-ahead. Pure numpy/scipy; reuses the project's cached yfinance history.

USAGE
    doppler run -- .venv/bin/python scripts/diffusion_bench.py
    doppler run -- .venv/bin/python scripts/diffusion_bench.py --folds 5 --horizons 21,63
"""

from __future__ import annotations

import argparse
from math import erf, sqrt

import numpy as np
from scipy import stats

from quantbots.research.data_fetch import DEFAULT_UNIVERSE, fetch_yf_history

ENTITIES = ["GOLD", "SILVER", "PLATINUM", "PALLADIUM", "COPPER", "WTI_OIL", "BRENT_OIL", "GASOLINE"]
FRACS = np.array([0.55, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.45])
N_SIMS = 30000


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2)))


def fit_student_t(rets: np.ndarray) -> tuple[float, float]:
    """(df, daily scale) so the t's std matches realized daily vol; df clamped to [3,15]."""
    dvol = float(np.std(rets))
    try:
        df = float(stats.t.fit(rets)[0])
    except Exception:  # noqa: BLE001
        df = 5.0
    df = min(max(df, 3.0), 15.0)
    return df, dvol / sqrt(df / (df - 2.0))


def simulate_terminal(process: str, rets: np.ndarray, spot: float, n_days: int,
                      rng: np.random.Generator, block_len: int = 10, df_kernel: float = 4.0) -> np.ndarray:
    if process == "student_t":
        df, scale = fit_student_t(rets)
        lr = (rng.standard_t(df, size=(N_SIMS, n_days)) * scale).sum(axis=1)
    elif process == "ksb":
        # Kernel-smoothed (block) bootstrap: resample the empirical body, then convolve
        # each day with a Student-t kernel at Silverman bandwidth, variance-corrected so
        # the per-day vol is preserved EXACTLY. The kernel's continuous fat tails let the
        # sum exceed any historically observed move (fixes the bootstrap's extrapolation
        # hole) while the empirical resample keeps the body calibration that wins.
        bl = min(block_len, len(rets))
        nb = int(np.ceil(n_days / bl))
        starts = rng.integers(0, len(rets) - bl + 1, size=(N_SIMS, nb))
        idx = starts[:, :, None] + np.arange(bl)[None, None, :]
        body = rets[idx].reshape(N_SIMS, nb * bl)[:, :n_days]
        s = float(np.std(rets))
        h = 0.9 * len(rets) ** (-0.2)                       # Silverman relative bandwidth (~0.18)
        z = rng.standard_t(df_kernel, size=(N_SIMS, n_days)) * sqrt((df_kernel - 2.0) / df_kernel)
        smoothed = (body + h * s * z) / sqrt(1.0 + h * h)   # variance-preserving smooth
        lr = smoothed.sum(axis=1)
    elif process == "hybrid":
        bl = min(block_len, len(rets))
        nb = int(np.ceil(n_days / bl))
        starts = rng.integers(0, len(rets) - bl + 1, size=(N_SIMS, nb))
        idx = starts[:, :, None] + np.arange(bl)[None, None, :]
        lr = rets[idx].reshape(N_SIMS, nb * bl)[:, :n_days].sum(axis=1)
        dvol = float(np.std(rets))
        lr = lr + rng.normal(0.0, 0.4 * dvol, size=(N_SIMS, n_days)).sum(axis=1)
    else:  # bootstrap
        bl = min(block_len, len(rets))
        nb = int(np.ceil(n_days / bl))
        starts = rng.integers(0, len(rets) - bl + 1, size=(N_SIMS, nb))
        idx = starts[:, :, None] + np.arange(bl)[None, None, :]
        lr = rets[idx].reshape(N_SIMS, nb * bl)[:, :n_days].sum(axis=1)
    return spot * np.exp(lr)


def evaluate(close: np.ndarray, folds: int, horizons: list[int], processes: list[str],
             cap: float) -> dict:
    """Expanding-window walk-forward. Returns per-process accumulators across folds."""
    logret_all = np.diff(np.log(close))
    n = len(close)
    # Expanding windows: train end grows; test is the next chunk.
    start = n // 3  # first train uses >=1/3 of history
    bounds = np.linspace(start, n - max(horizons) - 1, folds + 1).astype(int)

    acc = {p: {"brier": [], "pnl_cap": [], "pnl_uncap": [], "gap": []} for p in ["lognormal"] + processes}

    for k in range(folds):
        tr_end = bounds[k]
        te_end = bounds[k + 1]
        tr_ret = logret_all[:tr_end]
        if len(tr_ret) < 250:
            continue
        dm = tr_ret - tr_ret.mean()          # demeaned -> zero drift
        vol_daily = float(tr_ret.std())
        test_close = close[tr_end:te_end + max(horizons)]

        per = {p: {"brier": [], "pnl_cap": [], "pnl_uncap": [], "gap": []} for p in acc}
        for H in horizons:
            sigma_T = vol_daily * sqrt(H)
            # one terminal sim per (process, fold, horizon)
            sims = {}
            for p in processes:
                rng = np.random.default_rng(hash((p, k, H)) & 0xFFFFFFFF)
                sims[p] = simulate_terminal(p, dm, 1.0, H, rng)  # unit spot -> multiplicative
            for t in range(len(test_close) - H):
                cur = test_close[t]
                fut = test_close[t + H]
                if cur <= 0 or fut <= 0:
                    continue
                ratio = fut / cur
                for f in FRACS:
                    o = 1.0 if ratio > f else 0.0
                    m = 1.0 - ncdf(np.log(f) / sigma_T)         # lognormal P(exceed)
                    per["lognormal"]["brier"].append((m - o) ** 2)
                    per["lognormal"]["pnl_cap"].append(np.clip(m - 0.5, -cap, cap) * (o - 0.5))
                    per["lognormal"]["pnl_uncap"].append((m - 0.5) * (o - 0.5))
                    for p in processes:
                        d = float(np.mean(sims[p] > f))
                        d = min(max(d, 0.01), 0.99)
                        per[p]["brier"].append((d - o) ** 2)
                        per[p]["pnl_cap"].append(np.clip(d - 0.5, -cap, cap) * (o - 0.5))
                        per[p]["pnl_uncap"].append((d - 0.5) * (o - 0.5))
                        per[p]["gap"].append((d - m) * (o - m))  # diffusion vs lognormal-market
        for p in acc:
            if per[p]["brier"]:
                acc[p]["brier"].append(np.mean(per[p]["brier"]))
                acc[p]["pnl_cap"].append(np.mean(per[p]["pnl_cap"]))
                acc[p]["pnl_uncap"].append(np.mean(per[p]["pnl_uncap"]))
                acc[p]["gap"].append(np.mean(per[p]["gap"]))
    return acc


def summarize(name: str, a: dict) -> None:
    def stat(key, scale=1.0):
        v = np.array(a[key]) * scale
        if len(v) == 0:
            return "   n/a"
        mean = v.mean()
        worst = v.min()
        sharpe = mean / (v.std() + 1e-12)
        return f"{mean:+7.2f} (worst {worst:+6.2f}, sharpe {sharpe:+5.2f})"
    print(f"  {name:11} Brier {np.mean(a['brier']):.4f} | "
          f"PnL/1e4 cap {stat('pnl_cap', 1e4)} | uncap {stat('pnl_uncap', 1e4)}"
          + (f" | gap {np.mean(a['gap'])*1e4:+6.2f}" if a['gap'] else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--horizons", default="21,63", help="comma-sep trading-day horizons")
    ap.add_argument("--processes", default="student_t,bootstrap,hybrid")
    ap.add_argument("--cap", type=float, default=0.25, help="per-market |edge| cap (stake lever)")
    ap.add_argument("--period", default="13y")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    processes = [p.strip() for p in args.processes.split(",") if p.strip()]

    print(f"Walk-forward: {args.folds} folds, horizons {horizons}d, cap {args.cap}, "
          f"processes {processes}\n")

    grand = {p: {"brier": [], "pnl_cap": [], "pnl_uncap": [], "gap": []}
             for p in ["lognormal"] + processes}
    for e in ENTITIES:
        tkr = DEFAULT_UNIVERSE.get(e)
        if not tkr:
            continue
        try:
            df = fetch_yf_history(tkr, period=args.period)
            close = df["Close"].astype(float).to_numpy()
            close = close[np.isfinite(close) & (close > 0)]
        except Exception as ex:  # noqa: BLE001
            print(f"{e}: history failed ({ex})")
            continue
        if len(close) < 800:
            print(f"{e}: only {len(close)} pts, skip")
            continue
        acc = evaluate(close, args.folds, horizons, processes, args.cap)
        print(f"{e}  ({len(close)} pts)")
        for p in ["lognormal"] + processes:
            summarize(p, acc[p])
            for key in grand[p]:
                grand[p][key] += acc[p][key]
        print()

    print("=" * 88)
    print("FLEET TOTAL (all commodities, all folds)")
    for p in ["lognormal"] + processes:
        summarize(p, grand[p])
    print("\nDecision keys: standalone winner = highest capped PnL with non-negative worst-fold;")
    print("gap>0 => diffusion corrects the lognormal where it errs (monetizable vs a lognormal market).")


if __name__ == "__main__":
    main()
