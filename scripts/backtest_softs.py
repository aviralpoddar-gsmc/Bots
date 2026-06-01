"""Out-of-sample calibration backtest for the USDA softs bots.

The dry-run ROI is the model's *own* expectation — circular. This measures the
real thing: replay each bot's pricing over historical futures paths and ask
whether its probabilities are CALIBRATED (when it says 0.8, does the threshold get
hit ~80% of the time?) and whether they have SKILL (Brier beats baselines).

Method: for a monthly grid of anchor dates × horizons × strike-moneyness, compute
the bot's P(price_at_anchor+horizon > strike) using only info at the anchor, then
score against the realized outcome. Baselines: always-0.5, and (for cotton) the
zero-drift version of the same model, to isolate whether the USDA SUR drift helps.

Run: .venv/bin/python scripts/backtest_softs.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_softs import world_sur  # noqa: E402

HORIZONS = [0.25, 0.5, 1.0]
MONEYNESS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def _ndcdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def daily(ticker: str) -> pd.Series:
    import yfinance as yf
    s = yf.Ticker(ticker).history(period="max")["Close"].dropna()
    s.index = s.index.tz_localize(None)
    return s


def brier_and_calib(p: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    brier = float(np.mean((p - y) ** 2))
    # 5-bin reliability
    bins = np.clip((p * 5).astype(int), 0, 4)
    parts = []
    for b in range(5):
        m = bins == b
        if m.sum() >= 10:
            parts.append(f"[{b*0.2:.1f}-{(b+1)*0.2:.1f}) pred {p[m].mean():.2f} obs {y[m].mean():.2f} (n={m.sum()})")
    return brier, " | ".join(parts)


def cotton_drift(spot: float, sur: float | None) -> float:
    if sur is None or sur <= 0:
        return 0.0
    f_fund = 68.4 * (sur / 0.487) ** -0.39
    mu = 0.5 * math.log(f_fund / spot)
    return max(min(mu, 0.05), -0.05)


def free_sur_lookup() -> dict[int, float]:
    s = world_sur("cotton", exclude=("China",)).sur
    return {int(y): float(v) for y, v in s.items()}


def backtest_price(name: str, ticker: str, vol: float, use_drift: bool) -> None:
    px = daily(ticker)
    sur_by_my = free_sur_lookup() if use_drift else {}
    idx = px.index
    p_model, p_zero, y = [], [], []
    # monthly anchors with room for the longest horizon to realize
    anchors = pd.date_range(idx.min(), idx.max() - pd.Timedelta(days=400), freq="MS")
    for a in anchors:
        pos = idx.searchsorted(a)
        if pos >= len(idx):
            continue
        spot = float(px.iloc[pos])
        my = a.year if a.month >= 8 else a.year - 1
        sur = sur_by_my.get(my)
        mu = cotton_drift(spot, sur) if use_drift else 0.0
        for h in HORIZONS:
            tgt = a + pd.Timedelta(days=int(h * 365.25))
            tpos = idx.searchsorted(tgt)
            if tpos >= len(idx):
                continue
            future = float(px.iloc[tpos])
            sigma = max(vol * math.sqrt(h), 0.05)
            for mny in MONEYNESS:
                strike = spot * mny
                fair = spot * math.exp(mu * h)
                p_model.append(1 - _ndcdf(math.log(strike / fair) / sigma))
                p_zero.append(1 - _ndcdf(math.log(strike / spot) / sigma))
                y.append(1.0 if future > strike else 0.0)
    p_model, p_zero, y = map(np.array, (p_model, p_zero, y))
    n = len(y)
    b_model, calib = brier_and_calib(p_model, y)
    b_zero, _ = brier_and_calib(p_zero, y)
    b_base = float(np.mean((0.5 - y) ** 2))
    print(f"\n{'='*70}\n{name}  (vol={vol:.0%}, n={n}, drift={'ON' if use_drift else 'off'})\n{'='*70}")
    print(f"  Brier:  model={b_model:.4f}   zero-drift={b_zero:.4f}   always-0.5={b_base:.4f}")
    print(f"  Skill vs 0.5 baseline: {1 - b_model/b_base:+.1%}   "
          f"(lower Brier = better; drift delta vs zero = {b_zero - b_model:+.4f})")
    print(f"  Calibration: {calib}")


def backtest_coffee() -> None:
    cf = pd.read_csv(Path(__file__).resolve().parents[1] / "data/research/psd/psd_coffee.csv")
    dc = cf[cf.Attribute_Description == "Domestic Consumption"].groupby("Market_Year").Value.sum()
    g = (dc.pct_change() * 100).dropna()
    g = g[g.index >= 2000]
    mean, sigma = 1.43, 2.59
    thresholds = [0, 1, 2, 3, 5, 7, 10]
    p, y = [], []
    for thr in thresholds:
        for yr, realized in g.items():
            p.append(1 - _ndcdf((thr - mean) / sigma))
            y.append(1.0 if realized > thr else 0.0)
    p, y = np.array(p), np.array(y)
    b, calib = brier_and_calib(p, y)
    b_base = float(np.mean((0.5 - y) ** 2))
    print(f"\n{'='*70}\nCOFFEE consumption-growth (n={len(y)})\n{'='*70}")
    print(f"  Brier: model={b:.4f}  always-0.5={b_base:.4f}  skill={1 - b/b_base:+.1%}")
    print(f"  Calibration: {calib or '(bins too small)'}")


if __name__ == "__main__":
    backtest_price("COTTON (USDA drift)", "CT=F", 0.24, use_drift=True)
    backtest_price("COCOA (no USDA data)", "CC=F", 0.40, use_drift=False)
    backtest_coffee()
