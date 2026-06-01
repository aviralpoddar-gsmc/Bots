"""Prove the USDA fundamentals signal for soft commodities (cotton, coffee).

Phase-1 validation for the USDA-driven softs bots (docs/usda-softs-bots.md):

1. Build the WORLD stocks-to-use ratio (SUR) per marketing year from USDA FAS
   PSD bulk CSVs (data/research/psd/psd_<commodity>.csv).
2. Align it with the marketing-year average futures price (yfinance front-month).
3. Fit the contemporaneous elasticity  log(price) = a + b*log(SUR)  (b<0 expected).
4. Walk-forward (expanding-window) out-of-sample test: does the SUR->price map
   predict next observation better than a naive no-change baseline? This is the
   real bar — fundamentals must beat zero-drift OUT of sample.

No statsmodels dependency: OLS via numpy.linalg.lstsq, t-stats/R^2 computed by hand.
Run: .venv/bin/python scripts/research_softs.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PSD = Path(__file__).resolve().parents[1] / "data" / "research" / "psd"

# Marketing year start month (1-based) per commodity, and the futures ticker.
SPECS = {
    "cotton": {"ticker": "CT=F", "my_start": 8},   # US cotton MY: Aug-Jul
    "coffee": {"ticker": "KC=F", "my_start": 10},  # coffee MY ~ Oct-Sep
}


def world_sur(commodity: str, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """World stocks-to-use per marketing year = sum(Ending Stocks)/sum(Domestic use).

    `exclude` drops countries from the world aggregate (e.g. China for cotton,
    whose off-market state reserve decouples world SUR from price -> analysts use
    'free'/world-ex-China stocks-to-use).
    """
    df = pd.read_csv(PSD / f"psd_{commodity}.csv")
    if exclude:
        df = df[~df.Country_Name.isin(exclude)]
    use_attr = "Domestic Use" if commodity == "cotton" else "Domestic Consumption"
    keep = {"Ending Stocks": "stocks", use_attr: "use", "Production": "prod"}
    sub = df[df.Attribute_Description.isin(keep)].copy()
    sub["k"] = sub.Attribute_Description.map(keep)
    # Sum across all countries -> world total per (market year, attribute).
    world = sub.groupby(["Market_Year", "k"]).Value.sum().unstack("k")
    world = world.dropna(subset=["stocks", "use"])
    world = world[world.use > 0]
    world["sur"] = world.stocks / world.use
    return world[["stocks", "use", "prod", "sur"]]


def my_avg_price(commodity: str) -> pd.Series:
    """Marketing-year average front-month futures close, indexed by MY start year."""
    import yfinance as yf

    spec = SPECS[commodity]
    h = yf.Ticker(spec["ticker"]).history(period="max")["Close"]
    h.index = h.index.tz_localize(None)
    # Assign each daily obs to a marketing year: months >= my_start belong to that
    # calendar year's MY; earlier months belong to the previous year's MY.
    idx = h.index
    my = np.where(idx.month >= spec["my_start"], idx.year, idx.year - 1)
    return h.groupby(my).mean()


def ols(x: np.ndarray, y: np.ndarray):
    """Simple y = a + b*x OLS. Returns (a, b, r2, t_b, n)."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # std error of slope
    sigma2 = ss_res / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    se_b = math.sqrt(sigma2 / sxx) if sxx > 0 else float("nan")
    t_b = b / se_b if se_b else float("nan")
    return a, b, r2, t_b, n


def analyse(commodity: str, exclude: tuple[str, ...] = (), tag: str = "") -> None:
    print(f"\n{'='*70}\n{commodity.upper()}{tag}\n{'='*70}")
    sur = world_sur(commodity, exclude=exclude)
    price = my_avg_price(commodity)
    df = pd.DataFrame({"sur": sur.sur, "price": price}).dropna()
    df = df[(df.sur > 0) & (df.price > 0)]
    # restrict to the futures-overlap window
    df = df[df.index >= price.index.min()]
    print(f"overlap marketing years: {df.index.min()}-{df.index.max()}  (n={len(df)})")
    print(f"SUR range: {df.sur.min():.3f}-{df.sur.max():.3f}   "
          f"price range: {df.price.min():.1f}-{df.price.max():.1f}")

    lx, ly = np.log(df.sur.values), np.log(df.price.values)
    a, b, r2, t_b, n = ols(lx, ly)
    print(f"\nContemporaneous elasticity  log(price)=a+b*log(SUR):")
    print(f"  b (elasticity) = {b:+.3f}   t = {t_b:+.2f}   R^2 = {r2:.3f}   n = {n}")
    print(f"  interpretation: +10% SUR -> {b*0.10*100:+.1f}% price")

    # Walk-forward expanding window: fit on years < t, predict price_t from SUR_t.
    yrs = df.index.to_numpy()
    start = max(8, n // 2)  # need a few points to fit
    preds, actuals, naive = [], [], []
    for i in range(start, n):
        a_i, b_i, *_ = ols(lx[:i], ly[:i])
        preds.append(a_i + b_i * lx[i])      # model log-price prediction for year i
        actuals.append(ly[i])
        naive.append(ly[i - 1])              # naive: last year's price (zero-drift)
    preds, actuals, naive = map(np.array, (preds, actuals, naive))
    if len(preds):
        mape_model = float(np.mean(np.abs(np.exp(preds) - np.exp(actuals)) / np.exp(actuals)))
        mape_naive = float(np.mean(np.abs(np.exp(naive) - np.exp(actuals)) / np.exp(actuals)))
        # directional: does model predict the sign of YoY change?
        dir_model = np.sign(preds - naive) == np.sign(actuals - naive)
        hit = float(dir_model.mean())
        print(f"\nWalk-forward OOS (test years {yrs[start]}-{yrs[-1]}, n={len(preds)}):")
        print(f"  MAPE  model = {mape_model:.1%}   naive zero-drift = {mape_naive:.1%}   "
              f"-> model {'BEATS' if mape_model < mape_naive else 'LOSES vs'} naive")
        print(f"  YoY direction hit-rate = {hit:.0%}  (50% = coin flip)")


if __name__ == "__main__":
    # China's name in PSD is "China" — verify and use for the ex-China aggregate.
    analyse("cotton")
    analyse("cotton", exclude=("China",), tag="  (world-ex-China 'free' SUR)")
    analyse("coffee")
    print("\nNOTE: cocoa is NOT in USDA PSD (tracked by ICCO) — excluded here.")
