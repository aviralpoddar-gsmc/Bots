"""Append SHFE base/precious-metal carry to data/factors/carry.csv (free, keyless).

SHFE publishes a full daily settlement curve per metal at
https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{YYYYMMDD}.dat (JSON). We take
the two nearest delivery months per metal to compute front-vs-second carry — a Chinese-
market PROXY for copper/gold/silver/aluminum (SHFE-LME/COMEX basis exists, but carry as a
ratio is comparable). Sampled weekly 2018->2024-04 to align with the EIA energy carry
window. Maps SHFE product -> our commodity ticker so the cross-sectional validator can use it.

Run: .venv/bin/python scripts/build_shfe_carry.py
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# SHFE product id (without _f) -> our commodity ticker (validator maps these to yfinance)
PRODUCTS = {"cu": "HG", "au": "GC", "ag": "SI", "al": "AL"}
START, END = date(2018, 1, 1), date(2024, 4, 5)   # align with EIA energy window
URL = "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{}.dat"
HDRS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.shfe.com.cn/"}


def _carry_for_day(d: date) -> dict[str, float]:
    """Return {ticker: carry_ann} for one trade date, or {} if no file/holiday."""
    try:
        r = requests.get(URL.format(d.strftime("%Y%m%d")), headers=HDRS, timeout=20)
        if r.status_code != 200:
            return {}
        rows = r.json().get("o_curinstrument", [])
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for pid, tk in PRODUCTS.items():
        legs = [x for x in rows if str(x.get("PRODUCTID", "")).strip().lower().startswith(pid + "_")
                and x.get("SETTLEMENTPRICE") and float(x["SETTLEMENTPRICE"]) > 0]
        legs = sorted(legs, key=lambda x: str(x["DELIVERYMONTH"]))[:2]
        if len(legs) < 2:
            continue
        f, s = float(legs[0]["SETTLEMENTPRICE"]), float(legs[1]["SETTLEMENTPRICE"])
        out[tk] = (f - s) / f * 12.0    # backwardation>0, annualized monthly spacing
    return out


def main() -> None:
    recs = []
    d = START
    n_days = 0
    while d <= END:
        if d.weekday() == 2:  # weekly sample (Wednesdays)
            c = _carry_for_day(d)
            for tk, carry in c.items():
                recs.append({"date": d.isoformat(), "ticker": tk, "carry_ann": carry})
            n_days += 1
            time.sleep(0.15)
        d += timedelta(days=1)
    if not recs:
        print("SHFE: no data pulled"); return
    df = pd.DataFrame(recs)
    print(f"SHFE: {n_days} sampled days, {len(df)} metal-rows, tickers={sorted(df['ticker'].unique())}")
    # rolling z-score per ticker
    df = df.sort_values(["ticker", "date"])
    df["signal_zscore"] = df.groupby("ticker")["carry_ann"].transform(
        lambda s: (s - s.rolling(20, min_periods=10).mean()) / s.rolling(20, min_periods=10).std())
    for col in ("spot_close", "front_month_px", "second_month_px"):
        df[col] = ""        # not needed downstream (validator/fusion use signal_zscore)
    df["days_to_roll"] = 30
    cols = ["date", "ticker", "spot_close", "front_month_px", "second_month_px",
            "days_to_roll", "carry_ann", "signal_zscore"]
    path = Path("data/factors/carry.csv")
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    existing = existing[~existing["ticker"].isin(PRODUCTS.values())]   # replace any prior SHFE rows
    out = pd.concat([existing, df[cols]], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"merged -> {path}: {len(out)} rows, tickers now {sorted(out['ticker'].unique())}")


if __name__ == "__main__":
    main()
