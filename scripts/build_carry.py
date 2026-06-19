"""Build data/factors/carry.csv from EIA's keyless dnav futures files (per deep-research).

EIA Open Data is the only verified FREE, terms-compliant source for a futures curve, and
it covers ONLY WTI crude (RCLC1-4) and Henry Hub natural gas (RNGC1-4), 2018->2024-04
(NYMEX futures discontinued after 2024-04-05). Metals (HG/GC/SI) and Brent are NOT available
free (CME bot-blocked + no-scrape ToS, ICE paid, Stooq JS-PoW, Yahoo ToS) — so carry can be
built/validated for ENERGY only here. Public-domain data; derivative use permitted.

Run: .venv/bin/python scripts/build_carry.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# (ticker, EIA section, contract-1 series, contract-2 series)
SPECS = [
    ("CL", "pet", "RCLC1", "RCLC2"),
    ("NG", "ng", "RNGC1", "RNGC2"),
]
START = "2018-01-01"


def _series(section: str, code: str) -> pd.Series:
    url = f"https://www.eia.gov/dnav/{section}/hist_xls/{code}d.xls"
    df = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "px"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["px"].astype(float)


def main() -> None:
    out = []
    for tk, section, c1, c2 in SPECS:
        s1, s2 = _series(section, c1), _series(section, c2)
        df = pd.concat({"front_month_px": s1, "second_month_px": s2}, axis=1).dropna()
        df = df[df.index >= START]
        df["spot_close"] = df["front_month_px"]
        df["days_to_roll"] = 30
        # backwardation (front>second) -> positive carry; annualize monthly spacing
        df["carry_ann"] = (df["front_month_px"] - df["second_month_px"]) / df["front_month_px"] * 12.0
        mu = df["carry_ann"].rolling(252, min_periods=120).mean()
        sd = df["carry_ann"].rolling(252, min_periods=120).std()
        df["signal_zscore"] = (df["carry_ann"] - mu) / sd
        df["ticker"] = tk
        df["date"] = df.index.strftime("%Y-%m-%d")
        out.append(df)
        print(f"  {tk}: {len(df)} days {df['date'].iloc[0]}..{df['date'].iloc[-1]}")
    full = pd.concat(out, ignore_index=True)
    cols = ["date", "ticker", "spot_close", "front_month_px", "second_month_px",
            "days_to_roll", "carry_ann", "signal_zscore"]
    path = Path("data/factors/carry.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    full[cols].to_csv(path, index=False)
    print(f"wrote {len(full)} rows -> {path}  (ENERGY only; metals/Brent need a paid source)")


if __name__ == "__main__":
    main()
