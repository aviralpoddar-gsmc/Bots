"""Build data/factors/dcot.csv from the free CFTC Disaggregated COT (Socrata, no key).

Pulls Managed-Money positioning (2018→now) for the futures that map to our equities,
computes the net-position ratio + its weekly change + a rolling Z-score, and stamps the
point-in-time actionable_date (Tuesday report -> published Friday -> tradeable next Monday,
rolled forward over weekends). Run: .venv/bin/python scripts/build_dcot.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
# CFTC contract_market_code -> our factor ticker
CODES = {"085692": "HG", "067651": "CL", "088691": "GC", "084691": "SI", "023651": "NG"}
START = "2018-01-01"
ZWIN = 52  # weeks of burn-in for the Z-score


def _pull(code: str) -> pd.DataFrame:
    params = {"cftc_contract_market_code": code,
              "$where": f"report_date_as_yyyy_mm_dd >= '{START}T00:00:00.000'",
              "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": "100000"}
    rows = requests.get(URL, params=params, timeout=60).json()
    recs = []
    for r in rows:
        try:
            recs.append({
                "report_date": pd.Timestamp(r["report_date_as_yyyy_mm_dd"][:10]),
                "mm_long": float(r["m_money_positions_long_all"]),
                "mm_short": float(r["m_money_positions_short_all"]),
                "open_interest": float(r["open_interest_all"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(recs).sort_values("report_date")


def main() -> None:
    out = []
    for code, tk in CODES.items():
        df = _pull(code)
        if df.empty:
            print(f"  {tk}: no rows"); continue
        df["net_pos_ratio"] = (df["mm_long"] - df["mm_short"]) / df["open_interest"]
        df["1w_change_ratio"] = df["net_pos_ratio"].diff()
        mu = df["1w_change_ratio"].rolling(ZWIN, min_periods=ZWIN // 2).mean()
        sd = df["1w_change_ratio"].rolling(ZWIN, min_periods=ZWIN // 2).std()
        df["1w_change_z"] = (df["1w_change_ratio"] - mu) / sd
        df["publish_date"] = df["report_date"] + pd.Timedelta(days=3)        # Tue -> Fri
        act = df["report_date"] + pd.Timedelta(days=6)                        # -> next Monday
        df["actionable_date"] = act.where(act.dt.weekday < 5, act + pd.Timedelta(days=1))
        df["ticker"] = tk
        out.append(df)
        print(f"  {tk}: {len(df)} weeks {df['report_date'].min().date()}..{df['report_date'].max().date()}")
    full = pd.concat(out, ignore_index=True)
    cols = ["report_date", "publish_date", "actionable_date", "ticker", "mm_long",
            "mm_short", "open_interest", "net_pos_ratio", "1w_change_ratio", "1w_change_z"]
    for c in ("report_date", "publish_date", "actionable_date"):
        full[c] = full[c].dt.strftime("%Y-%m-%d")
    path = Path("data/factors/dcot.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    full[cols].to_csv(path, index=False)
    print(f"wrote {len(full)} rows -> {path}")


if __name__ == "__main__":
    main()
