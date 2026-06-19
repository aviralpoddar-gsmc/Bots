"""Multi-factor directional signals for commodity-linked equities (per the research brief).

Two kinds of factor source:
  - COMPUTED NOW from free point-in-time data: macro real-rates and the US dollar (FRED).
  - INGESTED from point-in-time CSVs the research delivers (carry term-structure, CFTC
    DCOT positioning) — placed in data/factors/, keyed to an `actionable_date` so the
    backtest has no lookahead (publication lags already baked into that column).

Each factor returns a daily signal (z-scored, sign-oriented so +1 = bullish the linked
equities). `fusion.py`/the forecast combine the validated ones into mu_view. The
factor_validate harness measures each one's forward-return IC + t-stat over long history
so we verify the research's claims on OUR data before trading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...research.data_fetch import fetch_fred_history

FACTOR_DIR = Path("data/factors")

# Equity baskets by driver (research §"Tradeable Universe").
PRECIOUS = ["GDX", "GDXJ", "NEM", "AEM", "WPM", "AG", "HL", "PAAS", "SIL", "FNV"]
INDUSTRIAL = ["FCX", "SCCO", "VALE", "TECK", "BHP", "COPX"]
ENERGY = ["XOM", "CVX", "COP", "OXY", "SLB", "HAL", "EQT", "AR"]


def _z(s, win: int = 252):
    import pandas as pd  # noqa: F401
    return (s - s.rolling(win, min_periods=win // 2).mean()) / s.rolling(win, min_periods=win // 2).std()


# --- macro factors computed from FRED (point-in-time, ~zero lag) -------------

def real_rate_signal(*, chg_days: int = 21):
    """Bullish-precious signal from the 10y real yield (FRED DFII10). Lower/falling real
    yields → bullish gold/silver, so signal = z-score of −Δ(real yield)."""
    import pandas as pd
    df = fetch_fred_history("DFII10")            # 10y TIPS real yield, daily %
    ry = pd.to_numeric(df["value"], errors="coerce").dropna()
    return _z(-ry.diff(chg_days)).dropna()


def dollar_signal(*, chg_days: int = 21):
    """Bullish-commodity signal from the broad USD index (FRED DTWEXBGS). Weaker/falling
    dollar → bullish commodities, so signal = z-score of −Δ(dollar)."""
    import pandas as pd
    df = fetch_fred_history("DTWEXBGS")
    dx = pd.to_numeric(df["value"], errors="coerce").dropna()
    return _z(-dx.diff(chg_days)).dropna()


# --- ingested point-in-time CSV factors (research deliverables) --------------

def carry_from_csv():
    """Dataset 1 schema: date,ticker,...,carry_ann,signal_zscore. Returns a wide DataFrame
    [date x commodity] of the carry z-score, or None if the file isn't present yet."""
    import pandas as pd
    p = FACTOR_DIR / "carry.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    col = "signal_zscore" if "signal_zscore" in df.columns else "carry_ann"
    return df.pivot(index="date", columns="ticker", values=col).sort_index()


def positioning_from_csv():
    """Dataset 2 schema: report_date,publish_date,actionable_date,ticker,...,1w_change_ratio.
    Indexed by ACTIONABLE_DATE (Friday/Mon), so no lookahead. Returns wide [date x commodity]
    of the 1w-change z-score (or the raw 1w_change_ratio), or None if not present."""
    import pandas as pd
    p = FACTOR_DIR / "dcot.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["actionable_date"])
    col = "1w_change_z" if "1w_change_z" in df.columns else "1w_change_ratio"
    return df.pivot(index="actionable_date", columns="ticker", values=col).sort_index()


def available_factors() -> dict[str, str]:
    """What's wired up right now (for the CLI to report)."""
    return {
        "real_rate": "computed (FRED DFII10) — precious",
        "dollar": "computed (FRED DTWEXBGS) — all commodities",
        "carry": "CSV data/factors/carry.csv" + ("" if (FACTOR_DIR / "carry.csv").exists() else " [MISSING]"),
        "positioning": "CSV data/factors/dcot.csv" + ("" if (FACTOR_DIR / "dcot.csv").exists() else " [MISSING]"),
    }
