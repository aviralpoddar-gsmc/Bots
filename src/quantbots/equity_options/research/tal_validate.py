"""Validate the tal multi-agent consensus signal with the ~4 months of history we have.

A standard walk-forward gate needs years. With only ~4 months we extract power by
construction: treat the consensus as a CROSS-SECTIONAL factor across ~8 commodities and
test it at daily frequency.

Two signals per material per day:
  - LEVEL  = avg P(exceeds) − 0.5  (biased by where thresholds sit, kept for reference)
  - CHANGE = Δ avg P over `chg_days`  (the crowd's REVISION; threshold placement cancels
             in the difference, so this is the clean directional signal)

Three measures vs the commodity's forward `h`-day return (commodity proxy, not the equity,
to remove idiosyncratic noise):
  1. Information Coefficient: daily cross-sectional Spearman rank-corr(signal, fwd_ret);
     mean IC + an overlap-deflated t-stat (n_eff = n_days / h).
  2. Long-short factor: every `h` days (NON-overlapping → independent), long the
     above-median-signal materials, short below; realize the next h-day return. Sharpe +
     t-stat over the independent periods.
  3. Hit rate of the factor periods.

This is a preliminary, small-sample read (~8 assets, ~4 months) — suggestive, NOT the
multi-year gate. It tells us whether the signal is worth carrying forward to a real gate
once the daily logger accrues history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# material -> commodity futures proxy (DEFAULT_UNIVERSE key)
PROXY = {"GOLD": "GOLD", "SILVER": "SILVER", "PLATINUM": "PLATINUM", "PALLADIUM": "PALLADIUM",
         "COPPER": "COPPER", "WTI": "WTI_OIL", "BRENT": "BRENT_OIL", "NATGAS": "NATGAS"}


@dataclass
class SignalResult:
    name: str
    mean_ic: float
    ic_t: float
    factor_sharpe_ann: float
    factor_t: float
    factor_hit: float
    n_days: int
    n_periods: int


def _proxy_closes(period: str = "1y"):
    import pandas as pd

    from ...research.data_fetch import DEFAULT_UNIVERSE, fetch_yf_history
    cols = {}
    for mat, key in PROXY.items():
        t = DEFAULT_UNIVERSE.get(key)
        try:
            cols[mat] = fetch_yf_history(t, period=period)["Close"].astype(float)
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(cols, axis=1).sort_index()


def _evaluate(signal, fwd, *, h: int) -> SignalResult | None:
    """signal, fwd: wide DataFrames [day x material], aligned. Returns IC + factor stats."""
    import pandas as pd
    from scipy.stats import spearmanr

    # daily cross-sectional IC
    ics = []
    for day in signal.index:
        s = signal.loc[day]; f = fwd.loc[day]
        m = s.notna() & f.notna()
        if m.sum() >= 4:
            ic, _ = spearmanr(s[m], f[m])
            if ic == ic:
                ics.append(ic)
    if len(ics) < 10:
        return None
    ics = np.array(ics)
    n_eff = max(len(ics) / h, 2)
    ic_t = ics.mean() / (ics.std(ddof=1) / math.sqrt(n_eff)) if ics.std() > 0 else 0.0

    # non-overlapping long-short factor, rebalanced every h days
    days = list(signal.index)
    rets = []
    for i in range(0, len(days) - h, h):
        day = days[i]
        s = signal.loc[day].dropna()
        f = fwd.loc[day]
        if len(s) < 4:
            continue
        med = s.median()
        longs = s[s > med].index
        shorts = s[s < med].index
        rl = f[longs].mean(); rs = f[shorts].mean()
        if rl == rl and rs == rs:
            rets.append(float(rl - rs))
    if len(rets) < 3:
        return None
    r = np.array(rets)
    periods_per_year = 252 / h
    sharpe = (r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year)) if r.std() > 0 else 0.0
    ft = r.mean() / (r.std(ddof=1) / math.sqrt(len(r))) if r.std() > 0 else 0.0
    return SignalResult("", float(ics.mean()), float(ic_t), float(sharpe), float(ft),
                        float((r > 0).mean()), len(ics), len(r))


def validate(consensus_df, *, h: int = 5, chg_days: int = 5) -> dict[str, SignalResult]:
    """consensus_df: from tal_snowflake.daily_material_consensus. Returns {signal: result}."""
    import pandas as pd
    if consensus_df is None or len(consensus_df) == 0:
        return {}
    wide = consensus_df.pivot(index="day", columns="material", values="avg_prob")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().reindex(columns=[m for m in PROXY if m in wide.columns])
    # align to business days, ffill the consensus
    closes = _proxy_closes()
    closes = closes.reindex(columns=wide.columns)
    idx = closes.index[(closes.index >= wide.index.min()) & (closes.index <= wide.index.max())]
    cons = wide.reindex(idx).ffill()
    px = closes.reindex(idx)
    fwd = px.shift(-h) / px - 1.0          # forward h-day return per material

    level = cons - 0.5
    change = cons.diff(chg_days)
    out = {}
    for name, sig in (("change", change), ("level", level)):
        res = _evaluate(sig.dropna(how="all"), fwd, h=h)
        if res:
            res.name = name
            out[name] = res
    return out
