"""Signal-level validation of macro/factor timing signals over long history.

For a daily macro signal (e.g. real-rate or dollar) and a basket of linked equities, this
measures whether the signal PREDICTS the basket's forward h-day return — the same question
the research answers, reproduced on OUR data and our (long) history:

  - IC: time-series correlation of signal_t vs the basket's forward h-day return, with an
        overlap-deflated t-stat (n_eff = n_days / h).
  - Timing factor: NON-overlapping h-day periods, long the basket when signal>0 (else flat);
        Sharpe + t-stat over the independent periods.
  - corr_to_momentum: how independent the signal is from 12-month commodity trend (we want low).

This validates the *signal*; the option-spread gate is the separate execution test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FactorResult:
    name: str
    ic: float
    ic_t: float
    timing_sharpe: float
    timing_t: float
    hit: float
    corr_to_momentum: float
    n_days: int
    n_periods: int


def _basket_fwd_return(equities, *, period: str, h: int):
    """Equal-weight basket forward h-day return series (business-day indexed)."""
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history
    cols = {}
    for e in equities:
        try:
            cols[e] = fetch_yf_history(e, period=period)["Close"].astype(float)
        except Exception:  # noqa: BLE001
            pass
    px = pd.concat(cols, axis=1).sort_index()
    fwd = (px.shift(-h) / px - 1.0).mean(axis=1)     # equal-weight basket forward return
    return fwd.dropna()


# CSV-factor commodity ticker -> yfinance futures proxy (for forward returns)
_TICK2YF = {"CL": "CL=F", "CO": "BZ=F", "HG": "HG=F", "GC": "GC=F", "SI": "SI=F",
            "NG": "NG=F", "AL": "ALI=F"}


def validate_csv_factor(panel, *, h: int = 21, period: str = "5y"):
    """Cross-sectional validation of a carry/positioning panel [date x commodity-ticker].

    Each date, does the factor rank the commodities in the order of their forward h-day
    returns? Reuses the cross-sectional IC + long-short engine. Returns a SignalResult
    (from tal_validate) or None if there isn't enough history (the gate that protects us
    from validating on a 2-row sample)."""
    import pandas as pd

    from ...research.data_fetch import fetch_yf_history
    from .tal_validate import _evaluate
    if panel is None or len(panel) == 0:
        return None
    panel = panel.copy(); panel.index = pd.to_datetime(panel.index)
    cols = [c for c in panel.columns if c in _TICK2YF]
    closes = {}
    for c in cols:
        try:
            closes[c] = fetch_yf_history(_TICK2YF[c], period=period)["Close"].astype(float)
        except Exception:  # noqa: BLE001
            pass
    if not closes:
        return None
    px = pd.concat(closes, axis=1).sort_index()
    idx = px.index[(px.index >= panel.index.min()) & (px.index <= panel.index.max())]
    if len(idx) < 2:
        return None
    sig = panel[list(closes)].reindex(idx).ffill()
    fwd = (px[list(closes)].shift(-h) / px[list(closes)] - 1.0).reindex(idx)
    return _evaluate(sig.dropna(how="all"), fwd, h=h)


def validate_signal(signal, equities, *, period: str = "12y", h: int = 21,
                    momentum_proxy: str | None = None) -> FactorResult | None:
    """signal: a daily pandas Series (the macro factor). equities: linked basket."""
    import pandas as pd

    fwd = _basket_fwd_return(equities, period=period, h=h)
    sig = signal.reindex(fwd.index).ffill()
    df = pd.concat({"sig": sig, "fwd": fwd}, axis=1).dropna()
    if len(df) < 60:
        return None
    s, f = df["sig"].to_numpy(), df["fwd"].to_numpy()
    ic = float(np.corrcoef(s, f)[0, 1]) if s.std() > 0 else 0.0
    n_eff = max(len(df) / h, 2)
    ic_t = ic * math.sqrt(n_eff)

    # non-overlapping timing factor: long basket when signal>0, else flat
    days = list(df.index)
    rets = []
    for i in range(0, len(days) - h, h):
        if df["sig"].iloc[i] > 0:
            rets.append(float(df["fwd"].iloc[i]))
        else:
            rets.append(0.0)
    r = np.array([x for x in rets])
    active = r[r != 0]
    if len(active) >= 3 and active.std() > 0:
        ppy = 252 / h
        sharpe = active.mean() / active.std(ddof=1) * math.sqrt(ppy)
        tt = active.mean() / (active.std(ddof=1) / math.sqrt(len(active)))
        hit = float((active > 0).mean())
    else:
        sharpe = tt = hit = 0.0

    # correlation to 12-month momentum of the driver
    corr_mom = float("nan")
    if momentum_proxy:
        try:
            from ...research.data_fetch import fetch_yf_history
            c = fetch_yf_history(momentum_proxy, period=period)["Close"].astype(float)
            mom = (c / c.shift(252) - 1.0)
            j = pd.concat({"sig": signal, "mom": mom}, axis=1).dropna()
            if len(j) > 60:
                corr_mom = float(j["sig"].corr(j["mom"]))
        except Exception:  # noqa: BLE001
            pass
    return FactorResult(name="", ic=ic, ic_t=ic_t, timing_sharpe=sharpe, timing_t=tt,
                        hit=hit, corr_to_momentum=corr_mom, n_days=len(df), n_periods=len(active))
