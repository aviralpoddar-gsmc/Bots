"""Discover which equities a commodity *predicts* — the universe screen.

tal has no equity↔commodity map (its ticker table is commodities only), so we discover
linkages statistically over a curated list of liquid, commodity-sensitive optionable
equities. For each (equity, commodity) we fit a lead-lag predictive regression

    r_eq(t) = a + b · r_commodity(t − k) + m · r_market(t) + e ,   k ∈ {0..max_lag}

pick the lag with the strongest commodity loading, and KEEP a pair only when it is both
**significant** (|t| ≥ t_min) and **out-of-sample stable** (same sign of b in both halves
of the history — the diffusion-bench "holds across folds" bar). k>0 means the commodity
*leads* the stock (genuinely predictive, not just contemporaneous). Reuses
`research.data_fetch.fetch_yf_history` and the OLS pattern from `research.pairs`.

The best surviving commodity per equity becomes that name's driver in the options config.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ...research.data_fetch import DEFAULT_UNIVERSE, fetch_yf_history

logger = logging.getLogger(__name__)

# Only TRUE commodities are valid predictors (DEFAULT_UNIVERSE also holds equity/benchmark
# entries like ALB/MP/SPX/TLT/BTC — using those as "commodities" yields spurious
# equity↔equity hits, e.g. TSLA 'predicted by' ALB).
COMMODITY_ENTITIES = [
    "GOLD", "SILVER", "PLATINUM", "PALLADIUM", "COPPER", "WTI_OIL", "BRENT_OIL",
    "NATGAS", "GASOLINE", "HEATING_OIL", "CORN", "WHEAT", "SOYBEANS", "COTTON",
    "SUGAR", "COCOA", "COFFEE",
]

# Curated liquid, optionable, commodity-sensitive equities across sectors. The screen
# decides WHICH commodity (if any) predicts each — these are just candidates.
CANDIDATE_EQUITIES: dict[str, str] = {
    # Precious / gold-silver miners
    "NEM": "Newmont", "GOLD": "Barrick", "AEM": "Agnico", "FNV": "Franco-Nevada",
    "WPM": "Wheaton", "GDX": "Gold miners ETF", "GDXJ": "Jr gold miners", "PAAS": "Pan American",
    "HL": "Hecla", "AG": "First Majestic", "SIL": "Silver miners ETF",
    # Copper / base / diversified
    "FCX": "Freeport", "SCCO": "Southern Copper", "TECK": "Teck", "VALE": "Vale",
    "RIO": "Rio Tinto", "BHP": "BHP", "COPX": "Copper miners ETF",
    # Steel / aluminum
    "NUE": "Nucor", "STLD": "Steel Dynamics", "CLF": "Cleveland-Cliffs", "X": "US Steel",
    "AA": "Alcoa", "CENX": "Century Aluminum",
    # Critical minerals / lithium / REE
    "MP": "MP Materials", "ALB": "Albemarle", "SQM": "SQM", "LAC": "Lithium Americas",
    "CCJ": "Cameco (uranium)", "URA": "Uranium ETF",
    # Energy E&P / majors / services
    "XOM": "Exxon", "CVX": "Chevron", "COP": "ConocoPhillips", "OXY": "Occidental",
    "DVN": "Devon", "EOG": "EOG", "MRO": "Marathon Oil", "APA": "APA", "FANG": "Diamondback",
    "SLB": "Schlumberger", "HAL": "Halliburton", "BKR": "Baker Hughes", "XLE": "Energy ETF",
    "USO": "Oil ETF", "OIH": "Oil services ETF",
    # Natural gas
    "LNG": "Cheniere", "EQT": "EQT", "AR": "Antero", "RRC": "Range", "CHK": "Chesapeake",
    # Refiners (crack spread)
    "VLO": "Valero", "MPC": "Marathon Petro", "PSX": "Phillips 66",
    # Airlines (jet fuel = oil, typically inverse)
    "DAL": "Delta", "UAL": "United", "AAL": "American", "LUV": "Southwest",
    # Autos / PGM consumers
    "GM": "GM", "F": "Ford", "TSLA": "Tesla",
    # Ag / fertilizer / processors
    "ADM": "ADM", "BG": "Bunge", "MOS": "Mosaic", "NTR": "Nutrien", "CF": "CF Industries",
    "CTVA": "Corteva", "DE": "Deere", "AGCO": "AGCO",
    # Chemicals / packaging
    "DOW": "Dow", "LYB": "LyondellBasell", "IP": "Intl Paper", "WLK": "Westlake",
}


@dataclass
class ScreenResult:
    equity: str
    commodity: str           # best commodity entity key (DEFAULT_UNIVERSE)
    lag: int                 # days the commodity LEADS the equity (0 = contemporaneous)
    beta: float              # commodity loading
    tstat: float             # |t| of the commodity loading (full sample)
    r2: float                # regression R²
    stable: bool             # same sign of beta in both halves of history
    n_obs: int

    @property
    def passes(self) -> bool:
        return self.stable and abs(self.tstat) >= 2.0


def _returns(ticker: str, period: str) -> "np.ndarray | None":
    try:
        df = fetch_yf_history(ticker, period=period)
        close = df["Close"].astype(float)
        return close, np.log(close / close.shift(1)).dropna()
    except Exception as e:  # noqa: BLE001
        logger.debug("screen: %s history failed (%s)", ticker, e)
        return None


def _ols_t(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with t-stats. Returns (coef, tstats, r2). X includes the intercept column."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        tstats = coef / np.where(se > 0, se, np.nan)
    except np.linalg.LinAlgError:
        tstats = np.full(k, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return coef, tstats, r2


def screen_equity(equity: str, *, commodities: list[str] | None = None, period: str = "5y",
                  market: str = "SPY", max_lag: int = 5) -> ScreenResult | None:
    """Find the commodity (and lag) that best predicts an equity. None if none qualify."""
    import pandas as pd

    commodities = commodities or COMMODITY_ENTITIES
    eq = _returns(equity, period)
    mk = _returns(market, period)
    if eq is None or mk is None:
        return None
    eq_ret, mk_ret = eq[1], mk[1]
    best: ScreenResult | None = None
    for c in commodities:
        cm = _returns(DEFAULT_UNIVERSE.get(c, c), period)
        if cm is None:
            continue
        cm_ret = cm[1]
        for lag in range(0, max_lag + 1):
            panel = pd.concat({"eq": eq_ret, "cm": cm_ret.shift(lag), "mk": mk_ret},
                              axis=1).dropna()
            if len(panel) < 250:
                continue
            y = panel["eq"].to_numpy()
            X = np.column_stack([np.ones(len(panel)), panel["cm"].to_numpy(), panel["mk"].to_numpy()])
            coef, tstats, r2 = _ols_t(y, X)
            beta_c, t_c = float(coef[1]), float(tstats[1])
            # OOS stability: same sign of the commodity loading in both halves.
            half = len(panel) // 2
            b1 = np.linalg.lstsq(X[:half], y[:half], rcond=None)[0][1]
            b2 = np.linalg.lstsq(X[half:], y[half:], rcond=None)[0][1]
            stable = (b1 > 0) == (b2 > 0) and abs(b1) > 1e-9 and abs(b2) > 1e-9
            if best is None or abs(t_c) > abs(best.tstat):
                best = ScreenResult(equity=equity, commodity=c, lag=lag, beta=beta_c,
                                    tstat=t_c, r2=r2, stable=stable, n_obs=len(panel))
    return best


def run_screen(*, equities: dict[str, str] | None = None, period: str = "5y",
               max_lag: int = 5) -> list[ScreenResult]:
    """Screen all candidate equities; return those with a significant, stable predictor,
    ranked by |t|."""
    equities = equities or CANDIDATE_EQUITIES
    out: list[ScreenResult] = []
    for tkr in equities:
        res = screen_equity(tkr, period=period, max_lag=max_lag)
        if res and res.passes:
            out.append(res)
        elif res:
            logger.info("screen: %s best=%s lag=%d t=%.1f stable=%s — rejected",
                        tkr, res.commodity, res.lag, res.tstat, res.stable)
    out.sort(key=lambda r: abs(r.tstat), reverse=True)
    return out
