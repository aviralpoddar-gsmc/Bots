"""Fetch deep daily history from public, keyless feeds for the research pipeline.

The runtime ingest sources (`quantbots ingest`) pull only the *latest*
observation — sufficient for the runners, useless for correlation work. This
module fetches multi-year daily history and caches it to disk so re-runs are
instant.

Sources:
- yfinance: daily OHLCV for commodities, ETFs, stocks, indices, currencies.
- FRED: macro series (rates, mortgages, inflation, money supply) via the
  keyless graph endpoint.
- NOAA: ENSO ONI monthly anomaly via the ASCII page already parsed in sources/noaa.py.

Each fetch returns a DataFrame indexed by date. The `fetch_universe` driver
builds one wide aligned panel from a name→ticker map.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "research" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(source: str, key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_").replace("=", "_").replace("^", "idx_")
    return CACHE_DIR / f"{source}__{safe}.pkl"


def _fresh_enough(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < max_age_hours


def fetch_yf_history(
    ticker: str, *, period: str = "5y", interval: str = "1d",
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """Daily OHLCV history from yfinance for one ticker.

    Returns DataFrame indexed by date with 'Close' column (others kept if present).
    Caches as parquet. Period strings: '1mo', '6mo', '1y', '2y', '5y', '10y', 'max'.
    """
    import yfinance as yf

    cache = _cache_path("yf", f"{ticker}_{period}_{interval}")
    if _fresh_enough(cache, max_age_hours):
        return pd.read_pickle(cache)
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        logger.warning("yfinance %s: empty response", ticker)
        return pd.DataFrame(columns=["Close"])
    # Normalize: tz-naive index, single Close column kept (others optional).
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index.name = "Date"
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    df.to_pickle(cache)
    return df


def fetch_fred_history(series_id: str, *, max_age_hours: float = 24.0) -> pd.DataFrame:
    """Full history for a FRED series. Returns DataFrame with 'value' column."""
    cache = _cache_path("fred", series_id)
    if _fresh_enough(cache, max_age_hours):
        return pd.read_pickle(cache)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    resp = requests.get(url, params={"id": series_id}, timeout=30)
    resp.raise_for_status()
    # FRED CSV header is currently `observation_date,<SERIES>` — keep this
    # auto-detecting in case they change column names again.
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date", series_id: "value"}).set_index("Date").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df.to_pickle(cache)
    return df


def fetch_noaa_oni_history(*, max_age_hours: float = 168.0) -> pd.DataFrame:
    """NOAA Oceanic Niño Index monthly anomaly."""
    cache = _cache_path("noaa", "oni")
    if _fresh_enough(cache, max_age_hours):
        return pd.read_pickle(cache)
    url = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows: list[tuple[str, float]] = []
    in_table = False
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Year"):
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for i, val in enumerate(parts[1:], start=1):
            try:
                rows.append((f"{year}-{i:02d}-15", float(val)))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["Date", "value"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.to_pickle(cache)
    return df


# The default research universe — what the clone has the most markets on, AND
# what yfinance has reliable daily data for. Keep this list curated; bad
# tickers pollute the correlation matrix with NaN.
DEFAULT_UNIVERSE: dict[str, str] = {
    # Precious metals (futures front-month)
    "GOLD":         "GC=F",
    "SILVER":       "SI=F",
    "PLATINUM":     "PL=F",
    "PALLADIUM":    "PA=F",
    # Base / industrial metals
    "COPPER":       "HG=F",
    # Energy
    "WTI_OIL":      "CL=F",
    "BRENT_OIL":    "BZ=F",
    "NATGAS":       "NG=F",
    "GASOLINE":     "RB=F",
    "HEATING_OIL":  "HO=F",
    # Soft commodities
    "CORN":         "ZC=F",
    "WHEAT":        "ZW=F",
    "SOYBEANS":     "ZS=F",
    "COTTON":       "CT=F",
    "SUGAR":        "SB=F",
    "COCOA":        "CC=F",
    "COFFEE":       "KC=F",
    # Equities — defense (clone has KTOS, AVAV, etc.)
    "KTOS":         "KTOS",
    "AVAV":         "AVAV",
    # Equities — critical minerals (clone has heavy MP Materials, REE Corp coverage)
    "MP":           "MP",
    "ALB":          "ALB",       # Albemarle, biggest pure-play lithium miner
    # Broad benchmarks for context
    "SPX":          "^GSPC",
    "DXY":          "DX=F",      # US Dollar Index
    "TLT":          "TLT",       # 20+ year Treasury ETF (rate sensitivity)
    # Crypto for completeness
    "BTC":          "BTC-USD",
}

# FRED macro series that pair nicely against commodities.
DEFAULT_FRED: dict[str, str] = {
    "FRED_MORTGAGE30US":   "MORTGAGE30US",  # US 30Y mortgage rate
    "FRED_DGS10":          "DGS10",          # 10Y treasury yield
    "FRED_DTWEXBGS":       "DTWEXBGS",       # Trade-weighted dollar index
    "FRED_VIXCLS":         "VIXCLS",         # VIX
}


def fetch_universe(
    yf_symbols: dict[str, str] | None = None,
    fred_series: dict[str, str] | None = None,
    *, period: str = "5y", include_oni: bool = True, max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """One aligned wide DataFrame of Close prices, indexed by date, one column
    per entity. NaN where a series is missing — analysis layer chooses what to
    do (dropna, ffill, drop column, etc.).
    """
    cols: dict[str, pd.Series] = {}
    yf_symbols = yf_symbols or DEFAULT_UNIVERSE
    fred_series = fred_series or DEFAULT_FRED

    for entity, ticker in yf_symbols.items():
        try:
            df = fetch_yf_history(ticker, period=period, max_age_hours=max_age_hours)
            if not df.empty and "Close" in df.columns:
                cols[entity] = df["Close"].astype(float)
        except Exception as e:  # noqa: BLE001
            logger.warning("yf %s (%s) failed: %s", entity, ticker, e)

    for entity, sid in fred_series.items():
        try:
            df = fetch_fred_history(sid, max_age_hours=max_age_hours)
            if not df.empty:
                cols[entity] = df["value"].astype(float)
        except Exception as e:  # noqa: BLE001
            logger.warning("fred %s (%s) failed: %s", entity, sid, e)

    if include_oni:
        try:
            df = fetch_noaa_oni_history()
            if not df.empty:
                cols["ENSO_ONI"] = df["value"].astype(float)
        except Exception as e:  # noqa: BLE001
            logger.warning("noaa ONI failed: %s", e)

    if not cols:
        return pd.DataFrame()
    panel = pd.concat(cols, axis=1).sort_index()
    panel.index.name = "Date"
    return panel
