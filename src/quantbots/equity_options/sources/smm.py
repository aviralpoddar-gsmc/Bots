"""Read SMM spot prices from tal's Snowflake, spliced with the local 3-year backfill.

tal ingests SMM into `SOURCES.SMM_SPOT_DAILY` (keyed by `SMM_SERIES_ID`, ex-VAT USD/tonne)
but only from ~2026-04-03. This repo carries a 3-year backfill (2023-06+) in
`research/data/metals_master_daily.csv`, already converted to ex-VAT (÷1.13) so it lines up
exactly with the Snowflake series (validated on the 55-day overlap).

`spot_series()` returns the combined daily series: Snowflake is authoritative from its start
date; the local backfill fills everything before it. Read-only — all access goes through
`tal_snowflake.query` (SELECT-only). Inventory is NOT in Snowflake (gated on SMM's tier).
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import tal_snowflake as ts

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[1] / "research" / "data"
_BACKFILL_CSV = _DATA_DIR / "metals_master_daily.csv"
CACHE_DIR = ts.CACHE_DIR  # reuse data/tal

# SMM_SERIES_ID -> column in metals_master_daily.csv that backfills it (ex-VAT).
BACKFILL_COL: dict[str, str] = {
    "SMM-Li-LC-001": "li_smm_usd_t",   # Battery-grade lithium carbonate
    "SMM-RE-OX-001": "ndpr_smm_usd_t",  # Pr-Nd oxide
}


def _q(s: str) -> str:
    """Escape a value for a Snowflake single-quoted literal. Quote-doubling alone
    is bypassable (backslash escapes inside Snowflake string literals), so escape
    backslashes first. The runner is SELECT-only, but stay airtight anyway."""
    return s.replace("\\", "\\\\").replace("'", "''")


def spot_series(series_id: str, *, splice: bool = True, max_age_hours: float = 12.0):
    """Daily ex-VAT USD/tonne spot for one SMM series, Snowflake + local backfill.

    Returns DataFrame indexed by `date` (DatetimeIndex) with columns
    `price_usd_t` and `source` ('snowflake' | 'backfill'). Snowflake wins on overlap;
    if Snowflake is unreachable the backfill alone is served. Raises
    tal_snowflake.TalUnavailable only when there is no data from either source.
    """
    import pandas as pd

    cache = CACHE_DIR / f"smm_spot_{series_id.replace('-', '_')}{'' if splice else '_sf_only'}.pkl"
    if ts._fresh(cache, max_age_hours):
        return pd.read_pickle(cache)

    try:
        rows = ts.query(
            "select to_char(OBSERVATION_DATE,'YYYY-MM-DD') D, PRICE_AVERAGE P "
            f"from SOURCES.SMM_SPOT_DAILY where SMM_SERIES_ID='{_q(series_id)}' "
            "and PRICE_AVERAGE is not null order by OBSERVATION_DATE")
    except ts.TalUnavailable:
        if not splice:
            raise
        rows = []  # Snowflake down — serve the local backfill alone
    sf = pd.DataFrame(rows)
    if len(sf):
        sf = sf.rename(columns={"D": "date", "P": "price_usd_t"})
        sf["date"] = pd.to_datetime(sf["date"]); sf["source"] = "snowflake"
        sf = sf.set_index("date")[["price_usd_t", "source"]]

    bf = pd.DataFrame()
    col = BACKFILL_COL.get(series_id)
    if splice and col and _BACKFILL_CSV.exists():
        raw = pd.read_csv(_BACKFILL_CSV, parse_dates=["date"]).set_index("date")
        if col in raw.columns:
            bf = raw[[col]].dropna().rename(columns={col: "price_usd_t"})
            bf["source"] = "backfill"
            if len(sf):  # keep backfill only strictly before Snowflake's first date
                bf = bf[bf.index < sf.index.min()]

    combined = pd.concat([bf, sf]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]  # snowflake wins ties
    if combined.empty:
        raise ts.TalUnavailable(f"no SMM data for {series_id} (Snowflake empty, no backfill)")
    combined["price_usd_t"] = combined["price_usd_t"].astype(float)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(cache)
    logger.info("smm spot %s: %d rows %s..%s (%d backfill + %d snowflake)", series_id,
                len(combined), combined.index.min().date(), combined.index.max().date(),
                (combined["source"] == "backfill").sum(), (combined["source"] == "snowflake").sum())
    return combined


def list_series(*, category: str | None = None, limit: int = 500):
    """Catalog of available SMM series in Snowflake (series_id, product, category, span)."""
    import pandas as pd
    where = f"where CATEGORY ilike '%{_q(category)}%' " if category else ""
    rows = ts.query(
        "select SMM_SERIES_ID, max(PRODUCT_NAME) PRODUCT, max(CATEGORY) CATEGORY, "
        "min(OBSERVATION_DATE) MIN_D, max(OBSERVATION_DATE) MAX_D, count(*) N "
        f"from SOURCES.SMM_SPOT_DAILY {where}group by 1 order by 1 limit {int(limit)}")
    return pd.DataFrame(rows)


def gfex_futures(variety: str = "lc", *, max_age_hours: float = 12.0):
    """Raw GFEX futures from Snowflake (per contract per day): settle, OI, volume, term curve.

    Snowflake covers from ~2025-11 only; the repo's research/data CSVs carry the 2023+ backfill
    of the derived active/term-slope series if deeper history is needed.
    """
    import pandas as pd
    cache = CACHE_DIR / f"gfex_{variety}.pkl"
    if ts._fresh(cache, max_age_hours):
        return pd.read_pickle(cache)
    rows = ts.query(
        "select to_char(OBSERVATION_DATE,'YYYY-MM-DD') D, CONTRACT_SYMBOL, DELIVERY_MONTH, "
        "SETTLEMENT, OPEN_INTEREST, VOLUME from SOURCES.GFEX_PRICES_RAW "
        f"where VARIETY='{_q(variety)}' order by OBSERVATION_DATE, DELIVERY_MONTH")
    df = pd.DataFrame(rows)
    if df.empty:
        return df  # don't cache emptiness for 12h
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df
