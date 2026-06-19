"""Read-only reader for tal's Snowflake warehouse (the `MIKHAIL` database).

tal holds exactly what this package was missing — a DIRECTIONAL signal — plus a
curated equity↔commodity map:

  - `PCF.TICKER_REFERENCE`  : TICKER_ID ↔ METAL/MATERIAL_QID ↔ SOURCE_SYMBOL ↔ ASSET_TYPE
  - `MEASURABLE.QUESTION`   : forecast questions (ID, SLUG, JSON_ID, QUESTION)
  - `MEASURABLE.QUESTION_TIMESERIES` : (QUESTION_ID, VALUE, DATE_PRODUCED) — tal's
    expectation VALUE over time, dated → usable with no lookahead.

The fenced `equity_options` package cannot import tal's own deps, so this reader SHELLS
OUT to `~/tal` and runs the query through tal's **sanctioned** `db.connection`
(get_snowflake_connection) under `doppler run` (dev_personal). It is strictly
READ-ONLY — it only issues SELECTs and never writes to tal. Results are cached to
`data/tal/*.parquet` (mirrors research/data_fetch) so backtests are reproducible and
Snowflake isn't hammered.

Access is via DEV_ROLE on the `MIKHAIL` database (the configured per-user dev clone does
not exist; `MIKHAIL` is the live warehouse DEV_ROLE can read). Override with
EQUITY_OPTIONS_TAL_DB / TAL_REPO env vars.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = _REPO_ROOT / "data" / "tal"
TAL_REPO = Path(os.environ.get("TAL_REPO", Path.home() / "tal"))
TAL_DB = os.environ.get("EQUITY_OPTIONS_TAL_DB", "MIKHAIL")
_UV_DEPS = ["snowflake-connector-python", "cryptography", "sqlalchemy", "snowflake-sqlalchemy"]

# Runner executed INSIDE ~/tal: uses tal's own sanctioned connection, SELECT-only.
_RUNNER = r'''
import os, json, datetime, decimal
from db.connection import get_snowflake_connection
sql = os.environ["EO_TAL_SQL"]
low = sql.strip().lower()
if not (low.startswith("select") or low.startswith("with") or low.startswith("show")):
    raise SystemExit("refusing non-SELECT query")
conn = get_snowflake_connection(); cur = conn.cursor()
cur.execute("use database " + os.environ.get("EO_TAL_DB", "MIKHAIL"))
cur.execute(sql)
cols = [d[0] for d in cur.description]
def enc(v):
    if isinstance(v, decimal.Decimal): return float(v)
    if isinstance(v, (datetime.datetime, datetime.date)): return v.isoformat()
    return v
rows = [{c: enc(v) for c, v in zip(cols, r)} for r in cur.fetchall()]
cur.close()
print(json.dumps(rows))
'''


class TalUnavailable(RuntimeError):
    """tal Snowflake could not be reached (no doppler/uv, auth, or network)."""


def query(sql: str, *, timeout: int = 300) -> list[dict]:
    """Run a read-only SELECT against tal's MIKHAIL db via its sanctioned connection.

    Raises TalUnavailable on any failure so callers can fall back gracefully.
    """
    if not TAL_REPO.exists():
        raise TalUnavailable(f"tal repo not found at {TAL_REPO}")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER)
        runner = f.name
    try:
        cmd = ["doppler", "run", "--", "env", f"PYTHONPATH={TAL_REPO}",
               "uv", "run", "--quiet"]
        for d in _UV_DEPS:
            cmd += ["--with", d]
        cmd += ["python", runner]
        env = {**os.environ, "EO_TAL_SQL": sql, "EO_TAL_DB": TAL_DB}
        proc = subprocess.run(cmd, cwd=str(TAL_REPO), env=env, capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0:
            raise TalUnavailable(f"tal query failed: {proc.stderr.strip()[-400:]}")
        out = proc.stdout.strip().splitlines()
        return json.loads(out[-1]) if out else []
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        raise TalUnavailable(f"tal query error: {e}") from e
    finally:
        os.unlink(runner)


# --- cached high-level pulls -------------------------------------------------

def _cache(name: str, max_age_hours: float):
    # Pickle (not parquet) to match research/data_fetch and avoid a pyarrow dep.
    return CACHE_DIR / f"{name}.pkl", max_age_hours


def _fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_hours


def ticker_reference(*, max_age_hours: float = 24.0):
    """tal's curated ticker↔metal map as a DataFrame (cached)."""
    import pandas as pd
    path, age = _cache("ticker_reference", max_age_hours)
    if _fresh(path, age):
        return pd.read_pickle(path)
    rows = query(
        "select TICKER_ID, TICKER_NAME, ASSET_TYPE, CATEGORY, METAL, MATERIAL_QID, "
        "SOURCE_SYMBOL, PRICE_SOURCE, IS_ACTIVE from PCF.TICKER_REFERENCE")
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def question_timeseries(question_ids: list[int], *, max_age_hours: float = 12.0):
    """tal's expectation VALUE over time for the given measurable question IDs.

    Returns DataFrame(QUESTION_ID, VALUE, DATE_PRODUCED) — dated, so callers slice to
    <= as_of for no-lookahead.
    """
    import pandas as pd
    if not question_ids:
        return pd.DataFrame(columns=["QUESTION_ID", "VALUE", "DATE_PRODUCED"])
    key = "ts_" + "_".join(str(q) for q in sorted(question_ids)[:8]) + f"_{len(question_ids)}"
    path, age = _cache(key, max_age_hours)
    if _fresh(path, age):
        return pd.read_pickle(path)
    ids = ",".join(str(int(q)) for q in question_ids)
    rows = query(f"select QUESTION_ID, VALUE, DATE_PRODUCED from MEASURABLE.QUESTION_TIMESERIES "
                 f"where QUESTION_ID in ({ids}) order by DATE_PRODUCED")
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def active_markets(*, min_bettors: int = 10, max_age_hours: float = 6.0):
    """Actively-traded (multi-agent) price markets: the informative crowd consensus.

    Filters to markets with >= min_bettors unique bettors and a USD price unit — these
    have real trading, so their LATEST_MARKET_PROBABILITY is a meaningful prediction (vs
    the stale 0.50/1.0 untraded markets). Returns DataFrame with prob/threshold/direction/
    bettors/volume/settlement + the market question (for metal classification)."""
    import pandas as pd
    path, age = _cache(f"active_mkts_{min_bettors}", max_age_hours)
    if _fresh(path, age):
        return pd.read_pickle(path)
    rows = query(
        "select MARKET_QUESTION, THRESHOLD, THRESHOLD_UNIT, THRESHOLD_DIRECTION, "
        "LATEST_MARKET_PROBABILITY, UNIQUE_BETTOR_COUNT, MANIFOLD_VOLUME, SETTLEMENT_DATE "
        "from MARKET.PREDICTION_MARKET "
        f"where UNIQUE_BETTOR_COUNT >= {int(min_bettors)} and LATEST_MARKET_PROBABILITY is not null "
        "and THRESHOLD_UNIT ilike 'USD%' and THRESHOLD_DIRECTION = 'exceeds'")
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


_MATERIAL_CASE = """
  case when MARKET_QUESTION ilike '%gold%' then 'GOLD'
       when MARKET_QUESTION ilike '%silver%' then 'SILVER'
       when MARKET_QUESTION ilike '%platinum%' then 'PLATINUM'
       when MARKET_QUESTION ilike '%palladium%' then 'PALLADIUM'
       when MARKET_QUESTION ilike '%copper%' then 'COPPER'
       when MARKET_QUESTION ilike '%brent%' then 'BRENT'
       when MARKET_QUESTION ilike '%wti%' or MARKET_QUESTION ilike '%crude%' then 'WTI'
       when MARKET_QUESTION ilike '%natural gas%' or MARKET_QUESTION ilike '%henry hub%' then 'NATGAS'
  end"""


def daily_material_consensus(*, min_bettors: int = 10, max_age_hours: float = 6.0):
    """Daily avg consensus probability per material over the available history window.

    Price-level markets only (excludes treatment-charge/premium/spread/margin/basis), so
    the day-over-day CHANGE is a clean directional revision (threshold placement cancels).
    Returns DataFrame(material, day, avg_prob, n_markets). ~8 materials x ~80 days.
    """
    import pandas as pd
    path, age = _cache(f"daily_consensus_{min_bettors}", max_age_hours)
    if _fresh(path, age):
        return pd.read_pickle(path)
    rows = query(f"""
      select {_MATERIAL_CASE} as material, to_char(date_trunc('day', h.DATE_RECORDED),'YYYY-MM-DD') as day,
             avg(h.PROBABILITY) as avg_prob, count(distinct h.MARKET_ID) as n_markets
      from MARKET.MARKET_PRICE_HISTORY h join MARKET.PREDICTION_MARKET pm on pm.ID = h.MARKET_ID
      where pm.UNIQUE_BETTOR_COUNT >= {int(min_bettors)} and pm.THRESHOLD_UNIT ilike 'USD%'
        and pm.THRESHOLD_DIRECTION = 'exceeds'
        and pm.MARKET_QUESTION not ilike '%treatment%' and pm.MARKET_QUESTION not ilike '%premium%'
        and pm.MARKET_QUESTION not ilike '%spread%' and pm.MARKET_QUESTION not ilike '%margin%'
        and pm.MARKET_QUESTION not ilike '%discount%' and pm.MARKET_QUESTION not ilike '%basis%'
        and pm.MARKET_QUESTION not ilike '%ratio%'
      group by 1, 2 having material is not null order by 1, 2""")
    df = pd.DataFrame(rows)
    if len(df):
        df.columns = [c.lower() for c in df.columns]  # Snowflake returns UPPERCASE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def find_questions(like: str, *, limit: int = 50) -> list[dict]:
    """Search measurable questions by SLUG/JSON_ID/text (ILIKE). Not cached (ad-hoc)."""
    safe = like.replace("'", "''")
    return query(
        "select ID, SLUG, JSON_ID, LEFT(QUESTION,160) as QUESTION from MEASURABLE.QUESTION "
        f"where SLUG ilike '%{safe}%' or JSON_ID ilike '%{safe}%' or QUESTION ilike '%{safe}%' "
        f"limit {int(limit)}")


# Commodity -> (MARKET_QUESTION name pattern, threshold unit) for SPOT-PRICE markets.
# Only the majors with a clean spot ladder; others abstain (no tal price view).
COMMODITY_MARKET_PATTERNS: dict[str, tuple[str, str]] = {
    "GOLD": ("gold spot price", "USD per troy oz"),
    "SILVER": ("silver spot price", "USD per troy oz"),
    "PLATINUM": ("platinum spot price", "USD per troy oz"),
    "PALLADIUM": ("palladium spot price", "USD per troy oz"),
    "COPPER": ("copper", "USD per tonne"),
    "WTI_OIL": ("wti", "USD"),
    "BRENT_OIL": ("brent", "USD"),
}


def commodity_price_markets(commodity: str, *, max_age_hours: float = 12.0):
    """Spot-price threshold markets for a major commodity (ID, THRESHOLD, direction,
    LATEST_MARKET_PROBABILITY, SETTLEMENT_DATE). Excludes premium/basis/spread/ratio
    questions. Cached. Empty DataFrame if the commodity has no clean ladder."""
    import pandas as pd
    pat = COMMODITY_MARKET_PATTERNS.get(commodity)
    if pat is None:
        return pd.DataFrame()
    name, unit = pat
    path, age = _cache(f"pm_{commodity}", max_age_hours)
    if _fresh(path, age):
        return pd.read_pickle(path)
    name_q = name.replace("'", "''")
    rows = query(
        "select ID, THRESHOLD, THRESHOLD_DIRECTION, LATEST_MARKET_PROBABILITY, "
        "SETTLEMENT_DATE from MARKET.PREDICTION_MARKET "
        f"where MARKET_QUESTION ilike '%{name_q}%' and THRESHOLD_UNIT = '{unit}' "
        "and THRESHOLD_DIRECTION = 'exceeds' and SETTLEMENT_DATE is not null "
        "and MARKET_QUESTION not ilike '%premium%' and MARKET_QUESTION not ilike '%basis%' "
        "and MARKET_QUESTION not ilike '%spread%' and MARKET_QUESTION not ilike '%ratio%' "
        "and MARKET_QUESTION not ilike '%discount%' and MARKET_QUESTION not ilike '%minus%'")
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def market_price_history(market_ids: list[int], *, max_age_hours: float = 12.0):
    """Probability history (MARKET_ID, PROBABILITY, DATE_RECORDED) for as-of reads."""
    import pandas as pd
    if not market_ids:
        return pd.DataFrame(columns=["MARKET_ID", "PROBABILITY", "DATE_RECORDED"])
    key = "pmh_" + "_".join(str(m) for m in sorted(market_ids)[:6]) + f"_{len(market_ids)}"
    path, age = _cache(key, max_age_hours)
    if _fresh(path, age):
        return pd.read_pickle(path)
    ids = ",".join(str(int(m)) for m in market_ids)
    rows = query("select MARKET_ID, PROBABILITY, DATE_RECORDED from MARKET.MARKET_PRICE_HISTORY "
                 f"where MARKET_ID in ({ids}) order by DATE_RECORDED")
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def probe() -> dict:
    """Summarize what's reachable — backs `eo tal-probe`."""
    out: dict = {"db": TAL_DB, "repo": str(TAL_REPO)}
    try:
        out["context"] = query("select current_database(), current_role()")
        for tbl in ("PCF.TICKER_REFERENCE", "MEASURABLE.QUESTION",
                    "MEASURABLE.QUESTION_TIMESERIES", "MARKET.PREDICTION_MARKET"):
            r = query(f"select count(*) as n from {tbl}")
            out[tbl] = r[0]["N"] if r else None
    except TalUnavailable as e:
        out["error"] = str(e)
    return out
