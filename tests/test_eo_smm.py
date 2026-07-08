"""SMM Snowflake reader: SQL escaping, splice cache keys, backfill fallback."""

import pandas as pd
import pytest

from quantbots.equity_options.sources import smm
from quantbots.equity_options.sources import tal_snowflake as ts


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolate cache dir + backfill csv; capture SQL sent to ts.query."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(smm, "CACHE_DIR", cache)
    csv = tmp_path / "metals_master_daily.csv"
    pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01", "2023-06-02", "2026-04-03"]),
        "li_smm_usd_t": [10000.0, 10100.0, 9000.0],
    }).to_csv(csv, index=False)
    monkeypatch.setattr(smm, "_BACKFILL_CSV", csv)
    sql_log = []

    def fake_query(sql, **kw):
        sql_log.append(sql)
        return [{"D": "2026-04-03", "P": 8800.0}, {"D": "2026-04-04", "P": 8900.0}]

    monkeypatch.setattr(smm.ts, "query", fake_query)
    return sql_log


def test_sql_escapes_quotes_and_backslashes(sandbox):
    smm.list_series(category="Li' OR 1=1 --")
    assert "'%Li'' OR 1=1 --%'" in sandbox[-1]
    smm.list_series(category="a\\'b")
    assert "a\\\\''b" in sandbox[-1]


def test_splice_flag_gets_its_own_cache_entry(sandbox):
    spliced = smm.spot_series("SMM-Li-LC-001", splice=True)
    unspliced = smm.spot_series("SMM-Li-LC-001", splice=False)
    assert (spliced["source"] == "backfill").any()
    assert not (unspliced["source"] == "backfill").any()  # stale spliced cache must not leak


def test_splice_trims_backfill_and_snowflake_wins(sandbox):
    df = smm.spot_series("SMM-Li-LC-001", splice=True)
    # backfill row on snowflake's start date (2026-04-03) must be dropped
    assert df.loc["2026-04-03", "source"] == "snowflake"
    assert float(df.loc["2026-04-03", "price_usd_t"]) == 8800.0
    assert df.loc["2023-06-01", "source"] == "backfill"


def test_backfill_fallback_when_snowflake_down(sandbox, monkeypatch):
    def down(sql, **kw):
        raise ts.TalUnavailable("snowflake down")
    monkeypatch.setattr(smm.ts, "query", down)
    df = smm.spot_series("SMM-Li-LC-001", splice=True)
    assert len(df) == 3 and (df["source"] == "backfill").all()


def test_raises_when_no_data_anywhere(sandbox, monkeypatch):
    def down(sql, **kw):
        raise ts.TalUnavailable("snowflake down")
    monkeypatch.setattr(smm.ts, "query", down)
    with pytest.raises(ts.TalUnavailable):
        smm.spot_series("SMM-XX-UNKNOWN", splice=True)  # no backfill column for it


def test_gfex_does_not_cache_empty_result(sandbox, monkeypatch):
    monkeypatch.setattr(smm.ts, "query", lambda sql, **kw: [])
    df = smm.gfex_futures("lc")
    assert df.empty
    assert not list(smm.CACHE_DIR.glob("gfex_*.pkl"))  # nothing pickled
