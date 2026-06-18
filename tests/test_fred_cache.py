"""FRED's public CSV endpoint is sometimes unreachable. fetch_history should
serve the last good data from the on-disk research cache (with a warning) rather
than leaving the backtest dead — but raise loudly when there's no cache at all."""

import pandas as pd
import pytest
import requests

from quantbots.research import data_fetch
from quantbots.sources import fred


def _seed_cache(path):
    df = pd.DataFrame(
        {"value": [6.1, 6.2, 6.3]},
        index=pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
    )
    df.index.name = "Date"
    df.to_pickle(path)


def _boom(*a, **k):
    raise requests.ConnectTimeout("network blocked")


def test_fetch_history_falls_back_to_cache_on_network_failure(monkeypatch, tmp_path):
    path = tmp_path / "fred__X.pkl"
    _seed_cache(path)
    monkeypatch.setattr(data_fetch, "_cache_path", lambda source, key: path)
    monkeypatch.setattr(fred.requests, "get", _boom)

    series = fred.fetch_history("X")

    assert series == [("2026-01-01", 6.1), ("2026-01-08", 6.2), ("2026-01-15", 6.3)]


def test_fetch_history_raises_when_no_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data_fetch, "_cache_path", lambda source, key: tmp_path / "missing.pkl")
    monkeypatch.setattr(fred.requests, "get", _boom)

    with pytest.raises(FileNotFoundError):
        fred.fetch_history("X")
