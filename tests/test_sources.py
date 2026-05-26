import io
from contextlib import contextmanager

from quantbots.sources import available, get_source
from quantbots.sources.base import Observation


class _Resp:
    def __init__(self, text="", payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_registry_lists_sources():
    assert {"stooq", "worldbank", "rss"} <= set(available())


def _stooq_csv(symbol, close):
    return (
        "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
        f"{symbol.upper()},2026-05-26,21:00:00,71.0,72.5,70.8,{close},12345\n"
    )


def test_stooq_parses_close(monkeypatch):
    # Stooq is queried one symbol per request; respond based on the `s` param.
    prices = {"cl.f": "72.34", "gc.f": "2405.1"}

    def fake_get(url, params=None, **k):
        sym = params["s"]
        return _Resp(text=_stooq_csv(sym, prices[sym]))

    monkeypatch.setattr("quantbots.sources.stooq.requests.get", fake_get)
    src = get_source("stooq", symbols={"WTI_OIL": "cl.f", "GOLD": "gc.f"})
    obs = {o.entity: o for o in src.fetch()}
    assert obs["WTI_OIL"].value == 72.34
    assert obs["GOLD"].value == 2405.1
    assert obs["WTI_OIL"].source == "stooq"


def test_stooq_skips_no_data(monkeypatch):
    monkeypatch.setattr(
        "quantbots.sources.stooq.requests.get",
        lambda *a, **k: _Resp(text=_stooq_csv("cl.f", "N/D")),
    )
    assert get_source("stooq", symbols={"WTI_OIL": "cl.f"}).fetch() == []


def test_worldbank_parses_latest(monkeypatch):
    payload = [{"page": 1}, [{"date": "2025", "value": 3.1}, {"date": "2024", "value": None}]]
    monkeypatch.setattr("quantbots.sources.worldbank.requests.get", lambda *a, **k: _Resp(payload=payload))
    src = get_source("worldbank", indicators=[{"entity": "US_CPI_YOY", "country": "US", "code": "FP.CPI.TOTL.ZG"}])
    obs = src.fetch()
    assert len(obs) == 1
    assert obs[0].entity == "US_CPI_YOY" and obs[0].value == 3.1


def test_rss_parses_items(monkeypatch):
    xml = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><title>Oil rises on supply fears</title><pubDate>Tue, 26 May 2026 10:00:00 GMT</pubDate><link>http://x/1</link></item>
      <item><title>Copper hits record</title><pubDate>Tue, 26 May 2026 11:00:00 GMT</pubDate><link>http://x/2</link></item>
    </channel></rss>"""

    @contextmanager
    def fake_urlopen(*a, **k):
        yield io.BytesIO(xml.encode())

    monkeypatch.setattr("quantbots.sources.rss.urllib.request.urlopen", fake_urlopen)
    src = get_source("rss", feeds=[{"entity": "NEWS", "url": "http://x/rss"}])
    obs = src.fetch()
    assert len(obs) == 2
    assert obs[0].text == "Oil rises on supply fears"
    assert obs[0].entity == "NEWS" and obs[0].value is None


def test_observation_as_row_roundtrips():
    o = Observation(source="s", entity="e", ts="2026-01-01T00:00:00", value=1.5)
    row = o.as_row()
    assert row["entity"] == "e" and row["value"] == 1.5
