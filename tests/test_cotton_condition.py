"""Tests for the production-weighted cotton condition index (bot #3 / Terminal 1):
the SIG_COTTON_COND_IDX processing signal and nass_crop's index→national fallback."""

from quantbots.processing.signals import (
    _COTTON_STATE_WEIGHTS,
    compute_cotton_condition_index,
)
from quantbots.strategies.nass_crop import NassCropStrategy


class FakeStore:
    """latest_observation over {entity: value} or {entity: (value, ts)}."""

    def __init__(self, values, default_ts="2026-06-01T00:00:00"):
        self.values = values
        self.default_ts = default_ts

    def latest_observation(self, entity, source=None):
        v = self.values.get(entity)
        if v is None:
            return None
        if isinstance(v, tuple):
            val, ts = v
        else:
            val, ts = v, self.default_ts
        return {"entity": entity, "value": val, "ts": ts}


def _idx(values):
    out = compute_cotton_condition_index(FakeStore(values))
    return out[0] if out else None


def test_index_is_production_weighted():
    # All five states present -> weighted mean by _COTTON_STATE_WEIGHTS.
    raw = {"TX": 40, "GA": 60, "AR": 70, "MS": 65, "NC": 55}
    o = _idx({f"NASS_COTTON_COND_GE_{s}": v for s, v in raw.items()})
    w = _COTTON_STATE_WEIGHTS
    expect = sum(raw[s] * w[s] for s in w) / sum(w.values())
    assert abs(o.value - expect) < 1e-9
    assert o.entity == "SIG_COTTON_COND_IDX" and o.payload["n_states"] == 5


def test_index_renormalizes_over_present_states():
    # Only Texas present -> index == Texas value (weights renormalized to that one state),
    # NOT diluted toward 0. This is the whole point: one stressed major state shows through.
    o = _idx({"NASS_COTTON_COND_GE_TX": 30})
    assert o.value == 30 and o.payload["n_states"] == 1


def test_index_excludes_prior_season_states():
    # TX returns LAST season's final reading (2025); the rest are current (2026).
    # The stale TX must be dropped so seasons aren't blended (it would flip the sign).
    o = _idx({
        "NASS_COTTON_COND_GE_TX": (36, "2025-47T00:00:00"),  # stale — excluded
        "NASS_COTTON_COND_GE_GA": (48, "2026-22T00:00:00"),
        "NASS_COTTON_COND_GE_AR": (74, "2026-22T00:00:00"),
        "NASS_COTTON_COND_GE_MS": (49, "2026-22T00:00:00"),
        "NASS_COTTON_COND_GE_NC": (60, "2026-22T00:00:00"),
    })
    assert "TX" not in o.payload["by_state"] and o.payload["n_states"] == 4
    w = _COTTON_STATE_WEIGHTS
    cur = {"GA": 48, "AR": 74, "MS": 49, "NC": 60}
    expect = sum(cur[s] * w[s] for s in cur) / sum(w[s] for s in cur)
    assert abs(o.value - expect) < 1e-9 and o.value > 50  # current-season -> bearish, not 45.7


def test_index_falls_back_to_national():
    o = _idx({"NASS_COTTON_COND_GE": 58})
    assert o.value == 58 and o.payload.get("fallback") == "national"


def test_index_absent_when_no_data():
    assert _idx({}) is None


def test_nass_crop_prefers_index_then_national():
    s = NassCropStrategy()  # defaults: cond_entity=SIG_COTTON_COND_IDX, fallback=national
    # Index present (30 vs ref 50 -> worse crop -> bullish drift, mu>0).
    s.bind(FakeStore({"SIG_COTTON_COND_IDX": 30, "NASS_COTTON_COND_GE": 55}))
    res = s.signal_drift(spot=80.0, price_entity="CME_COTTON", T=0.5)
    assert res is not None and res[0] > 0 and res[1]["condition"] == 30  # used the index, not national

    # Index missing -> fall back to the national print.
    s.bind(FakeStore({"NASS_COTTON_COND_GE": 62}))
    res = s.signal_drift(spot=80.0, price_entity="CME_COTTON", T=0.5)
    assert res is not None and res[1]["condition"] == 62  # 62 vs 50 -> bearish (mu<0)
    assert res[0] < 0
