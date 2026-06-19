"""Factor ingest (point-in-time CSV) + fusion blend — pure, no network."""

import numpy as np
import pandas as pd
import pytest

from quantbots.equity_options.research import factors as F
from quantbots.equity_options.research import fusion


def test_carry_csv_ingest(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FACTOR_DIR", tmp_path)
    pd.DataFrame({"date": ["2024-01-02", "2024-01-02"], "ticker": ["HG", "CL"],
                  "carry_ann": [0.06, -0.05], "signal_zscore": [1.8, -1.2]}).to_csv(
        tmp_path / "carry.csv", index=False)
    w = F.carry_from_csv()
    assert w is not None and "HG" in w.columns
    assert w["HG"].iloc[-1] == pytest.approx(1.8)   # uses signal_zscore column


def test_carry_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FACTOR_DIR", tmp_path)
    assert F.carry_from_csv() is None


def test_fusion_momentum_only(monkeypatch):
    # no CSVs, neutral macro -> fused drift ≈ momentum component, capped
    monkeypatch.setattr(fusion, "momentum_drift", lambda **k: (0.20, 0.8))
    monkeypatch.setattr(fusion, "_macro_series", lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(F, "carry_from_csv", lambda: None)
    monkeypatch.setattr(F, "positioning_from_csv", lambda: None)
    mu, comps = fusion.fused_drift(equity="FCX", commodity="COPPER", beta_c=1.0)
    assert "momentum" in comps and "carry" not in comps
    assert 0 < mu <= 0.35


def test_fusion_adds_carry(monkeypatch):
    monkeypatch.setattr(fusion, "momentum_drift", lambda **k: (0.10, 0.5))
    monkeypatch.setattr(fusion, "_macro_series", lambda c: pd.Series(dtype=float))
    idx = pd.to_datetime(["2024-01-02"])
    monkeypatch.setattr(F, "carry_from_csv", lambda: pd.DataFrame({"HG": [2.0]}, index=idx))
    monkeypatch.setattr(F, "positioning_from_csv", lambda: None)
    mu, comps = fusion.fused_drift(equity="FCX", commodity="COPPER", beta_c=1.0)
    assert "carry" in comps and comps["carry"] > 0      # bullish carry -> positive contribution
    # negative beta flips the carry contribution sign
    mu2, comps2 = fusion.fused_drift(equity="X", commodity="COPPER", beta_c=-1.0)
    assert comps2["carry"] < 0
