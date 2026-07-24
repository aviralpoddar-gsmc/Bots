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


def _neutral(monkeypatch):
    monkeypatch.setattr(fusion, "_macro_series", lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(F, "carry_from_csv", lambda: None)
    monkeypatch.setattr(F, "positioning_from_csv", lambda: None)


def test_fusion_skips_tal_without_spot(monkeypatch):
    # no spot -> tal factor never consulted (backward compatible with momentum-only)
    _neutral(monkeypatch)
    monkeypatch.setattr(fusion, "momentum_drift", lambda **k: (0.20, 0.8))
    from quantbots.equity_options.forecast import signal
    monkeypatch.setattr(signal, "tal_drift", lambda **k: (_ for _ in ()).throw(AssertionError()))
    mu, comps = fusion.fused_drift(equity="FCX", commodity="COPPER", beta_c=1.0)
    assert "tal" not in comps


def test_fusion_adds_tal_blends_toward_view(monkeypatch):
    _neutral(monkeypatch)
    monkeypatch.setattr(fusion, "momentum_drift", lambda **k: (0.10, 0.5))
    from quantbots.equity_options.forecast import signal
    monkeypatch.setattr(signal, "tal_drift", lambda **k: (0.30, 0.8))  # strong bullish view
    mu, comps = fusion.fused_drift(equity="FCX", commodity="COPPER", beta_c=1.0, spot=50.0)
    assert "tal" in comps and comps["tal"] == pytest.approx(0.30)
    # blend lands between momentum (0.10) and tal (0.30)
    assert 0.10 < mu < 0.30


def test_fusion_tal_zero_confidence_excluded(monkeypatch):
    _neutral(monkeypatch)
    monkeypatch.setattr(fusion, "momentum_drift", lambda **k: (0.10, 0.5))
    from quantbots.equity_options.forecast import signal
    monkeypatch.setattr(signal, "tal_drift", lambda **k: (0.0, 0.0))  # no usable ladder
    mu, comps = fusion.fused_drift(equity="FCX", commodity="COPPER", beta_c=1.0, spot=50.0)
    mu_ref, _ = fusion.fused_drift(equity="FCX", commodity="COPPER", beta_c=1.0)  # no tal at all
    assert "tal" not in comps
    assert mu == pytest.approx(mu_ref)   # zero-confidence tal must not change the blend
