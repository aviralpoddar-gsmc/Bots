"""IV surface fit + Breeden-Litzenberger risk-neutral density."""

import numpy as np
import pytest

from quantbots.equity_options.pricing.greeks import bsm_price
from quantbots.equity_options.pricing.surface import (
    _trapz,
    clean_chain_to_iv,
    fit_smile,
    risk_neutral_density,
)


def _flat_chain(s=100.0, t=0.5, r=0.02, iv=0.25):
    strikes = [80, 90, 95, 100, 105, 110, 120]
    quotes = []
    for k in strikes:
        for kind in ("call", "put"):
            quotes.append({"strike": k, "kind": kind,
                           "mid": bsm_price(s, k, t, r, iv, kind)})
    return quotes


def test_smile_recovers_flat_vol():
    s, t, r, iv = 100.0, 0.5, 0.02, 0.25
    strikes, ivs, wts = clean_chain_to_iv(_flat_chain(s, t, r, iv), s=s, t=t, r=r)
    smile = fit_smile(strikes, ivs, s=s, t=t, r=r, weights=wts)
    assert smile is not None
    # Reconstructed IV near ATM should be close to the flat input vol.
    assert float(smile.iv(100.0)) == pytest.approx(iv, abs=0.02)


def test_density_is_valid_distribution():
    s, t, r, iv = 100.0, 0.5, 0.02, 0.25
    strikes, ivs, wts = clean_chain_to_iv(_flat_chain(s, t, r, iv), s=s, t=t, r=r)
    smile = fit_smile(strikes, ivs, s=s, t=t, r=r, weights=wts)
    k_grid, dens = risk_neutral_density(smile, n_grid=400)
    assert np.all(dens >= 0.0)                      # non-negative
    assert _trapz(dens, k_grid) == pytest.approx(1.0, abs=1e-2)  # integrates to ~1
    # Mean of the density should sit near the forward (martingale).
    mean = _trapz(k_grid * dens, k_grid)
    fwd = s * np.exp(r * t)
    assert mean == pytest.approx(fwd, rel=0.05)


def test_fit_none_when_too_few_points():
    assert fit_smile(np.array([100.0]), np.array([0.2]), s=100, t=0.5, r=0.0) is None
