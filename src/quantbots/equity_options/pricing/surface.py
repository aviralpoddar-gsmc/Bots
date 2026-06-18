"""Per-expiry implied-vol smile fit and the risk-neutral density f_Q.

Why this exists: raw chain quotes are noisy and wide; reading a density straight off
them via Breeden-Litzenberger (f_Q(K) = e^{rT} d2C/dK2) explodes, because the second
difference amplifies noise. So we first fit a *smooth* smile, then differentiate the
smile-implied call-price curve on a dense grid. This mirrors the parent
`surface_arb._fit_normal_to_survival` idea (fit a smooth curve to a noisy ladder,
then read fair values off the fit) — here the fitted object is the vol smile.

The smile is a low-order polynomial in log-moneyness on TOTAL VARIANCE
(w = sigma^2 * T), which is smooth by construction and rarely produces butterfly
arbitrage at degree 2. We still clip the reconstructed density to be non-negative
and renormalize, so f_Q is always a valid distribution (a property the tests assert).

Per-contract trade edge does NOT need f_Q: by risk-neutral pricing the market mid
already equals e^{-rT} E_Q[payoff], so edge.py compares E_P[payoff] to the mid
directly. f_Q here is the smoothed/denoised view used for the drift-neutral edge,
density diagnostics, and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .greeks import bsm_price, implied_vol

# numpy 2.0 renamed trapz -> trapezoid; support both (extras pin numpy>=1.26).
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


@dataclass
class Smile:
    """A fitted per-expiry vol smile, callable at any strike."""

    t: float
    r: float
    q: float
    forward: float
    coeffs: np.ndarray         # polyfit of total variance vs log-moneyness
    k_min: float               # fit domain (log-moneyness), clamped beyond
    k_max: float

    def iv(self, strike: float | np.ndarray) -> np.ndarray:
        k = np.log(np.asarray(strike, dtype=float) / self.forward)
        k = np.clip(k, self.k_min, self.k_max)  # flat-extrapolate outside the fit
        w = np.polyval(self.coeffs, k)
        w = np.maximum(w, 1e-8)
        return np.sqrt(w / self.t)

    def call_price(self, strike: float | np.ndarray) -> np.ndarray:
        strike = np.atleast_1d(np.asarray(strike, dtype=float))
        ivs = np.atleast_1d(self.iv(strike))
        s = self.forward * np.exp(-(self.r - self.q) * self.t)  # spot from forward
        return np.array([bsm_price(s, float(k), self.t, self.r, float(v), "call", self.q)
                         for k, v in zip(strike, ivs)])


def fit_smile(strikes: np.ndarray, ivs: np.ndarray, *, s: float, t: float, r: float,
              q: float = 0.0, degree: int = 2, weights: np.ndarray | None = None) -> Smile | None:
    """Fit total variance w = iv^2 * t as a polynomial in log-moneyness.

    `strikes`/`ivs` are the cleaned per-expiry points (NaN/None already dropped).
    Returns None if there aren't enough points to fit the polynomial.
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    mask = np.isfinite(strikes) & np.isfinite(ivs) & (strikes > 0) & (ivs > 0)
    strikes, ivs = strikes[mask], ivs[mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[mask]
    if len(strikes) < degree + 1:
        return None
    forward = s * np.exp((r - q) * t)
    k = np.log(strikes / forward)
    w = ivs ** 2 * t
    deg = min(degree, len(strikes) - 1)
    coeffs = np.polyfit(k, w, deg, w=weights)
    return Smile(t=t, r=r, q=q, forward=float(forward), coeffs=coeffs,
                 k_min=float(k.min()), k_max=float(k.max()))


def risk_neutral_density(smile: Smile, *, n_grid: int = 400, lo_mult: float = 0.4,
                         hi_mult: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Breeden-Litzenberger density f_Q(K) = e^{rT} d2C/dK2 off the fitted smile.

    Returns (K_grid, density), with the density clipped non-negative and renormalized
    to integrate to 1 over the grid. The grid spans [lo_mult, hi_mult] x forward.
    """
    f = smile.forward
    k_grid = np.linspace(f * lo_mult, f * hi_mult, n_grid)
    c = smile.call_price(k_grid)
    # Second derivative via central differences on the (uniform) strike grid.
    d2c = np.gradient(np.gradient(c, k_grid), k_grid)
    dens = np.exp(smile.r * smile.t) * d2c
    dens = np.clip(dens, 0.0, None)
    area = _trapz(dens, k_grid)
    if area > 0:
        dens = dens / area
    return k_grid, dens


def clean_chain_to_iv(quotes: list[dict], *, s: float, t: float, r: float,
                      q: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert a list of option quotes to (strikes, ivs, vega_weights).

    Each quote dict needs: strike, kind ("call"/"put"), and a usable mid price under
    key "mid" (or bid/ask to average). Quotes that fail inversion are dropped. Puts
    and calls are both used (IV is put-call symmetric under BSM), giving a denser fit.
    Vega weights down-weight deep wings where IV is ill-conditioned.
    """
    from .greeks import greeks as _greeks

    strikes, ivs, wts = [], [], []
    for qd in quotes:
        strike = qd.get("strike")
        kind = qd.get("kind")
        mid = qd.get("mid")
        if mid is None and qd.get("bid") is not None and qd.get("ask") is not None:
            mid = 0.5 * (qd["bid"] + qd["ask"])
        if strike is None or kind not in ("call", "put") or mid is None or mid <= 0:
            continue
        iv = implied_vol(float(mid), s, float(strike), t, r, kind, q)
        if iv is None or not np.isfinite(iv):
            continue
        vega = _greeks(s, float(strike), t, r, iv, kind, q)["vega"]
        strikes.append(float(strike)); ivs.append(float(iv)); wts.append(max(vega, 1e-6))
    return np.array(strikes), np.array(ivs), np.array(wts)
