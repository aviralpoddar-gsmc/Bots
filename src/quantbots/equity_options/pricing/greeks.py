"""Black-Scholes-Merton pricing, greeks, and implied-volatility inversion.

Pure stdlib `math` for the scalar formulas (reusing the parent's `norm_cdf` so the
whole codebase shares one Gaussian), so this module has no hard dependency on the
quant extra. IV inversion uses Newton on vega with a bisection bracket fallback —
also pure-math — and an optional scipy Brentq if available for extra robustness.

Conventions:
- `s` spot, `k` strike, `t` years to expiry, `r` risk-free (cont.), `q` dividend yield,
  `sigma` annualized vol, `kind` in {"call","put"}.
- Greeks are returned in natural per-unit terms; the caller scales by contract
  multiplier (100) and position size. Vega is per 1.00 (100 vol-points) of sigma;
  theta is per year. `vega_pct` / `theta_day` helpers give the trader-friendly
  per-1-vol-point and per-calendar-day versions.
"""

from __future__ import annotations

import math

from ...strategies._model import norm_cdf

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("d1_d2 requires positive s, k, t, sigma")
    vol_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_t
    return d1, d1 - vol_t


def bsm_price(s: float, k: float, t: float, r: float, sigma: float, kind: str = "call",
              q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option (with continuous dividend q).

    Degenerate inputs (t<=0 or sigma<=0) collapse to the discounted intrinsic value
    so callers never hit a divide-by-zero at/after expiry."""
    if t <= 0 or sigma <= 0:
        intrinsic = max(s - k, 0.0) if kind == "call" else max(k - s, 0.0)
        return intrinsic * math.exp(-r * t) if t > 0 else intrinsic
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    disc_s = s * math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    if kind == "call":
        return disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)


def greeks(s: float, k: float, t: float, r: float, sigma: float, kind: str = "call",
           q: float = 0.0) -> dict[str, float]:
    """Delta, gamma, vega, theta, rho for a European option.

    - delta: dV/dS
    - gamma: d2V/dS2
    - vega:  dV/dsigma  (per 1.00 = 100 vol points; divide by 100 for per-point)
    - theta: dV/dt as a *decay* (per year; divide by 365 for per-day)
    - rho:   dV/dr per 1.00 (100 bps*100); divide by 100 for per-1%
    """
    if t <= 0 or sigma <= 0:
        # At expiry greeks are degenerate; report a step-function delta, rest 0.
        itm = (s > k) if kind == "call" else (s < k)
        return {"delta": (1.0 if kind == "call" else -1.0) if itm else 0.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    sqrt_t = math.sqrt(t)
    pdf = _norm_pdf(d1)
    disc_s = math.exp(-q * t)
    disc_k = math.exp(-r * t)
    gamma = disc_s * pdf / (s * sigma * sqrt_t)
    vega = s * disc_s * pdf * sqrt_t
    if kind == "call":
        delta = disc_s * norm_cdf(d1)
        theta = (-s * disc_s * pdf * sigma / (2 * sqrt_t)
                 - r * k * disc_k * norm_cdf(d2)
                 + q * s * disc_s * norm_cdf(d1))
        rho = k * t * disc_k * norm_cdf(d2)
    else:
        delta = -disc_s * norm_cdf(-d1)
        theta = (-s * disc_s * pdf * sigma / (2 * sqrt_t)
                 + r * k * disc_k * norm_cdf(-d2)
                 - q * s * disc_s * norm_cdf(-d1))
        rho = -k * t * disc_k * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price: float, s: float, k: float, t: float, r: float, kind: str = "call",
                q: float = 0.0, *, tol: float = 1e-6, max_iter: int = 100) -> float | None:
    """Invert BSM for sigma given a market price. Newton on vega, bisection fallback.

    Returns None when the price is below intrinsic / above the no-arb bound (no real
    IV exists) or the solver fails to converge.
    """
    if t <= 0 or price <= 0:
        return None
    disc_s = s * math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    # No-arbitrage bounds: max(disc_s - disc_k, 0) <= call <= disc_s (and put analog).
    if kind == "call":
        lo_bound, hi_bound = max(disc_s - disc_k, 0.0), disc_s
    else:
        lo_bound, hi_bound = max(disc_k - disc_s, 0.0), disc_k
    if price < lo_bound - 1e-9 or price > hi_bound + 1e-9:
        return None

    # Newton from a Brenner-Subrahmanyam ATM seed.
    sigma = max(math.sqrt(2 * math.pi / t) * price / s, 1e-3)
    for _ in range(max_iter):
        diff = bsm_price(s, k, t, r, sigma, kind, q) - price
        if abs(diff) < tol:
            return sigma
        v = s * math.exp(-q * t) * _norm_pdf(d1_d2(s, k, t, r, sigma, q)[0]) * math.sqrt(t)
        if v < 1e-10:
            break
        sigma -= diff / v
        if sigma <= 0 or sigma > 10:
            break  # diverged; hand off to bisection

    # Bisection fallback on a wide bracket (robust, always converges if a root exists).
    lo, hi = 1e-4, 10.0
    f_lo = bsm_price(s, k, t, r, lo, kind, q) - price
    f_hi = bsm_price(s, k, t, r, hi, kind, q) - price
    if f_lo * f_hi > 0:
        try:  # scipy Brent on the same bracket is marginally tighter when available
            from scipy.optimize import brentq
            return float(brentq(lambda x: bsm_price(s, k, t, r, x, kind, q) - price, lo, hi))
        except Exception:  # noqa: BLE001 - scipy missing or no sign change
            return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bsm_price(s, k, t, r, mid, kind, q) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
