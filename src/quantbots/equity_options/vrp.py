"""Delta-hedged variance-risk-premium (VRP) harvesting.

The one edge that survived the investigation: implied vol structurally exceeds realized
vol, so SELLING options earns a premium — but only if you strip out direction by
DELTA-HEDGING. An unhedged short straddle/condor is a directional bet (we proved it
loses); a delta-hedged one earns ≈ ∫ ½·Γ·S²·(σ_impl² − σ_real²) dt — the variance gap.

Why this is backtestable (the tal signal wasn't): the delta-hedged P&L depends on the
underlying's realized PATH (deep yfinance history) + the entry IV (Alpaca chain since
2024) — both cover the backtest window. We sell at the chain's entry IV, then hedge daily
along the realized path using BSM deltas at that (held-constant) entry IV.

Structures (defined-risk for L3): a short straddle is the purest VRP probe (used to test
whether the premium exists at all); the iron condor is the tradeable defined-risk version.
Entry filter: only sell when ATM IV exceeds our diffusion forecast vol by a margin.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .pricing.greeks import greeks as bsm_greeks

logger = logging.getLogger(__name__)
CONTRACT_MULTIPLIER = 100
_TRADING_DAYS = 252


@dataclass
class VrpLeg:
    strike: float
    kind: str           # call | put
    qty: int            # +long / -short (VRP structures are net short vega)
    entry_price: float  # per-share premium received(short)/paid(long) at entry
    iv: float           # entry implied vol (used for the hedge deltas)


def _intrinsic(S: float, strike: float, kind: str) -> float:
    return max(S - strike, 0.0) if kind == "call" else max(strike - S, 0.0)


def delta_hedged_pnl(legs: list[VrpLeg], S_path: np.ndarray, *, r: float, q: float = 0.0,
                     hedge_cost_bps: float = 1.0) -> float | None:
    """Realized $ P&L of holding `legs` to expiry while delta-hedging daily along S_path.

    S_path[0] = entry spot, S_path[-1] = expiry spot, one point per trading day. Hedge
    trades the underlying to flatten net option delta each day; `hedge_cost_bps` is
    per-trade slippage on notional. Returns P&L for ONE structure (×100 multiplier).
    """
    S_path = np.asarray(S_path, dtype=float)
    n = len(S_path) - 1
    if n < 1:
        return None
    cost = sum(l.qty * l.entry_price for l in legs)          # per-share net debit (<0 = credit)
    shares = 0.0
    hedge_cash = 0.0
    for i in range(n):
        t_rem = max((n - i) / _TRADING_DAYS, 1e-4)
        opt_delta = 0.0
        for l in legs:
            opt_delta += l.qty * bsm_greeks(S_path[i], l.strike, t_rem, r, l.iv, l.kind, q)["delta"]
        opt_delta *= CONTRACT_MULTIPLIER
        target = -opt_delta                                  # flatten total delta
        trade = target - shares
        hedge_cash -= trade * S_path[i]
        hedge_cash -= abs(trade) * S_path[i] * hedge_cost_bps * 1e-4
        shares = target
    # Liquidate the hedge + settle options at intrinsic.
    hedge_cash += shares * S_path[-1]
    hedge_cash -= abs(shares) * S_path[-1] * hedge_cost_bps * 1e-4
    payoff = sum(l.qty * _intrinsic(S_path[-1], l.strike, l.kind) for l in legs) * CONTRACT_MULTIPLIER
    return -cost * CONTRACT_MULTIPLIER + payoff + hedge_cash


# --- structure builders from a chain ----------------------------------------

def _nearest_row(rows: list[dict], kind: str, target_strike: float) -> dict | None:
    cands = [r for r in rows if r["kind"] == kind and r.get("iv") and r.get("mid")]
    return min(cands, key=lambda r: abs(r["strike"] - target_strike)) if cands else None


def short_straddle(chain: list[dict], spot: float) -> list[VrpLeg] | None:
    """Sell the ATM call + put — the purest VRP probe (naked; backtest-only)."""
    c = _nearest_row(chain, "call", spot)
    p = _nearest_row(chain, "put", spot)
    if not c or not p:
        return None
    return [VrpLeg(c["strike"], "call", -1, c.get("bid") or c["mid"], c["iv"]),
            VrpLeg(p["strike"], "put", -1, p.get("bid") or p["mid"], p["iv"])]


def iron_condor(chain: list[dict], spot: float, *, short_frac: float = 0.05,
                wing_frac: float = 0.10) -> list[VrpLeg] | None:
    """Defined-risk short vol: sell ~5% OTM strangle, buy ~10% OTM wings (tradeable L3)."""
    sp = _nearest_row(chain, "put", spot * (1 - short_frac))
    lp = _nearest_row(chain, "put", spot * (1 - wing_frac))
    sc = _nearest_row(chain, "call", spot * (1 + short_frac))
    lc = _nearest_row(chain, "call", spot * (1 + wing_frac))
    if not (sp and lp and sc and lc) or lp["strike"] >= sp["strike"] or lc["strike"] <= sc["strike"]:
        return None
    return [VrpLeg(sp["strike"], "put", -1, sp.get("bid") or sp["mid"], sp["iv"]),
            VrpLeg(lp["strike"], "put", +1, lp.get("ask") or lp["mid"], lp["iv"]),
            VrpLeg(sc["strike"], "call", -1, sc.get("bid") or sc["mid"], sc["iv"]),
            VrpLeg(lc["strike"], "call", +1, lc.get("ask") or lc["mid"], lc["iv"])]


def iron_fly(chain: list[dict], spot: float, *, wing_frac: float = 0.10) -> list[VrpLeg] | None:
    """Defined-risk short vol with an ATM body (sell ATM call+put, buy ±wing% wings).
    Much closer to a straddle than the OTM condor, so it keeps more of the VRP."""
    sc = _nearest_row(chain, "call", spot)
    sp = _nearest_row(chain, "put", spot)
    lc = _nearest_row(chain, "call", spot * (1 + wing_frac))
    lp = _nearest_row(chain, "put", spot * (1 - wing_frac))
    if not (sc and sp and lc and lp) or lc["strike"] <= sc["strike"] or lp["strike"] >= sp["strike"]:
        return None
    return [VrpLeg(sc["strike"], "call", -1, sc.get("bid") or sc["mid"], sc["iv"]),
            VrpLeg(sp["strike"], "put", -1, sp.get("bid") or sp["mid"], sp["iv"]),
            VrpLeg(lc["strike"], "call", +1, lc.get("ask") or lc["mid"], lc["iv"]),
            VrpLeg(lp["strike"], "put", +1, lp.get("ask") or lp["mid"], lp["iv"])]


def atm_iv(chain: list[dict], spot: float) -> float | None:
    c = _nearest_row(chain, "call", spot)
    p = _nearest_row(chain, "put", spot)
    ivs = [x["iv"] for x in (c, p) if x and x.get("iv")]
    return float(np.mean(ivs)) if ivs else None


# --- backtest ---------------------------------------------------------------

@dataclass
class VrpResult:
    underlying: str
    structure: str
    pnls: list[float]              # delta-hedged $ P&L per fold (filter-passed trades)
    iv_minus_rv: list[float]       # entry ATM IV − forecast realized vol per fold
    n_screened: int = 0           # folds evaluated

    def summary(self) -> dict:
        out = {"underlying": self.underlying, "structure": self.structure,
               "trades": len(self.pnls), "screened": self.n_screened}
        if self.pnls:
            a = np.asarray(self.pnls)
            sd = float(a.std())
            out["pnl_total"] = float(a.sum())
            out["pnl_mean"] = float(a.mean())
            out["sharpe"] = float(a.mean() / sd) if sd > 1e-9 else 0.0
            out["win_rate"] = float((a > 0).mean())
        if self.iv_minus_rv:
            out["avg_iv_minus_rv"] = float(np.mean(self.iv_minus_rv))
        return out


def _price_path(ticker: str, start, end) -> np.ndarray:
    import pandas as pd

    from ..research.data_fetch import fetch_yf_history  # noqa: E402
    df = fetch_yf_history(ticker, period="5y")
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df["Close"].astype(float).to_numpy()


def run_vrp_backtest(cfg, underlying: str, *, as_of_dates, horizon_days: int = 45,
                     structure: str = "straddle", iv_margin: float = 0.0,
                     chain_client=None) -> VrpResult:
    """Delta-hedged short-vol backtest for one underlying. Sells the structure at the
    historical chain's entry IV when ATM IV > forecast RV·(1+margin), then hedges daily
    along the realized path. Shorter default horizon (45d) — vol selling lives at the
    front of the curve."""
    from .backtest import close_asof, target_expiry
    from .forecast.underlying import _realized_vol
    from ..research.data_fetch import fetch_yf_history  # noqa: F401  (warms cache)
    from .sources import underlying as und_src
    from .sources.options_chain import ChainClient

    cc = chain_client or ChainClient(key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    r = und_src.risk_free_rate()
    res = VrpResult(underlying=underlying, structure=structure, pnls=[], iv_minus_rv=[])
    for as_of in as_of_dates:
        expiry = target_expiry(as_of, horizon_days)
        spot = close_asof(underlying, as_of.isoformat())
        realized_end = close_asof(underlying, expiry.isoformat())
        if spot is None or realized_end is None:
            continue
        res.n_screened += 1
        try:
            chain = cc.historical_chain(underlying, as_of=as_of, expiry=expiry, spot=spot, r=r)
        except Exception:  # noqa: BLE001
            continue
        if not chain:
            continue
        builder = {"straddle": short_straddle, "condor": iron_condor,
                   "fly": iron_fly}.get(structure, short_straddle)
        legs = builder(chain, spot)
        if not legs:
            continue
        aiv = atm_iv(chain, spot)
        rv = _realized_vol(underlying, as_of=as_of)
        if aiv is None:
            continue
        res.iv_minus_rv.append(aiv - rv)
        if aiv <= rv * (1.0 + iv_margin):    # only sell when IV is rich vs our RV
            continue
        path = _price_path(underlying, as_of, expiry)
        if len(path) < 5:
            continue
        pnl = delta_hedged_pnl(legs, path, r=r)
        if pnl is not None:
            res.pnls.append(pnl)
    return res
