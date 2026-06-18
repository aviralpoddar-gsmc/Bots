"""Structure / strike / expiry selection — the chooser.

For each expiry in the tradable DTE window we build one f_P sample (terminal-price
MC), then enumerate every candidate (structure, strike(s)) and price its
piecewise-linear payoff over that one sorted sample (the `diffusion_mc.group` trick:
one simulation prices the whole ladder, giving free cross-strike coherence). Each
candidate is scored by P&L Sharpe under f_P and filtered by the risk-limit gates
(liquidity floor, max relative spread, min premium, and the edge hurdle
E_P[payoff]·e^{-rT} > λ·cost). Verticals are included when `structures` allows L3.

Returns a ranked list of `Candidate`s; sizing/portfolio downstream turn the best of
them into orders.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np

from .edge import CONTRACT_MULTIPLIER, EdgeResult, Leg, evaluate
from .forecast.underlying import Forecast
from .pricing.greeks import greeks as bsm_greeks

logger = logging.getLogger(__name__)

ForecastFn = Callable[[float], Forecast | None]


@dataclass
class Candidate:
    underlying: str
    structure: str
    expiry: date
    dte: int
    legs: list[dict[str, Any]]        # [{symbol, strike, kind, qty, mid}, ...]
    cost_per_share: float             # net debit (>0) / credit (<0) per share
    premium: float                    # $ capital at risk (max loss) for one contract
    edge: EdgeResult
    net_greeks: dict[str, float]      # per-contract net delta/gamma/vega/theta
    forecast_vol: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_dollars(self) -> float:
        return self.edge.edge * CONTRACT_MULTIPLIER


def _liquid(row: dict, limits: dict) -> bool:
    if not row or row.get("mid") is None or row["mid"] <= 0:
        return False
    bid, ask, mid = row.get("bid"), row.get("ask"), row["mid"]
    # Require a real two-sided market. A 0/None bid means nobody is buying — the
    # quote is stale/indicative, NOT tradeable. Deep-ITM options on Alpaca's
    # indicative feed quote bid=0 with a tiny ask, which the model would misread as
    # "buy intrinsic value for pennies" (a confidently-wrong stale-quote trap).
    if not bid or bid <= 0 or not ask or ask <= 0:
        return False
    oi = row.get("open_interest")
    if oi is not None and oi < limits["min_open_interest"]:
        return False
    if (ask - bid) / mid > limits["max_rel_spread"]:
        return False
    return True


def _passes_intrinsic(row: dict, spot: float) -> bool:
    """Drop quotes whose mid sits below intrinsic value — impossible in a real
    market, so it's a stale/garbage quote (the other half of the deep-ITM trap)."""
    intrinsic = max(spot - row["strike"], 0.0) if row["kind"] == "call" \
        else max(row["strike"] - spot, 0.0)
    return row["mid"] >= 0.98 * intrinsic


def _leg_greeks(row: dict, s0: float, T: float, r: float, q: float) -> dict[str, float]:
    """Use the chain's greeks if present; otherwise compute from the row IV."""
    if row.get("delta") is not None and row.get("vega") is not None:
        return {k: float(row.get(k) or 0.0) for k in ("delta", "gamma", "vega", "theta")}
    iv = row.get("iv")
    if iv and T > 0:
        g = bsm_greeks(s0, row["strike"], T, r, float(iv), row["kind"], q)
        return {k: g[k] for k in ("delta", "gamma", "vega", "theta")}
    return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}


def _net_greeks(legs: list[Leg], rows: dict[tuple[float, str], dict], *, s0: float, T: float,
                r: float, q: float) -> dict[str, float]:
    net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for leg in legs:
        g = _leg_greeks(rows[(leg.strike, leg.kind)], s0, T, r, q)
        for k in net:
            net[k] += leg.qty * g[k] * CONTRACT_MULTIPLIER
    return net


def _exec_price(row: dict, qty: int) -> float:
    """Conservative executable price: pay the ASK on a buy, receive the BID on a sell.

    Pricing legs at mid overstates edge — you can't simultaneously buy and sell at
    mid. Crossing the spread (ask on longs, bid on shorts) is what an entry actually
    costs, and it's what kills the deep-ITM-vertical 'free money' mirage from wide
    quotes. Falls back to mid only if a side is missing (already filtered by _liquid)."""
    if qty > 0:
        return float(row.get("ask") or row["mid"])
    return float(row.get("bid") or row["mid"])


def _nearest(strikes: list[float], target: float) -> float | None:
    return min(strikes, key=lambda k: abs(k - target)) if strikes else None


def _condors(calls: list[dict], puts: list[dict], s0: float, width: float
             ) -> list[tuple[str, list[Leg]]]:
    """A few short iron condors at fixed OTM offsets (sell an OTM put spread + an OTM
    call spread). The canonical defined-risk variance-risk-premium harvest."""
    pmap = {p["strike"]: p for p in puts}
    cmap = {c["strike"]: c for c in calls}
    pstr = sorted(k for k in pmap if k < s0)        # OTM put strikes
    cstr = sorted(k for k in cmap if k > s0)        # OTM call strikes
    out: list[tuple[str, list[Leg]]] = []
    seen: set[tuple] = set()
    for pm, cm in ((0.93, 1.07), (0.90, 1.10), (0.96, 1.04)):
        sp = _nearest(pstr, s0 * pm)                # short put (sell)
        sc = _nearest(cstr, s0 * cm)                # short call (sell)
        if sp is None or sc is None:
            continue
        lp = _nearest([k for k in pstr if k < sp], sp - width)   # long put wing (buy)
        lc = _nearest([k for k in cstr if k > sc], sc + width)   # long call wing (buy)
        if lp is None or lc is None:
            continue
        key = (lp, sp, sc, lc)
        if key in seen:
            continue
        seen.add(key)
        legs = [Leg(sp, "put", -1, _exec_price(pmap[sp], -1)),
                Leg(lp, "put", +1, _exec_price(pmap[lp], +1)),
                Leg(sc, "call", -1, _exec_price(cmap[sc], -1)),
                Leg(lc, "call", +1, _exec_price(cmap[lc], +1))]
        out.append((f"IC{lp:g}/{sp:g}-{sc:g}/{lc:g}", legs))
    return out


def _enumerate(structure: str, calls: list[dict], puts: list[dict], *, s0: float,
               vertical_max_width: float) -> list[tuple[str, list[Leg]]]:
    """Yield (label, legs) candidates for one structure within one expiry.

    Debit structures BUY premium (long_*, bull_call/bear_put). Credit structures SELL
    premium (bull_put/bear_call/iron_condor) to harvest the variance risk premium —
    these short OTM legs, so they're true premium-selling spreads, not ITM.
    """
    out: list[tuple[str, list[Leg]]] = []
    if structure == "long_call":
        out += [(f"C{c['strike']:g}", [Leg(c["strike"], "call", +1, _exec_price(c, +1))])
                for c in calls]
    elif structure == "long_put":
        out += [(f"P{p['strike']:g}", [Leg(p["strike"], "put", +1, _exec_price(p, +1))])
                for p in puts]
    elif structure == "bull_call_spread":          # debit, bullish
        for i, lo in enumerate(calls):
            for hi in calls[i + 1:]:
                if hi["strike"] - lo["strike"] > vertical_max_width:
                    break
                out.append((f"BCS{lo['strike']:g}/{hi['strike']:g}",
                            [Leg(lo["strike"], "call", +1, _exec_price(lo, +1)),
                             Leg(hi["strike"], "call", -1, _exec_price(hi, -1))]))
    elif structure == "bear_put_spread":           # debit, bearish
        for i, lo in enumerate(puts):
            for hi in puts[i + 1:]:
                if hi["strike"] - lo["strike"] > vertical_max_width:
                    break
                out.append((f"BPS{hi['strike']:g}/{lo['strike']:g}",
                            [Leg(hi["strike"], "put", +1, _exec_price(hi, +1)),
                             Leg(lo["strike"], "put", -1, _exec_price(lo, -1))]))
    elif structure == "bull_put_spread":           # credit, bullish/neutral (sell OTM put)
        otm = [p for p in puts if p["strike"] <= s0]
        for i, hi in enumerate(otm):               # hi = short (higher strike, nearer spot)
            for lo in otm[:i]:                     # lo = long wing (lower strike)
                if hi["strike"] - lo["strike"] > vertical_max_width:
                    continue
                out.append((f"BPuS{hi['strike']:g}/{lo['strike']:g}",
                            [Leg(hi["strike"], "put", -1, _exec_price(hi, -1)),
                             Leg(lo["strike"], "put", +1, _exec_price(lo, +1))]))
    elif structure == "bear_call_spread":          # credit, bearish/neutral (sell OTM call)
        otm = [c for c in calls if c["strike"] >= s0]
        for i, lo in enumerate(otm):               # lo = short (lower strike, nearer spot)
            for hi in otm[i + 1:]:                 # hi = long wing (higher strike)
                if hi["strike"] - lo["strike"] > vertical_max_width:
                    break
                out.append((f"BCaS{lo['strike']:g}/{hi['strike']:g}",
                            [Leg(lo["strike"], "call", -1, _exec_price(lo, -1)),
                             Leg(hi["strike"], "call", +1, _exec_price(hi, +1))]))
    elif structure == "iron_condor":               # credit, neutral (pure VRP harvest)
        out += _condors(calls, puts, s0, vertical_max_width)
    return out


def select(underlying: str, chain: list[dict], forecast_fn: ForecastFn, *, r: float,
           q: float, limits: dict, structures: list[str],
           vertical_width_frac: float = 0.15,
           moneyness_band: tuple[float, float] = (0.6, 1.4)) -> list[Candidate]:
    """Rank tradable candidates for one underlying across its expiries."""
    by_expiry: dict[date, list[dict]] = defaultdict(list)
    for row in chain:
        if limits["min_dte"] <= row["dte"] <= limits["max_dte"] and _liquid(row, limits):
            by_expiry[row["expiry"]].append(row)

    candidates: list[Candidate] = []
    lo_band, hi_band = moneyness_band
    for expiry, rows in by_expiry.items():
        dte = rows[0]["dte"]
        T = dte / 365.25
        fc = forecast_fn(T)
        if fc is None:
            continue
        # Keep only near-the-money strikes (deep ITM/OTM are noise + illiquid) and
        # quotes that clear the intrinsic-value sanity check.
        rows = [r_ for r_ in rows
                if lo_band * fc.s0 <= r_["strike"] <= hi_band * fc.s0
                and _passes_intrinsic(r_, fc.s0)]
        calls = sorted((r_ for r_ in rows if r_["kind"] == "call"), key=lambda x: x["strike"])
        puts = sorted((r_ for r_ in rows if r_["kind"] == "put"), key=lambda x: x["strike"])
        row_index = {(r_["strike"], r_["kind"]): r_ for r_ in rows}
        width = fc.s0 * vertical_width_frac
        for structure in structures:
            for label, legs in _enumerate(structure, calls, puts, s0=fc.s0,
                                          vertical_max_width=width):
                res = evaluate(fc.terminal, legs, r=r, T=T)
                premium = res.max_loss * CONTRACT_MULTIPLIER   # capital at risk (debit or credit)
                if res.max_loss <= 1e-9:        # not a defined-risk structure
                    continue
                if premium < limits["min_premium"]:
                    continue
                if res.edge <= 0:               # must be priced favorably vs f_P
                    continue
                # Unified hurdle: expected edge per $ at risk must clear min_edge_return
                # (replaces the debit-only pv_payoff>=λ·cost test; works for credit too).
                if res.edge / res.max_loss < limits.get("min_edge_return", 0.0):
                    continue
                ng = _net_greeks(legs, row_index, s0=fc.s0, T=T, r=r, q=q)
                candidates.append(Candidate(
                    underlying=underlying, structure=structure, expiry=expiry, dte=dte,
                    legs=[{"symbol": row_index[(l_.strike, l_.kind)]["symbol"],
                           "strike": l_.strike, "kind": l_.kind, "qty": l_.qty, "mid": l_.mid}
                          for l_ in legs],
                    cost_per_share=res.cost, premium=premium, edge=res, net_greeks=ng,
                    forecast_vol=fc.sigma_fp, extra={"label": label, "mode": fc.mode}))
    candidates.sort(key=lambda c: c.edge.score, reverse=True)
    return candidates
