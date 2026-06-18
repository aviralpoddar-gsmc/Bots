"""Edge math: structure payoffs and the f_P-vs-market edge.

The key identity that keeps this honest and tradeable: by risk-neutral pricing the
market mid of any structure already equals e^{-rT} E_Q[payoff]. So the per-structure
edge is simply

    edge = e^{-rT} E_P[payoff]  -  market_cost
         = e^{-rT} ∫ payoff(S_T)·f_P(S_T) dS_T  -  e^{-rT} ∫ payoff·f_Q
         = e^{-rT} ∫ payoff·(f_P − f_Q),

the continuous generalization of `portfolio.ev_per_mana`'s (p − q). We evaluate the
f_P integral as the mean payoff over the Monte-Carlo terminal sample — no separate
f_Q needed for the trade decision (f_Q from `pricing.surface` is the smoothed
diagnostic / drift-neutral view).

Everything here is per-SHARE; the caller multiplies by the 100x contract multiplier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CONTRACT_MULTIPLIER = 100

# Structures we know how to build.
DEBIT_STRUCTURES = ("long_call", "long_put", "bull_call_spread", "bear_put_spread")
CREDIT_STRUCTURES = ("bull_put_spread", "bear_call_spread", "iron_condor")
STRUCTURES = DEBIT_STRUCTURES + CREDIT_STRUCTURES


def _single_payoff(terminal: np.ndarray, strike: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(terminal - strike, 0.0)
    return np.maximum(strike - terminal, 0.0)


@dataclass
class Leg:
    strike: float
    kind: str          # call | put
    qty: int           # +1 long, -1 short
    mid: float         # per-share premium


def structure_payoff(terminal: np.ndarray, legs: list[Leg]) -> np.ndarray:
    """Net per-share payoff at expiry of a multi-leg structure over the f_P sample."""
    payoff = np.zeros_like(terminal)
    for leg in legs:
        payoff = payoff + leg.qty * _single_payoff(terminal, leg.strike, leg.kind)
    return payoff


def structure_cost(legs: list[Leg]) -> float:
    """Net debit (>0) / credit (<0) per share to open the structure."""
    return float(sum(leg.qty * leg.mid for leg in legs))


def risk_profile(legs: list[Leg], cost: float) -> tuple[float, float]:
    """(max_profit, max_loss) per share for a defined-risk structure.

    The P&L = payoff(S_T) - cost is piecewise-linear with kinks only at the strikes,
    so its extremes occur at S=0, each strike, or S->large. Evaluating there gives the
    exact max/min for any vertical or condor (credit or debit). max_loss is returned as
    a positive number (the capital at risk = what sizing/caps bound)."""
    strikes = sorted({leg.strike for leg in legs})
    pts = np.array([0.0, *strikes, strikes[-1] * 3.0])
    pnl = structure_payoff(pts, legs) - cost
    return float(pnl.max()), float(max(-pnl.min(), 0.0))


@dataclass
class EdgeResult:
    cost: float            # net debit (>0) / credit (<0) per share
    ev_payoff: float       # E_P[payoff] (undiscounted, per share)
    pv_payoff: float       # e^{-rT} E_P[payoff]
    edge: float            # pv_payoff - cost (per share); >0 = priced favorably vs f_P
    sd_payoff: float       # SD_P[payoff]
    score: float           # PnL Sharpe under f_P:  E_P[PnL] / SD_P[PnL]
    kelly: float           # fractional Kelly on the PnL distribution
    win_prob: float        # P_P(PnL > 0)
    max_loss: float        # capital at risk per share (>0) — what sizing/caps bound
    max_profit: float      # best-case per share
    is_credit: bool        # True when we receive premium to open


def evaluate(terminal: np.ndarray, legs: list[Leg], *, r: float, T: float) -> EdgeResult:
    payoff = structure_payoff(terminal, legs)
    cost = structure_cost(legs)
    ev = float(np.mean(payoff))
    sd = float(np.std(payoff))
    pv = math.exp(-r * T) * ev
    edge = pv - cost                          # works for debit (cost>0) AND credit (cost<0)
    pnl = payoff - cost                       # per-share P&L distribution at expiry
    ev_pnl = float(np.mean(pnl))
    sd_pnl = float(np.std(pnl))
    score = ev_pnl / sd_pnl if sd_pnl > 1e-9 else 0.0
    # Kelly on the P&L distribution: f* ≈ E[PnL] / E[PnL^2] (clamped to [0,1]).
    e_pnl2 = float(np.mean(pnl ** 2))
    kelly = max(0.0, min(1.0, ev_pnl / e_pnl2)) if e_pnl2 > 1e-12 else 0.0
    win = float(np.mean(pnl > 0))
    max_profit, max_loss = risk_profile(legs, cost)
    return EdgeResult(cost=cost, ev_payoff=ev, pv_payoff=pv, edge=edge, sd_payoff=sd,
                      score=score, kelly=kelly, win_prob=win, max_loss=max_loss,
                      max_profit=max_profit, is_credit=cost < 0)
