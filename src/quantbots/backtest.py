"""Backtest a strategy's probability model against real historical data.

The goal: measure whether a bot is *accurate* (well-calibrated) and would have been
*profitable* — BEFORE risking any mana live.

Method (no resolved markets needed): take a historical series (e.g. FRED mortgage
rate, weekly since 1971). For each date `t` and a forecast horizon `h`, treat the
value at `t` as the bot's "current observation", build the threshold questions the
bot would face, get the bot's estimate, and check it against what *actually
happened* at `t+h`. That yields thousands of (predicted_probability, real_outcome)
pairs to score:

- **Brier score** — mean squared error of the probability (lower better). Compare
  to the 0.25 baseline of always saying 50%.
- **calibration** — do "70%" calls happen ~70% of the time? (reliability buckets)
- **simulated PnL / ROI / win-rate** — if the bot had bet (via real sizing) on each
  untraded-0.50 market, would it have made mana at resolution?

The strategy sees `closeTime = now + h` so its horizon-scaled volatility is correct,
while the *outcome* uses the real t→t+h move. Pure-stdlib scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .sizing import DEFAULT_LIMITS, compute_trade
from .strategies.base import Strategy
from .strategies.ladder import attach_ladder_fields

_YEAR_MS = 365.25 * 24 * 3600 * 1000


class FakeObs:
    """Stand-in for the store: serves the historical value as the latest obs."""
    def __init__(self, values: dict[str, float]):
        self.values = values

    def latest_observation(self, entity: str, source: str | None = None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v, "source": "backtest"} if v is not None else None


@dataclass
class BacktestResult:
    n: int = 0
    brier: float = 0.0
    baseline_brier: float = 0.25  # always-50% reference
    win_rate: float = 0.0
    bets: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    reliability: list[tuple[float, float, int]] = field(default_factory=list)  # (mean_pred, mean_outcome, count)

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_staked if self.total_staked else 0.0

    @property
    def skill(self) -> float:
        """Brier skill score vs the 50% baseline: >0 means better than a coin."""
        return 1.0 - self.brier / self.baseline_brier if self.baseline_brier else 0.0


def _reliability(pairs: list[tuple[float, int]], buckets: int = 10) -> list[tuple[float, float, int]]:
    out = []
    for b in range(buckets):
        lo, hi = b / buckets, (b + 1) / buckets
        sel = [(e, o) for e, o in pairs if (lo <= e < hi or (b == buckets - 1 and e == 1.0))]
        if sel:
            mp = sum(e for e, _ in sel) / len(sel)
            mo = sum(o for _, o in sel) / len(sel)
            out.append((round(mp, 3), round(mo, 3), len(sel)))
    return out


def _simulate(pairs: list[tuple[float, int]], limits: dict, liquidity: float) -> tuple[float, float, int, int]:
    staked = profit = 0.0
    wins = bets = 0
    for est, outcome in pairs:
        d = compute_trade(estimate=est, current_prob=0.5, position=None,
                          liquidity=liquidity, limits=limits)
        if not d:
            continue
        bets += 1
        stake = d["amount"]
        staked += stake
        won = (d["direction"] == "YES" and outcome == 1) or (d["direction"] == "NO" and outcome == 0)
        profit += stake if won else -stake
        wins += 1 if won else 0
    return staked, profit, wins, bets


def backtest(
    strategy: Strategy,
    entity: str,
    question_template: str,
    series: list[tuple[str, float]],
    horizon_steps: int,
    horizon_years: float,
    threshold_fracs: tuple[float, ...] = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
    limits: dict | None = None,
    liquidity: float = 100.0,
) -> BacktestResult:
    """Replay `strategy` over `series` and score calibration + simulated PnL.

    `question_template` must contain `{T}` and link to `entity` (so the strategy's
    own linker/model is exercised — we test what we'd deploy).
    """
    limits = limits or DEFAULT_LIMITS
    close_time = time.time() * 1000 + horizon_years * _YEAR_MS

    pairs: list[tuple[float, int]] = []
    for i in range(len(series) - horizon_steps):
        current = series[i][1]
        future = series[i + horizon_steps][1]
        if current is None or future is None or current <= 0:
            continue
        strategy.bind(FakeObs({entity: current}))
        for frac in threshold_fracs:
            threshold = round(current * frac, 4)
            market = attach_ladder_fields({
                "id": f"{i}:{frac}",
                "question": question_template.format(T=threshold),
                "probability": 0.5,
                "closeTime": close_time,
                "totalLiquidity": liquidity,
                "isResolved": False,
            })
            est = strategy.estimate([market]).get(market["id"])
            if est is None:
                continue
            pairs.append((est, 1 if future > threshold else 0))

    res = BacktestResult(n=len(pairs))
    if not pairs:
        return res
    res.brier = sum((e - o) ** 2 for e, o in pairs) / len(pairs)
    res.baseline_brier = sum((0.5 - o) ** 2 for e, o in pairs) / len(pairs)
    staked, profit, wins, bets = _simulate(pairs, limits, liquidity)
    res.total_staked, res.total_profit, res.bets = staked, profit, bets
    res.win_rate = wins / bets if bets else 0.0
    res.reliability = _reliability(pairs)
    return res
