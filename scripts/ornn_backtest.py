#!/usr/bin/env python
"""Quick ladder backtest of GPU compute-price markets off the Ornn index.

Pulls the Ornn Compute Price Index history (keyless) for each GPU SKU, then
replays the project's `backtest()` ladder harness with a zero-drift lognormal
pricer — the same model `commodity_spot`/`diffusion_mc` use on metals — to ask:
if we minted "compute index exceeds $T per GPU-hour by date Y" markets on the
clone, would the lognormal be calibrated and profitable against the realized
moves?

The lognormal's annual vol is estimated from the SKU's own daily log returns.

CAVEATS (it's a *quick* check, not the full walk-forward gate):
  - Vol is estimated in-sample on the whole series (no walk-forward split). The
    Ornn history is short (~3 months daily), so folds would be tiny; treat the
    Brier/PnL as indicative, not deployment-grade. scripts/diffusion_bench.py is
    the rigorous template once more history accrues.
  - Outcomes use the realized t -> t+h move; PnL bets from the 0.50 prior the
    clone's untraded markets sit at, sized by the real framework.

USAGE
    uv run python scripts/ornn_backtest.py
    uv run python scripts/ornn_backtest.py --horizon-days 30 --gpus GPU_H100_SXM,GPU_B200
"""

from __future__ import annotations

import argparse
import math

from quantbots.backtest import backtest
from quantbots.sources.ornn import _DEFAULT_GPUS, OrnnComputeSource
from quantbots.strategies._model import norm_cdf, years_to_close
from quantbots.strategies.base import Market, Strategy
from quantbots.strategies.ladder import parse_threshold

_TRADING_DAYS = 252.0
# Strikes span body AND tails. Near-spot strikes test calibration (Brier); the
# tail strikes test the realized edge that actually pays on the clone — far-OTM
# markets that sit at the untraded 0.50 prior while the model says ~0.05/0.95.
_FRACS = (0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3)


class _FakeObs:
    """Serve a single bound spot as the latest observation (backtest harness API)."""

    def __init__(self, values: dict[str, float]):
        self.values = values

    def latest_observation(self, entity: str, source: str | None = None):
        v = self.values.get(entity)
        return {"entity": entity, "value": v} if v is not None else None


class GpuLognormalStrategy(Strategy):
    """Zero-drift lognormal CDF over a price ladder for one GPU entity.

    Mirrors commodity_spot's math without the metals unit/keyword guards (the
    backtest harness controls the universe, so no text filtering is needed)."""

    name = "gpu_lognormal"

    def __init__(self, entity: str, annual_vol: float, min_vol: float = 0.05, **params):
        super().__init__(entity=entity, annual_vol=annual_vol, min_vol=min_vol, **params)
        self.entity = entity
        self.annual_vol = annual_vol
        self.min_vol = min_vol
        self._obs = None

    def bind(self, observations) -> None:
        self._obs = observations

    def estimate(self, group: list[Market]) -> dict[str, float]:
        if self._obs is None:
            return {}
        out: dict[str, float] = {}
        for m in group:
            parsed = parse_threshold(m.get("question", ""))
            if parsed is None:
                continue
            threshold, direction = parsed
            o = self._obs.latest_observation(self.entity)
            if not o or o.get("value") is None or o["value"] <= 0 or threshold <= 0:
                continue
            spot = o["value"]
            T = years_to_close(m)
            sigma = max(self.annual_vol * math.sqrt(T), self.min_vol)
            surv = 1.0 - norm_cdf(math.log(threshold / spot) / sigma)
            p = surv if direction == "exceeds" else 1.0 - surv
            out[m["id"]] = min(max(p, 0.01), 0.99)
        return out


def _series_for(entity: str, gpu: str) -> list[tuple[str, float]]:
    obs = OrnnComputeSource(gpus={entity: gpu}).fetch()
    pts = [(o.ts, o.value) for o in obs if o.value is not None]
    return sorted(pts, key=lambda x: x[0])


def _annual_vol(series: list[tuple[str, float]]) -> float:
    rets = [
        math.log(series[i][1] / series[i - 1][1])
        for i in range(1, len(series))
        if series[i - 1][1] > 0 and series[i][1] > 0
    ]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(_TRADING_DAYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--gpus", default="", help="comma-sep entities; default all")
    ap.add_argument("--fracs", default="", help="comma-sep strike fractions of spot; default body+tails")
    args = ap.parse_args()

    fracs = (
        tuple(float(x) for x in args.fracs.split(",") if x.strip())
        if args.fracs
        else _FRACS
    )

    entities = (
        [e.strip() for e in args.gpus.split(",") if e.strip()]
        if args.gpus
        else list(_DEFAULT_GPUS)
    )
    h_years = args.horizon_days / 365.25

    print(f"Ornn GPU compute-price ladder backtest — horizon {args.horizon_days}d, "
          f"strikes {fracs}\n")
    print(f"{'entity':22} {'n':>5} {'pts':>4} {'spot':>7} {'vol/yr':>7} "
          f"{'Brier':>7} {'skill':>7} {'win%':>6} {'bets':>5} {'ROI':>8}")
    print("-" * 92)

    agg_n = agg_brier = agg_base = 0.0
    agg_staked = agg_profit = agg_bets = agg_wins = 0.0
    for entity in entities:
        gpu = _DEFAULT_GPUS.get(entity, entity)
        try:
            series = _series_for(entity, gpu)
        except Exception as ex:  # noqa: BLE001
            print(f"{entity:22} fetch failed: {ex}")
            continue
        if len(series) < args.horizon_days + 5:
            print(f"{entity:22} only {len(series)} pts, skip")
            continue
        vol = _annual_vol(series)
        strat = GpuLognormalStrategy(entity=entity, annual_vol=vol)
        # backtest() rebinds the strategy per step; bind once for type-safety.
        strat.bind(_FakeObs({entity: series[-1][1]}))
        r = backtest(
            strategy=strat,
            entity=entity,
            question_template=(
                f"Will the {gpu} GPU compute price index exceed "
                "${T} per GPU-hour by 2026-12-31?"
            ),
            series=series,
            horizon_steps=args.horizon_days,
            horizon_years=h_years,
            threshold_fracs=fracs,
        )
        print(f"{entity:22} {r.n:5d} {len(series):4d} {series[-1][1]:7.2f} "
              f"{vol:7.2%} {r.brier:7.4f} {r.skill:+7.2%} {r.win_rate:6.1%} "
              f"{r.bets:5d} {r.roi:+8.2%}")
        agg_n += r.n
        agg_brier += r.brier * r.n
        agg_base += r.baseline_brier * r.n
        agg_staked += r.total_staked
        agg_profit += r.total_profit
        agg_bets += r.bets
        agg_wins += r.win_rate * r.bets

    print("-" * 92)
    if agg_n:
        brier = agg_brier / agg_n
        base = agg_base / agg_n
        skill = 1.0 - brier / base if base else 0.0
        win = agg_wins / agg_bets if agg_bets else 0.0
        roi = agg_profit / agg_staked if agg_staked else 0.0
        print(f"{'FLEET TOTAL':22} {int(agg_n):5d} {'':4} {'':7} {'':7} "
              f"{brier:7.4f} {skill:+7.2%} {win:6.1%} {int(agg_bets):5d} {roi:+8.2%}")
        print(f"\nBaseline Brier (always-50%) = {base:.4f}. "
              f"skill>0 => better than a coin; ROI = profit / staked.")


if __name__ == "__main__":
    main()
