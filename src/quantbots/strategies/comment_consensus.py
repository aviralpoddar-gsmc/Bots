"""Consensus strategy: trade toward the Platt-extremized crowd probability.

The judge cycle pools each covered market's bettor crowd (one implied probability
per distinct user — their latest bet's probAfter) into a mean, then extremizes it
with p̂ = σ(√3·logit(p̄)) — the AIA Forecaster paper's calibration fix for
ensembles that hedge toward 0.5 (arXiv:2511.07678). This bot simply treats the
extremized consensus as fair value; the runner's hold_band decides when the gap
to the market price is worth trading.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from .base import Market, Strategy
from .linker import link_market


class CommentConsensusStrategy(Strategy):
    name = "comment_consensus"
    description = (
        "Bridgewater-style crowd consensus: pools the bot crowd's implied "
        "probabilities per market, Platt-extremizes the mean (σ(√3·logit(p̄))), "
        "and trades toward it when the market price lags the crowd."
    )

    def __init__(self, **params: Any):
        super().__init__(**params)
        self._store: Any = None
        self.min_forecasters = int(params.get("min_forecasters", 3))
        self.max_age_hours = float(params.get("max_age_hours", 24.0))

    def bind(self, observations: Any) -> None:
        self._store = observations

    def estimate(self, group: list[Market]) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._store is None:
            return out
        cutoff = (datetime.now(UTC) - timedelta(hours=self.max_age_hours)).isoformat()
        for m in group:
            mid = str(m.get("id"))
            row = self._store.load_comment_consensus(mid)
            if not row or row["n_forecasters"] < self.min_forecasters:
                continue
            if row["computed_at"] < cutoff:
                continue  # stale consensus — the crowd may have moved
            out[mid] = row["p_extreme"]
            self._explanations[mid] = {
                "p_mean": round(row["p_mean"], 4),
                "p_extreme": round(row["p_extreme"], 4),
                "n_forecasters": row["n_forecasters"],
                "market_prob": m.get("probability"),
            }
        return out

    def correlation_key(self, market: Market) -> str:
        link = link_market(market)
        if link and link.entities:
            return link.entities[0]
        return str(market.get("id"))

    def explain(self, market_id: str) -> str | None:
        d = self._explanations.get(market_id)
        if not d:
            return None
        return (f"Crowd consensus of {d['n_forecasters']} independent bettors: mean "
                f"{d['p_mean']:.2f} → extremized {d['p_extreme']:.2f} "
                f"(σ(√3·logit), AIA Forecaster calibration).")
