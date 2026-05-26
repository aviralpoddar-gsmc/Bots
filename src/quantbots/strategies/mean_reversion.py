"""A no-LLM, no-extra-deps example strategy: single-market mean reversion.

Estimate = an exponential moving average of the market's own recent prices, so
the bot fades short-term moves back toward a slower trend. State (the EMA per
market) is held in memory across `estimate` calls within one process; for a
longer-lived bot you would persist it. This exists mainly as the simplest
possible reference implementation of the `Strategy` contract.
"""

from __future__ import annotations

from .base import Market, Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self, alpha: float = 0.2, **params: object):
        super().__init__(alpha=alpha, **params)
        self.alpha = alpha
        self._ema: dict[str, float] = {}

    def estimate(self, group: list[Market]) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in group:
            p = m.get("probability")
            if p is None:
                continue
            prev = self._ema.get(m["id"], p)
            ema = self.alpha * p + (1 - self.alpha) * prev
            self._ema[m["id"]] = ema
            # Fair value = the slower EMA; the runner trades the gap to it.
            out[m["id"]] = min(max(ema, 0.01), 0.99)
        return out
