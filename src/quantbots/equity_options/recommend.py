"""Phase-1 analytics: turn the config universe + live chains into ranked option
recommendations (and a capped portfolio allocation). NO execution here.

Pipeline per underlying: fetch chain (Alpaca) -> fit beta once -> build f_P per
expiry -> select candidates -> rank. Then `allocate` applies the portfolio caps and
greek budgets across underlyings to produce the actual ticket list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import EquityOptionsConfig, Underlying
from .forecast.underlying import Forecast, build_forecast
from .portfolio import Allocation, allocate
from .research.beta import fit_beta
from .selection import Candidate, select
from .sources import underlying as und_src
from .sources.options_chain import ChainClient

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    allocations: list[Allocation]
    candidates_by_underlying: dict[str, list[Candidate]]


def candidates_for(u: Underlying, chain: list[dict], cfg: EquityOptionsConfig, *,
                   r: float, q: float) -> list[Candidate]:
    """Rank tradable candidates for one underlying given its chain."""
    fcast = cfg.forecast
    diff = fcast.get("diffusion", {})
    spot = und_src.spot(u.ticker)
    if spot is None:
        logger.warning("recommend: no spot for %s — skip", u.ticker)
        return []
    beta = fit_beta(u.ticker, u.commodity, u.market_ticker,
                    lookback_days=u.beta_lookback_days,
                    shrink=fcast.get("beta_shrink", 0.3),
                    idio_floor=fcast.get("idio_floor", 0.05))
    if beta is None or beta.weak:
        logger.info("recommend: %s beta unusable — abstain", u.ticker)
        return []

    # Directional view → signed equity drift, tilting f_P so the selector picks a
    # bull-call spread in up-trends / bear-put in down-trends.
    #   "fused"    — momentum + tal price-consensus (+macro), the validated 2026-06-23 blend.
    #   "momentum" — commodity TSMOM only.
    #   "drift_neutral" — no tilt.
    # tal's view is horizon-dependent, so for "fused" the drift is computed per-T; for
    # "momentum" it's T-invariant (same value each call, so the per-T cache still holds).
    mode = fcast.get("mode", "momentum")
    lbs = fcast.get("momentum_lookbacks")
    lookbacks = tuple(lbs) if lbs else None
    min_strength = float(fcast.get("momentum_min_strength", 0.0))

    def drift_for(T: float) -> float:
        if mode == "fused":
            from .research.fusion import fused_drift
            mu, _ = fused_drift(equity=u.ticker, commodity=u.commodity, beta_c=beta.beta_c,
                                spot=spot, horizon_years=T, momentum_lookbacks=lookbacks,
                                momentum_min_strength=min_strength)
            return mu
        if mode == "momentum":
            from .forecast.direction import momentum_drift
            mu, _ = momentum_drift(commodity=u.commodity, beta_c=beta.beta_c,
                                   lookbacks=lookbacks, min_strength=min_strength)
            return mu
        return 0.0  # drift_neutral

    cache: dict[float, Forecast | None] = {}

    def forecast_fn(T: float) -> Forecast | None:
        key = round(T, 4)
        if key not in cache:
            mu_view = drift_for(T)
            cache[key] = build_forecast(
                ticker=u.ticker, commodity=u.commodity, market=u.market_ticker,
                s0=spot, T=T, r=r, q=q,
                mode="directional" if mu_view != 0.0 else "drift_neutral", mu_view=mu_view,
                n_sims=int(diff.get("n_sims", 20000)), period=diff.get("period", "10y"),
                process=diff.get("process", "ksb"), beta=beta)
        return cache[key]

    return select(u.ticker, chain, forecast_fn, r=r, q=q, limits=cfg.risk_limits,
                  structures=cfg.structures)


def recommend(cfg: EquityOptionsConfig, *, chain_client: ChainClient | None = None,
              bankroll: float | None = None, exclude: set[str] | None = None) -> Recommendation:
    """Produce ranked candidates + a capped allocation across the universe.

    `exclude` is the set of underlyings already held on the broker — skipped so a
    daily run never stacks a second structure onto a name we're already in.
    """
    chain_client = chain_client or ChainClient(key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    exclude = {t.upper() for t in (exclude or set())}
    r = und_src.risk_free_rate()
    limits = cfg.risk_limits
    by_underlying: dict[str, list[Candidate]] = {}
    commodity_of: dict[str, str] = {}
    for u in cfg.enabled_underlyings():
        commodity_of[u.ticker] = u.commodity
        if u.ticker in exclude:
            logger.info("recommend: %s already held — skip (no stacking)", u.ticker)
            continue
        try:
            chain = chain_client.get_chain(u.ticker, min_dte=limits["min_dte"],
                                           max_dte=limits["max_dte"])
        except Exception as e:  # noqa: BLE001 - one bad chain must not abort the run
            logger.warning("recommend: chain fetch failed for %s: %s", u.ticker, e)
            continue
        q = und_src.dividend_yield(u.ticker)
        cands = candidates_for(u, chain, cfg, r=r, q=q)
        if cands:
            by_underlying[u.ticker] = cands

    best = [c[0] for c in by_underlying.values() if c]
    best.sort(key=lambda c: c.edge.score, reverse=True)
    allocations = allocate(best, bankroll=bankroll or 100_000.0, limits=limits,
                           commodity_of=commodity_of)
    return Recommendation(allocations=allocations, candidates_by_underlying=by_underlying)
