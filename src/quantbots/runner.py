"""The bot loop: load -> prefilter -> group -> estimate -> size -> execute -> record.

This is shared infrastructure. The only strategy-specific call is
`strategy.estimate(group)`; everything else (sizing, execution, the trade ledger)
is identical for every bot. Defaults to dry-run for safety.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import BotConfig
from .manifold.client import ManifoldClient
from .portfolio import allocate, book_summary
from .resolvability import resolvability_score
from .sizing import compute_trade
from .store.db import Store
from .store.trades import ENTRY
from .strategies.base import Strategy

logger = logging.getLogger(__name__)
BATCH = 50
MAX_BET_ATTEMPTS = 4  # retry transiently-throttled bets this many sweeps
RETRY_BACKOFF = 2.0  # seconds, multiplied by attempt number (linear backoff)


def _is_throttle(resp_or_err: Any) -> bool:
    """True if a batch_bet response/error is a transient server throttle (the bet
    was not placed and is safe to retry), not a hard rejection."""
    txt = str(resp_or_err).lower()
    return "high volume" in txt or "try again" in txt or "queue" in txt or "rate limit" in txt


@dataclass
class RunResult:
    bot: str
    dry_run: bool
    n_markets: int
    signals: list[dict] = field(default_factory=list)
    orders_placed: int = 0
    errors: list[str] = field(default_factory=list)
    candidates: int = 0  # signals before budget/concentration allocation
    book: dict = field(default_factory=dict)  # allocator summary (staked, exp_profit, ...)


def _settlement_prob(market: dict) -> float | None:
    """Market probability to settle a resolved binary at: 1.0 / 0.0 / MKT prob."""
    res = market.get("resolution")
    if res == "YES":
        return 1.0
    if res == "NO":
        return 0.0
    if res == "MKT":
        return market.get("resolutionProbability")
    return None


def sync_resolutions(client: ManifoldClient, store: Store, bot_id: int) -> int:
    """For each open position, if the market has resolved, insert a synthetic
    RESOLUTION_CLOSE trade so PnL realizes with no special-case code."""
    closed = 0
    for market_id, pos in store.open_positions(bot_id).items():
        market = client.get_market(market_id)
        store.upsert_markets([market])
        if not market.get("isResolved"):
            continue
        prob = _settlement_prob(market)
        if prob is None:
            continue
        net_shares = pos["net_shares"]
        proceeds = net_shares * (prob if pos["direction"] == "YES" else 1 - prob)
        store.record_trade(
            bot_id=bot_id,
            market_id=market_id,
            trade_type="RESOLUTION_CLOSE",
            direction=pos["direction"],
            amount=proceeds,
            shares=net_shares,
            price_after=prob,
            reasoning=f"resolved {market.get('resolution')}",
        )
        closed += 1
    return closed


def _decide(bot: BotConfig, strategy: Strategy, markets: list[dict],
            positions: dict[str, dict]) -> list[dict]:
    min_resolv = float(bot.limits.get("min_resolvability", 0.0))
    signals: list[dict] = []
    for group in strategy.group(markets):
        estimates = strategy.estimate(group)
        for m in group:
            est = estimates.get(m["id"])
            if est is None or m.get("probability") is None:
                continue
            # Cancellation-aware: skip markets unlikely to ever resolve YES/NO.
            resolv = resolvability_score(m.get("question", ""))
            if resolv < min_resolv:
                continue
            decision = compute_trade(
                estimate=est,
                current_prob=m["probability"],
                position=positions.get(m["id"]),
                liquidity=m.get("totalLiquidity"),
                limits=bot.limits,
            )
            if decision:
                signals.append(
                    {
                        "market_id": m["id"],
                        "question": m.get("question"),
                        "current_prob": m["probability"],
                        "estimate": est,
                        "direction": decision["direction"],
                        "amount": decision["amount"],
                        "edge": abs(est - m["probability"]),
                        "group": strategy.correlation_key(m),
                        "resolvability": resolv,
                    }
                )
    return signals


def _allocate(
    signals: list[dict],
    limits: dict[str, Any],
    existing_total: float = 0.0,
    existing_group: dict[str, float] | None = None,
) -> list[dict]:
    """Size the run's book with the portfolio allocator: rank by EV per mana,
    fund best-first up to the total run budget, capping per-correlation-group
    exposure both within this run and cumulatively across runs (so repeated live
    runs can't over-accumulate in one underlying). Replaces the old edge-sorted
    greedy fill so a bot can deploy across thousands of markets safely.
    """
    total = limits.get("max_run_budget")
    per_group = limits.get("per_group_budget")
    # Allow a fraction-of-budget concentration cap (ergonomic; converted to mana).
    pct = limits.get("per_group_pct")
    if per_group is None and pct and total:
        per_group = float(total) * float(pct)
    return allocate(
        signals,
        total_budget=total,
        per_group_budget=per_group,
        min_ev=float(limits.get("min_ev", 0.0)),
        min_order_mana=float(limits.get("min_order_mana", 1)),
        max_total_exposure=limits.get("max_total_exposure"),
        max_group_exposure=limits.get("max_group_exposure"),
        existing_total=existing_total,
        existing_group=existing_group or {},
    )


def run_bot(
    *,
    bot: BotConfig,
    client: ManifoldClient,
    store: Store,
    strategy: Strategy,
    dry_run: bool = True,
) -> RunResult:
    bot_id = store.upsert_bot(
        bot.name, bot.strategy, {"limits": bot.limits, "params": bot.params}, bot.enabled
    )

    strategy.bind(store)  # give data-driven strategies a read handle to observations
    markets = strategy.prefilter(store.load_open_markets())
    positions = store.open_positions(bot_id)
    candidates = _decide(bot, strategy, markets, positions)
    # Tally already-deployed exposure (from the ledger) per correlation group so the
    # allocator can enforce cumulative across-run caps, not just per-run ones.
    market_group = {m["id"]: strategy.correlation_key(m) for m in markets}
    existing_group: dict[str, float] = {}
    existing_total = 0.0
    for mid, pos in positions.items():
        staked = pos.get("net_amount", pos.get("entry_amount", 0.0)) or 0.0
        g = market_group.get(mid, mid)
        existing_group[g] = existing_group.get(g, 0.0) + staked
        existing_total += staked
    # Portfolio allocation: EV-ranked, budget- and concentration-capped book.
    signals = _allocate(candidates, bot.limits, existing_total, existing_group)

    result = RunResult(bot=bot.name, dry_run=dry_run, n_markets=len(markets), signals=signals)
    result.candidates = len(candidates)
    result.book = book_summary(signals)
    if not signals:
        return result

    if dry_run:
        # Validate orders against the platform without moving mana. At scale we
        # validate a sample (each call is a request; the live path batches), enough
        # to catch systematic problems — auth, payload shape, closed markets.
        sample_n = int(bot.limits.get("dry_run_sample", 25))
        sample = signals if sample_n <= 0 else signals[:sample_n]
        for s in sample:
            try:
                client.place_bet(s["market_id"], s["direction"], s["amount"], dry_run=True)
            except Exception as e:  # noqa: BLE001 - surface, don't crash the loop
                result.errors.append(f"{s['market_id']}: {e}")
        return result

    # Live: place in batches of <=50, recording an ENTRY per fill. The server can
    # transiently throttle ("High volume of requests; please try again") — those
    # bets were NOT placed, so we collect and retry them with backoff (safe, no
    # double-bet). Hard errors (bad payload, insufficient balance) are not retried.
    by_id = {s["market_id"]: s for s in signals}

    def _record(resp: dict, s: dict) -> None:
        store.record_trade(
            bot_id=bot_id, market_id=s["market_id"], platform_bet_id=resp["betId"],
            trade_type=ENTRY, direction=s["direction"],
            amount=resp.get("amount", s["amount"]), shares=resp.get("shares", 0.0),
            price_before=resp.get("probBefore"), price_after=resp.get("probAfter"),
            llm_estimate=s["estimate"],
        )
        result.orders_placed += 1

    pending = list(signals)
    for attempt in range(MAX_BET_ATTEMPTS):
        throttled: list[dict] = []
        for i in range(0, len(pending), BATCH):
            chunk = pending[i : i + BATCH]
            bets = [
                {"contractId": s["market_id"], "outcome": s["direction"], "amount": s["amount"]}
                for s in chunk
            ]
            try:
                responses = client.batch_bet(bets)
            except Exception as e:  # noqa: BLE001 - whole-batch failure
                (throttled.extend(chunk) if _is_throttle(e)
                 else result.errors.append(str(e)))
                continue
            for resp in responses if isinstance(responses, list) else []:
                cid = resp.get("contractId") or resp.get("contract_id")
                s = by_id.get(cid) or (chunk[0] if len(chunk) == 1 else None)
                if "betId" in resp and s is not None:
                    _record(resp, s)
                elif s is not None and _is_throttle(resp):
                    throttled.append(s)  # not placed — safe to retry
                else:
                    result.errors.append(str(resp))
        pending = throttled
        if not pending or attempt == MAX_BET_ATTEMPTS - 1:
            break
        time.sleep(RETRY_BACKOFF * (attempt + 1))  # linear backoff between sweeps
    for s in pending:
        result.errors.append(f"{s['market_id']}: throttled after {MAX_BET_ATTEMPTS} attempts")
    return result
