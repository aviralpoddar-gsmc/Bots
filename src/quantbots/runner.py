"""The bot loop: load -> prefilter -> group -> estimate -> size -> execute -> record.

This is shared infrastructure. The only strategy-specific call is
`strategy.estimate(group)`; everything else (sizing, execution, the trade ledger)
is identical for every bot. Defaults to dry-run for safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import BotConfig
from .manifold.client import ManifoldClient
from .sizing import compute_trade
from .store.db import Store
from .store.trades import ENTRY
from .strategies.base import Strategy

logger = logging.getLogger(__name__)
BATCH = 50


@dataclass
class RunResult:
    bot: str
    dry_run: bool
    n_markets: int
    signals: list[dict] = field(default_factory=list)
    orders_placed: int = 0
    errors: list[str] = field(default_factory=list)


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
    signals: list[dict] = []
    for group in strategy.group(markets):
        estimates = strategy.estimate(group)
        for m in group:
            est = estimates.get(m["id"])
            if est is None or m.get("probability") is None:
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
                    }
                )
    return signals


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
    signals = _decide(bot, strategy, markets, positions)

    result = RunResult(bot=bot.name, dry_run=dry_run, n_markets=len(markets), signals=signals)
    if not signals:
        return result

    if dry_run:
        # Validate each order against the platform without moving mana.
        for s in signals:
            try:
                client.place_bet(s["market_id"], s["direction"], s["amount"], dry_run=True)
            except Exception as e:  # noqa: BLE001 - surface, don't crash the loop
                result.errors.append(f"{s['market_id']}: {e}")
        return result

    # Live: place in batches of <=50, then write an ENTRY row per filled order.
    by_id = {s["market_id"]: s for s in signals}
    for i in range(0, len(signals), BATCH):
        chunk = signals[i : i + BATCH]
        bets = [
            {"contractId": s["market_id"], "outcome": s["direction"], "amount": s["amount"]}
            for s in chunk
        ]
        try:
            responses = client.batch_bet(bets)
        except Exception as e:  # noqa: BLE001
            result.errors.append(str(e))
            continue
        for resp in responses if isinstance(responses, list) else []:
            cid = resp.get("contractId") or resp.get("contract_id")
            s = by_id.get(cid) or (chunk[0] if len(chunk) == 1 else None)
            if "betId" not in resp or s is None:
                result.errors.append(str(resp))
                continue
            store.record_trade(
                bot_id=bot_id,
                market_id=s["market_id"],
                platform_bet_id=resp["betId"],
                trade_type=ENTRY,
                direction=s["direction"],
                amount=resp.get("amount", s["amount"]),
                shares=resp.get("shares", 0.0),
                price_before=resp.get("probBefore"),
                price_after=resp.get("probAfter"),
                llm_estimate=s["estimate"],
            )
            result.orders_placed += 1
    return result
