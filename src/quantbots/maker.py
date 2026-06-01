"""Market-maker execution path — the 'maker' counterpart to runner.run_bot.

Why this is separate from the taker runner: a taker fires one market order and
records an ENTRY for the immediate fill. A *maker* posts limit orders that REST —
they fill over time (often zero shares at placement), so fills cannot be recorded
at placement. This module instead reconciles fills from the platform into the
ledger each cycle, then (re-)posts two-sided quotes around the strategy's fair
value with a TTL.

Per live cycle (see run_maker):
  1. reconcile fills — over EVERY market we have a resting order or position in
                       (not just this cycle's quotes), so orphaned markets don't
                       leak fills; record newly-filled tranches as ENTRY rows
  2. budget          — net of already-committed capital (filled inventory + the
                       reserved mana of still-resting orders)
  3. quote           — build two-sided bid (YES @ f-s) / ask (NO @ f+s) quotes,
                       capped per correlation group and by the inventory cap
  4. cancel + repost — cancel our resting quotes on managed markets; repost only
                       on markets that fully cancelled (a cancel failure must not
                       double-stack the book — the TTL clears the stale order)
  5. reconcile again — capture any leg that crossed and filled immediately

v1 is fixed spread (strategy.half_spread). Price skew and toxic-flow widening are
Phase 3 (docs/market-maker.md).

STATE-MODEL GUARDRAIL: resting orders are NEVER written to the trade ledger and
there is no RESTING trade_type (it would break every trade_type consumer). The
ledger records FILLS ONLY, as ENTRY rows. Resting state is live/ephemeral, read
from the API via client.get_open_limit_orders each cycle.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .config import BotConfig
from .manifold.client import ManifoldClient
from .resolvability import resolvability_score
from .store.db import Store
from .store.trades import ENTRY, group_positions, summarize_position

logger = logging.getLogger(__name__)
BATCH = 50


@dataclass
class Quote:
    """A two-sided quote around `fair` for one market. `bid`/`ask` are YES-prob
    limit prices: the bid is a YES limit at f-s (fills when price falls), the ask
    is a NO limit at f+s (fills when price rises). `sides` says which legs to post
    (both, or one if the inventory cap forces one-sided quoting)."""

    market_id: str
    question: str | None
    fair: float
    bid: float
    ask: float
    size: int
    sides: tuple[str, ...]
    resolvability: float


@dataclass
class MakerResult:
    bot: str
    dry_run: bool
    n_markets: int = 0
    quotes: list[Quote] = field(default_factory=list)
    legs_posted: int = 0
    cancelled: int = 0
    fills_recorded: int = 0
    reserved_mana: float = 0.0
    errors: list[str] = field(default_factory=list)


def build_maker_strategy(bot: BotConfig) -> Any:
    """Build the maker strategy for a maker-mode bot (maker mode on the runner).

    If the bot's strategy is already `market_maker` (the explicit wrapper), build
    it straight from params. Otherwise wrap the bot's OWN strategy as the
    fair-value source and take the maker execution knobs from `limits` — so any
    calibrated anchor (commodity_spot, stockpile_facts, ...) becomes a liquidity
    provider on its own account just by setting `maker: true`, with no separate
    bot and no double-stacking.
    """
    from .strategies import get_strategy

    if bot.strategy == "market_maker":
        return get_strategy("market_maker", **bot.params)
    lim = bot.limits
    return get_strategy(
        "market_maker",
        fair_value_source=bot.strategy,
        source_params=bot.params,
        base_half_spread=float(lim.get("half_spread", 0.04)),
        min_half_spread=float(lim.get("min_half_spread", 0.02)),
        inventory_cap=float(lim.get("inventory_cap", 200)),
        quote_ttl_hours=float(lim.get("quote_ttl_hours", 25)),
        max_markets=int(lim.get("max_markets", 10)),
    )


def _clamp_prob(p: float) -> float:
    """Round to whole-percent (the server only accepts <=2dp limitProb) and clamp
    to the tradeable (0.01, 0.99) band."""
    return min(max(round(p, 2), 0.01), 0.99)


def _quote_size(liquidity: float | None, limits: dict[str, Any]) -> int:
    """Per-leg mana: size to liquidity (nudge, don't slam), capped by max_order_size."""
    liq = max(liquidity or 0.0, 100.0)
    size = min(int(limits.get("max_order_size", 50)), int(liq * float(limits.get("liquidity_pct", 0.25))))
    return max(size, 0)


def _limit_float(limits: dict[str, Any], key: str, default: float) -> float:
    """Read a limit as a float, treating None/0 as 'unset' (use default)."""
    v = limits.get(key)
    return float(v) if v not in (None, 0) else default


def _ledger_state(
    store: Store, bot_id: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """Derive from the ledger:
      inv      — NET YES shares per market (YES + , NO - ) over OPEN legs only
      rec_sh   — shares already recorded per platform_bet_id (ENTRY rows): the
                 reconcile baseline that makes fill recording idempotent
      rec_amt  — mana already recorded per platform_bet_id (ENTRY rows)
      filled   — NET mana committed to open inventory (exposure)

    inv/filled net out EXIT/RESOLUTION_CLOSE rows (via summarize_position) so a
    refunded CANCEL — ~93% of clone resolutions — stops counting against exposure
    once `quantbots resolve` closes it. rec_* stay keyed on ENTRY rows only.
    """
    trades = store.trades_for_bot(bot_id)
    rec_sh: dict[str, float] = {}
    rec_amt: dict[str, float] = {}
    for t in trades:
        if t["trade_type"] != ENTRY:
            continue
        bet = t.get("platform_bet_id")
        if bet:
            rec_sh[bet] = rec_sh.get(bet, 0.0) + (t["shares"] or 0.0)
            rec_amt[bet] = rec_amt.get(bet, 0.0) + (t["amount"] or 0.0)
    inv: dict[str, float] = {}
    filled = 0.0
    for (mid, direction), pos_trades in group_positions(trades).items():
        s = summarize_position(pos_trades)
        if s["status"] != "OPEN":
            continue
        sign = 1.0 if direction == "YES" else -1.0
        inv[mid] = inv.get(mid, 0.0) + sign * s["net_shares"]
        filled += s["net_amount"]
    return inv, rec_sh, rec_amt, filled


def reconcile_fills(
    client: ManifoldClient,
    store: Store,
    bot_id: int,
    uid: str,
    market_ids: list[str],
    rec_sh: dict[str, float],
    rec_amt: dict[str, float],
) -> int:
    """Record newly-filled tranches of our limit orders as ENTRY rows.

    Idempotent: keyed on the platform bet id, it records only the delta
    (filled_now - already_recorded) in both shares and mana, then advances the
    baseline. Safe to run repeatedly; re-running with no new fills is a no-op.
    Returns the number of fill rows written.
    """
    n = 0
    for mid in market_ids:
        try:
            bets = client.get_bets(contractId=mid, userId=uid, limit=200)
        except Exception as e:  # noqa: BLE001 - a read failure must not unwind the cycle
            logger.warning("reconcile: get_bets failed for %s: %s", mid, e)
            continue
        for b in bets:
            if b.get("limitProb") is None:
                continue  # the maker only deals in limit orders
            filled_sh = float(b.get("shares") or 0.0)
            if filled_sh <= 0:
                continue
            bet_id = b.get("id")
            if not bet_id:
                continue
            d_sh = filled_sh - rec_sh.get(bet_id, 0.0)
            if d_sh <= 1e-9:
                continue  # nothing new filled on this order
            filled_amt = float(b.get("amount") or 0.0)
            d_amt = max(filled_amt - rec_amt.get(bet_id, 0.0), 0.0)
            store.record_trade(
                bot_id=bot_id,
                market_id=mid,
                platform_bet_id=bet_id,
                trade_type=ENTRY,
                direction=b.get("outcome", "YES"),
                amount=d_amt,
                shares=d_sh,
                price_before=b.get("probBefore"),
                price_after=b.get("probAfter"),
                reasoning="maker fill",
            )
            rec_sh[bet_id] = filled_sh
            rec_amt[bet_id] = filled_amt
            n += 1
    return n


def _select_markets(priced: list[dict], strategy: Any) -> list[dict]:
    """Pick the markets to quote: the most liquid, but DIVERSIFIED across
    correlation groups so the breadth cap can't load the whole book into one
    underlying (10 GOLD strikes). Round-robins the most-liquid market of each
    group, then the next, until max_markets is reached."""
    n = int(strategy.max_markets)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for m in priced:
        by_group[strategy.correlation_key(m)].append(m)
    for g in by_group.values():
        g.sort(key=lambda m: (m.get("totalLiquidity") or 0.0), reverse=True)
    # Visit groups most-liquid-first; take one market per group per pass.
    order = sorted(by_group.values(), key=lambda g: (g[0].get("totalLiquidity") or 0.0), reverse=True)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(i < len(g) for g in order):
        for g in order:
            if i < len(g):
                out.append(g[i])
                if len(out) >= n:
                    break
        i += 1
    return out


def build_quotes(
    strategy: Any,
    markets: list[dict],
    fair: dict[str, float],
    inv: dict[str, float],
    limits: dict[str, Any],
    min_resolv: float,
    budget: float,
) -> list[Quote]:
    """Turn fair values into postable two-sided quotes under the cycle's reserve
    budget and per-correlation-group caps. Drops: markets the source didn't price,
    below-resolvability markets, quotes that collapse to crossed or quote inside
    min_half_spread after the whole-percent clamp (near the 0.01/0.99 boundary),
    and sub-minimum sizes. Flips to one-sided past the inventory cap."""
    half = float(strategy.half_spread())
    min_half = float(getattr(strategy, "min_half_spread", half))
    cap = float(strategy.inventory_cap)
    min_order = float(limits.get("min_order_mana", 1))
    per_group_pct = limits.get("per_group_pct")
    group_budget = (
        float(per_group_pct) * budget if (per_group_pct and budget != math.inf) else math.inf
    )
    max_group_exp = limits.get("max_group_exposure")
    quotes: list[Quote] = []
    reserved = 0.0
    reserved_group: dict[str, float] = defaultdict(float)
    for m in markets:
        if reserved >= budget:
            break
        mid = m["id"]
        f = fair.get(mid)
        if f is None:
            continue
        resolv = resolvability_score(m.get("question", ""))
        if resolv < min_resolv:
            continue
        bid = _clamp_prob(f - half)
        ask = _clamp_prob(f + half)
        # Per-leg effective-spread floor: near 0.01/0.99 the clamp squashes one
        # side (or puts it on the wrong side of fair). Never quote inside min_spread.
        # (eps absorbs float noise so an exact-min spread like 0.58-0.54 still passes.)
        if bid >= ask or (f - bid) < min_half - 1e-9 or (ask - f) < min_half - 1e-9:
            continue
        size = _quote_size(m.get("totalLiquidity"), limits)
        if size < min_order:
            continue
        # Inventory cap: too long YES -> quote only the selling (NO/ask) side;
        # too long NO -> quote only the buying (YES/bid) side. Mean-reverts to flat.
        net = inv.get(mid, 0.0)
        if net >= cap:
            sides: tuple[str, ...] = ("ask",)
        elif net <= -cap:
            sides = ("bid",)
        else:
            sides = ("bid", "ask")
        legcost = size * len(sides)
        # `continue` not `break`: a later, cheaper (smaller / one-sided) quote may
        # still fit even when this one doesn't (mirrors portfolio.allocate).
        if reserved + legcost > budget:
            continue
        group = strategy.correlation_key(m)
        if reserved_group[group] + legcost > group_budget:
            continue  # per-correlation-group cap (no single underlying dominates)
        if max_group_exp is not None and reserved_group[group] + legcost > float(max_group_exp):
            continue
        reserved += legcost
        reserved_group[group] += legcost
        quotes.append(Quote(mid, m.get("question"), f, bid, ask, size, sides, resolv))
    return quotes


def _legs(quotes: list[Quote]) -> list[tuple[str, str, float, int]]:
    """Flatten quotes into individual order legs (market_id, outcome, limitProb, size)."""
    out: list[tuple[str, str, float, int]] = []
    for q in quotes:
        if "bid" in q.sides:
            out.append((q.market_id, "YES", q.bid, q.size))
        if "ask" in q.sides:
            out.append((q.market_id, "NO", q.ask, q.size))
    return out


def run_maker(
    *,
    bot: BotConfig,
    client: ManifoldClient,
    store: Store,
    strategy: Any,
    dry_run: bool = True,
) -> MakerResult:
    """Run one market-maker cycle. Dry-run (default) validates quote payloads
    without moving mana, cancelling, or writing to the ledger."""
    bot_id = store.upsert_bot(
        bot.name, bot.strategy, {"limits": bot.limits, "params": bot.params}, bot.enabled
    )
    strategy.bind(store)
    markets = strategy.prefilter(store.load_open_markets())

    # Fair value for every market the source can price, then a diversified breadth cap.
    fair: dict[str, float] = {}
    for grp in strategy.group(markets):
        fair.update(strategy.estimate(grp))
    priced = _select_markets([m for m in markets if m["id"] in fair], strategy)

    result = MakerResult(bot=bot.name, dry_run=dry_run, n_markets=len(priced))
    inv, rec_sh, rec_amt, filled = _ledger_state(store, bot_id)

    uid: str | None = None
    existing_mids: set[str] = set()
    resting_committed = 0.0
    if not dry_run:
        uid = client.get_me()["id"]
        try:
            all_open = client.get_open_limit_orders(user_id=uid)
        except Exception as e:  # noqa: BLE001
            all_open = []
            result.errors.append(f"list open orders: {e}")
        # Recent limit bets too: a limit order that FULLY filled between cycles
        # drops out of open-limit, so it (and its market) must be picked up here or
        # its terminal fill is lost. open-limit + inventory + recent fully covers it.
        try:
            recent = client.get_bets(userId=uid, limit=500)
        except Exception as e:  # noqa: BLE001
            recent = []
            result.errors.append(f"list recent bets: {e}")
        recent_mids = {
            b.get("contractId")
            for b in recent
            if b.get("limitProb") is not None and b.get("contractId")
        }
        existing_mids = (
            {o.get("contractId") for o in all_open if o.get("contractId")} | set(inv) | recent_mids
        )
        resting_committed = sum(
            max(float(o.get("orderAmount") or 0.0) - float(o.get("amount") or 0.0), 0.0)
            for o in all_open
        )
        # Reconcile fills accrued on existing orders/positions BEFORE re-quoting —
        # including orphaned markets no longer in this cycle's quote set.
        result.fills_recorded += reconcile_fills(
            client, store, bot_id, uid, sorted(existing_mids), rec_sh, rec_amt
        )
        inv, rec_sh, rec_amt, filled = _ledger_state(store, bot_id)

    # Per-cycle reserve budget = min(run budget, total-exposure headroom net of
    # capital already committed: filled inventory + reserved (resting) orders).
    budget = _limit_float(bot.limits, "max_run_budget", math.inf)
    max_total = bot.limits.get("max_total_exposure")
    if max_total is not None:
        budget = min(budget, max(float(max_total) - filled - resting_committed, 0.0))

    min_resolv = float(bot.limits.get("min_resolvability", 0.0))
    quotes = build_quotes(strategy, priced, fair, inv, bot.limits, min_resolv, budget)
    result.quotes = quotes
    if not quotes:
        return result

    ttl_ms = int(strategy.quote_ttl_hours * 3600 * 1000)
    legs = _legs(quotes)
    result.reserved_mana = float(sum(size for _, _, _, size in legs))

    if dry_run:
        # Validate a sample of legs against the platform without moving mana,
        # cancelling, or writing the ledger.
        sample_n = int(bot.limits.get("dry_run_sample", 25))
        for mid, outcome, lp, size in (legs if sample_n <= 0 else legs[:sample_n]):
            try:
                client.place_bet(
                    mid, outcome, size, limit_prob=lp, expires_millis_after=ttl_ms, dry_run=True
                )
            except Exception as e:  # noqa: BLE001 - surface, don't crash the loop
                result.errors.append(f"{mid} {outcome}@{lp}: {e}")
        return result

    # --- LIVE -----------------------------------------------------------------
    assert uid is not None
    quote_mids = {q.market_id for q in quotes}
    manage_mids = existing_mids | quote_mids

    # Cancel our resting quotes on managed markets. Track which markets fully
    # cleared so we never repost onto a market that still has a stale order
    # (a cancel failure leaves the TTL to expire it instead of double-stacking).
    cancel_failed: set[str] = set()
    for mid in sorted(manage_mids):
        try:
            for o in client.get_open_limit_orders(market_id=mid, user_id=uid):
                client.cancel_bet(o["id"])
                result.cancelled += 1
        except Exception as e:  # noqa: BLE001
            cancel_failed.add(mid)
            result.errors.append(f"cancel {mid}: {e}")

    # Post fresh two-sided quotes (only on fully-cleared markets), batched, each
    # with the TTL. No per-batch retry: a failed leg is simply re-quoted next cycle.
    bets = [
        {"contractId": mid, "outcome": outcome, "amount": size,
         "limitProb": lp, "expiresMillisAfter": ttl_ms}
        for mid, outcome, lp, size in legs
        if mid not in cancel_failed
    ]
    for i in range(0, len(bets), BATCH):
        chunk = bets[i : i + BATCH]
        try:
            resp = client.batch_bet(chunk)
            items = resp if isinstance(resp, list) else []
            for r in items:
                if isinstance(r, dict) and r.get("betId"):
                    result.legs_posted += 1
                else:
                    result.errors.append(f"leg rejected: {r}")
            if not items:
                result.errors.append(f"batch_bet returned no legs: {resp}")
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"batch_bet: {e}")

    # Reconcile again — a leg whose limit already crossed the price fills
    # immediately; capture it now rather than waiting a full cycle.
    result.fills_recorded += reconcile_fills(
        client, store, bot_id, uid, sorted(manage_mids), rec_sh, rec_amt
    )
    return result
