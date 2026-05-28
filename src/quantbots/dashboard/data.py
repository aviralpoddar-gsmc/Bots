"""Read-only data layer for the dashboard.

Pulls everything from the SQLite store and shapes it for the templates. No
network calls — the account balance is fetched by the server module so this
layer stays pure and unit-testable.

Every win-rate / loss-rate metric here is **refund-aware**: CANCEL resolutions
contribute realized PnL = 0 (the stake is refunded), so they are tallied as
"refunds" and excluded from the win-rate denominator. With ~93% of clone
resolutions hitting CANCEL, lumping refunds into the denominator made every
bot look like it had a 0% win rate even when it had never actually lost.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from ..config import BotConfig, load_bots
from ..store.db import Store
from ..store.pnl import bot_pnl, position_pnl
from ..store.trades import group_positions, summarize_position, trades_for_bot

# A position's realized PnL must clear this absolute threshold (mana) to count
# as a non-zero outcome. CANCEL refunds compute to 0 by construction but pick up
# float noise of ~1e-15; anything inside [-EPS, EPS] is treated as a refund/push.
PNL_EPS = 1e-6

# Short strategy taglines shown on each bot's card. Single sentence, edge-focused.
STRATEGY_BLURBS: dict[str, str] = {
    "commodity_spot":   "Lognormal-CDF pricing on metals/energy threshold ladders, anchored to live spot feeds.",
    "ladder_arb":       "Model-free isotonic monotonicity arbitrage across threshold ladders.",
    "term_structure":   "Kernel-smoothed term-structure coherence — fills stale dates from traded neighbours.",
    "commodity_futures": "Lognormal pricing on soft-commodity futures (ICE/CBOT) threshold markets.",
    "enso":             "Gaussian persistence model for NOAA ENSO/ONI climate-index markets.",
    "ensemble":         "Fuses ingested observations from multiple feeds into one fair value.",
    "mean_reversion":   "EMA mean-reversion on whatever the runner sees — model-free baseline.",
    "surface_arb":      "Normal-CDF parametric fit across a strike ladder.",
    "llm":              "Local-LLM forecaster (Ollama/llama.cpp) for the long tail of unlinked markets.",
}


def _wins_losses_refunds(trades: list[dict]) -> dict[str, int | None]:
    """Tally closed positions into (W / L / R) with a tolerance for float noise.

    Refunds (CANCEL resolutions) are NOT counted as losses — they're a separate
    category. Win rate = W / (W + L), so a bot with only refunds shows "—"
    instead of a misleading 0%.
    """
    w = l = r = 0
    for _key, pos_trades in group_positions(trades).items():
        if summarize_position(pos_trades)["status"] != "CLOSED":
            continue
        realized, _ = position_pnl(pos_trades, current_prob=None)
        if realized > PNL_EPS:
            w += 1
        elif realized < -PNL_EPS:
            l += 1
        else:
            r += 1
    rate = (w / (w + l)) if (w + l) else None
    return {"wins": w, "losses": l, "refunds": r, "rate": rate}


def _trade_aggregates(trades: list[dict]) -> dict[str, Any]:
    """Per-bot trade-shape metrics: count, total mana traded, average size, avg
    edge (|estimate − price_before|) of entries that had both numbers."""
    entries = [t for t in trades if t["trade_type"] == "ENTRY"]
    n = len(entries)
    total_mana = sum(t["amount"] for t in entries)
    avg_size = (total_mana / n) if n else 0.0
    edges = [
        abs(t["llm_estimate"] - t["price_before"])
        for t in entries
        if t["llm_estimate"] is not None and t["price_before"] is not None
    ]
    avg_edge = (sum(edges) / len(edges)) if edges else None
    return {
        "n_entries": n,
        "total_mana_traded": total_mana,
        "avg_size": avg_size,
        "avg_edge": avg_edge,
    }


def _bot_status_row(store: Store, cfg: BotConfig) -> dict[str, Any]:
    """Per-bot row for both leaderboard and bot card. Self-contained: one bot,
    one DB pass, no joins back into other bots."""
    b = store.get_bot(cfg.name)
    if not b:
        return {
            "name": cfg.name, "strategy": cfg.strategy, "enabled": cfg.enabled,
            "pnl": 0.0, "realized": 0.0, "unrealized": 0.0,
            "invested": 0.0, "open": 0, "closed": 0,
            "wins": 0, "losses": 0, "refunds": 0, "win_rate": None,
            "n_entries": 0, "total_mana_traded": 0.0, "avg_size": 0.0, "avg_edge": None,
            "last_trade_at": None, "exists": False, "bot_id": None,
        }
    bot_id = b["bot_id"]
    trades = trades_for_bot(store.conn, bot_id)
    pnl = bot_pnl(trades, current_prob=store.current_prob)
    wlr = _wins_losses_refunds(trades)
    agg = _trade_aggregates(trades)
    last_trade_at = max((t["date_executed"] for t in trades), default=None)
    return {
        "name": cfg.name,
        "strategy": cfg.strategy,
        "enabled": cfg.enabled,
        "bot_id": bot_id,
        "pnl": pnl["pnl"],
        "realized": pnl["realized_pnl"],
        "unrealized": pnl["unrealized_pnl"],
        "invested": pnl["total_invested"],
        "open": pnl["open_positions"],
        "closed": pnl["closed_positions"],
        "wins": wlr["wins"], "losses": wlr["losses"], "refunds": wlr["refunds"],
        "win_rate": wlr["rate"],
        "n_entries": agg["n_entries"],
        "total_mana_traded": agg["total_mana_traded"],
        "avg_size": agg["avg_size"],
        "avg_edge": agg["avg_edge"],
        "last_trade_at": last_trade_at,
        "exists": True,
    }


def overview(store: Store) -> dict[str, Any]:
    """Top-of-page totals across all bots."""
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    invested = sum(r["invested"] for r in rows)
    pnl_sum = sum(r["pnl"] for r in rows)
    return {
        "n_bots": sum(1 for r in rows if r["exists"]),
        "n_enabled": sum(1 for r in rows if r["enabled"]),
        "total_pnl": pnl_sum,
        "total_realized": sum(r["realized"] for r in rows),
        "total_unrealized": sum(r["unrealized"] for r in rows),
        "total_invested": invested,
        "total_open": sum(r["open"] for r in rows),
        "total_closed": sum(r["closed"] for r in rows),
        "total_wins": sum(r["wins"] for r in rows),
        "total_losses": sum(r["losses"] for r in rows),
        "total_refunds": sum(r["refunds"] for r in rows),
        "total_mana_traded": sum(r["total_mana_traded"] for r in rows),
        "n_trades": sum(r["n_entries"] for r in rows),
        "roi": (pnl_sum / invested) if invested else None,
    }


def leaderboard(store: Store) -> list[dict[str, Any]]:
    """All bots ranked by total PnL desc."""
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


def bot_detail(store: Store, name: str) -> dict[str, Any] | None:
    """Full per-bot view: status + recent trades + top exposures + PnL series."""
    cfgs = [c for c in load_bots() if c.name == name]
    if not cfgs:
        return None
    cfg = cfgs[0]
    row = _bot_status_row(store, cfg)
    row["blurb"] = STRATEGY_BLURBS.get(cfg.strategy, cfg.strategy)
    row["limits"] = cfg.limits
    row["params"] = cfg.params

    if not row["exists"]:
        row.update(recent_trades=[], exposures=[], pnl_series=[])
        return row
    bot_id = row["bot_id"]

    # Most recent ENTRYs (newest first), enriched with the cached question text.
    trades = trades_for_bot(store.conn, bot_id)
    entries = [t for t in trades if t["trade_type"] == "ENTRY"]
    recent = entries[-12:][::-1]
    row["recent_trades"] = [
        {
            "ts": t["date_executed"],
            "market_id": t["market_id"],
            "question": ((store.get_cached_market(t["market_id"]) or {}).get("question") or "")[:120],
            "direction": t["direction"],
            "amount": t["amount"],
            "price_before": t["price_before"],
            "price_after": t["price_after"],
            "estimate": t["llm_estimate"],
        }
        for t in recent
    ]

    # Top correlation-group exposures, by net_amount on open positions. We use
    # the first 6 words of the question as a cheap bucket key — sufficient for
    # surface readability without re-instantiating the strategy here.
    positions = store.open_positions(bot_id)
    exposure: dict[str, float] = defaultdict(float)
    for mid, pos in positions.items():
        q = ((store.get_cached_market(mid) or {}).get("question") or mid)
        bucket = " ".join(q.split()[:6])
        exposure[bucket] += pos.get("net_amount") or 0.0
    top = sorted(exposure.items(), key=lambda kv: -kv[1])[:6]
    max_amt = max((amt for _, amt in top), default=1.0) or 1.0
    row["exposures"] = [
        {"key": k, "amount": v, "pct_of_max": v / max_amt}
        for k, v in top
    ]
    row["total_open_exposure"] = sum(exposure.values())

    # Snapshot history -> chart points (chronological).
    snaps = store.conn.execute(
        "SELECT snapshot_date, pnl, realized_pnl, unrealized_pnl FROM pnl_snapshot "
        "WHERE bot_id=? ORDER BY snapshot_date", (bot_id,),
    ).fetchall()
    row["pnl_series"] = [
        {"date": s["snapshot_date"], "pnl": s["pnl"],
         "realized": s["realized_pnl"], "unrealized": s["unrealized_pnl"]}
        for s in snaps
    ]
    return row


def cumulative_pnl_series(store: Store) -> list[dict[str, Any]]:
    """Total PnL across all bots over time, from pnl_snapshot. Used for the
    hero chart. If no snapshots yet, returns []."""
    rows = store.conn.execute(
        "SELECT snapshot_date, SUM(pnl) AS pnl, SUM(realized_pnl) AS realized, "
        "SUM(unrealized_pnl) AS unrealized "
        "FROM pnl_snapshot GROUP BY snapshot_date ORDER BY snapshot_date"
    ).fetchall()
    return [
        {"date": r["snapshot_date"], "pnl": r["pnl"],
         "realized": r["realized"], "unrealized": r["unrealized"]}
        for r in rows
    ]


def activity_feed(store: Store, limit: int = 25) -> list[dict[str, Any]]:
    """Most recent trades across ALL bots — for the activity panel."""
    rows = store.conn.execute(
        """
        SELECT t.date_executed, t.market_id, t.trade_type, t.direction,
               t.amount, t.shares, t.price_before, t.price_after, t.llm_estimate,
               b.name AS bot_name
        FROM trade t JOIN bot b USING(bot_id)
        ORDER BY t.trade_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        cached = store.get_cached_market(r["market_id"]) or {}
        out.append({
            "ts": r["date_executed"],
            "bot": r["bot_name"],
            "type": r["trade_type"],
            "market_id": r["market_id"],
            "question": (cached.get("question") or "")[:100],
            "direction": r["direction"],
            "amount": r["amount"],
            "price_before": r["price_before"],
            "price_after": r["price_after"],
            "estimate": r["llm_estimate"],
        })
    return out


def humanize_age(iso_ts: str | None) -> str:
    """\"3m ago\" / \"2h ago\" / etc."""
    if not iso_ts:
        return "—"
    try:
        t = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
    except ValueError:
        return iso_ts
    delta = datetime.now(UTC) - t
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"
