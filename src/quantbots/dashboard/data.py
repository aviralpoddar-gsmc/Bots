"""Read-only data layer for the dashboard.

Pulls everything from the SQLite store and shapes it for the templates. No
network calls (the account balance is reachable via the live API but we don't
require it — pass a balance in from the caller if you want to show it). Every
function here is pure and trivial to unit-test against a fixture DB.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config import BotConfig, load_bots
from ..store.db import Store
from ..store.pnl import bot_pnl
from ..store.trades import group_positions, summarize_position, trades_for_bot

# Strategy docs that the dashboard surfaces verbatim. Keep these short — a single
# sentence each, focused on the EDGE the strategy claims. The Strategy class
# docstrings are too long for a card; these are the elevator pitch.
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


def _bot_status_row(store: Store, cfg: BotConfig) -> dict[str, Any]:
    """Per-bot row used by both the leaderboard and the bot card."""
    b = store.get_bot(cfg.name)
    if not b:
        return {
            "name": cfg.name, "strategy": cfg.strategy, "enabled": cfg.enabled,
            "pnl": 0.0, "realized": 0.0, "unrealized": 0.0,
            "invested": 0.0, "open": 0, "closed": 0, "win_rate": None,
            "last_trade_at": None, "exists": False,
        }
    bot_id = b["bot_id"]
    trades = trades_for_bot(store.conn, bot_id)
    pnl = bot_pnl(trades, current_prob=store.current_prob)
    last_trade_at = max((t["date_executed"] for t in trades), default=None)
    # Win rate over closed positions: % of resolved positions with positive realized PnL.
    closed_wins = closed_total = 0
    for (_mid, _dir), pos_trades in group_positions(trades).items():
        summary = summarize_position(pos_trades)
        if summary["status"] != "CLOSED":
            continue
        closed_total += 1
        # Realized for one position = exit proceeds - cost basis of exited shares.
        from ..store.pnl import position_pnl
        r, _ = position_pnl(pos_trades, current_prob=None)
        if r > 0:
            closed_wins += 1
    win_rate = (closed_wins / closed_total) if closed_total else None
    return {
        "name": cfg.name,
        "strategy": cfg.strategy,
        "enabled": cfg.enabled,
        "pnl": pnl["pnl"],
        "realized": pnl["realized_pnl"],
        "unrealized": pnl["unrealized_pnl"],
        "invested": pnl["total_invested"],
        "open": pnl["open_positions"],
        "closed": pnl["closed_positions"],
        "win_rate": win_rate,
        "last_trade_at": last_trade_at,
        "exists": True,
    }


def overview(store: Store) -> dict[str, Any]:
    """Top-of-page totals across all bots."""
    cfgs = load_bots()
    rows = [_bot_status_row(store, cfg) for cfg in cfgs]
    return {
        "n_bots": sum(1 for r in rows if r["exists"]),
        "n_enabled": sum(1 for r in rows if r["enabled"]),
        "total_pnl": sum(r["pnl"] for r in rows),
        "total_realized": sum(r["realized"] for r in rows),
        "total_unrealized": sum(r["unrealized"] for r in rows),
        "total_invested": sum(r["invested"] for r in rows),
        "total_open": sum(r["open"] for r in rows),
        "total_closed": sum(r["closed"] for r in rows),
    }


def leaderboard(store: Store) -> list[dict[str, Any]]:
    """All bots sorted by total PnL desc."""
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


def bot_detail(store: Store, name: str) -> dict[str, Any] | None:
    """Full per-bot view: status + recent trades + top correlation exposures."""
    cfgs = [c for c in load_bots() if c.name == name]
    if not cfgs:
        return None
    cfg = cfgs[0]
    row = _bot_status_row(store, cfg)
    row["blurb"] = STRATEGY_BLURBS.get(cfg.strategy, cfg.strategy)
    row["limits"] = cfg.limits
    row["params"] = cfg.params

    b = store.get_bot(cfg.name)
    if not b:
        row["recent_trades"] = []
        row["exposures"] = []
        row["pnl_series"] = []
        return row
    bot_id = b["bot_id"]

    # Recent trades — last 10 ENTRYs with cached question for readability.
    trades = trades_for_bot(store.conn, bot_id)
    entries = [t for t in trades if t["trade_type"] == "ENTRY"]
    recent = entries[-10:][::-1]  # newest first
    out_trades = []
    for t in recent:
        cached = store.get_cached_market(t["market_id"]) or {}
        out_trades.append({
            "ts": t["date_executed"],
            "market_id": t["market_id"],
            "question": (cached.get("question") or "")[:90],
            "direction": t["direction"],
            "amount": t["amount"],
            "price_before": t["price_before"],
            "price_after": t["price_after"],
            "estimate": t["llm_estimate"],
        })
    row["recent_trades"] = out_trades

    # Top correlation exposures — by raw correlation key (we don't reload the strategy here).
    # Sum open net_amount per market_id (positions are already in mana).
    positions = store.open_positions(bot_id)
    # Bucket by the cached question's prefix as a cheap proxy for the underlying
    # when we don't want to instantiate the strategy here. For commodity_spot and
    # the like, the strategy.correlation_key gives a tighter grouping — surfaced
    # in the bot's own page in a later iteration.
    exposure: dict[str, float] = defaultdict(float)
    for mid, pos in positions.items():
        cached = store.get_cached_market(mid) or {}
        # Use first 6 words of the question as a stable-ish bucket label.
        q = (cached.get("question") or mid)
        bucket = " ".join(q.split()[:6])
        exposure[bucket] += pos.get("net_amount") or 0.0
    top_expo = sorted(exposure.items(), key=lambda kv: -kv[1])[:8]
    row["exposures"] = [{"key": k, "amount": v} for k, v in top_expo]

    # PnL time series from snapshots — chronological for charting.
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


def activity_feed(store: Store, limit: int = 30) -> list[dict[str, Any]]:
    """Most recent trades across ALL bots — for the live activity panel."""
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
            "question": (cached.get("question") or "")[:80],
            "direction": r["direction"],
            "amount": r["amount"],
            "price_before": r["price_before"],
            "price_after": r["price_after"],
            "estimate": r["llm_estimate"],
        })
    return out


def humanize_age(iso_ts: str | None) -> str:
    """\"3m ago\" / \"2h ago\" / etc. Used for last-trade timestamps."""
    if not iso_ts:
        return "—"
    try:
        # SQLite stores naive ISO strings; assume UTC (record_trade uses _now).
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
