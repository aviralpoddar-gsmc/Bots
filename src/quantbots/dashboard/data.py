"""Read-only data layer for the dashboard.

Pulls everything from the SQLite store and shapes it for the templates. Pure
read; no mutations. Account balance + API latency are probed once per page load
by the server module so this layer stays sync and unit-testable.

Refund-aware: CANCEL resolutions contribute realized PnL = 0, so they're tallied
as "refunds" and excluded from win-rate denominators. With ~93% of clone
resolutions hitting CANCEL, lumping them in made every bot look 0%.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config import BotConfig, load_bots
from ..resolvability import resolvability_score
from ..store.db import Store
from ..store.pnl import bot_pnl, position_pnl
from ..store.trades import group_positions, summarize_position, trades_for_bot
from ..strategies import _REGISTRY, get_strategy

PNL_EPS = 1e-6  # tolerance for win/loss/refund classification

# Strategies are grouped by a coarse class for the distribution chart.
STRATEGY_CLASS: dict[str, str] = {
    "commodity_spot":   "Spot pricing",
    "commodity_futures": "Spot pricing",
    "ladder_arb":       "Structural arbitrage",
    "surface_arb":      "Structural arbitrage",
    "term_structure":   "Structural arbitrage",
    "ensemble":         "Data fusion",
    "enso":             "Macro / climate",
    "mean_reversion":   "Statistical baseline",
    "llm":              "Discretionary (LLM)",
}


# -----------------------------------------------------------------------------
# Per-bot helpers
# -----------------------------------------------------------------------------

def _wins_losses_refunds(trades: list[dict]) -> dict[str, Any]:
    """Closed positions split into W / L / R with a tolerance band.

    Refunds (CANCEL resolutions) compute to PnL = 0 by construction; with float
    noise they live in [-EPS, EPS] and would otherwise be mis-counted. Win rate
    = W / (W + L) so refunds don't drag a refund-heavy bot's rate to 0%.
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
    entries = [t for t in trades if t["trade_type"] == "ENTRY"]
    n = len(entries)
    total = sum(t["amount"] for t in entries)
    avg = (total / n) if n else 0.0
    edges = [
        abs(t["llm_estimate"] - t["price_before"])
        for t in entries
        if t["llm_estimate"] is not None and t["price_before"] is not None
    ]
    avg_edge = (sum(edges) / len(edges)) if edges else None
    yes_n = sum(1 for t in entries if t["direction"] == "YES")
    no_n = n - yes_n
    return {
        "n_entries": n,
        "total_mana_traded": total,
        "avg_size": avg,
        "avg_edge": avg_edge,
        "yes_entries": yes_n,
        "no_entries": no_n,
    }


def _max_drawdown(snaps: list[dict]) -> float | None:
    """Max peak-to-trough drawdown across a bot's PnL snapshots (in mana)."""
    if not snaps:
        return None
    peak = float("-inf")
    dd = 0.0
    for s in snaps:
        peak = max(peak, s["pnl"])
        dd = min(dd, s["pnl"] - peak)
    return dd if dd < 0 else 0.0


def _risk_block(row: dict[str, Any], cfg: BotConfig) -> dict[str, Any]:
    """Operator-attention diagnostics: exposure tier, concentration, inventory."""
    invested = row.get("invested") or 0.0
    cap_ceiling = float(cfg.limits.get("max_total_exposure") or 0.0)
    # Exposure tier — based on current open exposure vs the bot's own ceiling.
    expo_pct = (invested / cap_ceiling) if cap_ceiling else 0.0
    if cap_ceiling == 0:
        expo_tier = "Unbounded"
    elif expo_pct < 0.30:
        expo_tier = "Low"
    elif expo_pct < 0.70:
        expo_tier = "Medium"
    else:
        expo_tier = "High"
    # Concentration is filled by bot_detail (needs exposure breakdown); leave a placeholder.
    return {
        "exposure_tier": expo_tier,
        "exposure_pct_of_cap": expo_pct,
        "exposure_cap": cap_ceiling,
    }


def _bot_status(cfg: BotConfig, row: dict[str, Any]) -> str:
    """LIVE / PAUSED / DISABLED. Heuristic — refined as we add health signals."""
    if not cfg.enabled:
        return "DISABLED"
    last = row.get("last_trade_at")
    if not last:
        return "PAUSED"
    try:
        t = datetime.fromisoformat(last.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        if datetime.now(UTC) - t > timedelta(hours=24):
            return "PAUSED"
    except ValueError:
        pass
    return "LIVE"


def _bot_status_row(store: Store, cfg: BotConfig) -> dict[str, Any]:
    b = store.get_bot(cfg.name)
    base: dict[str, Any] = {
        "name": cfg.name, "strategy": cfg.strategy,
        "strategy_class": STRATEGY_CLASS.get(cfg.strategy, "Other"),
        "enabled": cfg.enabled,
        "pnl": 0.0, "realized": 0.0, "unrealized": 0.0,
        "invested": 0.0, "open": 0, "closed": 0,
        "wins": 0, "losses": 0, "refunds": 0, "win_rate": None,
        "n_entries": 0, "total_mana_traded": 0.0, "avg_size": 0.0, "avg_edge": None,
        "yes_entries": 0, "no_entries": 0,
        "last_trade_at": None, "bot_id": None,
        "exists": False, "max_drawdown": 0.0,
    }
    try:
        base["description"] = get_strategy(cfg.strategy, **cfg.params).description
    except Exception:  # noqa: BLE001 - strategies with missing extras shouldn't break the dash
        base["description"] = ""
    if not b:
        base["status"] = _bot_status(cfg, base)
        base["invested_mark"] = 0.0
        base["n_trades_all"] = 0
        base.update(_risk_block(base, cfg))
        return base
    bot_id = b["bot_id"]
    trades = trades_for_bot(store.conn, bot_id)
    pnl = bot_pnl(trades, current_prob=store.current_prob)
    wlr = _wins_losses_refunds(trades)
    agg = _trade_aggregates(trades)
    # Mark value of currently-open positions — this is what Manifold calls
    # "Invested": the mana you'd get back if you sold every open position at the
    # current mid-price. Differs from historical entry total (cost basis).
    invested_mark = 0.0
    for mid, pos in store.open_positions(bot_id).items():
        cur = store.current_prob(mid)
        if cur is None:
            continue
        invested_mark += pos["net_shares"] * (cur if pos["direction"] == "YES" else 1.0 - cur)
    snaps = store.conn.execute(
        "SELECT snapshot_date, pnl, realized_pnl, unrealized_pnl FROM pnl_snapshot "
        "WHERE bot_id=? ORDER BY snapshot_date", (bot_id,),
    ).fetchall()
    snap_dicts = [dict(s) for s in snaps]
    base.update({
        "bot_id": bot_id, "exists": True,
        "pnl": pnl["pnl"], "realized": pnl["realized_pnl"], "unrealized": pnl["unrealized_pnl"],
        "invested": pnl["total_invested"],     # historical cost basis (sum of entries)
        "invested_mark": invested_mark,         # current mark value of open positions
        "open": pnl["open_positions"], "closed": pnl["closed_positions"],
        "wins": wlr["wins"], "losses": wlr["losses"], "refunds": wlr["refunds"],
        "win_rate": wlr["rate"],
        "last_trade_at": max((t["date_executed"] for t in trades), default=None),
        "max_drawdown": _max_drawdown(snap_dicts) or 0.0,
        # All bet-style rows (entries + exits + partial exits). Matches what
        # Manifold's "Trades" column reports.
        "n_trades_all": sum(1 for t in trades if t["trade_type"] in ("ENTRY", "EXIT", "PARTIAL_EXIT")),
        **agg,
    })
    base["status"] = _bot_status(cfg, base)
    base.update(_risk_block(base, cfg))
    return base


# -----------------------------------------------------------------------------
# Aggregations exposed to the templates
# -----------------------------------------------------------------------------

def overview(store: Store, portfolio: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate totals across the whole fleet.

    The fleet trades on a SINGLE Manifold account, so the headline economics —
    profit, invested (mark-to-market), balance — are pulled DIRECTLY from the
    clone's `get-user-portfolio` endpoint when `portfolio` is supplied, rather
    than recomputed from the local ledger + (stale) price cache. Manifold's
    numbers are authoritative:

        active_capital = investmentValue        (server-side mark-to-market)
        total_pnl      = balance + investmentValue − totalDeposits   (= profit)
        roi            = total_pnl / totalDeposits

    The per-bot breakdowns below stay ledger-derived (Manifold can't attribute a
    shared account by bot). `ledger_*` fields expose what the ledger thinks so
    operators can spot drift, but the displayed headline trusts Manifold.
    """
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    invested_cost = sum(r["invested"] for r in rows)
    invested_mark = sum(r["invested_mark"] for r in rows)
    pnl_sum = sum(r["pnl"] for r in rows)
    realized_sum = sum(r["realized"] for r in rows)
    n_live = sum(1 for r in rows if r["status"] == "LIVE")
    n_trades_all = sum(r["n_trades_all"] for r in rows)

    out = {
        "n_bots": sum(1 for r in rows if r["exists"]),
        "n_enabled": sum(1 for r in rows if r["enabled"]),
        "n_live": n_live,
        "total_pnl": pnl_sum,
        "total_realized": realized_sum,
        "total_unrealized": sum(r["unrealized"] for r in rows),
        "total_invested": invested_cost,        # historical cost basis (informational)
        "active_capital": invested_mark,        # mark value of open positions
        "open_positions": sum(r["open"] for r in rows),
        "closed_positions": sum(r["closed"] for r in rows),
        "total_wins": sum(r["wins"] for r in rows),
        "total_losses": sum(r["losses"] for r in rows),
        "total_refunds": sum(r["refunds"] for r in rows),
        "total_mana_traded": sum(r["total_mana_traded"] for r in rows),
        "n_trades": n_trades_all,                # entries + exits = matches Manifold
        "n_entries": sum(r["n_entries"] for r in rows),
        "roi": (pnl_sum / invested_cost) if invested_cost else None,
        "avg_pnl_per_trade": (pnl_sum / n_trades_all) if n_trades_all else 0.0,
        "source": "ledger",                      # which numbers the headline trusts
    }
    if portfolio:
        balance = portfolio.get("balance") or 0.0
        deposits = portfolio.get("totalDeposits") or 0.0
        invest_value = portfolio.get("investmentValue") or 0.0
        account_profit = balance + invest_value - deposits
        # Headline numbers now come straight from Manifold.
        out["source"] = "manifold"
        out["account_balance"] = balance
        out["account_deposits"] = deposits
        out["account_profit"] = account_profit
        out["daily_profit"] = portfolio.get("dailyProfit")
        out["active_capital"] = invest_value      # authoritative mark-to-market
        out["total_pnl"] = account_profit          # authoritative profit
        # Keep the realized/unrealized split consistent with the authoritative
        # total: realized = settled cash (ledger), unrealized = the rest.
        out["total_realized"] = realized_sum
        out["total_unrealized"] = account_profit - realized_sum
        out["roi"] = (account_profit / deposits) if deposits else None
        # Preserve the ledger view for drift diagnostics (cache-staleness signal).
        out["ledger_pnl"] = pnl_sum
        out["ledger_active_capital"] = invested_mark
        out["pnl_drift"] = pnl_sum - account_profit
    return out


def portfolio_equity(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Equity curve straight from Manifold's portfolio history.

    Each snapshot already carries a server-computed `profit`
    (= balance + investmentValue − totalDeposits) — we just map it to the
    {ts, pnl} shape the chart expects. No ledger replay, no stale cache.
    """
    if not history:
        return []
    series = []
    for h in history:
        ts = h.get("timestamp")
        if ts is None:
            continue
        try:
            iso = datetime.fromtimestamp(int(ts) / 1000, UTC).isoformat()
        except (ValueError, OSError, OverflowError):
            continue
        series.append({"ts": iso, "pnl": h.get("profit") or 0.0})
    series.sort(key=lambda p: p["ts"])
    return series


def cache_age_seconds(store: Store) -> int | None:
    """How long ago the market cache was last touched, in seconds. None if empty."""
    row = store.conn.execute(
        "SELECT MAX(updated_at) AS u FROM market_cache"
    ).fetchone()
    if not row or not row["u"]:
        return None
    try:
        t = datetime.fromisoformat(row["u"].replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
    except ValueError:
        return None
    return int((datetime.now(UTC) - t).total_seconds())


def leaderboard(store: Store) -> list[dict[str, Any]]:
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


def bot_detail(store: Store, name: str) -> dict[str, Any] | None:
    cfgs = [c for c in load_bots() if c.name == name]
    if not cfgs:
        return None
    cfg = cfgs[0]
    row = _bot_status_row(store, cfg)
    row["limits"] = cfg.limits
    row["params"] = cfg.params
    if not row["exists"]:
        row.update(recent_trades=[], exposures=[], pnl_series=[], concentration_pct=0.0,
                   total_open_exposure=0.0, inventory_bias=None)
        return row
    bot_id = row["bot_id"]
    trades = trades_for_bot(store.conn, bot_id)

    # Recent ENTRYs (newest first)
    entries = [t for t in trades if t["trade_type"] == "ENTRY"]
    recent = entries[-12:][::-1]
    row["recent_trades"] = [{
        "ts": t["date_executed"],
        "market_id": t["market_id"],
        "question": ((store.get_cached_market(t["market_id"]) or {}).get("question") or "")[:140],
        "direction": t["direction"],
        "amount": t["amount"],
        "price_before": t["price_before"],
        "price_after": t["price_after"],
        "estimate": t["llm_estimate"],
    } for t in recent]

    # Exposures by bucketed question prefix (cheap proxy for the strategy's
    # correlation_key, without re-instantiating per market).
    positions = store.open_positions(bot_id)
    exposure: dict[str, float] = defaultdict(float)
    yes_amt = no_amt = 0.0
    for mid, pos in positions.items():
        q = ((store.get_cached_market(mid) or {}).get("question") or mid)
        bucket = " ".join(q.split()[:7])
        amt = pos.get("net_amount") or 0.0
        exposure[bucket] += amt
        if pos["direction"] == "YES":
            yes_amt += amt
        else:
            no_amt += amt
    top = sorted(exposure.items(), key=lambda kv: -kv[1])[:6]
    max_amt = max((amt for _, amt in top), default=1.0) or 1.0
    row["exposures"] = [
        {"key": k, "amount": v, "pct_of_max": v / max_amt}
        for k, v in top
    ]
    row["total_open_exposure"] = sum(exposure.values())
    row["concentration_pct"] = (top[0][1] / row["total_open_exposure"]) if (top and row["total_open_exposure"]) else 0.0
    total_dir = yes_amt + no_amt
    if total_dir > 0:
        bias_pct = (yes_amt - no_amt) / total_dir  # -1 (all NO) .. +1 (all YES)
        if abs(bias_pct) < 0.15:
            row["inventory_bias"] = ("Neutral", bias_pct)
        elif bias_pct > 0:
            row["inventory_bias"] = (f"YES-heavy ({yes_amt / total_dir:.0%})", bias_pct)
        else:
            row["inventory_bias"] = (f"NO-heavy ({no_amt / total_dir:.0%})", bias_pct)
    else:
        row["inventory_bias"] = None

    # PnL snapshot history -> chart points
    snaps = store.conn.execute(
        "SELECT snapshot_date, pnl, realized_pnl, unrealized_pnl FROM pnl_snapshot "
        "WHERE bot_id=? ORDER BY snapshot_date", (bot_id,),
    ).fetchall()
    row["pnl_series"] = [{"date": s["snapshot_date"], "pnl": s["pnl"]} for s in snaps]
    return row


def equity_curve(store: Store, days: int | None = None) -> list[dict[str, Any]]:
    """Per-trade cumulative REALIZED PnL across all bots, in trade-time order.

    Equity from snapshots is daily-resolution; this gives a per-trade curve so
    the chart has shape even on the first day. Unrealized swings don't show up
    here (they only land in snapshots) — the snapshot series fills that in for
    multi-day history.
    """
    rows = store.conn.execute(
        "SELECT trade_id, date_executed, trade_type, direction, amount, shares, "
        "price_before, price_after FROM trade ORDER BY trade_id"
    ).fetchall()
    cum = 0.0
    series = []
    for r in rows:
        # Realized contribution per trade row: EXITs and RESOLUTION_CLOSEs only.
        if r["trade_type"] in ("EXIT", "PARTIAL_EXIT", "RESOLUTION_CLOSE"):
            # We don't have per-trade cost basis here; approximate realized as
            # amount - (shares * cost) but cost isn't on the row. So use a
            # cheaper proxy: amount received - shares*price_before. For CANCEL
            # closes amount = shares*cps and price_after = cps, contributing 0.
            # For YES/NO closes amount = shares*settle, contributing positive
            # or negative vs entry price. Net effect: directionally correct.
            cum += (r["amount"] or 0) - (r["shares"] or 0) * (r["price_before"] or 0)
        series.append({"ts": r["date_executed"], "pnl": cum})
    return series


def strategy_distribution(store: Store) -> list[dict[str, Any]]:
    """Capital deployed per strategy class, for the allocation chart."""
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    by_class: dict[str, float] = defaultdict(float)
    for r in rows:
        by_class[r["strategy_class"]] += r["invested"]
    total = sum(by_class.values()) or 1.0
    out = [
        {"label": k, "amount": v, "pct": v / total}
        for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])
        if v > 0
    ]
    return out


def event_feed(store: Store, limit: int = 50) -> list[dict[str, Any]]:
    rows = store.conn.execute(
        """
        SELECT t.date_executed, t.market_id, t.trade_type, t.direction,
               t.amount, t.shares, t.price_before, t.price_after, t.llm_estimate,
               t.reasoning, b.name AS bot_name
        FROM trade t JOIN bot b USING(bot_id)
        ORDER BY t.trade_id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        cached = store.get_cached_market(r["market_id"]) or {}
        out.append({
            "ts": r["date_executed"], "bot": r["bot_name"], "type": r["trade_type"],
            "direction": r["direction"], "amount": r["amount"],
            "market_id": r["market_id"],
            "question": (cached.get("question") or "")[:100],
            "price_before": r["price_before"], "price_after": r["price_after"],
            "estimate": r["llm_estimate"], "reasoning": r["reasoning"],
        })
    return out


# -----------------------------------------------------------------------------
# Formatting helpers used in templates
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Endpoints added for the React dashboard
# -----------------------------------------------------------------------------

def strategy_index(store: Store) -> list[dict[str, Any]]:
    """One row per *strategy* (not per bot). Aggregates bots that use it."""
    rows = [_bot_status_row(store, cfg) for cfg in load_bots()]
    by_strategy: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = r["strategy"]
        agg = by_strategy.setdefault(s, {
            "name": s,
            "class": STRATEGY_CLASS.get(s, "Other"),
            "description": "",
            "bots": [],
            "total_pnl": 0.0,
            "total_trades": 0,
            "live_count": 0,
        })
        agg["bots"].append(r["name"])
        agg["total_pnl"] += r["pnl"]
        agg["total_trades"] += r["n_trades_all"]
        if r["status"] == "LIVE":
            agg["live_count"] += 1
        if not agg["description"]:
            agg["description"] = r.get("description") or ""
    # Surface registered strategies even if no bot uses them yet — operators
    # often want to see what's available before wiring it up.
    for name in _REGISTRY:
        if name in by_strategy:
            continue
        try:
            desc = get_strategy(name).description
        except Exception:  # noqa: BLE001
            desc = ""
        by_strategy[name] = {
            "name": name,
            "class": STRATEGY_CLASS.get(name, "Other"),
            "description": desc,
            "bots": [],
            "total_pnl": 0.0,
            "total_trades": 0,
            "live_count": 0,
        }
    out = list(by_strategy.values())
    out.sort(key=lambda x: (-x["total_pnl"], x["name"]))
    return out


def strategy_detail(store: Store, name: str) -> dict[str, Any] | None:
    for row in strategy_index(store):
        if row["name"] == name:
            return row
    return None


def markets_index(
    store: Store,
    *,
    page: int = 1,
    size: int = 50,
    q: str | None = None,
    min_resolvability: float | None = None,
) -> dict[str, Any]:
    """Paginated cached-market browser.

    Returns rows: question, type, current_prob, close_time, total_liquidity,
    resolvability score, and which bots currently hold a position. `q` filters
    by case-insensitive substring against the question + market_id.
    """
    page = max(1, int(page or 1))
    size = max(1, min(int(size or 50), 200))

    # Pull (id, question, prob, liquidity, close, raw) — we don't filter on
    # is_resolved so operators can audit settled rows too.
    where: list[str] = []
    args: list[Any] = []
    if q:
        where.append("(LOWER(question) LIKE ? OR LOWER(market_id) LIKE ?)")
        like = f"%{q.lower()}%"
        args.extend([like, like])
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Count first (small SQL hit).
    total_row = store.conn.execute(
        f"SELECT COUNT(*) AS c FROM market_cache {where_sql}", args
    ).fetchone()
    total = int(total_row["c"]) if total_row else 0

    rows = store.conn.execute(
        f"SELECT market_id, question, probability, total_liquidity, close_time, "
        f"raw_json FROM market_cache {where_sql} "
        f"ORDER BY total_liquidity DESC NULLS LAST, market_id "
        f"LIMIT ? OFFSET ?",
        [*args, size, (page - 1) * size],
    ).fetchall()

    # traded_by: bots that currently have an open position on each market.
    # One query per page is cheap enough (size ≤ 200).
    market_ids = [r["market_id"] for r in rows]
    by_market: dict[str, list[str]] = {mid: [] for mid in market_ids}
    if market_ids:
        ph = ",".join("?" for _ in market_ids)
        for tr in store.conn.execute(
            f"SELECT DISTINCT b.name, t.market_id FROM trade t "
            f"JOIN bot b USING(bot_id) WHERE t.market_id IN ({ph})",
            market_ids,
        ).fetchall():
            by_market[tr["market_id"]].append(tr["name"])

    out_rows = []
    import json as _json
    for r in rows:
        raw = r["raw_json"]
        kind = ""
        if raw:
            try:
                obj = _json.loads(raw) if isinstance(raw, str) else raw
                kind = (obj or {}).get("outcomeType") or (obj or {}).get("mechanism") or ""
            except Exception:  # noqa: BLE001
                kind = ""
        score = resolvability_score(r["question"] or "")
        if min_resolvability is not None and score < float(min_resolvability):
            continue
        close_iso = None
        if r["close_time"]:
            try:
                # close_time is unix ms in the Manifold API.
                close_iso = datetime.fromtimestamp(int(r["close_time"]) / 1000, UTC).isoformat()
            except Exception:  # noqa: BLE001
                close_iso = None
        out_rows.append({
            "id": r["market_id"],
            "question": r["question"] or "",
            "market_type": kind or "—",
            "current_prob": r["probability"],
            "close_time": close_iso,
            "total_liquidity": r["total_liquidity"],
            "resolvability": score,
            "traded_by": by_market.get(r["market_id"], []),
        })

    return {"rows": out_rows, "total": total, "page": page, "size": size}


def humanize_age(iso_ts: str | None) -> str:
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
    if s < 60: return f"{s}s ago"
    if s < 3600: return f"{s // 60}m ago"
    if s < 86400: return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def short_clock(iso_ts: str | None) -> str:
    """HH:MM:SS for the event feed."""
    if not iso_ts:
        return ""
    try:
        t = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return t.astimezone().strftime("%H:%M:%S")
    except ValueError:
        return iso_ts[:8]
