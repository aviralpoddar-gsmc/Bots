"""`quantbots` command line. Thin wrapper over the framework.

    quantbots health                 # prove API key + Cloudflare Access work
    quantbots refresh                # pull markets into the local cache
    quantbots ingest                 # fetch external data sources into the cache
    quantbots run --bot NAME         # dry-run a bot (default); add --live to trade
    quantbots status                 # dashboard: balance, per-bot PnL, exposure
    quantbots resolve --bot NAME     # close out resolved positions
    quantbots snapshot               # roll up PnL + print leaderboard
    quantbots strategies             # list registered strategies
    quantbots sources                # list registered data sources
    quantbots llm-bench              # rank local LLMs against real ground truth
    quantbots backtest               # measure a bot's calibration + PnL on history
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .config import load_bot, load_bots
from .manifold.client import ManifoldClient
from .runner import run_bot, sync_resolutions
from .sources import available as available_sources
from .sources.ingest import ingest as run_ingest
from .store.db import Store
from .strategies import available, get_strategy

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


def _client(api_key: str | None = None) -> ManifoldClient:
    return ManifoldClient(api_key=api_key)


@app.command()
def health() -> None:
    """Confirm the API key + Cloudflare Access both work (calls /me)."""
    me = _client().get_me()
    console.print(f"[green]OK[/] — @{me.get('username')} balance Ṁ{me.get('balance')}")


@app.command()
def status(bot: str = typer.Option("", "--bot", help="Limit to one bot (default: all)")) -> None:
    """Monitoring dashboard: live balance, per-bot PnL, and exposure by underlying."""
    me = _client().get_me()
    console.print(f"Account [cyan]@{me.get('username')}[/]  balance [green]Ṁ{me.get('balance'):,.0f}[/]")

    cfgs = [load_bot(bot)] if bot else load_bots()
    with Store() as store:
        table = Table(title="Bots — PnL (unrealized uses cached prices)")
        for col in ("bot", "open", "closed", "invested", "realized", "unrealized", "total PnL"):
            table.add_column(col, justify="right" if col != "bot" else "left")
        for cfg in cfgs:
            b = store.get_bot(cfg.name)
            if not b:
                continue
            p = store.bot_pnl(b["bot_id"])
            table.add_row(
                cfg.name, str(p["open_positions"]), str(p["closed_positions"]),
                f"Ṁ{p['total_invested']:,.0f}", f"Ṁ{p['realized_pnl']:,.0f}",
                f"Ṁ{p['unrealized_pnl']:,.0f}", f"Ṁ{p['pnl']:,.0f}",
            )
        console.print(table)

        # Exposure by correlation group (per bot) — concentration at a glance.
        for cfg in cfgs:
            b = store.get_bot(cfg.name)
            if not b:
                continue
            positions = store.open_positions(b["bot_id"])
            if not positions:
                continue
            try:
                strat = get_strategy(cfg.strategy, **cfg.params)
            except Exception:  # noqa: BLE001 - LLM/quant extras may be absent
                continue
            exposure: dict[str, float] = {}
            for mid, pos in positions.items():
                m = store.get_cached_market(mid) or {"id": mid}
                g = strat.correlation_key(m)
                exposure[g] = exposure.get(g, 0.0) + (pos.get("net_amount") or 0.0)
            if exposure:
                groups = ", ".join(f"{g}: Ṁ{v:,.0f}" for g, v in sorted(exposure.items(), key=lambda x: -x[1]))
                console.print(f"  [yellow]{cfg.name}[/] exposure — {groups}")


@app.command()
def strategies() -> None:
    """List registered strategies."""
    for name in available():
        console.print(f"• {name}")


@app.command()
def sources() -> None:
    """List registered data sources."""
    for name in available_sources():
        console.print(f"• {name}")


@app.command()
def link(limit: int = typer.Option(15, help="Sample links to print")) -> None:
    """Show which cached markets the linker maps to source entities (debug)."""
    from .strategies.linker import link_markets

    with Store() as store:
        markets = store.load_open_markets()
        links = link_markets(markets)
    console.print(f"linked {len(links)} / {len(markets)} cached markets")
    table = Table(show_lines=False)
    for col in ("market", "entities", "thr", "dir"):
        table.add_column(col)
    for lk in list(links.values())[:limit]:
        table.add_row(
            lk.question[:50],
            ", ".join(lk.entities),
            "" if lk.threshold is None else f"{lk.threshold:g}",
            lk.direction,
        )
    console.print(table)


@app.command()
def ingest(
    only: str = typer.Option("", "--only", help="Ingest just this one source"),
) -> None:
    """Fetch configured external data sources into the observations cache."""
    with Store() as store:
        result = run_ingest(store, only=only or None)
        entities = len(store.known_entities())
    for name, n in result.by_source.items():
        console.print(f"[green]{name}[/]: {n} observations")
    for name, err in result.errors.items():
        console.print(f"[red]{name} failed[/]: {err}")
    console.print(f"total {result.total} observations across {entities} entities")


@app.command()
def refresh(
    limit: int = typer.Option(1000, help="Max markets to pull"),
    search: str = typer.Option("", help="Optional search term to scope the universe"),
) -> None:
    """Pull markets from the clone into the local cache for the runner to read."""
    client = _client()
    if search:
        markets = client.search_markets(search, limit=limit)
    else:
        # Paginate (the API caps a page at 1000) until we hit `limit` or run out.
        markets, before = [], None
        while len(markets) < limit:
            page = client.list_markets(limit=min(1000, limit - len(markets)), before=before)
            if not page:
                break
            markets += page
            before = page[-1]["id"]
            if len(page) < 1000:
                break
    with Store() as store:
        n = store.upsert_markets(markets)
    console.print(f"[green]Cached[/] {n} markets")


@app.command()
def run(
    bot: str = typer.Option(..., "--bot", help="Bot name from config/bots.yaml"),
    live: bool = typer.Option(False, "--live", help="Actually place bets (default: dry-run)"),
    budget: float = typer.Option(0, "--budget", help="Override max mana this run may spend"),
) -> None:
    """Run one bot. Dry-run by default — validates orders without moving mana."""
    cfg = load_bot(bot)
    if budget > 0:
        cfg.limits["max_run_budget"] = budget
    if not cfg.api_key:
        raise typer.BadParameter(f"No key in env var {cfg.account_env!r}")
    strat = get_strategy(cfg.strategy, **cfg.params)
    with Store() as store:
        result = run_bot(
            bot=cfg, client=_client(cfg.api_key), store=store, strategy=strat, dry_run=not live
        )
    mode = "[red]LIVE[/]" if live else "[yellow]dry-run[/]"
    console.print(
        f"{mode} {result.bot}: funded {len(result.signals)} of {result.candidates} "
        f"candidate orders on {result.n_markets} markets"
    )
    book = result.book or {}
    if book:
        console.print(
            f"  book: staked [cyan]Ṁ{book.get('staked', 0):,.0f}[/] "
            f"exp.profit [green]Ṁ{book.get('exp_profit', 0):,.0f}[/] "
            f"(exp.ROI {book.get('exp_roi', 0):+.0%}) "
            f"across {len(book.get('groups', {}))} correlation groups"
        )
    _print_signals(result.signals[:25])
    if not live:
        console.print(f"validated a sample, {len(result.errors)} errors")
    else:
        console.print(f"[green]placed[/] {result.orders_placed} orders, {len(result.errors)} errors")
    for e in result.errors[:10]:
        console.print(f"  [red]err[/] {e}")


@app.command(name="llm-bench")
def llm_bench(
    models: str = typer.Option(
        "qwen3:8b,gemma3:latest,gemma4:latest", "--models",
        help="Comma-separated local model names to compare",
    ),
    asof: str = typer.Option("", "--asof", help="Date context for forecasts (default: today)"),
) -> None:
    """Rank local LLMs as forecasters, scored against our real data feeds.

    Run `quantbots ingest` first so there are ground-truth values to score against.
    """
    from datetime import UTC, datetime

    from .llm.bench import benchmark

    asof = asof or datetime.now(UTC).strftime("%B %Y")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    console.print(f"Benchmarking {model_list} as of [cyan]{asof}[/]")
    with Store() as store:
        scores = benchmark(model_list, asof, store=store)

    table = Table(title="Local LLM forecasting benchmark")
    for col in ("model", "valid", "coverage", "p50 err", "latency"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for s in scores:
        table.add_row(
            s.model,
            f"{s.valid}/{s.n}",
            f"{s.coverage:.0%}",
            f"{s.median_error:.3f}",
            f"{s.avg_latency:.1f}s",
        )
    console.print(table)
    if scores and scores[0].valid:
        console.print(f"[green]Best:[/] {scores[0].model} "
                      f"(coverage {scores[0].coverage:.0%}, p50 err {scores[0].median_error:.3f})")


# Backtest presets: a known historical series + the question the bot would face.
_BACKTEST_PRESETS = {
    "mortgage": {
        "strategy": "ensemble", "entity": "FRED_MORTGAGE30US", "fred_id": "MORTGAGE30US",
        "template": "Will the US 30-year fixed mortgage rate (Freddie Mac PMMS) exceed {T}%?",
        "steps_per_year": 52,
    },
    "housing": {
        "strategy": "ensemble", "entity": "FRED_HOUST1F", "fred_id": "HOUST1F",
        "template": "Will US single-family housing starts SAAR exceed {T} thousand units?",
        "steps_per_year": 12,
    },
}


@app.command()
def backtest(
    preset: str = typer.Option("mortgage", help=f"One of {list(_BACKTEST_PRESETS)}"),
    horizon_months: int = typer.Option(6, help="Forecast horizon to test"),
) -> None:
    """Measure a bot's calibration + simulated PnL on real historical data."""
    from .backtest import backtest as run_backtest
    from .config import load_bots
    from .sources.fred import fetch_history
    from .strategies import get_strategy

    p = _BACKTEST_PRESETS[preset]
    series = fetch_history(p["fred_id"])
    console.print(f"[cyan]{preset}[/]: {len(series)} points {series[0][0]}..{series[-1][0]}")

    # Use the configured params for that strategy if present.
    params = next((b.params for b in load_bots() if b.strategy == p["strategy"]), {})
    strat = get_strategy(p["strategy"], **params)
    steps = max(1, round(p["steps_per_year"] * horizon_months / 12))

    r = run_backtest(
        strat, p["entity"], p["template"], series,
        horizon_steps=steps, horizon_years=horizon_months / 12,
    )
    console.print(
        f"n={r.n}  [bold]Brier={r.brier:.4f}[/] (baseline {r.baseline_brier:.4f}, "
        f"skill {r.skill:+.1%})  win={r.win_rate:.1%}  "
        f"ROI={r.roi:+.1%}  staked={r.total_staked:.0f}  profit={r.total_profit:+.0f}"
    )
    tbl = Table(title="Calibration (reliability)")
    for c in ("predicted", "actual", "n"):
        tbl.add_column(c, justify="right")
    for mp, mo, cnt in r.reliability:
        tbl.add_row(f"{mp:.2f}", f"{mo:.2f}", str(cnt))
    console.print(tbl)


@app.command()
def resolve(bot: str = typer.Option(..., "--bot")) -> None:
    """Insert RESOLUTION_CLOSE trades for any of the bot's positions that resolved."""
    cfg = load_bot(bot)
    with Store() as store:
        bot_id = store.upsert_bot(cfg.name, cfg.strategy)
        n = sync_resolutions(_client(cfg.api_key), store, bot_id)
    console.print(f"[green]Closed[/] {n} resolved positions")


@app.command()
def snapshot() -> None:
    """Write a daily PnL snapshot for every configured bot, then print the board."""
    with Store() as store:
        for cfg in load_bots():
            bot_id = store.upsert_bot(cfg.name, cfg.strategy)
            store.write_snapshot(bot_id)
        board = store.leaderboard()
    table = Table(title="Leaderboard — PnL")
    for col in ("bot", "pnl", "realized", "unrealized", "open", "closed"):
        table.add_column(col, justify="right" if col != "bot" else "left")
    for row in board:
        table.add_row(
            row["name"],
            f"{row['pnl']:.1f}",
            f"{row['realized_pnl']:.1f}",
            f"{row['unrealized_pnl']:.1f}",
            str(row["open_positions"]),
            str(row["closed_positions"]),
        )
    console.print(table)


def _print_signals(signals: list[dict]) -> None:
    if not signals:
        return
    table = Table(show_lines=False)
    for col in ("market", "dir", "amount", "price", "estimate"):
        table.add_column(col)
    for s in signals[:25]:
        table.add_row(
            (s.get("question") or s["market_id"])[:50],
            s["direction"],
            str(s["amount"]),
            f"{s['current_prob']:.2f}",
            f"{s['estimate']:.2f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
