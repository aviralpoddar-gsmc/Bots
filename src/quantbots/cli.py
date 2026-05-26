"""`quantbots` command line. Thin wrapper over the framework.

    quantbots health                 # prove API key + Cloudflare Access work
    quantbots refresh                # pull markets into the local cache
    quantbots ingest                 # fetch external data sources into the cache
    quantbots run --bot NAME         # dry-run a bot (default); add --live to trade
    quantbots resolve --bot NAME     # close out resolved positions
    quantbots snapshot               # roll up PnL + print leaderboard
    quantbots strategies             # list registered strategies
    quantbots sources                # list registered data sources
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
    markets = client.search_markets(search, limit=limit) if search else client.list_markets(limit=limit)
    with Store() as store:
        n = store.upsert_markets(markets)
    console.print(f"[green]Cached[/] {n} markets")


@app.command()
def run(
    bot: str = typer.Option(..., "--bot", help="Bot name from config/bots.yaml"),
    live: bool = typer.Option(False, "--live", help="Actually place bets (default: dry-run)"),
) -> None:
    """Run one bot. Dry-run by default — validates orders without moving mana."""
    cfg = load_bot(bot)
    if not cfg.api_key:
        raise typer.BadParameter(f"No key in env var {cfg.account_env!r}")
    strat = get_strategy(cfg.strategy, **cfg.params)
    with Store() as store:
        result = run_bot(
            bot=cfg, client=_client(cfg.api_key), store=store, strategy=strat, dry_run=not live
        )
    mode = "[red]LIVE[/]" if live else "[yellow]dry-run[/]"
    console.print(f"{mode} {result.bot}: {len(result.signals)} signals on {result.n_markets} markets")
    _print_signals(result.signals)
    if not live:
        console.print(f"validated {len(result.signals)} orders, {len(result.errors)} errors")
    else:
        console.print(f"[green]placed[/] {result.orders_placed} orders, {len(result.errors)} errors")
    for e in result.errors[:10]:
        console.print(f"  [red]err[/] {e}")


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
