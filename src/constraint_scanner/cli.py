from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .api import DEFAULT_BASE_URL, ManifoldAPI
from .cache import DEFAULT_DB, MarketCache
from .detectors import (
    Violation,
    detect_answer_sum_violations,
    detect_duplicate_titles,
    detect_numeric_cdf_monotonicity,
    detect_pseudo_numeric_bounds,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# Outcome types whose answers (and thus arb signal) live only in the full-market
# payload — lite markets don't include answers. cpmm-multi-1 family.
MULTI_OUTCOME_TYPES = ["MULTIPLE_CHOICE", "NUMBER", "MULTI_NUMERIC", "DATE"]


async def _refresh(
    api_url: str,
    max_markets: int | None,
    concurrency: int,
    db_path: Path,
) -> tuple[int, int]:
    cache = MarketCache(db_path)
    total_lite = 0
    try:
        async with ManifoldAPI(api_url) as api:
            console.print(f"[cyan]Fetching markets from[/] {api.base_url}")
            batch: list = []
            async for m in api.iter_markets(max_markets=max_markets):
                batch.append(m)
                if len(batch) >= 500:
                    total_lite += cache.upsert_lite(batch)
                    console.print(f"  cached {total_lite} markets")
                    batch.clear()
            if batch:
                total_lite += cache.upsert_lite(batch)
                console.print(f"  cached {total_lite} markets")

            multi_ids: list[str] = []
            for outcome_type in MULTI_OUTCOME_TYPES:
                multi_ids.extend(cache.ids_needing_full(outcome_type))
            console.print(
                f"[cyan]Fetching full data for[/] {len(multi_ids)} multi-outcome markets"
            )
            full = await api.get_markets(multi_ids, concurrency=concurrency)
            n_full = cache.upsert_full(full)
            return total_lite, n_full
    finally:
        cache.close()


def _run_detectors(db_path: Path) -> dict[str, list[Violation]]:
    with MarketCache(db_path) as cache:
        lite = list(cache.iter_lite())
        full = list(cache.iter_full())
    return {
        "answer_sum": detect_answer_sum_violations(full),
        "numeric_cdf_monotonicity": detect_numeric_cdf_monotonicity(full),
        "duplicate_title": detect_duplicate_titles(lite),
        "pseudo_numeric_bounds": detect_pseudo_numeric_bounds(lite),
    }


def _print_summary(results: dict[str, list[Violation]]) -> None:
    table = Table(title="Constraint Scanner — Violations")
    table.add_column("Detector", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Top severity", justify="right")
    for kind, vs in results.items():
        top = max((v.severity for v in vs), default=0.0)
        table.add_row(kind, str(len(vs)), f"{top:.4f}")
    console.print(table)


def _print_top(results: dict[str, list[Violation]], n: int) -> None:
    for kind, vs in results.items():
        if not vs:
            continue
        sub = Table(title=f"Top {min(n, len(vs))} — {kind}", show_lines=False)
        sub.add_column("severity", justify="right")
        sub.add_column("market_ids")
        sub.add_column("detail")
        for v in sorted(vs, key=lambda x: x.severity, reverse=True)[:n]:
            sub.add_row(
                f"{v.severity:.4f}",
                ", ".join(v.market_ids[:3]) + ("…" if len(v.market_ids) > 3 else ""),
                json.dumps(v.detail, ensure_ascii=False)[:200],
            )
        console.print(sub)


def _write_report(results: dict[str, list[Violation]]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"scan-{ts}.json"
    payload = {
        "generated_at": ts,
        "counts": {kind: len(vs) for kind, vs in results.items()},
        "violations": {
            kind: [v.to_dict() for v in vs] for kind, vs in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


@app.command()
def scan(
    api_url: str = typer.Option(DEFAULT_BASE_URL, "--api-url", envvar="MANIFOLD_API_URL"),
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    limit: int | None = typer.Option(None, "--limit", help="Cap number of markets fetched"),
    concurrency: int = typer.Option(8, "--concurrency"),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh"),
    top: int = typer.Option(5, "--top", help="Print top-N violations per detector"),
) -> None:
    """Refresh market cache then run all constraint detectors."""
    if refresh:
        n_lite, n_full = asyncio.run(_refresh(api_url, limit, concurrency, db))
        console.print(f"[green]Refresh done[/]: {n_lite} lite, {n_full} full")
    else:
        console.print("[yellow]Skipping refresh[/], using existing cache")

    with MarketCache(db) as c:
        console.print(f"Cache size: {c.count()} markets")

    results = _run_detectors(db)
    _print_summary(results)
    _print_top(results, top)
    path = _write_report(results)
    console.print(f"[green]Report written:[/] {path}")


@app.command()
def report(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    top: int = typer.Option(10, "--top"),
) -> None:
    """Run detectors against the existing cache without refreshing."""
    with MarketCache(db) as c:
        console.print(f"Cache size: {c.count()} markets")
    results = _run_detectors(db)
    _print_summary(results)
    _print_top(results, top)
    path = _write_report(results)
    console.print(f"[green]Report written:[/] {path}")


if __name__ == "__main__":
    app()
