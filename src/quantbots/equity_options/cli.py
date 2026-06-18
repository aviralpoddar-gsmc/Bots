"""`eo` — the equity_options CLI. Separate entry point from `quantbots` (the clone).

Commands:
  eo chain --underlying FCX          # print a chain summary (needs Alpaca data keys)
  eo recommend [--underlying FCX]    # ranked recommendations + capped allocation (NO orders)
  eo trade [--paper] [--underlying]  # build + submit orders (dry by default) and record them
  eo positions [--paper]             # broker positions
  eo backtest --underlying FCX       # walk-forward calibration metrics
  eo snapshot [--paper]              # write greeks + pnl snapshot
  eo flatten --paper                 # cancel all orders + close all positions (kill switch)
  eo safety-check                    # verify the fence (no manifold import) + live refusal
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from . import config as cfg_mod
from .config import DRY, PAPER, load_config

app = typer.Typer(help="equity_options — REAL listed-options bot (Alpaca). See SAFETY.md.",
                  no_args_is_help=True)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _broker_mode(paper: bool, live: bool, cfg) -> str:
    if live:
        return cfg_mod.LIVE
    if paper:
        return PAPER
    return DRY if cfg.broker == DRY else cfg.broker


def _need_keys(cfg) -> None:
    """Fail cleanly (no traceback) when Alpaca market-data keys are missing."""
    if not (cfg.alpaca_key and cfg.alpaca_secret):
        console.print("[red]Missing Alpaca credentials.[/red] Set ALPACA_API_KEY and "
                      "ALPACA_SECRET_KEY (paper keys) in your env / Doppler, then retry.")
        raise typer.Exit(code=1)


def _held_underlyings(broker) -> set[str]:
    """Underlyings we shouldn't re-enter: those with an OPEN POSITION *or* a working
    ORDER on the broker (empty for dry-run). Covers the gap where a resting limit
    order isn't yet a position but would still stack if we re-traded the name."""
    from .occ import parse_occ
    from .positions import held_underlyings, structures_from_broker
    held: set[str] = set()
    try:
        held |= held_underlyings(structures_from_broker(broker.positions()))
    except Exception:  # noqa: BLE001 - dry broker / transient API issue
        pass
    try:
        for o in broker.list_orders(status="open", limit=200):
            for leg in (o.get("legs") or [o]):
                sym = leg.get("symbol")
                if sym:
                    try:
                        held.add(parse_occ(sym).underlying)
                    except ValueError:
                        pass
    except Exception:  # noqa: BLE001
        pass
    return held


@app.command()
def recommend(underlying: str = typer.Option(None, help="Limit to one ticker"),
              config: str = typer.Option(None, help="Path to equity_options.yaml")):
    """Rank tradable option candidates and show the capped allocation (no orders)."""
    from .recommend import recommend as run

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    if underlying:
        cfg.underlyings = [u for u in cfg.underlyings if u.ticker == underlying.upper()]
    rec = run(cfg)
    table = Table(title="Top option recommendations (per underlying)")
    for col in ("ticker", "structure", "expiry", "dte", "premium$", "edge$", "score",
                "win%", "fp_vol"):
        table.add_column(col)
    for tkr, cands in rec.candidates_by_underlying.items():
        c = cands[0]
        table.add_row(tkr, c.extra.get("label", c.structure), str(c.expiry), str(c.dte),
                      f"{c.premium:.0f}", f"{c.edge_dollars:.0f}", f"{c.edge.score:.2f}",
                      f"{c.edge.win_prob:.0%}", f"{c.forecast_vol:.0%}")
    console.print(table)
    console.print(f"\n[bold]Allocation[/bold] ({len(rec.allocations)} tickets):")
    for a in rec.allocations:
        console.print(f"  {a.candidate.underlying} {a.candidate.extra.get('label')} "
                      f"x{a.contracts}  premium ${a.premium_total:.0f}  "
                      f"score {a.candidate.edge.score:.2f}")


@app.command()
def chain(underlying: str = typer.Option(..., help="Underlying ticker"),
          config: str = typer.Option(None)):
    """Print a compact view of the live option chain for one underlying."""
    from .sources.options_chain import ChainClient

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    cc = ChainClient(key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    rows = cc.get_chain(underlying, min_dte=cfg.risk_limits["min_dte"],
                        max_dte=cfg.risk_limits["max_dte"])
    table = Table(title=f"{underlying.upper()} chain ({len(rows)} contracts)")
    for col in ("symbol", "expiry", "strike", "kind", "bid", "ask", "iv", "delta", "OI"):
        table.add_column(col)
    for r in rows[:60]:
        table.add_row(r["symbol"], str(r["expiry"]), f"{r['strike']:g}", r["kind"],
                      str(r.get("bid")), str(r.get("ask")),
                      f"{r['iv']:.2f}" if r.get("iv") else "-",
                      f"{r['delta']:.2f}" if r.get("delta") else "-",
                      str(r.get("open_interest") or "-"))
    console.print(table)


@app.command()
def trade(underlying: str = typer.Option(None), paper: bool = typer.Option(False),
          live: bool = typer.Option(False, help="REFUSED in this build"),
          config: str = typer.Option(None)):
    """Build orders from the allocation and submit them (DRY by default)."""
    from .execution.base import OptionOrder, OrderLeg
    from .execution.live import make_broker
    from .recommend import recommend as run
    from .store.db import OptionsStore

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)  # recommend fetches live chains regardless of broker mode
    if underlying:
        cfg.underlyings = [u for u in cfg.underlyings if u.ticker == underlying.upper()]
    mode = _broker_mode(paper, live, cfg)
    broker = make_broker(mode, key=cfg.alpaca_key, secret=cfg.alpaca_secret,
                         risk_limits_file=cfg.risk_limits_file)
    bankroll = broker.account_equity()
    # Position awareness: never stack a second structure on a name we already hold.
    exclude = _held_underlyings(broker)
    if exclude:
        console.print(f"[dim]Already held (skipping): {', '.join(sorted(exclude))}[/dim]")
    # Validation gate: only enter names whose walk-forward backtest PASSED recently.
    if cfg.gate.get("required", True):
        from .backtest import passing_tickers
        passing = passing_tickers(max_age_days=cfg.gate["max_age_days"])
        universe = {u.ticker for u in cfg.enabled_underlyings()}
        blocked = universe - passing - exclude
        if blocked:
            console.print(f"[yellow]Gate-blocked (no fresh backtest PASS): "
                          f"{', '.join(sorted(blocked))}[/yellow] — run `eo backtest` to refresh.")
        exclude = exclude | blocked
    rec = run(cfg, bankroll=bankroll, exclude=exclude)
    if not rec.allocations:
        console.print("[yellow]No allocations pass the gates. Nothing to do.[/yellow]")
        return
    store = OptionsStore()
    for a in rec.allocations:
        c = a.candidate
        legs = [OrderLeg(symbol=l["symbol"], side=("BUY" if l["qty"] > 0 else "SELL"))
                for l in c.legs]
        # Alpaca's net limit_price is a positive magnitude; debit vs credit is implied
        # by the per-leg buy/sell intents. cost_per_share is signed (credit < 0).
        order = OptionOrder(underlying=c.underlying, structure=c.structure, legs=legs,
                            qty=a.contracts, limit_price=round(abs(c.cost_per_share), 2))
        try:
            result = broker.submit(order)
        except Exception as e:  # noqa: BLE001 - one bad order must not abort the run
            console.print(f"[red]REJECTED[/red] {c.underlying} {c.extra.get('label')}: {e}")
            from .execution.base import OrderResult
            result = OrderResult(ticket_id=order.ticket_id, broker=mode, status="rejected")
        for l in c.legs:
            side = "BUY" if l["qty"] > 0 else "SELL"
            sign = -1 if side == "BUY" else 1
            store.record_leg(
                ticket_id=order.ticket_id, underlying=c.underlying, structure=c.structure,
                symbol=l["symbol"], trade_type="ENTRY", side=side, qty=a.contracts,
                fill_price=l["mid"], amount=sign * a.contracts * l["mid"] * 100,
                broker=mode, status=result.status, broker_order_id=result.broker_order_id,
                edge=c.edge.edge, forecast_vol=c.forecast_vol,
                reasoning=f"{c.extra.get('label')} score={c.edge.score:.2f}")
        console.print(f"[green]{mode.upper()}[/green] {c.underlying} "
                      f"{c.extra.get('label')} x{a.contracts} -> {result.status}")
    store.close()


@app.command()
def hedge(paper: bool = typer.Option(False), config: str = typer.Option(None)):
    """Delta-hedge open short-vol positions with the underlying shares (DRY by default).
    No-ops when there are no vol-neutral positions (directional spreads are left alone)."""
    from .execution.live import make_broker
    from .hedge import apply_hedges, compute_hedges
    from .sources.options_chain import ChainClient
    from .store.db import OptionsStore

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    mode = PAPER if paper else DRY
    broker = make_broker(PAPER, key=cfg.alpaca_key, secret=cfg.alpaca_secret)  # live positions
    cc = ChainClient(key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    with OptionsStore() as store:
        actions = compute_hedges(broker, cc, store)
        apply_hedges(broker, actions, dry=not paper)
    for a in actions:
        console.print(f"[cyan]hedge[/cyan] {a.side} {abs(a.trade_shares)} {a.underlying} "
                      f"(net Δ={a.net_option_delta:.0f}, held={a.current_shares:.0f})")
    console.print(f"hedge: {len(actions)} adjustment(s) ({mode}).")


@app.command()
def monitor(paper: bool = typer.Option(False), interval: int = typer.Option(300, help="seconds"),
            allow_entry: bool = typer.Option(True, help="open new (gated) positions intraday"),
            do_hedge: bool = typer.Option(True), max_minutes: int = typer.Option(0, help="0=run forever"),
            config: str = typer.Option(None)):
    """Continuous intraday loop: every `interval`s while the market is OPEN, reconcile
    fills, run exits (take-profit/stop/DTE on live marks), delta-hedge vol positions, and
    open new GATE-PASSING positions. Idles when the market is closed. Entries remain
    gate-controlled (set gate.required:false in config to override — not recommended)."""
    import subprocess
    import sys
    import time
    from datetime import UTC, datetime

    from .execution.live import make_broker

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    broker = make_broker(PAPER if paper else DRY, key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    base = [sys.executable, "-m", "quantbots.equity_options.cli"]
    pflag = ["--paper"] if paper else []
    console.print(f"[bold]eo monitor[/bold] interval={interval}s paper={paper} "
                  f"entry={allow_entry} hedge={do_hedge} — Ctrl-C to stop.")
    start = time.time()
    while True:
        if max_minutes and (time.time() - start) / 60 >= max_minutes:
            console.print("monitor: max_minutes reached, exiting."); break
        stamp = datetime.now(UTC).strftime("%H:%M:%SZ")
        try:
            is_open = broker.is_market_open()
        except Exception as e:  # noqa: BLE001
            console.print(f"[{stamp}] clock error: {e}"); is_open = False
        if is_open:
            console.print(f"[{stamp}] tick (market open)")
            if paper:
                subprocess.run(base + ["reconcile"], check=False)
            subprocess.run(base + ["manage"] + pflag, check=False)
            if do_hedge:
                subprocess.run(base + ["hedge"] + pflag, check=False)
            if allow_entry:
                subprocess.run(base + ["trade"] + pflag, check=False)
        else:
            console.print(f"[{stamp}] market closed — idle")
        time.sleep(interval)


@app.command()
def reconcile(config: str = typer.Option(None)):
    """Sync the local ledger to ACTUAL Alpaca fills (broker = source of truth)."""
    from .execution.alpaca import AlpacaPaperBroker
    from .store.db import OptionsStore

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    broker = AlpacaPaperBroker(key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    orders = broker.list_orders(status="all", limit=200)
    with OptionsStore() as store:
        n = store.reconcile_fills(orders)
    console.print(f"[green]Reconciled[/green] {n} ledger legs against {len(orders)} broker orders.")


@app.command()
def manage(paper: bool = typer.Option(False), config: str = typer.Option(None)):
    """Evaluate exit rules on open positions and close those that fire (DRY by default)."""
    from .execution.live import make_broker
    from .manage import build_close_order, exit_decisions
    from .positions import structures_from_broker
    from .sources import underlying as und_src
    from .store.db import OptionsStore

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    mode = PAPER if paper else DRY
    broker = make_broker(PAPER, key=cfg.alpaca_key, secret=cfg.alpaca_secret)  # need live positions
    structures = structures_from_broker(broker.positions())
    if not structures:
        console.print("No open option positions.")
        return
    spots = {s.underlying: (und_src.spot(s.underlying) or 0.0) for s in structures}
    decisions = exit_decisions(structures, rules=cfg.manage, spots=spots)
    if not decisions:
        console.print(f"[dim]{len(structures)} open positions; none meet an exit rule.[/dim]")
        return
    store = OptionsStore()
    for d in decisions:
        order = build_close_order(d.structure)
        verb = "PAPER-CLOSE" if paper else "DRY-CLOSE"
        if paper:
            try:
                result = broker.submit(order)
                status = result.status
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]CLOSE REJECTED[/red] {d.structure.underlying}: {e}")
                status = "rejected"
                result = None
        else:
            status = "intended"
            result = None
        for leg in order.legs:
            side = leg.side
            sign = -1 if side == "BUY" else 1
            store.record_leg(
                ticket_id=order.ticket_id, underlying=d.structure.underlying,
                structure="close", symbol=leg.symbol, trade_type="EXIT", side=side,
                qty=order.qty, fill_price=order.limit_price,
                amount=sign * order.qty * order.limit_price * 100, broker=mode,
                status=status, broker_order_id=getattr(result, "broker_order_id", None),
                reasoning=d.reason)
        console.print(f"[yellow]{verb}[/yellow] {d.structure.underlying} "
                      f"x{d.structure.contracts} -> {status}  ({d.reason})")
    store.close()


@app.command()
def positions(paper: bool = typer.Option(False), config: str = typer.Option(None)):
    """List broker positions (paper) or local open positions (dry)."""
    from .execution.live import make_broker
    from .store.db import OptionsStore

    cfg = load_config(config) if config else load_config()
    if paper:
        broker = make_broker(PAPER, key=cfg.alpaca_key, secret=cfg.alpaca_secret)
        for p in broker.positions():
            console.print(p)
    else:
        with OptionsStore() as store:
            for sym, pos in store.open_positions().items():
                console.print(f"{sym}: {pos['net_contracts']} contracts, "
                              f"net cash {pos['net_cash']:.0f}")


@app.command()
def backtest(underlying: str = typer.Option(None, help="Ticker (default: all enabled)"),
             horizon: int = typer.Option(90, help="Days to target expiry"),
             start: str = typer.Option("2024-03-01", help="First as-of date (>= Alpaca history)"),
             mode: str = typer.Option("momentum", help="forecast mode: momentum | tal | drift_neutral"),
             config: str = typer.Option(None)):
    """Walk-forward gate: Brier-skill vs the implied baseline + realized PnL/Sharpe.

    For each monthly as-of date it builds f_P with NO lookahead, reconstructs the
    historical chain from Alpaca bars, and scores against the realized terminal price.
    """
    from datetime import date, timedelta

    from .backtest import (
        ALPACA_OPTIONS_HISTORY_START,
        load_gate_results,
        monthly_as_of_dates,
        run_backtest,
        save_gate_results,
    )

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    tickers = ([underlying.upper()] if underlying
               else [u.ticker for u in cfg.enabled_underlyings()])
    gargs = {"min_trades": cfg.gate["min_trades"], "min_brier_skill": cfg.gate["min_brier_skill"],
             "min_sharpe": cfg.gate["min_sharpe"]}
    saved = load_gate_results().get("results", {})   # merge so single-ticker runs update one entry
    # Walk-forward window: from `start` (>= Alpaca history) to today - horizon - buffer,
    # so every fold's expiry has already realized.
    sy, sm, sd = (int(x) for x in start.split("-"))
    start_d = max(date(sy, sm, sd), date(2024, 3, 1))
    today = date.today()
    end_d = today - timedelta(days=horizon + 21)
    as_of_dates = monthly_as_of_dates(start_d, end_d)
    console.print(f"[dim]Alpaca options history starts {ALPACA_OPTIONS_HISTORY_START}; "
                  f"{len(as_of_dates)} monthly folds {start_d}..{end_d}, horizon {horizon}d.[/dim]")

    table = Table(title="Backtest gate (Brier-skill>0 AND PnL-Sharpe>0)")
    for col in ("ticker", "folds", "trades", "brier", "brier_skill", "crps",
                "pnl_total", "pnl_sharpe", "win%", "GATE"):
        table.add_column(col)
    for tkr in tickers:
        try:
            res = run_backtest(cfg, tkr, as_of_dates=as_of_dates, horizon_days=horizon, mode=mode)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]{tkr} backtest failed: {e}[/red]")
            continue
        s = res.summary()
        passed, reason = res.gate(**gargs)
        saved[tkr] = {"passed": passed, "brier_skill": s.get("brier_skill"),
                      "pnl_sharpe": s.get("pnl_sharpe"), "n_trades": s.get("n_trades"),
                      "reason": reason}
        table.add_row(
            tkr, str(s.get("folds", 0)), str(s.get("n_trades", 0)),
            f"{s.get('brier', float('nan')):.3f}", f"{s.get('brier_skill', float('nan')):+.3f}",
            f"{s.get('crps', float('nan')):.2f}", f"{s.get('pnl_total', 0):.0f}",
            f"{s.get('pnl_sharpe', 0):+.2f}", f"{s.get('pnl_win_rate', 0):.0%}",
            "[green]PASS[/green]" if passed else "[red]FAIL[/red]")
    save_gate_results(saved)
    console.print(table)
    n_pass = sum(1 for r in saved.values() if r.get("passed"))
    console.print(f"[dim]Saved gate results ({n_pass} PASS). `eo trade` only enters names "
                  f"that PASS a fresh gate when gate.required=true.[/dim]")


@app.command()
def snapshot(config: str = typer.Option(None)):
    """Write a PnL + greeks snapshot from the local ledger."""
    from .store.db import OptionsStore

    with OptionsStore() as store:
        positions = store.open_positions()
        premium = sum(abs(p["net_cash"]) for p in positions.values())
        store.write_pnl_snapshot(realized=0.0, unrealized=0.0, premium_at_risk=premium,
                                 open_positions=len(positions), closed_positions=0)
    console.print(f"[green]Snapshot written[/green] ({len(positions)} open positions).")


@app.command(name="cancel-orders")
def cancel_orders(config: str = typer.Option(None)):
    """Cancel all OPEN (working/unfilled) orders. Leaves positions intact. Used by the
    daily cycle to clear stale limit orders before re-pricing."""
    from .execution.alpaca import AlpacaPaperBroker

    cfg = load_config(config) if config else load_config()
    _need_keys(cfg)
    AlpacaPaperBroker(key=cfg.alpaca_key, secret=cfg.alpaca_secret).cancel_all()
    console.print("[green]Cancelled[/green] all open orders (positions untouched).")


@app.command()
def flatten(paper: bool = typer.Option(False), config: str = typer.Option(None)):
    """KILL SWITCH: cancel all open orders and close all positions on the account."""
    from .execution.live import make_broker

    cfg = load_config(config) if config else load_config()
    if not paper:
        console.print("[yellow]Dry mode: nothing to flatten. Use --paper.[/yellow]")
        return
    broker = make_broker(PAPER, key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    broker.cancel_all()
    broker.close_all()
    console.print("[red]Flattened[/red]: cancelled all orders and closed all positions.")


@app.command()
def screen(period: str = typer.Option("5y"), max_lag: int = typer.Option(5),
           write: bool = typer.Option(False, help="Write discovered universe to config"),
           top: int = typer.Option(40, help="Keep top-N by |t| when writing")):
    """Discover which equities a commodity predicts (lead-lag screen) and optionally
    write the discovered universe that `eo trade`/`eo backtest` will then use."""
    import yaml as _yaml

    from . import config as cfg_mod
    from .research.screen import run_screen

    results = run_screen(period=period, max_lag=max_lag)
    table = Table(title=f"Commodity→equity predictors ({len(results)} passed: |t|≥2 + sign-stable)")
    for col in ("equity", "commodity", "lead(d)", "beta", "t", "R²", "n"):
        table.add_column(col)
    for r in results:
        table.add_row(r.equity, r.commodity, str(r.lag), f"{r.beta:+.2f}",
                      f"{r.tstat:+.1f}", f"{r.r2:.2f}", str(r.n_obs))
    console.print(table)
    if write:
        kept = results[:top]
        doc = {"underlyings": [
            {"ticker": r.equity, "commodity": r.commodity, "market_ticker": "SPY",
             "name": f"{r.commodity} predictor lag{r.lag} t{r.tstat:+.1f}",
             "beta_lookback_days": 504}
            for r in kept]}
        cfg_mod.DISCOVERED_UNIVERSE.write_text(_yaml.safe_dump(doc, sort_keys=False))
        console.print(f"[green]Wrote {len(kept)} underlyings[/green] -> "
                      f"{cfg_mod.DISCOVERED_UNIVERSE} (now drives the universe).")


@app.command(name="metals-matrix")
def metals_matrix(min_corr: float = typer.Option(0.2, help="hide |corr| below this"),
                  period: str = typer.Option("3y"), top: int = typer.Option(15),
                  csv: str = typer.Option(None, help="write full matrix to this CSV path")):
    """Correlation matrix of commodity-equities vs each metal (one ranked table per metal)."""
    from .research.metal_matrix import build_matrix, dominant_metal, per_metal

    m = build_matrix(period=period)
    if csv:
        m.to_csv(csv)
        console.print(f"[dim]full matrix -> {csv}[/dim]")
    for metal, series in per_metal(m, min_corr=min_corr).items():
        table = Table(title=f"{metal} — equities by return correlation")
        table.add_column("equity"); table.add_column("corr")
        for eq, c in series.head(top).items():
            color = "green" if c > 0 else "red"
            table.add_row(eq, f"[{color}]{c:+.2f}[/{color}]")
        console.print(table)
    dom = dominant_metal(m)
    console.print("[bold]Dominant metal per equity[/bold] (|corr| max):")
    from collections import defaultdict
    by_metal = defaultdict(list)
    for eq, (metal, c) in sorted(dom.items(), key=lambda x: -abs(x[1][1])):
        by_metal[metal].append(f"{eq}({c:+.2f})")
    for metal, eqs in by_metal.items():
        console.print(f"  [cyan]{metal}[/cyan]: {', '.join(eqs)}")


@app.command(name="tal-probe")
def tal_probe():
    """Report what tal Snowflake data is reachable (ticker map + expectations)."""
    from .sources import tal_snowflake as tal

    info = tal.probe()
    if info.get("error"):
        console.print(f"[red]tal unavailable:[/red] {info['error']}")
        raise typer.Exit(code=1)
    console.print(f"[bold]tal db[/bold]={info['db']}  context={info.get('context')}")
    for k, v in info.items():
        if k.startswith(("PCF", "MEASURABLE", "MARKET")):
            console.print(f"  {k}: {v:,}")
    df = tal.ticker_reference()
    eq = df[df["ASSET_TYPE"] == "EQUITY"] if "ASSET_TYPE" in df else df
    console.print(f"[green]ticker_reference[/green]: {len(df)} rows, "
                  f"{len(eq)} equities, {df['METAL'].notna().sum()} with a metal link")


@app.command(name="safety-check")
def safety_check():
    """Verify the fence: no `manifold` import under equity_options, and live refuses."""
    import ast
    from pathlib import Path

    pkg = Path(__file__).resolve().parent
    offenders: list[str] = []
    for py in pkg.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            if any("manifold" in m for m in mods):
                offenders.append(f"{py.relative_to(pkg.parent)}: imports {mods}")
    if offenders:
        console.print("[red]FENCE VIOLATION[/red]: equity_options imports manifold:")
        for o in offenders:
            console.print(f"  {o}")
        raise typer.Exit(code=1)
    console.print("[green]OK[/green] no manifold imports under equity_options.")

    from .execution.live import LiveBroker, LiveTradingRefused
    try:
        LiveBroker()
        console.print("[red]FAIL[/red]: LiveBroker did not refuse.")
        raise typer.Exit(code=1)
    except LiveTradingRefused:
        console.print("[green]OK[/green] LiveBroker refuses real-money trading.")


if __name__ == "__main__":
    app()
