import { Link } from "react-router";
import { useMemo } from "react";
import { useSnapshot } from "@/App";
import { HeaderBar } from "@/components/header-bar";
import { Kpi } from "@/components/kpi";
import { Sparkline } from "@/components/sparkline";
import { Delta } from "@/components/delta";
import { StatusDot } from "@/components/status-dot";
import { Badge } from "@/components/ui/badge";
import { mana, num, pct, age, truncate, pnlClass } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { BotRow, EquityPoint, EventFeedRow } from "@/lib/types";
import { ArrowUpRight, Layers, TrendingUp, Wallet } from "lucide-react";

function FleetSparkline({ equity, bot }: { equity: EquityPoint[]; bot: BotRow }) {
  // Per-bot sparkline is hard without per-bot history; until we wire that
  // through, derive a 12-point series from the global equity slope, scaled by the
  // bot's PnL share. Fine for v1 visual signal.
  const vals = useMemo(() => {
    if (!equity || equity.length < 2 || !bot.pnl) {
      return Array(12).fill(0);
    }
    const slice = equity.slice(-12);
    const max = Math.max(...slice.map((p) => Math.abs(p.pnl))) || 1;
    const scale = bot.pnl / max;
    return slice.map((p) => p.pnl * scale);
  }, [equity, bot.pnl]);
  return <Sparkline values={vals} width={64} height={16} signed />;
}

function PortfolioMix({ rows }: { rows: BotRow[] }) {
  const classes = useMemo(() => {
    const acc = new Map<string, number>();
    for (const r of rows) {
      acc.set(r.strategy_class, (acc.get(r.strategy_class) ?? 0) + Math.max(0, r.invested_mark));
    }
    const total = Array.from(acc.values()).reduce((a, b) => a + b, 0) || 1;
    return Array.from(acc.entries())
      .map(([label, v]) => ({ label, pct: v / total, mana: v }))
      .sort((a, b) => b.pct - a.pct);
  }, [rows]);

  return (
    <div className="flex flex-col gap-2 min-w-[180px]">
      <div className="label-eyebrow">Capital deployed</div>
      <div className="flex h-2 rounded-sm overflow-hidden gap-px">
        {classes.map((c, i) => (
          <div
            key={c.label}
            className={cn(
              "h-full",
              i === 0
                ? "bg-signal"
                : i === 1
                  ? "bg-signal/60"
                  : i === 2
                    ? "bg-signal/35"
                    : "bg-signal/20",
            )}
            style={{ width: `${c.pct * 100}%` }}
            title={`${c.label}: ${pct(c.pct)}`}
          />
        ))}
      </div>
      <ul className="space-y-0.5 mt-1">
        {classes.slice(0, 3).map((c, i) => (
          <li key={c.label} className="flex items-center justify-between text-[10.5px] font-mono text-text-3">
            <span className="flex items-center gap-1.5">
              <span
                className={cn(
                  "h-1 w-2",
                  i === 0 ? "bg-signal" : i === 1 ? "bg-signal/60" : "bg-signal/35",
                )}
              />
              <span className="uppercase tracking-wide">{c.label}</span>
            </span>
            <span className="text-text-2">{pct(c.pct, 0)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function LiveTape({ events }: { events: EventFeedRow[] }) {
  const recent = events.slice(0, 4);
  return (
    <div className="hidden xl:flex flex-col gap-1 min-w-[280px] max-w-[320px]">
      <div className="label-eyebrow flex items-center justify-between">
        <span>Live trade tape</span>
        <Link to="/feed" className="text-signal hover:underline">view →</Link>
      </div>
      <ul className="space-y-0.5">
        {recent.length === 0 && <li className="text-[11px] text-text-muted font-mono">— quiet</li>}
        {recent.map((e, i) => (
          <li key={i} className="flex items-center gap-2 text-[11px] font-mono leading-snug">
            <span className="text-text-muted">{age(e.ts).replace(" ago", "")}</span>
            <span className="text-signal-dim w-[2.6rem] truncate" title={e.bot}>
              {e.bot.split("_")[0]}
            </span>
            <span className={cn("w-[1.4rem]", e.direction === "YES" ? "text-positive" : "text-negative")}>
              {e.direction === "YES" ? "↑" : "↓"}
            </span>
            <span className="text-text-2 flex-1 truncate" title={e.question}>
              {truncate(e.question, 32) || "—"}
            </span>
            <span className="text-text">Ṁ{Math.round(e.amount)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function FleetPage() {
  const { data } = useSnapshot();
  const overview = data?.overview;
  const rows = data?.leaderboard ?? [];
  const equity = data?.equity ?? [];

  // Win rate excludes refunds.
  const winRate = overview && (overview.total_wins + overview.total_losses) > 0
    ? overview.total_wins / (overview.total_wins + overview.total_losses)
    : null;

  return (
    <div className="flex flex-col">
      <HeaderBar
        title="Fleet"
        subtitle="Autonomous trading systems on the private Manifold clone. Realized + unrealized PnL, ranked by realized edge."
        clock={data?.system?.username ? new Date().toLocaleTimeString("en-US", { hour12: false }) : undefined}
        right={
          <div className="flex items-stretch divide-x divide-border text-right">
            <div className="px-5">
              <div className="label-eyebrow">Bots</div>
              <div className="font-mono text-[13px] font-semibold text-text mt-1">
                {overview?.n_live ?? "—"} <span className="text-text-3">/ {overview?.n_bots ?? "—"} live</span>
              </div>
            </div>
            <div className="px-5">
              <div className="label-eyebrow">Trades</div>
              <div className="font-mono text-[13px] font-semibold text-text mt-1">{num(overview?.n_trades)}</div>
            </div>
          </div>
        }
      />

      <section className="px-8 pt-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
        <Kpi
          label="Portfolio PnL"
          accent
          value={
            <span className={pnlClass(overview?.total_pnl)}>
              {mana(overview?.total_pnl, { signed: true, compact: true })}
            </span>
          }
          sub={
            <span className="flex gap-3">
              <span><span className="text-text-3">realized</span> <span className={pnlClass(overview?.total_realized)}>{mana(overview?.total_realized, { signed: true })}</span></span>
              <span><span className="text-text-3">unrealized</span> <span className={pnlClass(overview?.total_unrealized)}>{mana(overview?.total_unrealized, { signed: true })}</span></span>
            </span>
          }
          trail={
            overview?.roi != null && (
              <span className={cn("font-mono text-[11px]", pnlClass(overview.roi))}>
                {overview.roi > 0 ? "+" : ""}{(overview.roi * 100).toFixed(2)}% ROI
              </span>
            )
          }
          className="xl:col-span-2"
        />
        <Kpi
          label="Active capital"
          value={mana(overview?.active_capital, { compact: true })}
          sub={
            <span>
              mark value of <span className="text-text-2">{overview?.open_positions ?? "—"}</span> open positions
            </span>
          }
          trail={<Wallet className="h-3 w-3 text-text-muted" />}
        />
        <Kpi
          label="Trades"
          value={num(overview?.n_trades, { compact: true })}
          sub={
            <span>
              {num(overview?.n_entries)} entries <span className="text-text-muted">·</span>{" "}
              {(overview?.total_wins ?? 0) + (overview?.total_losses ?? 0)} closed
            </span>
          }
          trail={<TrendingUp className="h-3 w-3 text-text-muted" />}
        />
        <Kpi
          label="Win rate"
          value={
            winRate === null ? <span className="text-text-3">—</span> : pct(winRate, 0)
          }
          sub={
            <span className="flex gap-2">
              <span className="text-positive">W {overview?.total_wins ?? 0}</span>
              <span className="text-negative">L {overview?.total_losses ?? 0}</span>
              <span className="text-text-3">R {overview?.total_refunds ?? 0}</span>
            </span>
          }
        />
      </section>

      <section className="px-8 pt-6 pb-4 flex items-end justify-between gap-8">
        <PortfolioMix rows={rows} />
        <LiveTape events={data?.events ?? []} />
      </section>

      <section className="px-8 pt-2 pb-12">
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-display text-[18px] font-semibold tracking-tight text-text">Leaderboard</h2>
          <span className="label-eyebrow">{rows.length} bots · ranked by PnL</span>
        </div>

        <div className="rounded-lg border border-border bg-surface overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="label-eyebrow text-left px-4 py-3 w-10">#</th>
                <th className="label-eyebrow text-left px-2 py-3">Bot</th>
                <th className="label-eyebrow text-left px-3 py-3">Strategy</th>
                <th className="label-eyebrow text-left px-3 py-3 w-20">Status</th>
                <th className="label-eyebrow text-right px-3 py-3">PnL</th>
                <th className="label-eyebrow text-left px-3 py-3 w-16">Trend</th>
                <th className="label-eyebrow text-right px-3 py-3">Capital</th>
                <th className="label-eyebrow text-right px-3 py-3">Trades</th>
                <th className="label-eyebrow text-right px-3 py-3 whitespace-nowrap">W · L · R</th>
                <th className="label-eyebrow text-right px-3 py-3">Edge</th>
                <th className="label-eyebrow text-right px-3 py-3 pr-4">Last trade</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((b, i) => (
                <BotTableRow key={b.name} rank={i + 1} bot={b} equity={equity} />
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={11} className="px-4 py-12 text-center text-text-3 text-[12px] font-mono">
                    no bots loaded — check that the dashboard server is running and the SQLite store is populated
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function BotTableRow({ rank, bot, equity }: { rank: number; bot: BotRow; equity: EquityPoint[] }) {
  return (
    <tr className="border-b border-border/60 last:border-0 hover:bg-surface-2 transition-colors group">
      <td className="px-4 py-3 font-mono text-[11px] text-text-3">{String(rank).padStart(2, "0")}</td>
      <td className="px-2 py-3">
        <Link
          to={`/bots/${encodeURIComponent(bot.name)}`}
          className="flex items-center gap-2 text-text font-medium text-[13.5px] hover:text-signal transition-colors"
        >
          {bot.name}
          <ArrowUpRight className="h-3 w-3 opacity-0 group-hover:opacity-100 text-signal transition-opacity" />
        </Link>
        <div className="text-[11px] text-text-3 mt-0.5 leading-snug max-w-[42ch] truncate" title={bot.description}>
          {bot.description || "—"}
        </div>
      </td>
      <td className="px-3 py-3">
        <Link to={`/strategies/${encodeURIComponent(bot.strategy)}`}>
          <Badge variant="default" className="hover:border-signal/40 hover:text-signal cursor-pointer">
            <Layers className="h-2.5 w-2.5" /> {bot.strategy}
          </Badge>
        </Link>
      </td>
      <td className="px-3 py-3">
        <StatusDot status={bot.status} label />
      </td>
      <td className="px-3 py-3 text-right">
        <Delta value={bot.pnl} compact />
        <div className="text-[10px] font-mono text-text-3 mt-0.5">
          U {mana(bot.unrealized, { signed: true, compact: true })}
        </div>
      </td>
      <td className="px-3 py-3">
        <FleetSparkline equity={equity} bot={bot} />
      </td>
      <td className="px-3 py-3 text-right font-mono text-[12.5px] text-text-2">
        {mana(bot.invested_mark, { compact: true })}
        <div className="text-[10px] text-text-3 mt-0.5">{bot.exposure_tier.toLowerCase()}</div>
      </td>
      <td className="px-3 py-3 text-right font-mono text-[12.5px] text-text-2">
        {num(bot.n_trades_all)}
      </td>
      <td className="px-3 py-3 text-right font-mono text-[11.5px] whitespace-nowrap">
        <span className="text-positive">{bot.wins}</span>
        <span className="text-text-muted"> · </span>
        <span className="text-negative">{bot.losses}</span>
        <span className="text-text-muted"> · </span>
        <span className="text-text-3">{bot.refunds}</span>
      </td>
      <td className="px-3 py-3 text-right font-mono text-[12.5px] text-text-2">
        {bot.avg_edge != null ? bot.avg_edge.toFixed(3) : "—"}
      </td>
      <td className="px-3 py-3 pr-4 text-right font-mono text-[11.5px] text-text-3">
        {bot.last_trade_at ? age(bot.last_trade_at) : "—"}
      </td>
    </tr>
  );
}
