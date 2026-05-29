import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, AlertTriangle } from "lucide-react";
import { HeaderBar } from "@/components/header-bar";
import { EquityChart } from "@/components/equity-chart";
import { StatusDot } from "@/components/status-dot";
import { Delta } from "@/components/delta";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { age, mana, num, pct, truncate, prob, pnlClass } from "@/lib/format";
import type { BotDetail } from "@/lib/types";
import { cn } from "@/lib/utils";

export function BotDetailPage() {
  const { name = "" } = useParams();
  const [bot, setBot] = useState<BotDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setBot(null);
    setErr(null);

    async function load() {
      try {
        const data = await api.bot(name);
        if (mounted) setBot(data);
      } catch (e) {
        if (mounted) setErr((e as Error).message);
      }
    }
    load();
    const id = window.setInterval(load, 10_000);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, [name]);

  if (err) {
    return (
      <div className="px-8 py-16 text-center">
        <p className="text-text-3 font-mono text-[12px]">error loading bot: {err}</p>
        <Link to="/" className="text-signal font-mono text-[11px] mt-4 inline-block">← back to fleet</Link>
      </div>
    );
  }
  if (!bot) {
    return (
      <div className="px-8 py-16 text-center">
        <p className="text-text-3 font-mono text-[12px]">loading {name}…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <HeaderBar
        title={bot.name}
        subtitle={bot.description || "—"}
        right={
          <div className="flex items-stretch gap-4">
            <Link to="/" className="text-text-3 hover:text-text font-mono text-[11px] uppercase tracking-wider flex items-center gap-1.5">
              <ArrowLeft className="h-3 w-3" /> fleet
            </Link>
          </div>
        }
      />

      {/* Hero: PnL + equity curve */}
      <section className="px-8 pt-6">
        <div className="grid grid-cols-12 gap-4 mb-4">
          <div className="col-span-12 lg:col-span-3 rounded-xl border border-border bg-surface p-5 flex flex-col gap-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="label-eyebrow">PnL · realized + unrealized</div>
                <div className={cn("num text-[44px] font-bold leading-none mt-2", pnlClass(bot.pnl))}>
                  {mana(bot.pnl, { signed: true, compact: true })}
                </div>
              </div>
              <StatusDot status={bot.status} label />
            </div>
            <div className="grid grid-cols-2 gap-3 text-[12px]">
              <div>
                <div className="label-eyebrow">Realized</div>
                <Delta value={bot.realized} className="text-[14px] mt-1 block" />
              </div>
              <div>
                <div className="label-eyebrow">Unrealized</div>
                <Delta value={bot.unrealized} className="text-[14px] mt-1 block" />
              </div>
              <div>
                <div className="label-eyebrow">Capital</div>
                <div className="font-mono text-[14px] mt-1 text-text">{mana(bot.invested_mark, { compact: true })}</div>
              </div>
              <div>
                <div className="label-eyebrow">Drawdown</div>
                <div className={cn("font-mono text-[14px] mt-1", pnlClass(bot.max_drawdown))}>
                  {bot.max_drawdown ? mana(bot.max_drawdown, { compact: true }) : "—"}
                </div>
              </div>
            </div>
            <div className="border-t border-border pt-3 grid grid-cols-3 gap-3 text-[11px]">
              <div>
                <div className="label-eyebrow">Win</div>
                <div className="font-mono text-text-2 mt-1">{bot.win_rate != null ? pct(bot.win_rate, 0) : "—"}</div>
              </div>
              <div>
                <div className="label-eyebrow">W / L / R</div>
                <div className="font-mono mt-1">
                  <span className="text-positive">{bot.wins}</span>
                  <span className="text-text-muted"> · </span>
                  <span className="text-negative">{bot.losses}</span>
                  <span className="text-text-muted"> · </span>
                  <span className="text-text-3">{bot.refunds}</span>
                </div>
              </div>
              <div>
                <div className="label-eyebrow">Trades</div>
                <div className="font-mono text-text-2 mt-1">{num(bot.n_trades_all)}</div>
              </div>
            </div>
          </div>

          <div className="col-span-12 lg:col-span-9 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-[16px] font-semibold tracking-tight text-text">Equity curve</h2>
              <span className="label-eyebrow">{bot.pnl_series.length} snapshots</span>
            </div>
            <EquityChart data={bot.pnl_series} height={260} />
          </div>
        </div>

        {/* Right-rail strip: risk + concentration */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          <RiskTile
            label="Exposure"
            value={bot.exposure_tier}
            sub={`${pct(bot.exposure_pct_of_cap)} of cap`}
            tone={bot.exposure_tier === "High" ? "warn" : undefined}
          />
          <RiskTile
            label="Concentration"
            value={pct(bot.concentration_pct, 0)}
            sub="top group share"
            tone={bot.concentration_pct > 0.4 ? "warn" : undefined}
          />
          <RiskTile
            label="Inventory"
            value={bot.inventory_bias?.[0] ?? "Neutral"}
            sub={
              bot.inventory_bias
                ? `${bot.yes_entries} Y / ${bot.no_entries} N`
                : `${bot.yes_entries} Y / ${bot.no_entries} N`
            }
            tone={
              bot.inventory_bias && Math.abs(bot.inventory_bias[1]) > 0.5 ? "warn" : undefined
            }
          />
          <RiskTile
            label="Last trade"
            value={bot.last_trade_at ? age(bot.last_trade_at) : "—"}
            sub={`avg ${bot.avg_edge != null ? bot.avg_edge.toFixed(3) : "—"} edge`}
            tone={bot.status === "PAUSED" ? "warn" : undefined}
          />
        </div>
      </section>

      <section className="px-8 pb-12">
        <Tabs defaultValue="trades">
          <TabsList>
            <TabsTrigger value="trades">Recent trades</TabsTrigger>
            <TabsTrigger value="exposures">Top exposures</TabsTrigger>
            <TabsTrigger value="strategy">Strategy</TabsTrigger>
            <TabsTrigger value="params">Params & limits</TabsTrigger>
          </TabsList>

          <TabsContent value="trades">
            <RecentTradesTable bot={bot} />
          </TabsContent>

          <TabsContent value="exposures">
            <ExposuresList bot={bot} />
          </TabsContent>

          <TabsContent value="strategy">
            <StrategyPanel bot={bot} />
          </TabsContent>

          <TabsContent value="params">
            <ParamsPanel bot={bot} />
          </TabsContent>
        </Tabs>
      </section>
    </div>
  );
}

function RiskTile({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "warn";
}) {
  return (
    <div className={cn(
      "rounded-lg border bg-surface px-4 py-3 flex flex-col gap-1",
      tone === "warn" ? "border-warn/40" : "border-border",
    )}>
      <div className="flex items-center justify-between">
        <span className="label-eyebrow">{label}</span>
        {tone === "warn" && <AlertTriangle className="h-3 w-3 text-warn" />}
      </div>
      <div className="font-mono text-[15px] text-text">{value}</div>
      {sub && <div className="text-[10.5px] text-text-3 font-mono">{sub}</div>}
    </div>
  );
}

function RecentTradesTable({ bot }: { bot: BotDetail }) {
  if (!bot.recent_trades.length) {
    return <EmptyState>no trades yet — bot may be paused, in dry-run, or freshly added.</EmptyState>;
  }
  return (
    <div className="rounded-lg border border-border bg-surface overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            <th className="label-eyebrow text-left px-4 py-3">Time</th>
            <th className="label-eyebrow text-left px-3 py-3 w-14">Dir</th>
            <th className="label-eyebrow text-left px-3 py-3">Market</th>
            <th className="label-eyebrow text-right px-3 py-3">Size</th>
            <th className="label-eyebrow text-right px-3 py-3">Price</th>
            <th className="label-eyebrow text-right px-3 py-3 pr-4">Est</th>
          </tr>
        </thead>
        <tbody>
          {bot.recent_trades.map((t, i) => (
            <tr key={i} className="border-b border-border/40 last:border-0 hover:bg-surface-2 transition-colors">
              <td className="px-4 py-3 font-mono text-[11px] text-text-3">{age(t.ts)}</td>
              <td className="px-3 py-3">
                <Badge variant={t.direction === "YES" ? "positive" : "negative"}>{t.direction}</Badge>
              </td>
              <td className="px-3 py-3 text-[12.5px] text-text-2 max-w-[60ch] truncate" title={t.question}>
                {truncate(t.question, 80) || "—"}
              </td>
              <td className="px-3 py-3 text-right font-mono text-[12px] text-text">{mana(t.amount)}</td>
              <td className="px-3 py-3 text-right font-mono text-[11.5px] text-text-3">
                {prob(t.price_before)} <span className="text-text-muted">→</span>{" "}
                <span className="text-text-2">{prob(t.price_after)}</span>
              </td>
              <td className="px-3 py-3 pr-4 text-right font-mono text-[11.5px] text-signal-dim">
                {t.estimate != null ? prob(t.estimate) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExposuresList({ bot }: { bot: BotDetail }) {
  if (!bot.exposures.length) {
    return <EmptyState>no open positions.</EmptyState>;
  }
  return (
    <div className="rounded-lg border border-border bg-surface p-5">
      <div className="flex items-end justify-between mb-4">
        <div>
          <div className="label-eyebrow">Top exposures by mana</div>
          <div className="text-[11px] text-text-3 mt-1 font-mono">
            total open: {mana(bot.total_open_exposure, { compact: true })} ·{" "}
            top group: {pct(bot.concentration_pct, 0)}
          </div>
        </div>
        <span className="label-eyebrow">{bot.exposures.length} groups</span>
      </div>
      <ul className="space-y-3">
        {bot.exposures.map((e) => (
          <li key={e.key}>
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-[13px] text-text-2 truncate mr-3" title={e.key}>{e.key}</span>
              <span className="font-mono text-[13px] text-text">{mana(e.amount)}</span>
            </div>
            <div className="h-1 bg-surface-3 rounded-sm overflow-hidden">
              <div
                className="h-full bg-signal/70 transition-all"
                style={{ width: `${(e.pct_of_max * 100).toFixed(1)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function StrategyPanel({ bot }: { bot: BotDetail }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-6 max-w-[80ch]">
      <div className="flex items-center gap-3 mb-4">
        <Badge variant="signal">{bot.strategy}</Badge>
        <Badge>{bot.strategy_class}</Badge>
      </div>
      <p className="text-[14px] text-text-2 leading-relaxed">{bot.description || "—"}</p>
      <div className="mt-6 pt-4 border-t border-border flex items-center gap-2 text-[11px] font-mono text-text-3">
        <span>read the source:</span>
        <code className="text-signal-dim">
          src/quantbots/strategies/{bot.strategy}.py
        </code>
      </div>
    </div>
  );
}

function ParamsPanel({ bot }: { bot: BotDetail }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <CodeBlock title="Params" data={bot.params} />
      <CodeBlock title="Limits" data={bot.limits} />
    </div>
  );
}

function CodeBlock({ title, data }: { title: string; data: Record<string, unknown> }) {
  const entries = Object.entries(data ?? {}).sort(([a], [b]) => a.localeCompare(b));
  return (
    <div className="rounded-lg border border-border bg-surface overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <span className="label-eyebrow">{title}</span>
        <span className="font-mono text-[10px] text-text-muted">{entries.length} keys</span>
      </div>
      <div className="font-mono text-[12px] divide-y divide-border/40">
        {entries.length === 0 && (
          <div className="px-4 py-3 text-text-3">empty</div>
        )}
        {entries.map(([k, v]) => (
          <div key={k} className="px-4 py-2 flex items-baseline gap-4 hover:bg-surface-2">
            <span className="text-text-3 min-w-[14ch]">{k}</span>
            <span className="text-text-2 truncate flex-1" title={JSON.stringify(v)}>
              {typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-12 text-center">
      <p className="text-text-3 font-mono text-[12px] uppercase tracking-wider">{children}</p>
    </div>
  );
}
