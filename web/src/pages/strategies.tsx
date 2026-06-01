import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Layers } from "lucide-react";
import { HeaderBar } from "@/components/header-bar";
import { Badge } from "@/components/ui/badge";
import { Delta } from "@/components/delta";
import { api } from "@/lib/api";
import { mana, num, pnlClass } from "@/lib/format";
import type { StrategyInfo } from "@/lib/types";
import { cn } from "@/lib/utils";

export function StrategiesPage() {
  const [rows, setRows] = useState<StrategyInfo[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const data = await api.strategies();
        if (mounted) setRows(data);
      } catch (e) {
        if (mounted) setErr((e as Error).message);
      }
    }
    load();
  }, []);

  return (
    <div className="flex flex-col">
      <HeaderBar
        title="Strategies"
        subtitle="The thesis layer. Each strategy is implemented once, then instantiated as one or more bots with different params."
      />

      <section className="px-8 py-8">
        {err && <div className="text-negative font-mono text-[11px] mb-4">{err}</div>}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {rows.map((s) => (
            <Link
              key={s.name}
              to={`/strategies/${encodeURIComponent(s.name)}`}
              className="group rounded-lg border border-border bg-surface p-5 hover:border-border-strong transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <Badge variant="signal">
                  <Layers className="h-2.5 w-2.5" /> {s.name}
                </Badge>
                <span className={cn("font-mono text-[14px] font-semibold", pnlClass(s.total_pnl))}>
                  {mana(s.total_pnl, { signed: true, compact: true })}
                </span>
              </div>
              <h3 className="font-display text-[15px] font-semibold text-text mb-2 group-hover:text-signal transition-colors">
                {s.class}
              </h3>
              <p className="text-[12.5px] text-text-3 leading-snug line-clamp-3 mb-4">
                {s.description || "—"}
              </p>
              <div className="flex items-center justify-between font-mono text-[10.5px] text-text-3 uppercase tracking-wider pt-3 border-t border-border">
                <span>{s.bots.length} bot{s.bots.length === 1 ? "" : "s"}</span>
                <span>{num(s.total_trades)} trades</span>
                <span className={s.live_count > 0 ? "text-signal" : "text-text-muted"}>
                  {s.live_count > 0 ? `${s.live_count} live` : "off"}
                </span>
              </div>
            </Link>
          ))}
          {rows.length === 0 && !err && (
            <div className="col-span-full text-center py-12 text-text-3 font-mono text-[12px] uppercase tracking-wider">
              loading…
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export function StrategyDetailPage() {
  const { name = "" } = useParams();
  const [s, setS] = useState<StrategyInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setS(null);
    setErr(null);
    async function load() {
      try {
        const data = await api.strategy(name);
        if (mounted) setS(data);
      } catch (e) {
        if (mounted) setErr((e as Error).message);
      }
    }
    load();
  }, [name]);

  if (err) {
    return (
      <div className="px-8 py-16 text-center">
        <p className="text-text-3 font-mono text-[12px]">error: {err}</p>
        <Link to="/strategies" className="text-signal font-mono text-[11px] mt-4 inline-block">← back</Link>
      </div>
    );
  }
  if (!s) {
    return (
      <div className="px-8 py-16 text-center">
        <p className="text-text-3 font-mono text-[12px]">loading…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <HeaderBar
        title={s.class}
        subtitle="Strategy class · the thesis these bots act on."
        right={
          <Link to="/strategies" className="text-text-3 hover:text-text font-mono text-[11px] uppercase tracking-wider flex items-center gap-1.5">
            <ArrowLeft className="h-3 w-3" /> strategies
          </Link>
        }
      />

      <section className="px-8 py-8 max-w-[80ch]">
        <div className="flex items-center gap-3 mb-6">
          <Badge variant="signal"><Layers className="h-2.5 w-2.5" /> {s.name}</Badge>
          <span className="font-mono text-[11px] text-text-3">{s.bots.length} bot{s.bots.length === 1 ? "" : "s"}</span>
        </div>

        <div className="rounded-lg border border-border bg-surface p-6 mb-6">
          <div className="label-eyebrow mb-3">Thesis</div>
          <p className="text-[14px] text-text-2 leading-relaxed">{s.description || "—"}</p>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-6">
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="label-eyebrow">Aggregate PnL</div>
            <Delta value={s.total_pnl} className="text-[18px] mt-1.5 block" compact />
          </div>
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="label-eyebrow">Trades</div>
            <div className="font-mono text-[18px] font-bold text-text mt-1.5">{num(s.total_trades)}</div>
          </div>
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="label-eyebrow">Live</div>
            <div className={cn("font-mono text-[18px] font-bold mt-1.5", s.live_count > 0 ? "text-signal" : "text-text-3")}>
              {s.live_count} / {s.bots.length}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-surface overflow-hidden">
          <div className="border-b border-border px-5 py-3">
            <div className="label-eyebrow">Bots using this strategy</div>
          </div>
          <ul className="divide-y divide-border/40">
            {s.bots.map((b) => (
              <li key={b}>
                <Link
                  to={`/bots/${encodeURIComponent(b)}`}
                  className="flex items-center justify-between px-5 py-3 hover:bg-surface-2 transition-colors"
                >
                  <span className="font-mono text-[13px] text-text">{b}</span>
                  <span className="font-mono text-[10.5px] text-signal-dim uppercase tracking-wider">open →</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-6 text-[10.5px] font-mono uppercase tracking-wider text-text-3">
          source: <code className="text-signal-dim">src/quantbots/strategies/{s.name}.py</code>
        </div>
      </section>
    </div>
  );
}
