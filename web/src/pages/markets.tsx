import { useEffect, useState } from "react";
import { HeaderBar } from "@/components/header-bar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { age, prob, truncate, pct } from "@/lib/format";
import type { MarketRow } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight, Filter } from "lucide-react";
import { Link } from "react-router";

const PAGE = 50;

export function MarketsPage() {
  const [rows, setRows] = useState<MarketRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [q, setQ] = useState("");
  const [minRes, setMinRes] = useState<number | "all">("all");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    async function load() {
      try {
        const data = await api.markets({
          page,
          size: PAGE,
          q: q.trim() || undefined,
          min_resolvability: minRes === "all" ? undefined : minRes,
        });
        if (!mounted) return;
        setRows(data.rows);
        setTotal(data.total);
        setErr(null);
      } catch (e) {
        if (mounted) setErr((e as Error).message);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    const id = window.setTimeout(load, 200);
    return () => {
      mounted = false;
      window.clearTimeout(id);
    };
  }, [page, q, minRes]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE));

  return (
    <div className="flex flex-col">
      <HeaderBar
        title="Markets"
        subtitle="The cached universe of markets on the clone. Resolvability score gates which bots are allowed to trade them."
        right={
          <div className="flex items-stretch gap-4 text-right">
            <div className="px-3">
              <div className="label-eyebrow">Total</div>
              <div className="font-mono text-[13px] font-semibold text-text mt-1">{total.toLocaleString()}</div>
            </div>
          </div>
        }
      />

      <section className="px-8 pt-6 pb-3 flex flex-wrap items-center gap-3">
        <input
          value={q}
          onChange={(e) => {
            setPage(1);
            setQ(e.target.value);
          }}
          placeholder="search markets…"
          className="bg-surface border border-border focus:border-signal focus:outline-none rounded-md px-3 py-1.5 text-[12px] font-mono text-text placeholder:text-text-muted min-w-[260px]"
        />
        <div className="flex items-center gap-1">
          <Filter className="h-3 w-3 text-text-3" />
          <span className="label-eyebrow mr-2">Resolvability</span>
          {(
            [
              ["all", "all"],
              [0.5, "≥ 50%"],
              [0.3, "≥ 30%"],
              [0.1, "≥ 10%"],
            ] as [number | "all", string][]
          ).map(([v, label]) => (
            <button
              key={label}
              type="button"
              onClick={() => {
                setPage(1);
                setMinRes(v);
              }}
              className={cn(
                "font-mono text-[10.5px] uppercase tracking-wider px-2 py-1 rounded-sm border transition-colors",
                minRes === v
                  ? "border-signal text-signal bg-signal-bg/40"
                  : "border-border text-text-3 hover:text-text",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1 || loading}
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>
          <span className="font-mono text-[11px] text-text-3">
            {page} / {totalPages}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages || loading}
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
        </div>
      </section>

      <section className="px-8 pb-12">
        {err && <div className="text-negative font-mono text-[11px] mb-4">{err}</div>}
        <div className="rounded-lg border border-border bg-surface overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="label-eyebrow text-left px-4 py-3">Market</th>
                <th className="label-eyebrow text-left px-3 py-3 w-[5rem]">Type</th>
                <th className="label-eyebrow text-right px-3 py-3 w-[5rem]">Prob</th>
                <th className="label-eyebrow text-right px-3 py-3 w-[5rem]">Resolv</th>
                <th className="label-eyebrow text-left px-3 py-3 w-[16rem]">Traded by</th>
                <th className="label-eyebrow text-right px-3 py-3 pr-4 w-[6rem]">Closes</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.id} className="border-b border-border/40 last:border-0 hover:bg-surface-2 transition-colors">
                  <td className="px-4 py-3 text-[12.5px] text-text-2 max-w-[60ch] truncate" title={m.question}>
                    {truncate(m.question, 96) || m.id}
                  </td>
                  <td className="px-3 py-3">
                    <Badge variant="default">{m.market_type}</Badge>
                  </td>
                  <td className="px-3 py-3 text-right font-mono text-[12px] text-text">
                    {prob(m.current_prob)}
                  </td>
                  <td className="px-3 py-3 text-right font-mono text-[12px]">
                    <ResolvBadge v={m.resolvability} />
                  </td>
                  <td className="px-3 py-3">
                    <div className="flex flex-wrap gap-1">
                      {m.traded_by.slice(0, 3).map((b) => (
                        <Link key={b} to={`/bots/${encodeURIComponent(b)}`}>
                          <Badge variant="signal" className="hover:border-signal cursor-pointer">{b}</Badge>
                        </Link>
                      ))}
                      {m.traded_by.length > 3 && (
                        <Badge variant="ghost">+{m.traded_by.length - 3}</Badge>
                      )}
                      {m.traded_by.length === 0 && <span className="text-text-muted font-mono text-[10.5px]">—</span>}
                    </div>
                  </td>
                  <td className="px-3 py-3 pr-4 text-right font-mono text-[11px] text-text-3">
                    {m.close_time ? age(m.close_time).replace(" ago", "") : "—"}
                  </td>
                </tr>
              ))}
              {rows.length === 0 && !loading && (
                <tr>
                  <td colSpan={6} className="px-4 py-16 text-center text-text-3 text-[12px] font-mono uppercase tracking-wider">
                    no markets match
                  </td>
                </tr>
              )}
              {loading && rows.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-16 text-center text-text-3 text-[12px] font-mono uppercase tracking-wider">
                    loading…
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

function ResolvBadge({ v }: { v: number | null }) {
  if (v == null) return <span className="text-text-muted">—</span>;
  if (v >= 0.5) return <span className="text-positive">{pct(v, 0)}</span>;
  if (v >= 0.2) return <span className="text-warn">{pct(v, 0)}</span>;
  return <span className="text-text-3">{pct(v, 0)}</span>;
}
