import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { HeaderBar } from "@/components/header-bar";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { age, clock, mana, prob, truncate } from "@/lib/format";
import type { EventFeedRow } from "@/lib/types";
import { cn } from "@/lib/utils";

const TYPE_BADGES: Record<string, "default" | "positive" | "negative" | "warn" | "signal"> = {
  ENTRY: "signal",
  EXIT: "positive",
  PARTIAL_EXIT: "warn",
  RESOLUTION_CLOSE: "default",
  RESOLUTION_REFUND: "default",
};

export function FeedPage() {
  const [rows, setRows] = useState<EventFeedRow[]>([]);
  const [filter, setFilter] = useState<string>("ALL");
  const [search, setSearch] = useState("");

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const data = await api.feed(300);
        if (mounted) setRows(data);
      } catch {
        // silent — header bar shows global status
      }
    }
    load();
    const id = window.setInterval(load, 8_000);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, []);

  const bots = useMemo(() => {
    const set = new Set<string>();
    for (const r of rows) set.add(r.bot);
    return Array.from(set).sort();
  }, [rows]);

  const filtered = useMemo(() => {
    let r = rows;
    if (filter !== "ALL") r = r.filter((x) => x.bot === filter);
    if (search.trim()) {
      const q = search.toLowerCase();
      r = r.filter(
        (x) =>
          x.question.toLowerCase().includes(q) ||
          x.bot.toLowerCase().includes(q) ||
          x.market_id.toLowerCase().includes(q),
      );
    }
    return r;
  }, [rows, filter, search]);

  return (
    <div className="flex flex-col">
      <HeaderBar
        title="Trade tape"
        subtitle="Live stream of every order written by the fleet. Newest first. Refreshes every 8 seconds."
        right={
          <div className="flex items-stretch gap-4 text-right">
            <div className="px-3">
              <div className="label-eyebrow">Total</div>
              <div className="font-mono text-[13px] font-semibold text-text mt-1">{rows.length}</div>
            </div>
            <div className="px-3">
              <div className="label-eyebrow">Showing</div>
              <div className="font-mono text-[13px] font-semibold text-text mt-1">{filtered.length}</div>
            </div>
          </div>
        }
      />

      <section className="px-8 pt-6 pb-3 flex flex-wrap items-center gap-3">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="search market, bot, or id…"
          className="bg-surface border border-border focus:border-signal focus:outline-none rounded-md px-3 py-1.5 text-[12px] font-mono text-text placeholder:text-text-muted min-w-[260px]"
        />
        <div className="flex items-center gap-1 overflow-x-auto">
          <button
            type="button"
            onClick={() => setFilter("ALL")}
            className={cn(
              "font-mono text-[10.5px] uppercase tracking-wider px-2 py-1 rounded-sm border transition-colors",
              filter === "ALL"
                ? "border-signal text-signal bg-signal-bg/40"
                : "border-border text-text-3 hover:text-text",
            )}
          >
            all
          </button>
          {bots.map((b) => (
            <button
              key={b}
              type="button"
              onClick={() => setFilter(b)}
              className={cn(
                "font-mono text-[10.5px] uppercase tracking-wider px-2 py-1 rounded-sm border transition-colors whitespace-nowrap",
                filter === b
                  ? "border-signal text-signal bg-signal-bg/40"
                  : "border-border text-text-3 hover:text-text",
              )}
            >
              {b}
            </button>
          ))}
        </div>
      </section>

      <section className="px-8 pb-12">
        <div className="rounded-lg border border-border bg-surface overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="label-eyebrow text-left px-4 py-3 w-[7rem]">Time</th>
                <th className="label-eyebrow text-left px-3 py-3 w-[10rem]">Bot</th>
                <th className="label-eyebrow text-left px-3 py-3 w-[5rem]">Type</th>
                <th className="label-eyebrow text-left px-3 py-3 w-[3.5rem]">Dir</th>
                <th className="label-eyebrow text-left px-3 py-3">Market</th>
                <th className="label-eyebrow text-right px-3 py-3 w-[5rem]">Size</th>
                <th className="label-eyebrow text-right px-3 py-3 w-[9rem]">Price</th>
                <th className="label-eyebrow text-right px-3 py-3 pr-4 w-[4.5rem]">Est</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e, i) => (
                <tr key={i} className="border-b border-border/40 last:border-0 hover:bg-surface-2 transition-colors">
                  <td className="px-4 py-2.5 font-mono text-[11px] text-text-3 whitespace-nowrap">
                    <span className="text-text-2">{clock(e.ts)}</span>
                    <div className="text-[10px] text-text-muted">{age(e.ts)}</div>
                  </td>
                  <td className="px-3 py-2.5">
                    <Link
                      to={`/bots/${encodeURIComponent(e.bot)}`}
                      className="font-mono text-[11.5px] text-text hover:text-signal transition-colors"
                    >
                      {e.bot}
                    </Link>
                  </td>
                  <td className="px-3 py-2.5">
                    <Badge variant={TYPE_BADGES[e.type] ?? "default"}>{e.type.replace("_", " ")}</Badge>
                  </td>
                  <td className="px-3 py-2.5">
                    <Badge variant={e.direction === "YES" ? "positive" : "negative"}>{e.direction}</Badge>
                  </td>
                  <td className="px-3 py-2.5 text-[12.5px] text-text-2 max-w-[60ch] truncate" title={e.question || e.market_id}>
                    {truncate(e.question || e.market_id, 80) || "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-[12px] text-text">{mana(e.amount)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-[11px] text-text-3 whitespace-nowrap">
                    {prob(e.price_before)} <span className="text-text-muted">→</span>{" "}
                    <span className="text-text-2">{prob(e.price_after)}</span>
                  </td>
                  <td className="px-3 py-2.5 pr-4 text-right font-mono text-[11px] text-signal-dim">
                    {e.estimate != null ? prob(e.estimate) : "—"}
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-4 py-16 text-center text-text-3 text-[12px] font-mono uppercase tracking-wider">
                    no trades match the current filter
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
