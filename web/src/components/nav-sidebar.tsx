import { NavLink } from "react-router";
import { Activity, BookOpen, Globe, LineChart } from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusDot } from "./status-dot";
import type { Snapshot } from "@/lib/types";
import { age, mana } from "@/lib/format";

const NAV = [
  { to: "/", label: "Fleet", icon: LineChart, end: true },
  { to: "/feed", label: "Feed", icon: Activity },
  { to: "/strategies", label: "Strategies", icon: BookOpen },
  { to: "/markets", label: "Markets", icon: Globe },
];

interface Props {
  snapshot: Snapshot | null;
  connected: boolean;
}

export function NavSidebar({ snapshot, connected }: Props) {
  const sys = snapshot?.system;
  const ov = snapshot?.overview;
  return (
    <aside className="h-screen sticky top-0 w-[224px] shrink-0 border-r border-border bg-surface flex flex-col">
      <div className="px-5 pt-6 pb-5 border-b border-border">
        <span className="font-mono text-[12px] tracking-wider uppercase text-text">
          quantbots<span className="text-text-3">/fleet</span>
        </span>
        <div className="mt-2 text-[10px] font-mono uppercase tracking-wider text-text-3">
          v0.1 · ops console
        </div>
      </div>

      <nav className="flex-1 flex flex-col gap-px py-3">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                "mx-2 px-3 py-2 flex items-center gap-2.5 rounded-md text-[12px] transition-colors",
                "font-mono uppercase tracking-wider",
                isActive
                  ? "bg-signal-bg/50 text-signal border-l-2 border-signal pl-[10px]"
                  : "text-text-3 hover:text-text hover:bg-surface-2",
              )
            }
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border px-5 py-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="label-eyebrow">Stream</span>
          <StatusDot status={connected ? "LIVE" : "PAUSED"} />
        </div>
        <div className="flex items-center justify-between">
          <span className="label-eyebrow">API</span>
          <span className={cn(
            "font-mono text-[10px] uppercase tracking-wider",
            sys?.status === "LIVE" ? "text-positive" : "text-warn",
          )}>
            {sys?.status ?? "—"} · {sys?.latency_ms ?? "—"}ms
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="label-eyebrow">Cache</span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-text-3">
            {snapshot?.cache_age_s != null ? `${Math.floor(snapshot.cache_age_s / 60)}m` : "—"}
          </span>
        </div>
        <div className="border-t border-border pt-3 mt-1">
          <div className="label-eyebrow">Operator</div>
          <div className="font-mono text-[12px] text-text mt-1">{sys?.username ?? "—"}</div>
          <div className="font-mono text-[10.5px] text-text-3 mt-0.5">
            {mana(sys?.balance, { compact: true })} balance
          </div>
        </div>
        <div className="text-[10px] font-mono uppercase tracking-wider text-text-muted mt-2 pt-2 border-t border-border">
          updated {snapshot?.ts ? age(snapshot.ts) : "—"}
        </div>
      </div>
    </aside>
  );
}
