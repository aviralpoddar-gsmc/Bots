import { useMemo } from "react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from "recharts";
import { mana } from "@/lib/format";

interface Point {
  date: string;
  pnl: number;
}

interface Props {
  data: Point[];
  height?: number;
  /** Whether the line/fill should switch color by current sign. */
  signed?: boolean;
}

export function EquityChart({ data, height = 220, signed = true }: Props) {
  const ds = useMemo(
    () => data.map((p) => ({ ts: p.date, pnl: p.pnl, label: p.date.slice(0, 10) })),
    [data],
  );
  const last = ds.at(-1)?.pnl ?? 0;
  const color = signed
    ? last > 0
      ? "var(--color-positive)"
      : last < 0
        ? "var(--color-negative)"
        : "var(--color-signal-dim)"
    : "var(--color-signal-dim)";

  if (ds.length < 2) {
    return (
      <div
        className="rounded-lg border border-border bg-surface flex items-center justify-center text-text-3 text-[11.5px] font-mono uppercase tracking-wider"
        style={{ height }}
      >
        not enough snapshots yet — equity curve appears once the bot has 2+ daily snapshots
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-surface" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={ds} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="eq-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.28} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="label"
            axisLine={{ stroke: "var(--color-border)" }}
            tickLine={false}
            tick={{ fill: "var(--color-text-3)", fontFamily: "var(--font-mono)", fontSize: 10 }}
            minTickGap={48}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fill: "var(--color-text-3)", fontFamily: "var(--font-mono)", fontSize: 10 }}
            tickFormatter={(v) => mana(v as number, { compact: true })}
            width={56}
          />
          <ReferenceLine y={0} stroke="var(--color-border-strong)" strokeDasharray="2 4" />
          <Tooltip
            cursor={{ stroke: "var(--color-signal-dim)", strokeWidth: 1, strokeDasharray: "2 2" }}
            contentStyle={{
              background: "var(--color-surface-3)",
              border: "1px solid var(--color-border-strong)",
              borderRadius: 6,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              padding: "6px 8px",
            }}
            labelStyle={{ color: "var(--color-text-3)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}
            itemStyle={{ color: "var(--color-text)" }}
            formatter={(v) => [mana(v as number, { signed: true }), "PnL"]}
          />
          <Area
            type="monotone"
            dataKey="pnl"
            stroke={color}
            strokeWidth={1.5}
            fill="url(#eq-fill)"
            dot={false}
            activeDot={{ r: 3, fill: color, stroke: "var(--color-bg)", strokeWidth: 1 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
