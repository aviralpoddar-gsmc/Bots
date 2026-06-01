import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  trail?: React.ReactNode;
  accent?: boolean;
  className?: string;
}

export function Kpi({ label, value, sub, trail, accent, className }: Props) {
  return (
    <div
      className={cn(
        "relative rounded-xl border bg-surface px-5 py-4 flex flex-col gap-2 min-h-[112px]",
        accent ? "border-signal/30" : "hairline",
        className,
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="label-eyebrow">{label}</span>
        {trail}
      </div>
      <div className="num text-[28px] xl:text-[32px] font-bold leading-none text-text">{value}</div>
      {sub && <div className="text-[11.5px] text-text-3 mt-0.5">{sub}</div>}
      {accent && (
        <span className="absolute top-3 right-3 h-1.5 w-1.5 rounded-full bg-signal animate-[pulse-signal_2.4s_infinite]" />
      )}
    </div>
  );
}
