import { cn } from "@/lib/utils";
import { delta, pnlClass } from "@/lib/format";

interface Props {
  value: number | null | undefined;
  className?: string;
  compact?: boolean;
  arrow?: boolean;
}

export function Delta({ value, className, compact, arrow = true }: Props) {
  const cls = pnlClass(value);
  const isPos = (value ?? 0) > 0;
  const isNeg = (value ?? 0) < 0;
  return (
    <span className={cn("font-mono font-semibold inline-flex items-center gap-1", cls, className)}>
      {arrow && (
        <span aria-hidden className="text-[10px]">
          {isPos ? "▲" : isNeg ? "▼" : "·"}
        </span>
      )}
      {delta(value, { compact })}
    </span>
  );
}
