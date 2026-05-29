import { cn } from "@/lib/utils";
import type { BotStatus } from "@/lib/types";

interface Props {
  status: BotStatus;
  label?: boolean;
  className?: string;
}

const tone: Record<BotStatus, { dot: string; text: string; label: string }> = {
  LIVE: { dot: "bg-signal shadow-[0_0_0_0_rgba(0,217,255,0.5)] animate-[pulse-signal_2.4s_infinite]", text: "text-signal", label: "Live" },
  PAUSED: { dot: "bg-text-3", text: "text-text-3", label: "Paused" },
  DISABLED: { dot: "bg-text-muted", text: "text-text-muted", label: "Off" },
};

export function StatusDot({ status, label = false, className }: Props) {
  const t = tone[status];
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", t.dot)} />
      {label && (
        <span className={cn("font-mono text-[10px] uppercase tracking-wider", t.text)}>
          {t.label}
        </span>
      )}
    </span>
  );
}
