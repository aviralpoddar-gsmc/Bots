import { RefreshCw } from "lucide-react";
import { Button } from "./ui/button";
import { api } from "@/lib/api";
import { useState } from "react";

interface Props {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  /** UTC clock — useful as an "alive" signal. */
  clock?: string;
}

export function HeaderBar({ title, subtitle, right, clock }: Props) {
  const [busy, setBusy] = useState(false);
  async function refresh() {
    setBusy(true);
    try {
      await api.refresh();
    } catch {
      // surfaced via toast layer if we add one; for now silent
    } finally {
      setBusy(false);
    }
  }
  return (
    <header className="flex items-end justify-between gap-6 px-8 pt-8 pb-6 border-b border-border">
      <div>
        <h1 className="font-display text-[26px] font-semibold tracking-tight leading-none text-text">
          {title}
        </h1>
        {subtitle && (
          <p className="text-[13px] text-text-3 mt-2 leading-snug max-w-[58ch]">{subtitle}</p>
        )}
      </div>
      <div className="flex items-stretch gap-4">
        {right}
        {clock && (
          <div className="text-right">
            <div className="label-eyebrow">Local</div>
            <div className="font-mono text-[13px] font-semibold text-text mt-1">{clock}</div>
          </div>
        )}
        <Button variant="ghost" size="default" onClick={refresh} disabled={busy} title="Refresh market cache (~30s)">
          <RefreshCw className={busy ? "h-3 w-3 animate-spin" : "h-3 w-3"} />
          {busy ? "syncing" : "sync"}
        </Button>
      </div>
    </header>
  );
}
