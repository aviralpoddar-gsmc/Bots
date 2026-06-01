// Number formatting — every number on the dashboard goes through here.

const MANA = "Ṁ";
const COMPACT = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const FULL = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const DEC2 = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function mana(n: number | null | undefined, opts: { signed?: boolean; compact?: boolean } = {}): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  const sign = opts.signed ? (n > 0 ? "+" : n < 0 ? "−" : "") : "";
  const fmt = opts.compact && abs >= 10_000 ? COMPACT : FULL;
  return `${sign}${MANA}${fmt.format(abs)}`;
}

export function num(n: number | null | undefined, opts: { signed?: boolean; compact?: boolean; decimals?: number } = {}): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  const sign = opts.signed ? (n > 0 ? "+" : n < 0 ? "−" : "") : "";
  if (opts.decimals !== undefined) {
    return `${sign}${new Intl.NumberFormat("en-US", { minimumFractionDigits: opts.decimals, maximumFractionDigits: opts.decimals }).format(abs)}`;
  }
  const fmt = opts.compact && abs >= 10_000 ? COMPACT : FULL;
  return `${sign}${fmt.format(abs)}`;
}

export function pct(n: number | null | undefined, decimals = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(decimals)}%`;
}

export function prob(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return DEC2.format(n);
}

export function delta(n: number | null | undefined, opts: { compact?: boolean } = {}): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return mana(n, { signed: true, compact: opts.compact });
}

// "12:47:03" local time
export function clock(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    return new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z").toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso.slice(11, 19);
  }
}

// "5m ago", "1h ago", "3d ago" — single-letter unit, compact.
export function age(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const t = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z").getTime();
    const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  } catch {
    return "—";
  }
}

export function truncate(s: string, n: number): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export function pnlClass(n: number | null | undefined): string {
  if (n === null || n === undefined) return "text-text-3";
  if (n > 0) return "text-positive";
  if (n < 0) return "text-negative";
  return "text-text-3";
}
