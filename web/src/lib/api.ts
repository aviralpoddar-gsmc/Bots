import type {
  BotDetail,
  EventFeedRow,
  MarketRow,
  Snapshot,
  StrategyInfo,
} from "./types";

const API = "/api";

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${path}`);
  return (await r.json()) as T;
}

export const api = {
  snapshot: () => jget<Snapshot>("/snapshot"),
  bot: (name: string) => jget<BotDetail>(`/bots/${encodeURIComponent(name)}`),
  feed: (limit = 200) => jget<EventFeedRow[]>(`/feed?limit=${limit}`),
  strategies: () => jget<StrategyInfo[]>("/strategies"),
  strategy: (name: string) => jget<StrategyInfo>(`/strategies/${encodeURIComponent(name)}`),
  markets: (params?: { page?: number; size?: number; q?: string; min_resolvability?: number }) => {
    const sp = new URLSearchParams();
    if (params?.page) sp.set("page", String(params.page));
    if (params?.size) sp.set("size", String(params.size));
    if (params?.q) sp.set("q", params.q);
    if (params?.min_resolvability !== undefined)
      sp.set("min_resolvability", String(params.min_resolvability));
    const qs = sp.toString();
    return jget<{ rows: MarketRow[]; total: number; page: number; size: number }>(
      `/markets${qs ? `?${qs}` : ""}`,
    );
  },
  refresh: async () => {
    const r = await fetch(`${API}/refresh`, { method: "POST" });
    if (!r.ok) throw new Error(`refresh failed: ${r.status}`);
    return r.json();
  },
};
