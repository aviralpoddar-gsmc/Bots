import { useEffect, useState } from "react";
import type { Snapshot } from "@/lib/types";
import { api } from "@/lib/api";

interface StreamState {
  data: Snapshot | null;
  error: string | null;
  connected: boolean;
  lastUpdate: number | null;
}

// Subscribes to /api/stream (SSE) and falls back to /api/snapshot polling on failure.
// Snapshot updates fire every 5 s server-side.
export function useStream(): StreamState & { refresh: () => Promise<void> } {
  const [state, setState] = useState<StreamState>({
    data: null,
    error: null,
    connected: false,
    lastUpdate: null,
  });

  async function refresh() {
    try {
      const snap = await api.snapshot();
      setState((s) => ({ ...s, data: snap, error: null, lastUpdate: Date.now() }));
    } catch (e) {
      setState((s) => ({ ...s, error: (e as Error).message }));
    }
  }

  useEffect(() => {
    let es: EventSource | null = null;
    let mounted = true;
    let pollTimer: number | null = null;

    function startPolling() {
      if (pollTimer != null) return;
      const tick = async () => {
        if (!mounted) return;
        await refresh();
        pollTimer = window.setTimeout(tick, 10_000);
      };
      tick();
    }

    try {
      es = new EventSource("/api/stream");
      es.onopen = () => {
        if (!mounted) return;
        setState((s) => ({ ...s, connected: true, error: null }));
      };
      es.onmessage = (ev) => {
        if (!mounted) return;
        try {
          const snap = JSON.parse(ev.data) as Snapshot;
          setState((s) => ({ ...s, data: snap, lastUpdate: Date.now(), error: null }));
        } catch (e) {
          // ignore single bad frame
        }
      };
      es.onerror = () => {
        if (!mounted) return;
        setState((s) => ({ ...s, connected: false }));
        // SSE will auto-retry; also kick off polling as backup
        startPolling();
      };
    } catch {
      startPolling();
    }
    // Initial fetch for instant first paint.
    refresh();

    return () => {
      mounted = false;
      if (es) es.close();
      if (pollTimer != null) window.clearTimeout(pollTimer);
    };
  }, []);

  return { ...state, refresh };
}
