import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router";
import { TooltipProvider } from "@/components/ui/tooltip";
import { NavSidebar } from "@/components/nav-sidebar";
import { useStream } from "@/hooks/use-stream";
import { FleetPage } from "@/pages/fleet";
import { BotDetailPage } from "@/pages/bot-detail";
import { FeedPage } from "@/pages/feed";
import { StrategiesPage, StrategyDetailPage } from "@/pages/strategies";
import { MarketsPage } from "@/pages/markets";
import { createContext, useContext } from "react";
import type { Snapshot } from "@/lib/types";

interface StreamCtx {
  data: Snapshot | null;
  connected: boolean;
  refresh: () => Promise<void>;
}

const StreamContext = createContext<StreamCtx>({
  data: null,
  connected: false,
  refresh: async () => {},
});

export function useSnapshot() {
  return useContext(StreamContext);
}

function Shell() {
  const stream = useStream();
  return (
    <StreamContext.Provider value={{ data: stream.data, connected: stream.connected, refresh: stream.refresh }}>
      <div className="flex min-h-screen">
        <NavSidebar snapshot={stream.data} connected={stream.connected} />
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>
    </StreamContext.Provider>
  );
}

export default function App() {
  return (
    <TooltipProvider delayDuration={300}>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<FleetPage />} />
            <Route path="/bots/:name" element={<BotDetailPage />} />
            <Route path="/feed" element={<FeedPage />} />
            <Route path="/strategies" element={<StrategiesPage />} />
            <Route path="/strategies/:name" element={<StrategyDetailPage />} />
            <Route path="/markets" element={<MarketsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  );
}
