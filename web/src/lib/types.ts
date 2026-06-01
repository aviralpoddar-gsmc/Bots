// Types mirror src/quantbots/dashboard/data.py shapes.

export type BotStatus = "LIVE" | "PAUSED" | "DISABLED";

export interface SystemStatus {
  username: string | null;
  balance: number | null;
  totalDeposits: number | null;
  latency_ms: number | null;
  status: "LIVE" | "DEGRADED" | "UNKNOWN";
}

export interface Overview {
  n_bots: number;
  n_enabled: number;
  n_live: number;
  total_pnl: number;
  total_realized: number;
  total_unrealized: number;
  total_invested: number;
  active_capital: number;
  open_positions: number;
  closed_positions: number;
  total_wins: number;
  total_losses: number;
  total_refunds: number;
  total_mana_traded: number;
  n_trades: number;
  n_entries: number;
  roi: number | null;
  avg_pnl_per_trade: number;
  account_balance?: number;
  account_deposits?: number;
  account_profit?: number;
  daily_profit?: number | null;
  // "manifold" when the headline numbers came straight from get-user-portfolio;
  // "ledger" when the API was unreachable and we fell back to the local store.
  source?: "manifold" | "ledger";
  // Ledger view, kept for drift diagnostics when source === "manifold".
  ledger_pnl?: number;
  ledger_active_capital?: number;
  pnl_drift?: number;
}

export interface RiskBlock {
  exposure_tier: "Low" | "Medium" | "High" | "Unbounded";
  exposure_pct_of_cap: number;
  exposure_cap: number;
}

export interface BotRow extends RiskBlock {
  name: string;
  strategy: string;
  strategy_class: string;
  enabled: boolean;
  status: BotStatus;
  exists: boolean;
  description: string;
  pnl: number;
  realized: number;
  unrealized: number;
  invested: number;
  invested_mark: number;
  open: number;
  closed: number;
  wins: number;
  losses: number;
  refunds: number;
  win_rate: number | null;
  n_entries: number;
  total_mana_traded: number;
  avg_size: number;
  avg_edge: number | null;
  yes_entries: number;
  no_entries: number;
  last_trade_at: string | null;
  max_drawdown: number;
  n_trades_all: number;
  bot_id: number | null;
}

export interface RecentTrade {
  ts: string;
  market_id: string;
  question: string;
  direction: "YES" | "NO";
  amount: number;
  price_before: number | null;
  price_after: number | null;
  estimate: number | null;
}

export interface Exposure {
  key: string;
  amount: number;
  pct_of_max: number;
}

export interface PnlPoint {
  date: string;
  pnl: number;
}

export interface EquityPoint {
  ts: string;
  pnl: number;
}

export interface BotDetail extends BotRow {
  limits: Record<string, unknown>;
  params: Record<string, unknown>;
  recent_trades: RecentTrade[];
  exposures: Exposure[];
  pnl_series: PnlPoint[];
  concentration_pct: number;
  total_open_exposure: number;
  inventory_bias: [string, number] | null;
}

export interface StrategyDistribution {
  label: string;
  amount: number;
  pct: number;
}

export interface EventFeedRow {
  ts: string;
  bot: string;
  type: string;
  direction: "YES" | "NO";
  amount: number;
  market_id: string;
  question: string;
  price_before: number | null;
  price_after: number | null;
  estimate: number | null;
  reasoning: string | null;
}

export interface Snapshot {
  overview: Overview;
  leaderboard: BotRow[];
  events: EventFeedRow[];
  distribution: StrategyDistribution[];
  equity: EquityPoint[];
  system: SystemStatus;
  cache_age_s: number | null;
  ts: string;
}

export interface StrategyInfo {
  name: string;
  class: string;
  description: string;
  bots: string[];
  total_pnl: number;
  total_trades: number;
  live_count: number;
}

export interface MarketRow {
  id: string;
  question: string;
  market_type: string;
  current_prob: number | null;
  close_time: string | null;
  total_liquidity: number | null;
  resolvability: number | null;
  traded_by: string[];
}
