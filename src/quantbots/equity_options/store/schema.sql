-- equity_options store (SQLite). SEPARATE DB from the clone store.
--
-- Same design principle as the parent: `option_trade` is an append-only ledger and
-- the single source of truth. Positions and PnL are DERIVED by aggregation.

-- Contract metadata cache (one row per OCC symbol seen).
CREATE TABLE IF NOT EXISTS option_contract (
    symbol      TEXT PRIMARY KEY,    -- OCC symbol
    underlying  TEXT NOT NULL,
    expiry      TEXT NOT NULL,       -- ISO date
    strike      REAL NOT NULL,
    kind        TEXT NOT NULL,       -- call | put
    multiplier  INTEGER NOT NULL DEFAULT 100,
    last_mid    REAL,
    last_iv     REAL,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_oc_underlying ON option_contract(underlying);

-- Append-only ledger. ONE ROW PER LEG. Legs of one structure share `ticket_id`.
CREATE TABLE IF NOT EXISTS option_trade (
    trade_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       TEXT NOT NULL,       -- groups legs placed together
    underlying      TEXT NOT NULL,
    structure       TEXT NOT NULL,       -- long_call | bull_call_spread | ...
    symbol          TEXT NOT NULL,       -- OCC symbol of THIS leg
    trade_type      TEXT NOT NULL,       -- ENTRY | EXIT | EXPIRY_CLOSE
    side            TEXT NOT NULL,       -- BUY | SELL
    qty             INTEGER NOT NULL,    -- contracts (>0)
    fill_price      REAL,                -- per-share premium of this leg
    amount          REAL NOT NULL,       -- cash flow: negative=paid, positive=received
    multiplier      INTEGER NOT NULL DEFAULT 100,
    broker          TEXT NOT NULL,       -- dry | paper | live
    broker_order_id TEXT,
    status          TEXT NOT NULL,       -- intended | submitted | filled | canceled | rejected
    edge            REAL,                -- per-share model edge at entry
    forecast_vol    REAL,
    reasoning       TEXT,
    date_executed   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ot_ticket ON option_trade(ticket_id);
CREATE INDEX IF NOT EXISTS idx_ot_symbol ON option_trade(symbol);
CREATE INDEX IF NOT EXISTS idx_ot_underlying ON option_trade(underlying);

-- Portfolio-level greek snapshot (for risk monitoring / dashboard).
CREATE TABLE IF NOT EXISTS greeks_snapshot (
    snapshot_ts  TEXT PRIMARY KEY,
    net_delta    REAL NOT NULL,
    net_gamma    REAL NOT NULL,
    net_vega     REAL NOT NULL,
    net_theta    REAL NOT NULL,
    positions    INTEGER NOT NULL,
    premium_at_risk REAL NOT NULL
);

-- Daily PnL roll-up for the leaderboard / equity curve.
CREATE TABLE IF NOT EXISTS option_pnl_snapshot (
    snapshot_date    TEXT PRIMARY KEY,
    realized_pnl     REAL NOT NULL,
    unrealized_pnl   REAL NOT NULL,
    pnl              REAL NOT NULL,
    premium_at_risk  REAL NOT NULL,
    open_positions   INTEGER NOT NULL,
    closed_positions INTEGER NOT NULL
);
