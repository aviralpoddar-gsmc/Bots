-- quantbots store schema (SQLite).
--
-- Design principle (ported from TAL): the `trade` table is an append-only
-- ledger and is the single source of truth. Positions and PnL are DERIVED from
-- it by aggregation — never mutate a position directly.

-- One row per bot you run.
CREATE TABLE IF NOT EXISTS bot (
    bot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT UNIQUE NOT NULL,
    enabled  INTEGER NOT NULL DEFAULT 1,
    strategy TEXT NOT NULL,          -- 'surface_arb', 'mean_reversion', 'llm', ...
    config   TEXT                    -- JSON: limits + strategy params
);

-- Append-only. THE source of truth for everything.
CREATE TABLE IF NOT EXISTS trade (
    trade_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id          INTEGER NOT NULL,
    market_id       TEXT NOT NULL,       -- manifold contractId
    platform_bet_id TEXT,                -- betId from the bet response
    trade_type      TEXT NOT NULL,       -- ENTRY | EXIT | PARTIAL_EXIT | RESOLUTION_CLOSE
    direction       TEXT NOT NULL,       -- YES | NO
    amount          REAL NOT NULL,       -- mana spent (ENTRY) or notional returned (EXIT)
    shares          REAL NOT NULL,       -- shares filled
    price_before    REAL,                -- prob before
    price_after     REAL,                -- prob after (or 1.0/0.0 on resolution)
    llm_estimate    REAL,                -- bot's fair-value estimate (nullable)
    reasoning       TEXT,
    date_executed   TEXT NOT NULL,       -- ISO-8601 UTC
    FOREIGN KEY (bot_id) REFERENCES bot(bot_id)
);
CREATE INDEX IF NOT EXISTS idx_trade_bot ON trade(bot_id);
CREATE INDEX IF NOT EXISTS idx_trade_pos ON trade(bot_id, market_id, direction);

-- Daily roll-up for the leaderboard.
CREATE TABLE IF NOT EXISTS pnl_snapshot (
    bot_id           INTEGER NOT NULL,
    snapshot_date    TEXT NOT NULL,      -- ISO date
    realized_pnl     REAL NOT NULL,
    unrealized_pnl   REAL NOT NULL,
    pnl              REAL NOT NULL,
    total_invested   REAL NOT NULL,
    open_positions   INTEGER NOT NULL,
    closed_positions INTEGER NOT NULL,
    PRIMARY KEY (bot_id, snapshot_date)
);

-- Optional cache of live-ish market state (populated from the API or websocket).
CREATE TABLE IF NOT EXISTS market_cache (
    market_id         TEXT PRIMARY KEY,
    question          TEXT,
    probability       REAL,
    total_liquidity   REAL,
    is_resolved       INTEGER NOT NULL DEFAULT 0,
    resolution        TEXT,
    close_time        INTEGER,
    last_updated_time INTEGER,
    raw_json          TEXT,
    updated_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_market_resolved ON market_cache(is_resolved);
