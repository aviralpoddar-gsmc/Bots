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

-- Ingested external data. Append-ish: dedup on (source, entity, ts). A numeric
-- observation sets `value`; a text one (news headline) sets `text`. `entity` is
-- the canonical key for the observed quantity (e.g. "WTI_OIL", "US_CPI_YOY").
CREATE TABLE IF NOT EXISTS observations (
    source      TEXT NOT NULL,
    entity      TEXT NOT NULL,
    ts          TEXT NOT NULL,     -- ISO-8601 of the observation time
    value       REAL,
    text        TEXT,
    payload     TEXT,              -- JSON of the raw record
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (source, entity, ts)
);
CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity, ts);
CREATE INDEX IF NOT EXISTS idx_obs_source ON observations(source, ts);

-- Adversarial comment judging (comments/judge.py). One row per judged comment —
-- the PRIMARY KEY doubles as the "already judged" dedupe. verdict follows the
-- Bridgewater rule: `unsound` ONLY when the comment's factual claims contradict
-- the supplied evidence, never on logic-vibes alone. replied_at / faded_at stay
-- NULL in dry-run; they record the live actions once those are enabled.
CREATE TABLE IF NOT EXISTS comment_verdict (
    comment_id     TEXT PRIMARY KEY,
    market_id      TEXT NOT NULL,
    author         TEXT,
    entity         TEXT,              -- linked entity (e.g. GOLD), if any
    verdict        TEXT NOT NULL,     -- sound | unsound | noise
    confidence     TEXT NOT NULL,     -- high | medium | low
    factual_errors TEXT,              -- JSON list of contradicted claims
    reply_draft    TEXT,              -- drafted reply (posted only when live)
    evidence       TEXT,              -- JSON evidence pack shown to the judge
    bet_id         TEXT,              -- the commenter's attached bet, if any
    bet_outcome    TEXT,              -- YES | NO
    bet_amount     REAL,
    judged_by      TEXT NOT NULL,     -- bot name
    judged_at      TEXT NOT NULL,
    replied_at     TEXT,
    faded_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_cv_market ON comment_verdict(market_id, judged_at);

-- Bridgewater-style consensus per market: mean of the bettor crowd's implied
-- probabilities, then Platt-extremized p̂ = σ(√3·logit(p̄)) (LLM/crowd forecasts
-- hedge toward 0.5; extremization was the paper's biggest calibration lever).
CREATE TABLE IF NOT EXISTS comment_consensus (
    market_id     TEXT PRIMARY KEY,
    p_mean        REAL NOT NULL,
    p_extreme     REAL NOT NULL,
    n_forecasters INTEGER NOT NULL,
    computed_at   TEXT NOT NULL
);
