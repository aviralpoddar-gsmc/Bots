# quantbots — Bot infrastructure for the private Manifold clone

## What this repo is

A **reusable framework** for building and running quant trading bots against the
company's private Manifold clone. It is shared infrastructure: bot authors write
*only* a strategy (`Strategy.estimate(group) -> {market_id: prob}`); the client,
sizing, execution, trade ledger, and PnL accounting are provided and should not
need to be touched.

This is **not** a single bot — it is the platform other people build bots on.
Keep the core (`manifold/`, `store/`, `sizing.py`, `portfolio.py`,
`resolvability.py`, `runner.py`) generic and strategy-agnostic. Strategy-specific
logic lives only under `strategies/`.

## The platform (fixed facts)

A self-hosted fork of Manifold Markets, so the standard **Manifold v0 REST API**
applies. Play-money mana (M$), CPMM market maker, no real money.

| Property        | Value                                          |
| --------------- | ---------------------------------------------- |
| API base        | `https://manifold.mikhailtal.dev/api/v0`       |
| Web base        | `https://manifold.mikhailtal.dev`              |
| WebSocket       | `wss://manifold.mikhailtal.dev/api/ws`         |
| Bet auth header | `Authorization: Key <MANIFOLD_CLONE_API_KEY>`  |
| Access headers  | `CF-Access-Client-Id`, `CF-Access-Client-Secret` |
| Rate limit      | 500 requests / minute / IP                     |

Every request needs the two Cloudflare Access headers **and** the API key.

## ⚠️ The cancellation reality (read before reasoning about PnL)

The clone has **no oracle** — resolution is a creator action (an automated job on
the operator's account). Of 9,578 resolved markets, **~93% resolve CANCEL/N-A
(bets refunded), only ~7% YES/NO.** A market settles YES/NO only when its named
source actually publishes a verifiable value: commodity **prices** (LBMA/LME/COMEX)
resolve reliably (LBMA precious metals ~100%); company **production/demand** figures
almost never do (~0%). Consequences:

- **Paper EV ≠ realized edge.** Any expected-profit number that ignores
  cancellation overstates reality. The core multiplies EV by a resolvability score
  (`resolvability.py`) so bots concentrate capital where edge actually pays out.
- Cancellation refunds bets (~capital-neutral), so the cost of a cancel-prone
  market is mostly *opportunity cost* (tied-up budget), not loss.
- The deepest, most persistent mispricings tend to be in markets that **won't**
  resolve (nobody corrects them) — chasing raw edge is an adverse-selection trap.

This is the single most important fact for setting expectations. See
`resolvability.py` and the README "Cancellation reality" section.

## ⚠️ Safety rules (non-negotiable)

- **Trade on the clone ONLY.** The client is hard-coded to the clone base URL and
  must never accept a "platform" argument that could point at public
  `manifold.markets`. Do not add one.
- **Default to `dry_run=True`** in any new entry point until a human confirms.
- A `dry_run` bet validates auth + payload without moving mana — the safest first
  call.

## Local compute only (for LLM strategies)

LLM strategies must use **locally-running models** (Ollama / llama.cpp / vLLM /
LiteLLM proxy → local) — no hosted inference APIs (OpenAI/Anthropic/Gemini cloud)
until bots are demonstrably profitable. The `llm/` client speaks the
OpenAI-compatible protocol but points at a **local** endpoint. Flag anything that
would require hosted inference.

## Secrets

Provided via env / Doppler — never commit them. See `.env.example`:
`MANIFOLD_CLONE_API_KEY`, `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`, and
optionally `MANIFOLD_CLONE_ADMIN_API_KEY`.

## Layout

```
src/quantbots/
  manifold/client.py     # clone-only v0 client (auth + payloads, rate limit, retries). Correctness-critical.
  manifold/websocket.py  # live price cache (optional extra: realtime)
  store/schema.sql       # trade / bot / pnl_snapshot / market_cache
  store/db.py            # connection + schema init + PnL/snapshot helpers
  store/trades.py        # append-only writes + position aggregation
  store/pnl.py           # realized/unrealized formulas + daily snapshot
  sizing.py              # 1/3-push per-market sizing + caps. Correctness-critical.
  portfolio.py           # CORE: EV-per-mana ranking, concentration caps, realized-EV allocation
  resolvability.py       # CORE: cancellation-aware score P(market resolves YES/NO) from question text
  backtest.py            # replay a strategy on historical series -> Brier/calibration/PnL
  runner.py              # the bot loop: load -> prefilter -> group -> estimate -> size -> allocate -> execute -> record
  strategies/base.py     # Strategy ABC — the ONE seam bot authors implement (estimate/prefilter/group/correlation_key)
  strategies/_model.py   # shared lognormal helpers (norm_cdf, years_to_close)
  strategies/ladder.py   # parse threshold/direction + measurable key from question text
  strategies/linker.py   # map market question -> source entity + threshold (heuristic)
  strategies/mean_reversion.py # no-LLM example: EMA mean reversion
  strategies/surface_arb.py    # no-LLM: fit a normal CDF to a strike ladder (quant extra)
  strategies/ensemble.py       # fuse ingested observations -> fair value (deterministic)
  strategies/enso.py           # climate bot: ENSO/ONI markets, Gaussian persistence
  strategies/commodity_futures.py  # soft-commodity (ag) futures price markets, lognormal
  strategies/commodity_spot.py # LIVE: metals/energy spot-price ladders, strict unit/currency guard
  strategies/ladder_arb.py     # LIVE: model-free monotonicity arb across thresholds (isotonic/PAVA)
  strategies/term_structure.py # LIVE: time-axis coherence — smooth stale dates from traded neighbours
  strategies/llm.py            # local-LLM example: percentile -> CDF
  sources/base.py        # Source ABC: fetch() -> list[Observation] (external data)
  sources/{stooq,fred,noaa,worldbank,rss}.py  # keyless feeds: prices+equities+softs / US macro / climate / global macro / news
  sources/ingest.py      # fetch configured sources -> observations cache
  llm/{client,health,bench}.py # local OpenAI-compatible client + Ollama watchdog + model benchmark
  config.py              # env/secret + bots.yaml/sources.yaml loading (merges sizing.DEFAULT_LIMITS)
  cli.py                 # `quantbots` entry point (health/refresh/ingest/run/status/resolve/snapshot/backtest/llm-bench)
  dashboard/server.py    # Flask: serves web/dist bundle + JSON+SSE API at /api/*
  dashboard/data.py      # read-only aggregations over the SQLite store
config/bots.yaml         # per-bot: strategy, limits, account env var
config/sources.yaml      # per-source: feed config
scripts/manual_trade.py  # connection smoke test
scripts/daily_cycle.sh   # ops: resolve -> refresh -> ingest -> run --live -> snapshot (all bots)
scripts/com.quantbots.daily.plist  # launchd agent: run daily_cycle.sh at 09:00 daily
web/                     # Vite + React 19 + TS + Tailwind v4 dashboard. `bun install && bun run build` → web/dist/, served by Flask at /.
```

## Dashboard

`quantbots dashboard` serves the React SPA from `web/dist/` plus a JSON+SSE API
at `/api/*`. Pages: `/` (fleet leaderboard), `/bots/[name]` (detail + equity
curve), `/feed` (live trade tape), `/strategies` + `/strategies/[name]`,
`/markets`. The SSE channel `/api/stream` pushes a fresh snapshot every 5 s.

Dev: `cd web && bun run dev` (Vite on :5173 proxies `/api/*` to Flask :8000).
Build: `cd web && bun run build` (output: `web/dist/`).

**Always read `DESIGN.md` before changing dashboard visuals.** It defines the
Mission Control aesthetic — IBM Plex Mono+Sans, cyan signal accent, green/red
PnL on a near-black surface — and the anti-slop rules (no purple gradients, no
3-column SaaS icon grids, no Inter/Roboto). Deviations need explicit approval.

## How to add a bot (the only thing most authors do)

1. Write a `Strategy` subclass in `strategies/` implementing `estimate(group)`.
   Optionally override `prefilter`, `group`, and `correlation_key` (the last lets
   the portfolio allocator cap exposure across correlated markets).
2. Register it in `strategies/__init__.py`'s `_REGISTRY`.
3. Add an entry to `config/bots.yaml` pointing at the strategy + an account env var.
4. `quantbots run --bot <name>` to validate (dry-run by default), then add `--live`.

The runner automatically applies the portfolio allocator (EV ranking, concentration
caps) and resolvability weighting to every bot — authors don't wire those up.

See `README.md` for the full guide.
