# quantbots — Bot infrastructure for the private Manifold clone

## What this repo is

A **reusable framework** for building and running quant trading bots against the
company's private Manifold clone. It is shared infrastructure: bot authors write
*only* a strategy (`Strategy.estimate(group) -> {market_id: prob}`); the client,
sizing, execution, trade ledger, and PnL accounting are provided and should not
need to be touched.

This is **not** a single bot — it is the platform other people build bots on.
Keep the core (`manifold/`, `store/`, `sizing.py`, `runner.py`) generic and
strategy-agnostic. Strategy-specific logic lives only under `strategies/`.

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
  manifold/client.py     # Phase 0 — clone-only v0 client (auth + payloads). Correctness-critical.
  manifold/websocket.py  # Phase 2.5 — live price cache (optional extra: realtime)
  store/schema.sql       # trade / bot / pnl_snapshot / market_cache
  store/db.py            # connection + schema init
  store/trades.py        # append-only writes + position aggregation
  store/pnl.py           # realized/unrealized formulas (port verbatim) + daily snapshot
  sizing.py              # 1/3-push sizing + caps (port verbatim). Correctness-critical.
  runner.py              # the bot loop: load -> group -> estimate -> size -> execute -> record
  strategies/base.py     # Strategy ABC — the ONE seam bot authors implement
  strategies/surface_arb.py    # no-LLM example: fit a distribution to a strike ladder
  strategies/mean_reversion.py # no-LLM example: EMA mean reversion
  strategies/llm.py            # local-LLM example: percentile -> CDF
  strategies/ensemble.py       # fuse ingested observations -> fair value (deterministic)
  strategies/linker.py         # map market question -> source entity + threshold (heuristic)
  sources/base.py        # Source ABC: fetch() -> list[Observation] (external data)
  sources/{stooq,worldbank,rss}.py  # keyless feeds: prices / macro / news
  sources/ingest.py      # fetch configured sources -> observations cache
  llm/client.py          # OpenAI-compatible client pointed at a LOCAL endpoint
  llm/health.py          # Ollama wedge probe + watchdog
  config.py              # env/secret + bots.yaml loading
  cli.py                 # `quantbots` entry point
config/bots.yaml         # per-bot: strategy, limits, account env var
scripts/manual_trade.py  # Phase 1 connection smoke test
```

## How to add a bot (the only thing most authors do)

1. Write a `Strategy` subclass in `strategies/` implementing `estimate(group)`.
2. Register it in `strategies/__init__.py`'s `REGISTRY`.
3. Add an entry to `config/bots.yaml` pointing at the strategy + an account env var.
4. `quantbots run --bot <name> --dry-run` to validate, then drop `--dry-run`.

See `README.md` for the full guide.
