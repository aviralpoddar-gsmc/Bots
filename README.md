# quantbots

Reusable framework for building **quant trading bots** on the company's private
Manifold clone (`manifold.mikhailtal.dev`). It's play-money (mana, M$), CPMM, no
real money — a bot sandbox.

The whole point: **a bot author writes only a strategy.** The client, sizing,
execution, trade ledger, and PnL accounting are shared infrastructure you don't
touch. The seam is one method:

```python
class Strategy:
    def estimate(self, group: list[Market]) -> dict[str, float]:
        """Return {market_id: your_fair_value_probability}."""
```

---

## ⚠️ Safety rules (read first)

- **Clone only.** `ManifoldClient` is hard-wired to the clone URL. There is no
  "platform" argument and you must never add one — bots must never touch public
  `manifold.markets`.
- **Dry-run by default.** `quantbots run` validates orders without moving mana
  unless you pass `--live`. A `dry_run` bet is the safest possible first call.
- **Local compute only for LLMs.** LLM strategies use a *local* model (Ollama /
  llama.cpp / vLLM / local LiteLLM). No hosted inference until bots are
  profitable.

---

## Setup

```bash
uv sync                       # core only
uv sync --extra quant         # + numpy/scipy (surface_arb)
uv sync --extra llm           # + openai client (local LLM strategies)
uv sync --extra realtime      # + websockets (live price cache)

cp .env.example .env          # then fill in secrets (or source from Doppler)
```

Secrets (env / Doppler — never commit): `MANIFOLD_CLONE_API_KEY`,
`CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`. See **What to request from the
team** below if you don't have these.

---

## Phase 1 — prove the connection

```bash
quantbots health                                   # calls /me; proves key + CF Access
python scripts/manual_trade.py --slug <open-slug>            # dry-run a 10-mana bet
python scripts/manual_trade.py --slug <open-slug> --execute  # place it for real
```

When `health` prints your username + balance, auth works.

## Phase 2 — run a bot

```bash
quantbots refresh                      # pull markets into the local SQLite cache
quantbots ingest                       # pull external data sources into the cache
quantbots run --bot surface_arb_1      # DRY-RUN: prints intended orders, no mana moved
quantbots run --bot surface_arb_1 --live   # actually trade
quantbots resolve --bot surface_arb_1  # close out any resolved positions
quantbots snapshot                     # roll up PnL + print the leaderboard
```

`run` defaults to dry-run. Always dry-run a new bot first and eyeball the signals.

## Data sources (information bots trade on)

Bots trade on *information*: external feeds are ingested into the store's
`observations` table, and strategies read them to form a fair-value view of a
market. Sources mirror the strategy pattern — small independent modules under
`sources/`, listed by `quantbots sources`, configured in `config/sources.yaml`,
fetched by `quantbots ingest`.

Built-in (keyless): `stooq` (commodity/FX/index prices + equities + soft-commodity
futures), `fred` (US macro series via the public CSV — mortgage rate, housing
starts, ...), `noaa` (climate indices — ENSO/Oceanic Niño Index), `worldbank`
(global macro: CPI, GDP, unemployment), `rss` (news headlines). To add one:
implement a `Source` subclass (`fetch() -> list[Observation]`), register it in
`sources/__init__.py`, and add it to `config/sources.yaml`.

An **Observation** is the normalized unit — `value` for numbers, `text` for news,
keyed by `entity` (the canonical thing observed, e.g. `WTI_OIL`, `US_CPI_YOY`) so
different feeds can describe the same quantity. Run `ingest` on its own schedule
(cron / `quantbots`-in-a-loop), independent of trading.

> Note: a data-source *API key* (FRED, a news API) is fine — that's data, not
> hosted inference. The local-compute-only rule applies to the *model/reasoning*
> step, not to pulling data.

---

## How to add a bot (the only thing most authors do)

1. **Write the strategy.** Create `src/quantbots/strategies/my_strategy.py`:

   ```python
   from .base import Market, Strategy

   class MyStrategy(Strategy):
       name = "my_strategy"

       def estimate(self, group: list[Market]) -> dict[str, float]:
           # `group` is a list of raw Manifold market dicts. Return your fair
           # value (0..1) per market id. Omit a market to abstain.
           return {m["id"]: my_fair_value(m) for m in group}

       # Optional: narrow the universe / decide what's evaluated together.
       # def prefilter(self, markets): ...
       # def group(self, markets): ...
   ```

2. **Register it** — add one line to `src/quantbots/strategies/__init__.py`:

   ```python
   "my_strategy": "quantbots.strategies.my_strategy:MyStrategy",
   ```

3. **Configure it** in `config/bots.yaml`:

   ```yaml
   - name: my_bot
     strategy: my_strategy
     account_env: MANIFOLD_CLONE_API_KEY   # env var holding this bot's key
     limits: { max_order_size: 50, hold_band: 0.05 }
     params: { my_param: 1.0 }             # passed to MyStrategy(**params)
   ```

4. **Run it:** `quantbots run --bot my_bot` (dry-run), then `--live`.

That's it. You never touch the client, sizing, ledger, or PnL.

---

## What the framework gives you

| Piece | Where | What it does |
| --- | --- | --- |
| Connection | `manifold/client.py` | Clone-only v0 client: auth + CF Access headers, rate limiting, retries, bet/sell. |
| Sizing | `sizing.py` | "Push 1/3 toward your estimate," capped by order size, liquidity %, and price impact. The gap = conviction. |
| Bot loop | `runner.py` | load → prefilter → group → `estimate` → size → execute → record. |
| Ledger + PnL | `store/` | Append-only `trade` table is the source of truth; positions & PnL are derived. SQLite. |
| Strategies | `strategies/` | `surface_arb` (distribution fit), `mean_reversion` (EMA), `llm` (local model). |
| Local LLM | `llm/` | OpenAI-compatible client pointed at a local endpoint + Ollama health watchdog. |

**Sizing in one line:** target price = `current + (estimate - current)/3`, then the
order is the *min* of {mana to reach target, max order size, liquidity %, max
price-impact move}. No Kelly, no confidence knob.

**PnL model:** YES share → worth `prob`, NO share → worth `1 - prob`. Realized
from EXIT rows, unrealized by marking remaining shares (per-share cost basis, so
`realized + unrealized` = true total). Resolution = a synthetic `RESOLUTION_CLOSE`
trade at prob 1.0/0.0 — no special-case code.

---

## Reference strategies

- **`surface_arb`** (no LLM, `quant` extra) — fits a normal CDF to a measurable's
  strike ladder and trades strikes back toward the fitted curve / monotonicity.
  The "stat-arb" starting point. Needs threshold/direction parsed from questions
  (`strategies/ladder.py` does this heuristically).
- **`mean_reversion`** (no deps) — fades a market toward an EMA of its own price.
  Simplest reference implementation.
- **`ensemble`** (no LLM) — trades on ingested data. The **linker**
  (`strategies/linker.py`) maps a market's question to source `entity` keys +
  threshold/direction; the strategy converts each linked observation into
  `P(quantity clears threshold)` (lognormal, tunable `sigma`) and combines them as
  a weighted average. Single-source = restrict `entity_map` to one entity;
  multi-source = leave the full map. Has a plausibility guard (`max_ratio`) and
  exclusion keywords so it skips mis-linked markets instead of trading bogus
  signals. Inspect coverage with `quantbots link`.
- **`enso`** (no LLM) — climate bot on ENSO/Oceanic Niño Index markets, fed by
  `noaa`. Uses a **Gaussian persistence** model (additive, handles negative
  values) — a different model from `ensemble`'s lognormal, because the ONI isn't
  a positive price. Self-contained linking (only ONI markets).
- **`commodity_futures`** (no LLM) — soft-commodities bot on ICE/CBOT ag-futures
  price markets (cotton, sugar, wheat, corn, cocoa), fed by `stooq`. Lognormal
  price-threshold model with its own catalog; self-contained.
- **`llm`** (`llm` extra, local model) — one call per measurable returns a
  percentile distribution; each strike is read off the interpolated CDF. The
  "make the bot smarter" move: reason about the *quantity*, not 30 yes/no
  questions.

Each data-driven bot does its **own** linking and only acts on its domain's
markets, so multiple bots can run side by side without stepping on each other.

### Linking & market coverage

Trading on data hinges on **linking** a market to the right source `entity`. The
linker (`strategies/linker.py`) is deterministic, with three matchers:

1. **Stock tickers** — `"... (WULF) stock price ..."` → `STOCK_WULF` (auto-general,
   paired with a Stooq `wulf.us` feed). No per-ticker curation.
2. **Curated catalog** — a precise phrase → exact series entity (e.g. single-family
   housing starts → `FRED_HOUST1F`). Add a line per series as feeds are added.
3. **Commodity keywords** — broad names, suppressed by exclusion keywords when the
   question is about an out-of-scope metric (volume/production/reserves/share).

We started with the cleanest deterministic numeric clusters: **US macro via FRED**
(30y mortgage rate, housing starts) and **equity prices via Stooq** (WULF). These
link cleanly because the entity + threshold + unit parse unambiguously. Against the
live set the linker now maps ~48 markets and the `ensemble` bot produces sensible
data-driven estimates on them.

The platform's *long tail* (rare-earth balances in t REO, company revenue/leverage,
niche indices like Cotlook A) needs **(a)** domain-specific sources matching those
quantities and **(b)** semantic linking (local-LLM/embeddings) to distinguish
entity + metric + unit + date. The pipeline — sources → observations → linker →
ensemble → sizing → execution — is done and validated end-to-end; widening it is
now a matter of adding sources + catalog entries.

---

## Choosing a local model (`quantbots llm-bench`)

"Which local model should the `llm` bot use?" is answered empirically, not by
vibes. `quantbots llm-bench` asks each candidate model for a percentile
distribution of quantities we **already know the true value of** (from the
observations cache — mortgage rate, cotton, ONI, gold, oil, ...), then scores:

- **validity** — fraction returning parseable, monotonic percentiles
- **coverage** — fraction where the true value lands in the model's p10–p90
  (calibration; well-calibrated ≈ 80%)
- **p50 error** — median point accuracy
- **latency** — avg seconds per forecast

```bash
quantbots ingest                 # get ground-truth values first
quantbots llm-bench --models qwen3:8b,gemma3:latest,gemma4:latest
```

Fully local and reproducible. The `llm` bot's edge is the **long tail** the
deterministic bots can't link (no data feed) — it reasons about the underlying
quantity, returns a percentile distribution, and reads each strike off the CDF.

## Local LLM hosts (gotchas)

- Set `num_ctx=32768` — Ollama defaults to 2048 and **silently truncates**,
  breaking JSON mode (`llm/client.py` does this for you).
- Ollama can wedge: `/api/tags` still 200s while generation is stuck. Probe with
  `llm/health.py` (hits `/api/generate`) and restart the server on failure. Set
  `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_MAX_QUEUE=32`.

---

## Testing

```bash
uv run pytest          # core (sizing, pnl, ledger, ladder) — no extras needed
```

---

## What to request from the team (can't self-serve)

- **Cloudflare Access service token** (`CF_ACCESS_CLIENT_ID` / `_SECRET`) scoped
  for your machine/CI.
- A **`MANIFOLD_CLONE_API_KEY`** for a bot account (a `@*Manual` key works to
  start), or `MANIFOLD_CLONE_ADMIN_API_KEY` to mint dedicated bot accounts + mana.
- Confirmation of **which markets your bots may trade** (a test tag or a few
  low-liquidity markets) so you don't step on the production fleet.

---

## Status / roadmap

- ✅ Phase 0–2 infra: client, store, sizing, runner, PnL, two no-LLM strategies,
  CLI, dry-run path. Validated against synthetic fixtures + a fake client.
- ⏳ Not yet exercised against the live clone (needs CF Access + key).
- ⏳ `manifold/websocket.py` live price cache (scaffold for the `realtime` extra).
- ⏳ Multi-model LiteLLM proxy + mana-per-dollar leaderboard (Phase 3 stretch).
