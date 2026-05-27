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

## ⚠️ The cancellation reality (read this before trusting any PnL number)

The clone has **no oracle** — a market is resolved by an automated job on the
operator's account, and it only settles **YES/NO when the named source actually
publishes a verifiable value.** Measured over 9,578 resolved markets:

- **~93% resolve CANCEL/N-A** (bets refunded), only **~7% resolve YES/NO.**
- Which ones is **predictable**: commodity **prices** resolve (LBMA precious metals
  **~100%**, LME ~39%, generic spot ~22%); company **production/demand** figures
  almost never do (**~0–1%**).

So **paper EV massively overstates realized edge.** The framework handles this for
you: `resolvability.py` scores each market's P(resolve), and the allocator ranks and
gates on **realized EV = paper EV × resolvability**, concentrating capital on
markets that actually settle (set a per-bot `min_resolvability` floor to hard-skip
the rest). Cancellation refunds bets, so the cost of a cancel-prone market is mostly
*opportunity cost* (tied-up budget), not loss — but the deepest, most persistent
mispricings tend to be exactly the markets that **won't** resolve, so chasing raw
edge is an adverse-selection trap. **When you reason about a bot's profit, reason in
realized (resolvability-weighted) terms.**

## Current live fleet (as of 2026-05-27)

Three bots run live, automatically, via the daily cycle — all trade only markets
likely to resolve, size to liquidity, and share the portfolio allocator:

| Bot | Strategy | Edge | Universe |
| --- | --- | --- | --- |
| `commodity_spot_1` | `commodity_spot` | Data-anchored fair value: prices the ladder off live spot + horizon-scaled vol | gold/silver/platinum/palladium/copper/WTI/Brent/RBOB spot-price markets |
| `ladder_arb_1` | `ladder_arb` | Model-free: enforces monotonicity across thresholds on one date | any numeric ladder (concentrated on price ladders by resolvability) |
| `term_structure_1` | `term_structure` | Model-free: smooths a metric+threshold across resolution dates, filling stale 0.50s from traded neighbours | same, time axis |

Monitor with `quantbots status`. They're scheduled by
`scripts/com.quantbots.daily.plist` (launchd, 09:00 daily) running
`scripts/daily_cycle.sh`; `launchctl unload …` to stop, `QUANTBOTS_LIVE=0` for a dry
cycle. **Both scripts hardcode the repo path** (`/Users/mikhail/Bots`) — set
`QUANTBOTS_REPO` / edit the plist to match your machine before installing.

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
quantbots refresh --limit 70000        # pull the whole market universe into the local SQLite cache
quantbots ingest                       # pull external data sources into the cache
quantbots run --bot commodity_spot_1   # DRY-RUN: prints the allocated book, no mana moved
quantbots run --bot commodity_spot_1 --live   # actually trade
quantbots status                       # dashboard: balance, per-bot PnL, exposure by underlying
quantbots resolve --bot commodity_spot_1  # realize PnL on resolved positions
quantbots snapshot                     # roll up PnL + print the leaderboard
```

`run` defaults to dry-run. Always dry-run a new bot first and eyeball the book (it
prints `funded X of Y candidates`, staked mana, realized expected profit, and the
number of correlation groups). The default cache is only the newest ~8.8k markets —
`refresh --limit 70000` caches all ~62k so bots see the full universe.

To run everything on a schedule, use `scripts/daily_cycle.sh` (see "Cancellation
reality" above for the launchd setup).

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
| Portfolio allocator | `portfolio.py` | Ranks signals by **realized** EV/mana, funds best-first under budget + per-correlation-group concentration caps. Turns thousands of per-market signals into one capital-efficient book. |
| Resolvability | `resolvability.py` | Scores P(market resolves YES/NO vs CANCEL); the allocator weights EV by it so capital lands on markets that actually settle. See "Cancellation reality." |
| Bot loop | `runner.py` | load → prefilter → group → `estimate` → size → allocate → execute → record. Live executor retries throttled bets with backoff. |
| Ledger + PnL | `store/` | Append-only `trade` table is the source of truth; positions & PnL are derived. SQLite. |
| Strategies | `strategies/` | Live: `commodity_spot`, `ladder_arb`, `term_structure`. Reference: `surface_arb`, `mean_reversion`, `ensemble`, `enso`, `commodity_futures`, `llm`. |
| Local LLM | `llm/` | OpenAI-compatible client pointed at a local endpoint + Ollama health watchdog. |
| Ops | `scripts/` | `daily_cycle.sh` (resolve→refresh→ingest→run→snapshot) on a launchd daily schedule; `quantbots status` dashboard. |

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
- **`commodity_spot`** (no LLM) — trades genuine *spot-price* threshold ladders for
  the metals/energy commodities we have a live Stooq feed for (gold, silver,
  platinum, palladium, copper, WTI, Brent, RBOB gasoline). Same lognormal family as
  `commodity_futures`, but with a **strict unit/currency guard**: a commodity only
  matches when the threshold's quoted unit confirms the right benchmark *and*
  currency, and operational metrics (production/demand/AISC/spread/...) are
  hard-excluded. This is what keeps it out of confidently-wrong bets — silver feed
  is cents/oz (÷100→$/oz), copper is cents/lb (×22.0462→$/MT), "natural gas" markets
  are EUR/MWh European gas (excluded, wrong benchmark), palladium "koz" is a volume,
  nickel "CNY/t" is yuan, "zinc sulfate" is a chemical, not LME zinc. Horizon-capped
  to ~1y (where the zero-drift model is backtested). **Live since 2026-05-27.**
- **`ladder_arb`** (no LLM, **no extras** — pure stdlib) — model-free structural
  arbitrage. Within each (metric, resolution-date) ladder, survival P(value >
  threshold) must be non-increasing in threshold; where the market violates that,
  betting toward an isotonic-regression (PAVA) fit is +EV at resolution using the
  market's own prices as the bound. Domain-agnostic (any numeric ladder, ~50k
  markets). Date-aware grouping (fixes the `measurable_key` date-collapse), weights
  informative strikes so untraded 0.50 defaults don't drag the curve, and refuses
  to trade strikes pinned at the clamp (they anchor the fit but may be locked).
  **Live since 2026-05-27.**
- **`term_structure`** (no LLM, **no extras** — pure stdlib) — the time-axis
  complement to `ladder_arb`. Holds a metric+threshold fixed and looks across
  resolution dates: P(value > K) should trace a *smooth* curve in time, so stale
  0.50 dates sitting next to traded neighbours are mispriced. A Gaussian kernel
  smoother fills each stale date from the *traded* anchors only (never the reverse,
  never overriding a traded price), with a 0.5 prior so dates far from any anchor
  aren't confidently extrapolated. **Live since 2026-05-27.**
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

The clone is a **metals/mining/critical-minerals** market (~62k markets, ~96%
untraded at 0.50): LME/COMEX metal prices, spreads, inventories, rare-earth oxide
ratios, and mining-company production/sales. The live `commodity_spot` bot links the
spot-price subset it has a Stooq feed for; the model-free `ladder_arb` and
`term_structure` bots are domain-agnostic (they exploit internal price coherence and
need no feed at all), and the resolvability filter concentrates all of them on the
price markets that actually resolve.

The platform's *long tail* (company production/demand, niche indices) mostly
**cancels** (see "Cancellation reality") and is deprioritized automatically. A
future local-LLM bot could price the small resolvable slice of it that has no clean
data feed. The pipeline — sources → observations → linker → strategy → sizing →
allocator → execution — is done and validated end-to-end; widening it is a matter of
adding sources + catalog entries.

---

## Validate before deploying (`quantbots backtest`)

A bot should prove it's **accurate** and **profitable** on history before it
trades real mana. `quantbots backtest` replays a bot's probability model across
decades of real data (e.g. FRED mortgage rate, weekly since 1971): for each date
`t` and horizon `h` it forms the threshold questions the bot would face, takes the
bot's estimate, and checks it against what *actually happened* at `t+h`. That gives
thousands of (predicted probability, real outcome) pairs to score:

- **Brier score** vs the 0.25 always-50% baseline (and a **skill** %)
- **calibration** — do "70%" calls happen ~70% of the time? (reliability buckets)
- **simulated PnL / ROI / win-rate** under the bot's real sizing

```bash
quantbots backtest --preset mortgage --horizon-months 6
```

This is also how parameters get **tuned**: sweeping the ensemble's volatility on
mortgage data showed the default (0.5) was 3× too high (underconfident); 0.15 is
near-perfectly calibrated (+41% Brier skill). Volatility is **per-asset**
(`entity_vol` in config) — rates ~0.15, housing ~0.20, equities higher — each
calibrated by its own backtest. **No bot should go live until it shows positive
skill here.**

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
uv run pytest          # full suite — sizing, portfolio, resolvability, runner, PnL,
                       # ledger, ladder, and every strategy. No extras needed.
```

Strategy tests are pure-Python and fixture-free (e.g. `test_commodity_spot.py`,
`test_ladder_arb.py`, `test_term_structure.py`, `test_resolvability.py`,
`test_portfolio.py`, `test_runner.py`) — copy one as a template for a new bot.

---

## What to request from the team (can't self-serve)

- **Cloudflare Access service token** (`CF_ACCESS_CLIENT_ID` / `_SECRET`) scoped
  for your machine/CI.
- A **`MANIFOLD_CLONE_API_KEY`** for a bot account (a `@*Manual` key works to
  start), or `MANIFOLD_CLONE_ADMIN_API_KEY` to mint dedicated bot accounts + mana.
- Confirmation of **which markets your bots may trade** (a test tag or a few
  low-liquidity markets) so you don't step on the production fleet.

---

## Scaling & operations

Trading the clone at scale (~62k markets, mostly untraded at 0.50) is a *portfolio*
problem, not a per-market one. `portfolio.py` turns a bot's pile of per-market
signals into a capital-efficient book:

- **Realized-EV ranking** — base EV/mana is `(p−q)/q` for YES, `(q−p)/(1−q)` for NO,
  then multiplied by resolvability (`p_resolve`). So capital flows to the deepest
  mispricings *that actually settle* — not paper edge stuck in markets that cancel.
- **Concentration caps** — per-run *and* across-run, keyed by `Strategy.correlation_key`
  (e.g. all gold strikes are one bet on the gold price). Bounds cumulative exposure
  over repeated live runs (`max_total_exposure`, `max_group_exposure`).
- **Liquidity-aware, gentle sizing** — `liquidity_pct` + `max_price_impact` size each
  order to the market and nudge the price a few points; scale comes from *breadth*
  (many markets), and repeated runs converge a mispriced market toward fair value.

Net effect (measured): adding the resolvability weighting roughly doubled
realized expected profit *per mana staked* and freed budget that was being parked in
deep-but-unresolvable production/demand mispricings — the model-free bots' funded
books shifted to ~90% price markets.

**Run continuously:** `scripts/daily_cycle.sh` chains `resolve → refresh → ingest →
run --live → snapshot`. Scheduled daily via `scripts/com.quantbots.daily.plist`
(launchd). `quantbots status` is the monitoring dashboard (balance, per-bot PnL,
exposure by underlying). Set `QUANTBOTS_LIVE=0` for a dry cycle; `launchctl unload`
to stop.

## Status / roadmap

- ✅ Core infra: client, store, sizing, **portfolio allocator**, **resolvability
  filter**, runner, PnL, CLI, backtester, dry-run path. Throttle-retry on the live
  executor. Full test suite (`uv run pytest`) green.
- ✅ **Live on the clone since 2026-05-27** — three bots deployed at scale via the
  portfolio allocator + resolvability weighting + daily launchd cycle + `status`
  dashboard: `commodity_spot` (data-anchored prices), `ladder_arb` (across-threshold
  monotonicity), `term_structure` (across-date coherence).
- ⏳ A **cancellation predictor from the description** (parse each market's named
  resolution sources for a sharper resolvability score than question-text-only).
- ⏳ Further arbitrage forms scoped but not built: cross-currency/venue (FX-linked),
  cross-commodity lead-lag, spread/ratio-vs-components. (Exact-duplicate markets are
  a verified, separate issue — ~2,485 identical-question pairs at different prices.)
- ⏳ A local-LLM fundamental estimator for the resolvable slice of the long tail.
- ⏳ `manifold/websocket.py` live price cache (scaffold for the `realtime` extra).
