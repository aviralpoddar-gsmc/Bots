# Market-Maker bot — design

> Status: **design / not yet built.** This maps out what the market-maker will do
> and how it fits the framework, so we can productionize it deliberately.

## Why it exists (the role it fills)

Every bot in the fleet today is a **taker**: it reads a fair value, hits the AMM
with a single market order, and moves the price a little. Two consequences:

1. **No standing liquidity.** Nothing rests in the book, so other traders (and our
   own taker bots) face a thin liq-100 AMM. `stockpile_facts` choked at ~Ṁ2k for
   exactly this reason — AMM slippage on a thin pool is brutal.
2. **Prices don't stick.** A taker nudges and stops; the price drifts back.

The market-maker is the missing **maker** role. It posts **two-sided resting limit
orders around a fair value**, which:

- adds real order-book depth → markets become tradeable (the liquidity goal);
- pins the price at fair value as fills arrive (the correct-pricing goal);
- earns the bid/ask spread;
- lets every *taker* bot deploy far more capital into the now-deeper book.

It is the keystone of the flywheel: **anchors create a fair value → the maker turns
it into standing liquidity → propagators (ladder_arb / term_structure / coherence)
spread it across correlated markets → the market is usefully priced and liquid.**

## Core mechanism

For a market with fair value `f` and a chosen `half_spread`, post two limit orders:

| Order | Manifold form | Meaning |
|---|---|---|
| **Bid** | `{outcome: YES, limitProb: f − s, amount: A}` | buy YES if price falls to `f−s` |
| **Ask** | `{outcome: NO,  limitProb: f + s, amount: A}` | buy NO if price rises to `f+s` (≡ sell YES at `f+s`) |

(`limitProb` is always the YES-probability. A YES limit fills when the market drops
to it; a NO limit fills when the market rises to it. Together they are a resting
book quoting `f−s … f+s`.) Sellers of YES hit our bid; buyers of YES hit our ask.
We capture `2s` per round-trip and, because `f` is our fair value, every fill is at
a favorable price.

Invariant: never quote crossed (`bid < ask`), always two-sided unless deliberately
skewing for inventory, never quote inside `min_spread`.

## Fair value source — maker mode is an *execution style*, not a new model

Key design decision: the maker does **not** need its own forecasting model. The
fleet already produces fair values via `Strategy.estimate()`. So:

- **v1 — standalone `market_maker` bot** wrapping one configured fair-value source
  (e.g. `commodity_spot` / `stockpile_facts` / `import_reliance`) plus a neutral
  fallback. Trades the markets that source can price.
- **Long-term — "maker mode" on the runner.** A per-bot `maker: true` flag changes
  *execution* only: instead of one market order toward `estimate`, the runner posts
  a YES limit at `estimate − s` and a NO limit at `estimate + s`. Then **any
  calibrated anchor strategy can provide liquidity around its own fair value** for
  free (e.g. `commodity_spot` in maker mode deepens every metal it prices). This is
  the clean end state; the standalone bot is the stepping stone.

Two operating modes:

- **Anchored MM** (high value): a model has a view → quote tightish around `f`,
  providing liquidity *and* converging the price. Use on the markets we already
  price well.
- **Neutral MM** (broad, lower value): no model view → quote a **wide** book around
  the current price (or 0.50) for pure liquidity provision, directionally neutral.
  Only on decent-resolvability markets, small and wide, to avoid being picked off.

## Spread model

`half_spread = max(min_spread, base + k_unc·σ_f + k_res·(1 − resolvability))`

Wider when: the fair value is uncertain (`σ_f`), the market is cancel-prone (less
upside to tight quotes), or recent flow is toxic (see below). `min_spread` is set so
expected spread capture stays positive net of fees. Tight quotes converge price
fast but get picked off; wide quotes are safe but trade less — tune per family.

## Inventory management

Fills accumulate a position. Track net YES/NO per market from the ledger and:

- **Skew quotes by inventory** (Avellaneda–Stoikov-lite): if long YES, shift both
  quotes down so we're more eager to sell than buy, mean-reverting inventory toward
  flat. Prevents one-sided accumulation when flow is informed.
- **Cap inventory** per market and total (reuse `portfolio.allocate` exposure caps
  + `max_group_exposure`). Beyond the cap, quote one-sided (only the
  inventory-reducing side).

## Order lifecycle — TTL re-quoting

Limit orders rest until filled. To keep quotes fresh (anti-stale-pickoff) without an
explicit cancel call, **post every quote with a TTL** and re-quote each cycle:

1. Each run: recompute `f`, `σ_f`, spread, inventory skew.
2. Post fresh bid/ask with `expiresMillisAfter ≈ 25h` (slightly over the daily
   cadence) so yesterday's quotes self-expire as today's replace them.
3. No crossed/duplicate quotes because the old ones expire.

This avoids needing order cancellation for v1. Active management (cancel + re-post
intraday) is a later optimization.

### Client prerequisites (must add before building)

The clone client supports `limitProb` but **not** order expiry or cancellation. Add:

- `expires_millis_after` (and/or `expires_at`) param on `place_bet` / `batch_bet`
  → the TTL above. **Required for v1.**
- *(optional, for active management)* `cancel_bet(bet_id)` and a way to list open
  unfilled limit orders (`get_bets(kinds="open-limit")` or equivalent).

Verify the clone honors these (it's a Manifold fork; upstream supports
`expiresMillisAfter` on bets and `POST /bet/cancel/:id`).

## Risk controls (non-negotiable)

- `min_spread` floor → never quote for negative expected capture.
- Inventory caps per market + total; one-sided quoting past the cap.
- **Toxic-flow widening**: if one side keeps filling and the price trends through it
  (we're being run over), widen and/or skew that side; pause the market if extreme.
- Resolvability-aware sizing: MM on ~0% markets is mostly opportunity cost (cancel
  refunds), so prioritize decent-resolvability families; keep cancel-prone MM small.
- Clone-only host (inherited), `dry_run` validation first, conservative caps until a
  human confirms — same safety rules as every other bot.

## Profit & failure model

- **+ Spread capture** (`2s` per round-trip) and **+ convergence** (fills are at
  prices better than fair when `f` is right).
- **− Adverse selection**: informed traders fill us then the price moves further —
  the main loss; mitigated by spread, skew, TTL re-quoting, toxic-flow widening.
- **− Inventory risk**: holding a position into a YES/NO resolution that goes
  against us — mitigated by inventory caps + skew + only anchoring to calibrated `f`.
- **− Opportunity cost**: capital locked in resting orders on cancel-prone markets.

## Measurement

Per bot: fill rate, realized spread captured, inventory turnover, PnL split
(spread vs convergence vs inventory). System level: book depth added, fraction of
markets with live two-sided quotes, and price-convergence / Brier improvement on the
markets it quotes.

## Phased rollout

1. **Anchored MM on our best markets** — quote around `stockpile_facts` /
   `commodity_spot` / `import_reliance` fair values with a TTL and small inventory
   caps. Deepens exactly where we have alpha and unblocks taker deployment (fixes
   the Ṁ2k throttle). Lowest risk, immediate payoff.
2. **Neutral wide MM** to broaden liquidity across more decent-resolvability families.
3. **Inventory skew + toxic-flow detection** (Avellaneda–Stoikov-lite); later,
   intraday cancel/re-post for active quoting.
4. **Promote to runner "maker mode"** so every calibrated anchor provides liquidity
   around its own fair value.

## Config sketch (`config/bots.yaml`)

```yaml
- name: market_maker_1
  strategy: market_maker          # wraps a fair-value source + neutral fallback
  account_env: MARKET_MAKER_1_API_KEY   # own account: @MarketMakerBot (AP)
  enabled: false                  # dry-run validate first
  limits:
    max_total_exposure: 20000
    max_group_exposure: 2000
    min_resolvability: 0.0        # liquidity provision is cancel-safe
  params:
    fair_value_source: commodity_spot   # reuse an existing calibrated estimate()
    base_half_spread: 0.04
    min_half_spread: 0.02
    k_uncertainty: 0.5
    k_resolvability: 0.10
    inventory_cap: 200            # max |position| per market before one-sided
    quote_ttl_hours: 25
    neutral_fallback: false       # v1: only quote where the source has a view
```

## Open questions to resolve before building

1. Does the clone honor `expiresMillisAfter` / `bet/cancel`? (verify against the API)
2. Fees/rebates on limit orders — do they change `min_spread`?
3. Does the AMM auto-fill limit orders as the pool price crosses them, or only
   against incoming taker orders? (affects fill dynamics and convergence speed)
4. Maker-mode-on-runner vs standalone strategy for v1 (recommend standalone first).
