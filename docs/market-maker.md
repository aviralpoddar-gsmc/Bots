# Market-Maker bot — design

> ## ⚖️ Verdict (2026-06-01): BUILT & WORKS, but SHELVED — marginal value on this platform
> The maker is mechanically complete and live-verified (Phases 0–2 + canary + Phase 4
> maker-mode). But measured against live data, **both of its value levers are near-nil here**:
> - **Spread capture: 0.** Of 5 canary fills, 0 came from a real counterparty
>   (`matchedBetId` set) — 100% were AMM crossings. The clone has no organic two-sided
>   flow to make a market *for*; resting quotes sat unhit. A maker can't earn a spread
>   nobody trades against.
> - **Better execution: ≤1.2% of stake.** commodity_spot_1's 1,503 market-order fills
>   move price only ~2pts mean (median 1.3, mana-wtd 2.4), ≈1.22% slippage vs mid —
>   because it already sizes to liquidity and caps impact at 5pts. Limit-capped maker
>   execution would save a fraction of that.
>
> **Recommendation:** keep the code (it's reusable, reviewed infra) but leave
> `market_maker_1` `enabled:false`; do NOT add it to daily_cycle, do NOT flip
> commodity_spot_1 to maker-mode, do NOT build Phase 3 — the platform doesn't reward
> any of it. Revisit only if the clone develops real trader flow. The lasting win from
> this effort is the verified client primitives (limit TTL/cancel/open-order listing)
> and the `maker:true` runner mode, available if ever needed.


> Status: **Phase 2 built — standalone v1 maker (2026-06-01).** Phase 0 verified
> the clone's limit-order behavior; Phase 1 added the client primitives; Phase 2
> ships the `market_maker` strategy + `maker.run_maker` execution path (reconcile
> loop, diversified two-sided quoting, inventory + group + exposure caps), the
> `quantbots make` CLI, and `market_maker_1` in bots.yaml (enabled:false). Adversarially
> reviewed (14 findings fixed); 183 tests green; live dry-run clean. NOT YET LIVE:
> mint @MarketMakerBot, then a small --live canary. Phase 3 = price skew +
> toxic-flow widening + maker-mode-on-runner.

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

> **Verified live (2026-06-01):** the clone honors `expiresMillisAfter` via a
> periodic expiry **sweep** (a 60s-TTL probe order expired between t=78s and
> t=150s), i.e. an order clears within ~1–2 min *past* its nominal `expiresAt`,
> not instantly. Negligible at a 25h TTL, so the self-expiry plan holds. Reserved
> mana is refunded on both expiry and cancel (probe balance returned to baseline).

### Client prerequisites — ✅ BUILT & VERIFIED (Phase 1, 2026-06-01)

The clone client supported `limitProb` but not order expiry or cancellation.
Added to `manifold/client.py` and verified live against the clone:

- `place_bet(..., expires_millis_after=, expires_at=)` — TTL params (integer ms),
  limit-order-only. **Verified:** `expiresAt = createdTime + expires_millis_after`.
- `batch_bet` — now documents the `expiresMillisAfter`/`expiresAt`/`limitProb`
  pass-through so both quote legs post in one call.
- `cancel_bet(bet_id)` — `POST bet/cancel/:betId` (the **bet** id, not contract id).
  **Verified:** primary path works; cancelled order reads `isCancelled=True`.
- `get_open_limit_orders(market_id=, user_id=)` — wraps
  `get_bets(userId, kinds="open-limit")`. **Verified:** returns resting orders
  with `orderAmount` (total) / `amount` (filled-so-far) / `isFilled` / `fills`.

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

## Phase 0 findings — answered live (2026-06-01)

Probed on `@CommoditySpotBot` against gold market `pq0IhPdlAh` with self-cleaning
M$1 orders. Original open questions, now resolved:

1. **Honors `expiresMillisAfter` / `bet/cancel`?** ✅ Both. Expiry via a periodic
   sweep (~1–2 min lag past `expiresAt`); cancel via `POST bet/cancel/:betId`
   (primary path, no fallback needed). Refunds reserved mana.
2. **Fees/rebates on limit orders?** None observed — `creatorFee`, `platformFee`,
   `liquidityFee` were all `0` on both resting and crossing-fill responses. So
   `min_spread` is governed by adverse selection, not fees (but re-confirm on a
   real two-sided fill before trusting at scale).
3. **Auto-fill as the pool price crosses?** ✅ Yes — a crossing limit fills
   immediately against the AMM (`fills[].matchedBetId = None`) and walks the price
   **up to but not past** its `limitProb` (e.g. limit 0.67 → probAfter 0.619). A
   resting limit far from price stays unfilled (`amount=0, shares=0, fills=[]`).
   This is exactly the convergence dynamic the maker relies on.

## Remaining open questions

1. ~~Maker-mode-on-runner vs standalone for v1~~ → **standalone shipped** (Phase 2).
2. Confirm zero-fee on a real **two-sided round-trip fill** (the probe only saw
   resting + crossing, not a maker round-trip) — verify on the first live canary.
3. ~~Reserved-mana accounting~~ → **done**: `run_maker` subtracts resting
   `orderAmount` (via `get_open_limit_orders`) and net filled inventory from the
   `max_total_exposure` headroom each cycle.

## Phase 2 — what shipped (v1)

- `strategies/market_maker.py` — `MarketMakerStrategy` wraps a fair-value source
  (default `commodity_spot`), delegating prefilter/estimate/correlation; adds
  `half_spread` and the maker params.
- `maker.py` — `run_maker`: reconcile fills (over open-orders ∪ inventory ∪ recent
  bets, so no orphaned market leaks), net-exposure budget, diversified breadth
  selection, two-sided quoting with per-group + inventory + per-leg-spread caps,
  cancel-then-repost (repost only fully-cleared markets), fills-only ledger writes.
- `quantbots make --bot NAME` (dry-run default), `market_maker_1` in bots.yaml.
- Adversarial review fixed: orphaned-market reconcile/cancel, cancel-failure
  double-stack, two-sided resolution (`open_position_legs` + per-leg
  `sync_resolutions`), gross→net exposure, reserved-mana budget, group caps,
  budget `continue`-not-`break`, near-boundary spread squash, honest leg counts.

### Go-live checklist (→ live)  — Phases 0–2 + canary DONE 2026-06-01
1. ~~Mint `@MarketMakerBot` + `MARKET_MAKER_1_API_KEY`~~ ✅ done (Doppler dev/stg/prd).
2. ~~`quantbots run --bot market_maker_1` (dry-run) — quotes + 0 errors~~ ✅.
3. ~~Tiny `--live` canary; verify fills reconcile~~ ✅ (idempotent, both legs handled).
4. To fully deploy: flip `enabled: true` and add `market_maker_1` to the existing
   `BOTS=(...)` array in `scripts/daily_cycle.sh`. **No separate `make` step** —
   `quantbots run` auto-routes maker-mode bots, and the resolve loop already closes
   both legs via `open_position_legs`. Before that: disable `commodity_spot_1` on
   the overlapping legs (or flip it to `maker: true` and retire this bot) so the two
   don't double-stack the same view.

## Phase 4 — maker mode on the runner (built 2026-06-01)

`maker: true` on ANY bot switches execution only: `quantbots run` auto-routes it to
the maker path, wrapping the bot's own `strategy` as the fair-value source (knobs
from `limits`). No wrapper strategy, no separate bot — any calibrated anchor
becomes a liquidity provider on its own account. `market_maker_1` now ships as
`strategy: commodity_spot` + `maker: true`. The end-state (Phase 4 fully realized)
is to set `maker: true` on `commodity_spot_1` itself, retiring the dedicated MM bot
and eliminating overlap. Remaining: price skew + toxic-flow widening (Phase 3
risk controls), measure real (non-AMM) counterparty fills before scaling.
