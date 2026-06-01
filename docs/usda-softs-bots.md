# USDA-driven soft-commodity bots — research + design (cotton / cocoa / coffee)

**Status:** BUILT (dry-run, `enabled:false`) · **Date:** 2026-06-01 · **Decision inputs:** data
scope = NASS + FAS PSD + WASDE; modelling = deterministic signals first; build all three bots.

## Build status (2026-06-01)

Phase-1 signal proven (§2b) and all three bots built + tested + validated end-to-end against the
live 50k-market cache. Nothing is live (`enabled:false`, not in `daily_cycle.sh`).

- **Sources:** `sources/fas_psd.py` (keyless bulk CSV → `PSD_COTTON_FREE_SUR`, `PSD_COTTON_SUR`,
  `PSD_COFFEE_CONS`, `PSD_COFFEE_CONS_GROWTH`) and `sources/nass.py` (QuickStats, keyed/optional,
  skips gracefully without `NASS_API_KEY`). Registered in `sources/__init__.py` + `sources.yaml`.
- **Strategies:** `cotton_fundamental` (futures-anchored lognormal + bounded ex-China-SUR drift +
  optional NASS condition), `cocoa_fundamental` (vol-calibrated zero-drift; no USDA data),
  `coffee_consumption` (FAS consumption-growth Normal CDF). Registered + 3 bot entries in
  `bots.yaml` (`enabled:false`).
- **Tests:** `tests/test_softs_usda.py` (12 cases) incl. the basis/spread-rejection regression.
  Full suite green (194 passed).
- **Live validation:** after `quantbots ingest --only fas_psd`, cotton matches **18 outright price
  markets** (basis/spread traps correctly excluded), cocoa 20, coffee 15, with coherent edges.
- **Known fix applied:** cotton/cocoa now reject basis/spread/differential markets (e.g. "Cotlook
  A minus ICE basis exceed 18 cents/lb") — pricing those off the ~76¢ outright was a
  confidently-wrong trap.
- **Calibration note:** live ex-China SUR parsed at ~0.443 vs the regression `sur_ref` 0.487; the
  drift is capped and directional so this is robust, but re-fit `sur_ref`/`price_ref` on the final
  ingested series before go-live (open question #4).

## Performance (2026-06-01)

**Dry-run** (via `quantbots run`, keys aliased to the shared clone key, 0 errors, no mana moved):

| Bot | Funded | Staked | Exp. profit | Exp. ROI |
| --- | --- | ---: | ---: | ---: |
| cotton | 12/12 on 18 mkts | Ṁ61 | Ṁ15 | +25% |
| cocoa | 12/12 on 20 mkts | Ṁ63 | Ṁ11 | +18% |
| coffee | 9/15 on 15 mkts | Ṁ90 | Ṁ3 | +4% (resolvability tax) |

**Out-of-sample calibration backtest** (`scripts/backtest_softs.py`) — replay each bot's pricing over
historical futures paths (2000–2024, monthly anchors × 3 horizons × 7 strikes, n≈6,400) and score
probabilities against realized outcomes. This is the honest "how they perform" (the dry-run ROI is
the model's own expectation and is circular):

| Bot | Brier | vs always-0.5 | Skill | Calibration |
| --- | ---: | ---: | ---: | --- |
| **Cotton** (USDA drift) | **0.1482** | 0.2500 | **+40.7%** | near-perfect (0.10→0.12, 0.50→0.52, 0.94→0.91) |
| Cocoa (zero-drift) | 0.1488 | 0.2500 | +40.5% | biased high (0.69→0.86, 0.89→0.97) |
| Coffee (consumption) | 0.1845 | 0.2500 | +26.2% | ok, small n (0.49→0.58) |

Findings:
- **Cotton is the standout: excellent calibration + real skill, and the USDA ex-China-SUR drift
  measurably helps** — it lowers Brier from 0.1537 (zero-drift) to 0.1482 (Δ +0.0055). The weak
  annual R² still translates into a genuine pricing improvement once expressed as a capped drift.
  This bot is ready to deploy.
- **Cocoa has skill but is directionally biased**: over a sample dominated by cocoa's secular bull
  it under-predicts the upside. Confirmed this is a *drift* problem, not a vol problem — raising vol
  to 0.50–0.55 or adding momentum does NOT fix the high-bin bias and worsens Brier. Keep vol 0.40,
  size small; the real fix is an ICCO fundamental/trend drift (Bot 4 work).
- **Coffee has modest skill** but the markets resolve ~0% (resolvability ~0.04), so the +4% dry-run
  ROI and +26% Brier skill are largely academic until/unless those markets settle.

## Original brief

This doc is the research record and the bot blueprint. Nothing here is wired live. It
extends the existing `commodity_futures` / `commodity_spot` lognormal-CDF pattern with a
**fundamentals-informed fair value and drift** sourced from USDA, rather than the current
zero-drift-anchored-to-spot assumption.

---

## 0. The reality check that shapes everything

The repo's market census (from `market_cache`) for these three commodities:

| Commodity | Markets on clone | Dominant question type | Verdict |
| --- | ---: | --- | --- |
| **Cotton** | ~132 | ICE Cotton No.2 **futures-price** thresholds (`exceed X cents/lb on DATE`) + acreage/abandonment | **Primary target.** Price markets resolve; NASS is world-class on US cotton |
| **Cocoa** | ~251 | Cocoa **futures-price** thresholds + "ECOM cocoa volume" | **Secondary.** Futures resolve; NASS irrelevant → lean on FAS/WASDE/ICCO. Volume markets ≈0% resolvability |
| **Coffee** | ~60 | **Volume / market-size only** ("ECOM coffee volume", "specialty coffee market size"). **No futures markets; KC not in price feed** | **Park it.** Demand/volume metrics resolve ≈0% (cancel). Not worth a fundamentals bot until price markets exist |

Two structural facts from `resolvability.py` and CLAUDE.md, restated because they govern the
whole design:

1. **~93% of clone resolutions are CANCEL.** A market settles YES/NO only when its named
   source publishes a verifiable value. **Price** markets with a strong source
   (ICE/CME/COMEX) score ~0.22→0.35; **production / demand / volume / acreage** markets
   score ~0.02 and almost never pay out.
2. **The adverse-selection trap:** USDA fundamentals map *most directly* onto the cotton
   **acreage / production / abandonment** markets — which are exactly the ones that won't
   resolve. The markets that *do* pay out are the **futures-price** thresholds, where
   fundamentals help only **indirectly**: fundamentals → fair-value price forecast (level +
   drift) → CDF → P(price > threshold).

**Therefore the edge is not "trade USDA acreage markets."** It is: *use USDA fundamentals to
beat the zero-drift lognormal on the cotton (and, more weakly, cocoa) futures-price ladders
that already resolve.* Everything below serves that thesis.

---

## 1. Data sources — what to ingest and how

### 1.1 USDA NASS QuickStats API (cotton & US crops) — *verified*

- **Endpoint:** `GET https://quickstats.nass.usda.gov/api/api_GET/` returning JSON / CSV /
  XML. Companion endpoints `get_param_values` and `get_counts`.
- **Auth:** free API key, obtained by agreeing to the Terms of Service + email
  (`quickstats.nass.usda.gov/api`). Key passed as `?key=...`.
- **Query model — "What / Where / When":** `commodity_desc` (e.g. `COTTON`),
  `statisticcat_desc` (`AREA PLANTED`, `AREA HARVESTED`, `PRODUCTION`, `YIELD`,
  `CONDITION`, `PRICE RECEIVED`), `short_desc`, `agg_level_desc`
  (`NATIONAL`/`STATE`/`COUNTY`), `year`, `reference_period_desc`, `freq_desc`. Numeric
  operators supported via suffix, e.g. `year__GE=2015`.
- **Hard limit:** a single call returns **≤ 50,000 records** (error otherwise) — page by
  year/state. This drives the ingestion design (per-series narrow queries, not bulk pulls).

**Cotton series + release calendar (the catalysts):**

| Report | Cadence | What it pins | Why it moves price |
| --- | --- | --- | --- |
| Prospective Plantings | end-Mar | intended acreage | first acreage surprise of the year |
| Acreage | end-Jun | actual planted acreage | confirms/refutes March intentions |
| Crop Progress & Condition | **weekly**, 4pm ET, Apr 1–Nov 30 | 5-bucket condition (very poor→excellent), % planted/harvested | running yield-surprise proxy |
| Crop Production | monthly Aug→ | yield & production forecast | revises the US supply side |
| Crop Production Annual Summary | Jan | final production | settles the crop year |
| Cotton Ginnings | semi-monthly in season | bales ginned | high-frequency output confirmation |
| Agricultural Prices | monthly | price received by farmers | farm-price anchor (basis to ICE) |

### 1.2 USDA FAS PSD Online (global — cotton, coffee, cocoa) — *verified*

- **Why:** the **world balance sheet** (production, consumption, exports, **ending stocks**,
  → **stocks-to-use**) that NASS does not provide. This is the only USDA home for cocoa &
  coffee fundamentals.
- **Access:** FAS Open Data API, base path **`/OpenData`** on the FAS apps host
  (`apps.fas.usda.gov/opendatawebv2/`); also bulk CSV downloads. The API also bundles ESR
  (export sales) and GATS (trade). Requires a free FAS API key. A community Python wrapper
  exists (`usda-fas-sdk` on PyPI) — reference only; we'll write our own thin client.
- **Coverage:** series back to **1960** for most commodities. Commodity codes:
  cotton `2631000`, green coffee `0711100`, cocoa (bean) code in the `0813...`/cocoa group
  (confirm exact code at build time — see open questions).
- **Companion circulars (PDF, monthly):** *Cotton: World Markets and Trade* and the
  coffee/cocoa circulars carry the headline balance numbers and USDA's narrative. Example
  pulled during research: cotton **MY2026/27 consumption 121.7M vs production 116.0M bales**
  → a **stock drawdown** (structurally bullish); coffee **MY2025/26 ending stocks ~20.1M
  bags** alongside ICO prices that had nearly tripled.

### 1.3 WASDE (the cotton catalyst) — *verified*

- Monthly **World Agricultural Supply and Demand Estimates** from WAOB, fixed published
  calendar, release ~**9th–12th** of each month. Contains the **US + world cotton balance
  sheet** (the official ending-stocks / stocks-to-use print).
- It is **the single largest scheduled price catalyst for cotton** — the report is the event
  the whole market re-prices around. For a bot this matters twice: (a) the *level* it prints
  sets fair value; (b) the *days around it* are when futures-market quotes are most likely to
  be stale/мispriced relative to the new balance sheet.

### 1.4 Coffee & cocoa fundamentals not in USDA — *verified caveats*

- **No USDA price forecast exists for coffee or cocoa** (ERS season-average forecasts cover
  **upland cotton only**, built from futures + NASS).
- The *real* coffee drivers — **Brazil/Vietnam weather and frost** — and cocoa drivers —
  **Ivory Coast & Ghana port arrivals, grindings, weather** — are only **partly** captured by
  PSD's annual balance sheet. Higher-frequency truth lives in **ICO** (coffee) and **ICCO**
  (cocoa quarterly bulletins) data, plus FAS **GAIN** attaché reports (e.g. semi-annual
  Coffee reports for Brazil/Vietnam/Colombia).
- Consequence: cocoa/coffee fundamental signals are **slower and coarser** than cotton's, and
  the stocks→price link for coffee is **correlational, not a fitted elasticity**. Size
  accordingly.

---

## 2. From fundamentals to a price-threshold probability

The bot needs `P(price_at_close > threshold)`. Today `commodity_futures` does:

```
sigma = max(annual_vol * sqrt(T), min_vol)
surv  = 1 - norm_cdf( log(threshold / spot) / sigma )      # zero-drift lognormal
```

It anchors to **today's futures price with zero drift**. The USDA upgrade replaces `spot`
with a **fundamentals-adjusted fair value** `F` and adds a **drift** `mu`:

```
F     = current_futures * exp( mu_fund * T )               # fundamentals tilt the center
sigma = max( annual_vol * sqrt(T), min_vol )
surv  = 1 - norm_cdf( ( log(threshold / F) ) / sigma )
```

where `mu_fund` is a **bounded** annualized log-drift derived from the balance sheet.

### 2.1 The core signal: stocks-to-use → price level

The most robust, literature-backed fundamental price driver is the **stocks-to-use ratio
(SUR)** = ending stocks / total use. **Low SUR ⇒ high/firm price; high SUR ⇒ soft price** —
the relationship is convex (prices spike non-linearly when SUR gets tight). Implementation:

1. Pull PSD ending stocks + total use → `SUR_now` for the marketing year covering the close
   date. Compute the historical mean/percentile of SUR from the 1960+ PSD series.
2. Map SUR deviation to a **fair-value tilt** via a calibrated (log-)inverse relationship —
   fit `log(price) ≈ a + b·log(SUR)` on PSD history (`b < 0`). The fitted `b` is the
   empirical **price elasticity to stocks-to-use**; cotton has a clean fit (it's exactly what
   ERS uses), cocoa weaker, coffee weakest.
3. `mu_fund` = clamp( (log(F_target) − log(current_futures)) / horizon, ±cap ). The cap (e.g.
   ±15%/yr) keeps a single noisy balance sheet from producing a runaway drift.

### 2.2 Secondary signals (cotton, in-season)

- **Crop condition Δ:** week-over-week change in "good+excellent" % is a yield-surprise proxy
  → nudges US production → nudges SUR. Use as a small `mu` adjustment Apr–Nov only.
- **Acreage surprise:** Acreage (Jun) minus Prospective Plantings (Mar), and either vs
  trade-expectation → step change in supply outlook around release.
- **WASDE surprise / event drift:** when a WASDE print moves the balance sheet, futures
  quotes on the clone may lag for hours/days — a transient mispricing the CDF will flag.

### 2.3 The volatility band

Per-commodity `annual_vol` (realized, from the price feed history — same approach as
`commodity_spot.vols`). Rough starting priors to calibrate at build time (open question #3):
cotton ~22–28%/yr, arabica coffee ~35–45%/yr (frost tail), cocoa ~30–40%/yr (recently far
higher). Keep `min_vol` floor and the `max_horizon_years` guard from `commodity_spot` — the
fundamentals tilt is only credible inside ~1 marketing year.

### 2.4 Honest limits

- Cotton fair value from the cointegration literature targets the **US farm price**, which
  carries a **basis** to the ICE settlement the clone markets reference. We must add/estimate
  that basis or anchor `mu` to the futures series directly and use PSD only for the *drift*,
  not the *level* (preferred — avoids the basis problem).
- The cointegration result rests on **one peer-reviewed paper** (rolling OLS beat the
  Hoffman-Meyer futures-only forecast 10/12 months, MAPE 2008–2023, WASDE most useful
  Nov–May). Treat as directional support, not gospel.

---

## 2b. Phase-1 empirical results (2026-06-01) — *signal proven, with caveats*

Ran `scripts/research_softs.py`: world stocks-to-use from FAS PSD bulk CSVs (1960+) vs
marketing-year-average front-month futures (yfinance, overlap 1999–2025, n=27). OLS by hand
(numpy), expanding-window walk-forward out-of-sample.

| Commodity | Elasticity `b` | t-stat | R² | OOS MAPE vs naive | OOS direction hit-rate |
| --- | ---: | ---: | ---: | --- | ---: |
| **Coffee** (world SUR) | **−0.76** | **−7.5** | **0.69** | 20.2% vs 19.7% (loses on level) | **79%** |
| **Cotton** (world SUR) | +0.03 | 0.1 | 0.00 | 18.1% vs 15.7% (loses) | 71% |
| **Cotton** (world **ex-China** "free" SUR) | −0.39 | −0.9 | 0.03 | **15.6% vs 15.7% (beats)** | 71% |
| Cocoa | — not in USDA PSD (ICCO only) — | | | | |

Realized annualized vol (daily log-returns): cotton 30% full / 21% last-2y; coffee 34%/37%;
cocoa 35%/**62%** (recent regime is extreme).

**What the data actually says — three honest conclusions:**

1. **Coffee has the strongest, cleanest fundamental signal** (b=−0.76, t=−7.5, R²=0.69): a +10%
   stocks-to-use move ⇒ −7.6% price, highly significant. **But coffee has no resolvable price
   markets on the clone** (§0). The best signal has nothing to trade. Painful but decisive.
2. **Raw *world* cotton SUR is uninformative** (R²≈0) — the China state-reserve distortion is
   real and large. **World-ex-China ("free") SUR is the correct metric:** it flips the sign
   negative (−0.39) and is the only model that **beats the zero-drift naive out-of-sample**.
   But the contemporaneous fit is still **weak and not significant** (R²≈0.03, t≈−0.9).
3. **Across both, the level forecast is noisy but the *direction* is reliable** (71–79% vs 50%
   coin-flip). ⇒ The USDA fundamental belongs in the model as a **small, bounded directional
   drift**, NOT a precise fair-value level. The earlier ±15%/yr drift cap is too aggressive;
   the data supports **±3–5%/yr** for cotton.

**Net:** the annual balance sheet is a *modest directional tilt + a vol anchor* for cotton, not
a standalone alpha. The bigger USDA edge for cotton is likely **event-driven** (WASDE
ending-stocks revisions + stale clone quotes) and **in-season crop condition**, which annual
data can't measure. This reshapes the build (see revised §4).

---

## 3. Pipeline design (ingestion)

Two new `Source` subclasses, mirroring `stooq.py`/`fred.py`, both emitting `Observation`s into
the existing `observations` table (so strategies read them via `strategy.bind(store)` →
`store.latest_observation(entity)` / `load_observations`). No core changes.

### 3.1 `sources/nass.py` (`name="nass"`)

- Params: API key (via env), list of series specs `{entity, commodity_desc, statisticcat_desc,
  short_desc, agg_level_desc, freq_desc}`.
- `fetch()`: one narrow QuickStats call per series (respecting the 50k cap), take the latest
  reference period, emit `Observation(source="nass", entity=..., ts=..., value=..., payload={raw})`.
- Entities, e.g. `NASS_COTTON_CONDITION_GE` (good+excellent %), `NASS_COTTON_AREA_PLANTED`,
  `NASS_COTTON_PRODUCTION`, `NASS_COTTON_PRICE_RECVD`.

### 3.2 `sources/fas_psd.py` (`name="fas_psd"`)

- Params: FAS API key, commodity/attribute specs (cotton/coffee/cocoa × production,
  consumption, exports, ending stocks).
- `fetch()`: pull latest marketing-year values, **compute stocks-to-use in the source**, emit
  `entity` like `PSD_COTTON_SUR`, `PSD_COTTON_ENDSTOCKS`, `PSD_COCOA_SUR`, `PSD_COFFEE_SUR`,
  plus a `fetch_history()` for backtest/elasticity fitting (mirrors `fred.fetch_history`).
- Optionally a `WASDE`-event entity (last release date + cotton ending-stocks print) for the
  event-drift signal.

`config/sources.yaml` gets two new blocks; `data/quantbots.sqlite` `observations` table
absorbs them with no schema change.

---

## 4. Bot designs (revised after Phase-1 results)

Phase-1 changes the priority order. Coffee (best signal) is blocked on market availability;
cotton (only resolvable market) has a weak annual signal that works only ex-China and only as a
small directional drift. So the build is **one modest cotton bot done honestly**, plus a clearly
higher-value **event overlay**, with coffee/cocoa explicitly parked on documented blockers.

### Bot 1 — `cotton_fundamental` — **build first, but size it as a modest edge**

- **Strategy:** generalize `commodity_futures` with an optional `drift` term (new
  `strategies/cotton_fundamental.py` subclass is cleaner). Anchor the lognormal **level** to the
  ICE Cotton No.2 futures feed (`CME_COTTON`, already in `stooq`) — *not* to PSD, which sidesteps
  the farm-price↔ICE basis problem (§2.4). Add a **bounded `mu_fund`** from **world-ex-China**
  `PSD_COTTON_FREE_SUR` (sign −, the only model that beat naive OOS), plus an in-season
  `NASS_COTTON_CONDITION` nudge.
- **Drift cap: ±3–5%/yr** (NOT ±15% — the R²≈0.03 fit cannot support more). The drift only ever
  *tilts* the zero-drift CDF; it cannot dominate it.
- **Vol:** anchor `annual_vol≈0.24` (between 21% last-2y and 30% full), `min_vol=0.05`,
  `max_horizon_years=1.25`.
- **Universe:** cotton **price-threshold** markets only (reuse `commodity_futures` regex +
  `parse_threshold`); resolvability gate `min_resolvability≈0.2` excludes acreage/abandonment.
- **Honest framing:** this is a *better-calibrated* `commodity_futures` (real vol + a small
  correct-direction tilt + event awareness), not a new alpha. Expected edge is incremental.
- **Guards:** `correlation_key="COTTON"` so the allocator caps total cotton exposure.

### Bot 2 — `cotton_wasde_event` — **likely the higher-value play**

- The annual fit can't see what the literature says is cotton's biggest catalyst: the **WASDE
  ending-stocks revision**. Overlay that, in the ±2 trading days around each WASDE (~9th–12th),
  re-prices cotton futures ladders against the *change* in the balance sheet and catches clone
  quotes that haven't updated. Lower frequency, higher conviction.
- Needs a WASDE-event entity in the FAS/WASDE source (last release + cotton ending-stocks print
  + prior print → surprise). Build after Bot 1's plumbing exists; it reuses the same source.

### Bot 3 — coffee — **parked on a market-availability blocker (not a signal blocker)**

- Signal is the **strongest** of the three (b=−0.76, R²=0.69) — but the clone has **no coffee
  price markets** and KC isn't in the feed. Two unlocks, both outside this bot: (a) coffee
  **price-threshold** markets get created on the clone, (b) add `CME_COFFEE` (`kc.f`) to `stooq`.
  If you control market creation, *creating coffee price markets is the single highest-ROI move*
  — it turns our best signal into a tradeable one.

### Bot 4 — cocoa — **parked on a data blocker**

- Cocoa is **not in USDA PSD** (ICCO tracks it). Needs a new `sources/icco.py` (quarterly
  bulletin: grindings, arrivals, stocks) before any fundamental bot. Futures markets do exist on
  the clone (~251) and `CME_COCOA` is in the feed, so a *vol-only* (no-drift) `commodity_futures`
  entry could trade cocoa today without ICCO — but that's just the existing strategy, no USDA edge.

---

## 5. Validation plan (before any `--live`)

1. **Elasticity fit:** fit `log(price) ~ log(SUR)` per commodity on PSD 1960+ history; report
   `b`, R², and out-of-sample stability. Kill the drift term for any commodity where the fit
   is not significant (expected: cotton yes, cocoa weak, coffee no).
2. **Backtest** via `backtest.py`: replay `cotton_fundamental` vs the plain zero-drift
   `commodity_futures` on historical cotton series → Brier / calibration / PnL. The bar:
   fundamentals must **beat zero-drift out-of-sample**, not just in-sample.
3. **Dry-run** on the live clone universe (default), inspect `explain()` output on real
   markets, confirm resolvability gating excludes the acreage/volume traps.
4. Only then flip `enabled: true` + add to `scripts/daily_cycle.sh` BOTS array (note the
   pair_trading lesson: enabling without adding to the cron array = silently never runs), with
   its own minted bot account (`scripts/mint_bot_accounts.py`).

---

## 6. Open questions (carry into build)

1. Exact FAS PSD commodity code + attribute IDs for cocoa bean and green coffee, and per-series
   start years (for the elasticity backtest).
2. Documented cocoa stocks-to-use → price elasticity (ICCO/PSD) — is there a usable fit, or do
   we run cocoa drift-free and rely on vol only?
3. Calibrated annualized vol + seasonality per commodity for the CDF band (cotton/coffee/cocoa).
4. Cotton **basis**: anchor `mu` to the ICE futures series directly (preferred) vs. modelling
   the farm-price→ICE basis. Decision needed before Bot 1 sizing.

---

## 7. Sources (verified, 3/3 adversarial vote unless noted)

- NASS QuickStats API — quickstats.nass.usda.gov/api ; nass.usda.gov/developer
- WASDE — usda.gov Office of the Chief Economist / WAOB
- FAS PSD Open Data — apps.fas.usda.gov/opendatawebv2 ; fas.usda.gov/data/databases-applications
- FAS circulars — apps.fas.usda.gov/psdonline/circulars/{cotton,coffee}.pdf
- ERS season-average price forecasts (cotton only) — ers.usda.gov/data-products/season-average-price-forecasts
- Cotton spot/futures cointegration — Journal of Agricultural and Applied Economics (Cambridge)
- NASS Crop Progress & Condition — nass.usda.gov Guide to NASS Surveys
- Cocoa price modelling — Tothmihaly (afjare.org) ; ICCO Feb-2026 Quarterly Bulletin
- Coffee — FAS GAIN Coffee Semi-annual (Brazil) ; ICO public market data
- Quant mapping — arXiv:1802.01393 ; ERS err80 (futures→fair-value)
