# GSMC Internship — Complete Work Inventory

*Aviral Poddar (UMass) — identities: `aviralpoddar-gsmc`, `mikhailtal-design`, `Mikhail Tal` — May–July 2026.*

Organized by category so items can be included/excluded independently. Every entry is
self-contained: what it is, the model/approach in full, status, and measured numbers.
All figures pulled from repos, git history, live databases, and PR records as of 2026-07-22.

**Categories**
1. [Clone trading bots — the quantbots fleet](#1-clone-trading-bots--the-quantbots-fleet)
2. [Core platform infrastructure (quantbots framework)](#2-core-platform-infrastructure)
3. [Equity options — real-money carve-out](#3-equity-options--real-money-carve-out)
4. [Local-compute LLM infrastructure](#4-local-compute-llm-infrastructure)
5. [tal platform contributions (PRs)](#5-tal-platform-contributions)
6. [Research repos](#6-research-repos)
7. [Agentic systems (sandbox-agents)](#7-agentic-systems)
8. [Data engineering & market creation](#8-data-engineering--market-creation)
9. [Headline numbers & presentation notes](#9-headline-numbers--presentation-notes)

---

## 1. Clone trading bots — the quantbots fleet

All trade play-money mana (Ṁ) on the private Manifold clone. Fleet totals (live DB,
2026-07-22): **23,616 trades · Ṁ227,271 staked · 3,463 distinct markets · 1,441 resolutions
realized · 106,915 markets cached (~62k universe, ~96% untraded at 0.50)**. Every bot has its
own minted clone account. Every bot automatically gets sizing, portfolio allocation, and
resolvability weighting from the framework (§2).

### 1a. Structural / model-free arbitrage bots

#### ladder_arb_1 — monotonicity arbitrage
- **Status:** LIVE since 2026-05-27 · **3,939 trades · Ṁ25,265** (the fleet's biggest trader)
- **Model:** Within a (metric, resolution-date) strike ladder, P(value > K) must be
  non-increasing in K. Fits the nearest coherent curve by **weighted isotonic regression via
  PAVA** (pool-adjacent-violators, O(n), pure stdlib) and trades off-curve strikes toward the
  fit. Informative weighting: traded/moved strikes weight ×5, untraded 0.50s ×1;
  clamp-pinned strikes anchor the fit but are never traded; date-aware grouping prevents
  collapsing different expiries into one ladder.
- **Why it matters:** domain-agnostic — needs no data feed, no opinion about the world, only
  internal coherence. Covers ~50k markets.
- **File:** `src/quantbots/strategies/ladder_arb.py`

#### term_structure_1 — time-axis coherence
- **Status:** LIVE since 2026-05-27 · **1,715 trades · Ṁ4,718**
- **Model:** The orthogonal axis to ladder_arb: hold (metric, threshold) fixed, vary
  resolution date — P(value > K) should trace a smooth curve in time. A **Gaussian kernel
  smoother** (bandwidth 6 months) fills stale 0.50-priced dates from *traded anchors only*,
  with a 0.5-prior pseudo-count so it never extrapolates confidently far from data. Requires
  ≥2 anchors and ≥3 dates before acting.
- **File:** `src/quantbots/strategies/term_structure.py`

#### stockpile_grid_arb_1 — 2-D monotone surface arbitrage
- **Status:** LIVE · **22 trades · Ṁ112** (deliberately small: resolvability ~0.03)
- **Model:** U.S. strategic-materials vault-procurement ladders form a 2-D grid (strike ×
  expiry). Survival probability must be **monotone-down in strike AND monotone-up in expiry**
  (procurement is cumulative). Fits the nearest 2-D monotone surface by **cyclic isotonic
  projection** — alternating weighted PAVA passes down strike-lines and up expiry-lines to
  convergence. Cancel-safe breadth play.
- **File:** `src/quantbots/strategies/stockpile_grid_arb.py`

#### conditional_arb_1 — Fréchet-bound coherence on conditional markets
- **Status:** LIVE (built 2026-06-19) · **28 trades · Ṁ170**
- **Model:** For "IF [A]=YES: B" markets pricing c = P(B|A), the law of total probability
  gives **Fréchet bounds: max(0, (a+b−1)/a) ≤ c ≤ min(1, b/a)**. The nested case
  (quantity implies predicate, e.g. zinc≥3400 ⟹ zinc≥2800) is **exact: c = b/a**. Trades
  only the conditional toward the band; abstains if the legs are survival-inverted (lets
  ladder_arb repair them first) or the predicate is < 0.05. Resolvability = product of both
  legs. Downside is capital-neutral (predicate-NO → CANCEL refund).
- **Validation:** 3 zinc triangles fire, rhodium correctly suppressed.
- **Files:** `src/quantbots/strategies/conditional_arb.py`, `tests/test_conditional_arb.py`

#### surface_arb_1 — parametric ladder fitting
- **Status:** LIVE · **225 trades · Ṁ7,296**
- **Model:** Fits a **normal CDF** to a measurable's strike ladder and trades strikes toward
  the fitted curve — the parametric cousin of ladder_arb, and the fleet's stat-arb starting
  point. `strategies/ladder.py` parses threshold/direction from question text.
- **File:** `src/quantbots/strategies/surface_arb.py`

#### semantic_arb — LLM-linked logical arbitrage
- **Status:** built; targets the duplicate-market problem
- **Model:** Cross-market relations the structural bots can't link syntactically. A local LLM
  asserts only relations true **by meaning alone** — equivalent (P(a)=P(b)), negation
  (1−P(b)), implies (P(a)≤P(b)), exclusive (P(a)+P(b)≤1). Market prices are **never shown to
  the model** (no anchoring); cheap O(n) candidate blocking by shared-rarest-token + date;
  self-consistency voting over a temperature spread; then **POCS (iterated convex
  projection)** onto the feasible region with partial correction (strength 0.7).
- **Target:** ~2,485 verified byte-identical duplicate market sets trading at different prices.
- **File:** `src/quantbots/strategies/semantic_arb.py`

#### stockpile_facts_1 / stockpile_coherence_1 — reference-lookup + policy coherence
- **Status:** LIVE · 624 trades · Ṁ4,106 (facts) / 3 trades · Ṁ17 (coherence)
- **Model (facts):** answers U.S. strategic-materials *fact* markets from curated public
  record: the USGS 2022 Critical Minerals list (50 minerals, baked-in frozenset) → P=0.93/0.07;
  National Defense Stockpile held positions (curated from GAO-24-106959 / CRS R47833) →
  0.85/0.12; abstains where the record is ambiguous. The alpha bot of the 3-bot stockpile set.
- **Model (coherence):** consistency across buffer-stock/policy markets.
- **Files:** `src/quantbots/strategies/stockpile_facts.py`, `stockpile_coherence.py`

### 1b. Stochastic-process pricing bots

#### diffusion_mc_1 — kernel-smoothed bootstrap Monte Carlo  ★ promoted
- **Status:** LIVE since 2026-06-04 as @DiffusionMcBot, **promoted over commodity_spot_1** ·
  **3,388 trades · Ṁ22,811**
- **Model:** prices P(price > K at T) from a simulated terminal distribution. Default
  process **"ksb" = kernel-smoothed block bootstrap**: resample ~10-trading-day blocks of
  *demeaned* (zero-drift) historical daily log-returns (blocks preserve volatility
  clustering), convolve each day with a **variance-preserving Student-t kernel** at Silverman
  bandwidth h = 0.9·n^(−0.2) so simulations can exceed any historically observed move (fixes
  the plain bootstrap's bounded-tail hole), compound over round(T·252) days. Alternative
  processes: plain bootstrap, fitted Student-t (df per commodity via scipy, clamped 3–15),
  Gaussian-jitter hybrid. Calibrated on 10y yfinance history; falls back to the parent
  lognormal when history is missing.
- **Validation (the gate story):** first walk-forward gate said *no out-of-sample edge*;
  diagnosed the artifact — an uncapped-from-0.50 backtest — re-ran under the framework's real
  per-market stake cap, and the verdict **reversed**: beats the lognormal on **Brier, PnL,
  Sharpe AND worst-fold across 8 commodities × 5 folds**; edge peaks at 21–63-day horizons.
  The winning lever was the per-market cap.
- **Files:** `src/quantbots/strategies/diffusion_mc.py`, `scripts/diffusion_bench.py`

#### commodity_spot_1 — data-anchored lognormal (retired parent)
- **Status:** RETIRED 2026-06-04 (resolve-only; superseded by diffusion_mc) · 2,102 trades ·
  Ṁ14,865 · was LIVE from 2026-05-27
- **Model:** zero-drift lognormal on genuine spot-price ladders: P = 1 − Φ(ln(K/S)/σ√T),
  σ = max(annual_vol·√T, min_vol); per-commodity vols (oils 0.39–0.45, copper 0.21, gold
  0.16, silver 0.30, Pt 0.22, Pd 0.30); horizon capped at 1.25y. Its lasting contribution is
  the **strict unit/currency guard** (the "confidently-wrong firewall"), inherited by every
  subclass: requires the quoted unit (per troy oz / /MT / /bbl / /gal), rejects foreign
  currency (CNY/EUR/yuan/RMB) and chemical compounds (sulfate/oxide/carbonate); feed→market
  conversion factors (silver ×0.01 cents/oz→$/oz, copper ×22.0462 cents/lb→$/MT); excludes
  European natural gas (wrong benchmark).
- **File:** `src/quantbots/strategies/commodity_spot.py`

#### pair_trading_1 — cointegration convergence overlay
- **Status:** LIVE since 2026-06-01 · **2,088 trades · Ṁ13,841** (was silently excluded from
  the cron BOTS array at first — found and fixed)
- **Model:** subclasses commodity_spot; a research layer fits OLS hedge ratio β, spread
  mean/std, and **Ornstein-Uhlenbeck half-life** per pair. Expected spread convergence
  E[s_T − s_0] = (μ − s_0)(1 − e^(−θT)), θ = ln2/half-life, injected as a log-drift into the
  lognormal, damped by reversion_capture = 0.5, attributed to one leg (the partner is the
  martingale anchor). Fires only past entry_z = 1.5. 11 default resolvable metal/energy pairs
  (GOLD/SILVER, WTI/BRENT, PLATINUM/PALLADIUM, …).
- **File:** `src/quantbots/strategies/pair_trading.py`, `scripts/research_pairs.py`

#### enso_1 — climate persistence
- **Status:** LIVE · **637 trades · Ṁ5,892**
- **Model:** ENSO/Oceanic Niño Index markets (NOAA data). **Additive Gaussian persistence**
  (ONI is not a positive price, so no lognormal): P(ONI_future > T) = 1 − Φ((T − V)/σ),
  σ = monthly_vol·√months. Self-contained question linker.
- **File:** `src/quantbots/strategies/enso.py`

#### commodity_1 — soft-commodity futures
- **Status:** LIVE · **829 trades · Ṁ8,455**
- **Model:** same lognormal family as commodity_spot but for ag futures price markets
  (cotton/sugar/wheat/corn/cocoa; Stooq/ICE/CBOT catalog).
- **File:** `src/quantbots/strategies/commodity_futures.py`

#### mean_reverter — reference implementation
- **Status:** LIVE (reference) · 6 trades · Ṁ61
- **Model:** fades the market toward an EMA of its own price — the simplest no-LLM example,
  kept as the "how to write a bot" template.
- **File:** `src/quantbots/strategies/mean_reversion.py`

#### ensemble_1 — deterministic multi-source fusion
- **Status:** LIVE · **866 trades · Ṁ9,943**
- **Model:** linker maps question → entity + threshold; each numeric observation contributes
  P = 1 − Φ(ln(T/V)/σ) (lognormal), fused as a source-weighted average. Per-entity vol
  (`entity_vol`), plausibility guard (max_ratio = 20 drops mis-links), horizon-scaled σ.
- **Validation:** backtest-driven retune of default vol 0.5 → 0.15 gave **+41% Brier skill**
  on the FRED mortgage series.
- **File:** `src/quantbots/strategies/ensemble.py`

### 1c. Fundamental / single-source signal bots

All share `_signal_base.SignalDriftStrategy`: each bot draws alpha from exactly **one**
external source, expressed as a bounded annualized log-drift on the shared price anchor,
priced through the lognormal CDF; abstains unless the drift clears `min_drift`.

#### cotton_fundamental_1 / fas_fundamental — USDA stocks-to-use drift
- **Status:** LIVE · **433 trades · Ṁ9,496**
- **Model:** USDA FAS PSD world-ex-China cotton stocks-to-use ratio → ±3–5%/yr drift on ICE
  cotton. **The one fundamental signal that beat zero-drift out of sample** (ex-China SUR
  b = −0.39). Research also found coffee SUR elasticity b = −0.76 (t = −7.5, R² = 0.69) —
  strongest signal in the complex but untradeable (no resolvable coffee markets).
- **Validation:** cotton Brier 0.1482 = **+40.7% skill** vs always-0.5, near-perfect
  calibration; USDA drift lowers Brier 0.1537 → 0.1482; direction hit-rates 71–79%.

#### cocoa_fundamental_1 — vol-anchored cocoa
- **Status:** LIVE · 137 trades · Ṁ14,362
- **Model:** vol-anchored zero-drift lognormal (no USDA PSD coverage for cocoa; ICCO only).
  Backtest Brier 0.1488 = +40.5% skill (high-biased, documented).

#### coffee_consumption_1 — FAS consumption growth
- **Status:** LIVE · 48 trades · Ṁ8,000
- **Model:** FAS consumption-growth normal CDF. Backtest +26.2% skill but ~0% resolvability
  in practice — kept small.

#### fas_balance_1 — balance carry-forward
- **Status:** LIVE · 430 trades · Ṁ4,218
- **Model:** carries FAS balance-sheet figures forward to cover far-dated cotton *quantity*
  markets that would otherwise sit untouched at 0.50.

#### cftc_softs_1 — positioning drift
- **Status:** LIVE · **548 trades · Ṁ18,044**
- **Model:** CFTC Disaggregated Commitments of Traders positioning → bounded drift.

#### weather_cocoa_1 — growing-region weather anomaly
- **Status:** LIVE (T1 of the ags-weather push, shipped 2026-06-02) · 232 trades · Ṁ15,000
- **Model:** open-meteo growing-region weather anomalies → cocoa drift.

#### nass_cotton_1 — crop-condition index
- **Status:** LIVE (abstains without a NASS key) · 11 trades · Ṁ637
- **Model:** **production-weighted per-state cotton condition index** from USDA NASS
  QuickStats.

#### news_drift_1 ("007") — news-driven drift
- **Status:** LIVE since 2026-06-09 as @Bot007 · **2,760 trades · Ṁ20,691**
- **Model:** a **local LLM digests commodity news RSS** (Investing.com / OilPrice / Mining)
  into per-commodity signed direction signals SIG_<COM>_NEWS ∈ [−1,1], confidence-weighted
  and **recency-decayed (36h half-life)**, applied as a small bounded drift (k = 0.08).
  Abstains unless ≥2 fresh, directionally-clear headlines. Rejected GDELT as a source. Also
  fixed the SignalDriftStrategy closed-market 403 bug en route.
- **Files:** `src/quantbots/strategies/news_drift.py`, `scripts/validate_news_leadlag.py`

#### wasde_event — report-surprise overlay
- **Status:** built, gated (abstains until the next WASDE print)
- **Model:** WASDE cotton ending-stocks revision surprise as an event overlay.

### 1d. LLM forecasting bots

#### llm_forecaster / llm_ag_coverage — percentile → CDF
- **Status:** llm_forecaster LIVE (local qwen3) · 118 trades · Ṁ785; llm_ag_coverage
  shelved after its coverage push · 1,832 trades · Ṁ14,575
- **Model:** **one local-model call per measurable** returns p10/p25/p50/p75/p90; fit a
  normal (μ = p50; σ averaged from the 10–90 span [z = 2.5631] and IQR [z = 1.3490]),
  widened ×1.5 against measured local-model overconfidence (benchmarked 57–71% p10–p90
  coverage vs ideal 80%); then **every strike on the ladder is read off the analytic CDF** —
  one call prices the whole ladder. Confidence cap 0.80. Model selection is empirical via
  `quantbots llm-bench` (validity/coverage/p50-error/latency); also A/B'd Qwythos-9B vs
  gemma4 with a full dry-run diff (kept gemma4) and fixed 2 crash bugs found in the process.
- **File:** `src/quantbots/strategies/llm.py`, `src/quantbots/llm/bench.py`

#### mercury_ensemble_1 — Bayesian-mixture calibration  (hosted-inference exception)
- **Status:** LIVE since 2026-06-11 as @MercuryEnsembleBot (owner-approved hosted-inference
  experiment; engine-agnostic, reverts to local if it doesn't beat the local baseline) ·
  224 trades · Ṁ1,465, scoped to resolvable price markets
- **Model:** samples Mercury (Inception Labs diffusion LM) **N=20× over temperatures
  0.4–1.0**; mixes per-sample CDFs into a posterior-predictive p̄ = (1/N)Σpᵢ; decomposes
  uncertainty by the **law of total variance** — epistemic = Var[pᵢ] (sampler disagreement),
  aleatoric = p̄(1−p̄) − epistemic; a **direction-agreement gate** abstains when <70% of
  samples agree on side (split jury); **disagreement-shrinkage** pulls toward the market:
  confidence = clamp(1 − epistemic/τ), estimate = market + (p̄ − market)·confidence
  (τ = 0.04); min quorum 12 valid samples.
- **Files:** `src/quantbots/strategies/mercury_ensemble.py`, `_mixture.py`,
  `docs/mercury-ensemble-calibration.md`

### 1e. The comment society (Bridgewater-pattern judging fleet)

Built 2026-07-08 on the AIA Forecaster paper (arXiv:2511.07678). Clone context: ~1M comments
from 66 tal bots.

#### consensus_1 — extremized comment consensus
- **Status:** LIVE 2026-07-08 · 357 trades · Ṁ2,313 · 108 consensus rows computed
- **Model:** pools each market's bot-bettor crowd (one implied probability per user = their
  latest bet's probAfter; requires ≥3 forecasters) into a mean, then **Platt-extremizes**:
  p̂ = σ(√3·logit(p̄)) — the paper's correction for ensembles that hedge toward 0.5 — and
  trades toward the extremized consensus.
- **File:** `src/quantbots/strategies/comment_consensus.py`, `src/quantbots/comments/consensus.py`

#### The adversarial judge (pipeline, not a bot)
- **Status:** LIVE — own launchd loop (`com.quantbots.comments.plist`,
  `scripts/comment_judge_cycle.sh`); verdicts to date: **797 sound · 211 unsound (high
  confidence) · 501 noise**
- **Model:** implements the paper's *negative* result correctly — critiquing forecaster
  *reasoning* is worse than nothing; value comes only from **gathering independent evidence
  and overriding at high confidence**. The judge builds an evidence pack from ingested feeds
  (stooq/LBMA/FRED/NOAA + parsed threshold, unit-converted) and marks a comment `unsound`
  **only if its factual claims contradict the evidence numbers** — never for "bad reasoning".
  Hard anti-false-positive rules: market price is never evidence; no self-derived unit
  conversions; ≤3-day staleness cap on high-confidence verdicts. All local inference
  (qwen3:32b).
- **Files:** `src/quantbots/comments/{judge,cycle}.py`

#### adversary_metals_1 — comment fade
- **Status:** LIVE 2026-07-08 · 9 trades · Ṁ61
- **Model:** turns actionable verdicts (unsound + high-confidence + attached bet) into a
  fair-value tilt **against** the bad commenter's position — but `_fights_anchor` only fades
  bets on the *wrong side of the data anchor*: wrong-reasoning-but-right-conclusion is never
  faded. Shift 0.05 per verdict, cap 0.12.
- **File:** `src/quantbots/strategies/comment_fade.py`

### 1f. Built, validated, deliberately NOT live (the discipline shelf)

#### market_maker_1 — shelved on measurement
- **Status:** SHELVED 2026-06-01, kept enabled:false; @MarketMakerBot minted; 5 trades · Ṁ70
- **Model:** fully built and live-verified through canary — two-sided limit book at f ± s
  around the diffusion_mc fair value, **Avellaneda-Stoikov-lite inventory skew**, TTL
  re-quoting via expiresMillisAfter ≈ 25h, toxic-flow widening. Uses limit-order client
  primitives added for it (place/cancel/get-open).
- **Why shelved:** measured marginal value — canary got **0/5 fills from real counterparties**
  (100% AMM crossings; the clone has no organic two-sided flow) and taker slippage was
  already only ~1.2% of stake. Both value levers near-nil. Working code killed with data.
- **Files:** `src/quantbots/maker.py` (462 LOC), `docs/market-maker.md`

#### cocoa_atlantic — Atlantic-Niño SST drift
- **Status:** built 2026-06-02, validated sign (−1) but **weak/nonlinear** → kept
  enabled:false. Source verdicts from the same push: open-meteo/CPC keep; CHIRPS/tidbits/
  Kalshi cut.

#### drought_cotton / cocoa_stocks — unconventional-data bots
- **Status:** built + validated 2026-06-02, both enabled:false. US Drought Monitor DSCI
  index; ICE certified cocoa-stocks .xls.

#### dup_arb — removed per owner decision
- The ~2,485 exact-duplicate market sets are real, but the dedicated dup_arb strategy was
  removed; the surface remains a target for semantic_arb.

---

## 2. Core platform infrastructure

The reusable framework everything in §1 runs on (~19,200 LOC Python; 55 test files, green).

- **The one seam** (`strategies/base.py`): bot authors implement
  `estimate(group) → {market_id: prob}`; optional prefilter/group/correlation_key/explain.
- **Clone-only API client** (`manifold/client.py`): base URL hard-wired to the clone
  (deliberate safety invariant — no "platform" argument exists); Cloudflare Access + API-key
  auth; 500 req/min limit; dry-run bet validation; batch bets; limit-order primitives.
- **Sizing** (`sizing.py`): the **1/3-push rule** — target = current + (estimate−current)/3 —
  under four caps: mana-to-target, max_order_size 50, 33% of pool liquidity, 10% max price
  impact (LMSR approximation b ≈ liquidity/4, cost |b·Δlogit|); hold band 0.05 kills churn.
  No Kelly, no confidence knob — the gap is the conviction.
- **Portfolio allocator** (`portfolio.py`): greedy knapsack on EV-per-mana (YES (p−q)/q, NO
  (q−p)/(1−q)) × resolvability = **realized EV**; per-run and per-correlation-group budgets
  plus across-run exposure caps from the ledger.
- **Resolvability core** (`resolvability.py`): see §9 headline — the cancellation-reality
  insight (93% of 9,578 resolutions CANCEL) turned into a question-text score in [0.01,0.99]
  calibrated to observed decided-rates (price 0.22, exchange-settled 0.35, precious+price
  0.90, LBMA 0.97, production 0.006…); conditionals = product of legs. Measured effect:
  **~2× expected profit per mana**, model-free books shifted to ~90% price markets.
- **Backtest harness** (`backtest.py`): replay over real historical series → Brier vs 0.25
  baseline, skill, 10-bucket reliability, simulated PnL/ROI under real sizing;
  `--strategy/--limit/--n-samples/--param` overrides for calibration A/B.
- **Store/ledger** (`store/`): append-only `trade` table as source of truth; positions/PnL
  derived; resolution = synthetic RESOLUTION_CLOSE trade at 1.0/0.0; **CANCEL closes at cost
  basis** (realized 0) — the accounting design that makes 93% cancellation a non-event.
  Post-incident hardening (2026-05-28): cache-first resolution reads, refresh-before-resolve
  (was N+1 and ignored CANCEL).
- **Runner** (`runner.py`): load → prefilter → group → estimate → resolvability gate → size →
  allocate → execute → record; batches ≤50 with throttle-aware retries; **every bet posts a
  justification comment** (fair value, market price, signed edge, expected ROI, strategy
  explanation; optional local-LLM prose).
- **Dashboard**: React 19 + Vite + Tailwind v4 SPA ("Mission Control": IBM Plex, cyan signal,
  green/red PnL on near-black, per DESIGN.md) over Flask JSON+SSE — fleet leaderboard, bot
  detail + equity curve, live trade tape, strategies, markets; snapshot every 5s over SSE.
- **Ops**: launchd agents — daily cycle 09:00 (resolve → refresh → ingest → process → run
  --live → snapshot, all 24 bots; retired bots resolve-only via RESOLVE_ALSO), separate
  comment-judge and equity-options loops; Doppler-managed secrets; `quantbots status`
  monitoring; logs actively running through 2026-07-22.
- **CLI** (`cli.py`, 609 LOC): health / refresh / ingest / process / run / status / resolve /
  snapshot / backtest / llm-bench / link / sources / judge-comments / make.

---

## 3. Equity options — real-money carve-out

Fenced package `src/quantbots/equity_options/` trading **real listed options via Alpaca
paper** — an owner-approved exception to the clone-only rule, with safety as architecture.

**Safety design**
- May not import the Manifold code — **enforced by an AST import-scan test**
  (`tests/test_eo_safety.py`); absent from runner/CLI/registry/daily-cycle; own `eo` CLI
  (typer, 860 LOC), own SQLite, own launchd loops.
- Staged execution ladder dry → paper → gated-live; `execution/live.py` is a **refusing
  stub** (raises without owner-approval env + committed risk-limits file, and refuses even
  then) — **paper is the ceiling by construction**. Ungated kill switch: `eo flatten --paper`.

**Models**
- **Pricing/edge core:** per-structure edge = e^(−rT)·∫payoff·(f_P − f_Q) — the physical
  measure f_P is the same diffusion_mc Monte-Carlo density; f_Q is the option-implied
  density. Structures: long call/put, bull/bear call/put spreads, iron condor.
- **VRP harvesting** (`vrp.py`): delta-hedged variance-risk premium —
  PnL ≈ ∫½·Γ·S²·(σ²_impl − σ²_real)dt; sells vol only when ATM IV exceeds the diffusion
  forecast; iron condor as the defined-risk expression.
- **TSMOM v2** (`forecast/direction.py`): commodity time-series momentum, 12-month return
  skipping the last month, **multi-lookback blend (3/6/12m)** + a **trend-quality regime
  filter** (strength = |trend|/vol; abstain below threshold — momentum's whipsaw mode),
  propagated to equities via screened betas (shrink 0.7, cap ±0.35). Motivated by a live
  lesson: the book was systematically short a trending-up gold-miner complex.
- **Multi-factor research layer** (`research/factors.py`, `fusion.py`): momentum + FRED macro
  (real rate DFII10 for precious, dollar DTWEXBGS otherwise) + carry (point-in-time CSV) +
  CFTC positioning (**indexed by actionable date — no lookahead**) + tal consensus; fused by
  t-stat-prior weights renormalized over available factors; every factor must pass forward-
  return IC validation (`factor_validate.py`) before it trades.
- **Critical-minerals producer thesis** (`research/tal_signals.py`, `screen.py`): the
  consumer-cost metal→equity thesis **fails** (sector beta); the **producer side is strong
  and right-signed** (lithium→ALB/SQM t≈9–10). Multi-agent tal-market consensus tilt per
  material → bull/bear spreads on curated producers (copper→FCX/SCCO, gold→GDX/NEM/AEM/WPM,
  silver→AG/PAAS, lithium→ALB/SQM, rare-earth→MP…), weighted by equity↔metal correlation.

**The incident and the risk system**
2026-07-07: Alpaca paper re-marked the whole book to zero overnight (**−$17k phantom equity
cliff**, zero fills); the ledger disagreed with the broker and the bot re-armed into the
crater. Rebuilt around **broker-truth**: full-history reconciliation, ghost-leg settlement,
and a **latched circuit breaker** (`breaker.py` — trips on equity drop or ledger-open legs
absent at broker; persists in the store until a human `eo resume`). It has since latched
itself correctly in production (2026-07-13, "reconcile settled 2 filled ghost legs").

**The empirical result**
Bi-weekly walk-forward gate (Brier-skill ≥ 0.02 AND ≥12 trades): **only precious-metal
producers pass** — AEM +0.107 skill / 0.52 Sharpe, PAAS +0.078/0.45, FNV +0.06/0.34,
WPM +0.054/0.67, NEM +0.05/0.47, GDX +0.024/0.41. **Every energy name fails** (XOM −0.023,
CVX −0.036, COP −0.029, SLB −0.025); base metals marginal (SCCO +0.023 skill but −0.05
Sharpe). Live ledger: 260 legs, realized −$3,606 / total −$4,866 on $5,510 premium-at-risk;
87 closed / 2 open; posture = Path-1 break-even wind-down of the legacy book.

**Data pipeline:** SMM lithium-carbonate + NdPr-oxide (ex-VAT ÷1.13 alignment to tal's
SMM_SPOT_DAILY), GFEX lithium futures, tal Snowflake reader (read-only SELECTs under doppler,
parquet-cached); `metals_master_daily.csv` (727 rows) with back-cast segments (lithium←ALB
R²=0.86, NdPr←REMX R²=0.71) **explicitly quarantined from backtests** (circularity).

---

## 4. Local-compute LLM infrastructure

The "run good models on owned hardware" thread — spanning ~/Bots, ~/tal, and standalone ops.

- **Mac Studio model hosting (tal PR #817, MERGED 2026-04-16):** wired a self-hosted Ollama
  Gemma on the Mac Studio into tal's LiteLLM proxy as a bot backend; wrote
  `docs/operations/local-model-hosts.md`; secrets action + deploy wiring + safety-check test.
  First appearance of the owned-hardware thread.
- **Hardening (tal PR #863, MERGED 2026-04-21):** robust local-model bot calls in
  `utils/llm.py` / `trading_runner.py` (timeouts, think-mode off), provider tests.
- **Energy pricing / the weighted die (tal PR #1666, MERGED 2026-06-26):** the fleet samples
  models ∝ inverse measured $/call; local models bill $0 through the proxy so the die could
  never reprice them. `modelrank/energy.py` prices a local token by electricity:
  **$/token = (watts/1000) × ($/kWh) ÷ (tokens_per_sec × 3600)** — measured on the Mac
  Studio M3 Ultra (~68 W wall via powermetrics, ~89 tok/s gemma4 decode, $0.13/kWh),
  marginal-only so it's comparable to a cloud invoice; flows into the same mana-per-dollar
  leaderboard column; site methodology updated (local costs starred as estimates). Also added
  o3 and gemini-2.5-pro to the pool.
- **qwen3:8b on the die (local branches `feat/local-qwen3-bot-pool` /
  `feat/local-model-energy-pricing`):** measured ~93 tok/s, energy-priced, cold-start weight
  0.2, verified strict-JSON + tool-calling against the live host. Unmerged.
- **vLLM on Apple Metal (`~/vllm-metal-work`, May 2026):** scripts to serve Qwen3-32B-MLX-4bit
  on :8081 — vLLM ops scaffolding on Apple silicon.
- **Gemma keep-warm ops (`~/local-gemma-launchd`, April 2026):** launchd agents keeping a
  local Gemma-4/Ollama model warm + a concurrency benchmark.
- **quantbots LLM stack:** OpenAI-compatible client pointed at local endpoints only
  (local-only rule; mercury is the one sanctioned exception), an Ollama health watchdog, and
  **`llm-bench`** — the empirical model-selection harness (validity/coverage/error/latency)
  used for the qwen3/gemma4/Qwythos decisions.
- **Model reliability engineering:** percentile-anchoring and scale-sanity guards in the llm
  strategy, resilient per-ladder calls, configurable timeout/num_ctx, abstain-on-null fixes —
  the practical craft of making 8–32B local models produce tradeable numbers.
- **Open issue filed:** tal #871 — persist LiteLLM token usage for bot LLM evaluations (the
  observability gap behind the die).

---

## 5. tal platform contributions

*tal = GSMC's critical-minerals knowledge-graph + prediction-market platform (measurables →
markets → globe/surface UI → ~66-bot fleet, Snowflake-backed). Gross scale authored: ~50k+
insertions.*

**Merged PRs (8):** #817, #863, #877, #1211, #1390, #1588, #1599, #1666.

- **#877 — Compact measurable bot prompts (MERGED, +1,780/−132, 17 files):** rewrote
  `sources/bots/trading_strategies.py` (+610) and `research.py` (+345); added
  `context_budgets.py`, prompt-diagnostics DB migration (V124), `utils/bots/manage.py`,
  ~447 lines of strategy tests.
- **#1211 — Ebola/DRC measurables (MERGED, +2,008):** 6 measurables → 12 markets for the
  2026 DRC Bundibugyo outbreak (border closures, Bunagana crossing at 3 horizons, national
  emergency, M23/Goma, Alphamin Bisie suspension).
- **#1390 — Surface UI (MERGED, +219):** overlay live clone prices on post-trade refresh;
  plus an indium analyst report + essay.
- **#1588 / #1599 — company coverage (MERGED):** JPT Opto-electronics (688025.SS), Mitsui
  Kinzoku (5706.T).
- **#1256 — HMM regime-detection bot (authored, closed; +3,969, 127 unit tests):** the most
  technically ambitious PR — a Hidden Markov Model (**Baum-Welch + Viterbi in pure numpy**)
  over a 6-hour evidence corpus with QUIET/SIGNAL/STORM hidden states; **routing-only**
  (boosts wake-up scores of bots whose measurables are in a non-QUIET regime; never trades);
  layered false-positive defenses (state-separation guard, inference quality gate,
  freshness/confidence SQL filters, dormant by default); 6 new DB tables (migrations
  V162/V163). Walk-forward backtest, no leakage: **precision@100 = 76% vs 52% legacy**.
- **#1258 — lead/lag universe expansion (authored, closed; +225):** after each LLM
  classification pass, auto-insert 1-hop neighbours from MEASURABLE_LEAD_PLAUSIBILITY
  (score ≥ 50) into BOT_MARKET_UNIVERSE, bidirectionally.
- **#1283 — Cotton complex (authored, closed; +9,514):** 12 measurables → **153 markets
  published to the clone** — ICE Cotton No.2, Cotlook A, the cotton-polyester substitution
  spread (structural demand destruction), cash-futures squeeze basis, USDA S&D, long-run
  fiber share.
- **#1273 — Deep measurables (authored, closed; +22,805/−213 — largest single PR):** Solaris
  digital infrastructure (8 measurables / 71 markets: Oracle/IBM infra revenue, SRU cadence,
  OpenIndiana, illumos-gate commits), Siemens Energy gas-turbine backlog (24 markets),
  onshore wind EIA + DOE repowering (24), AI-datacenter electricity share + subsea-cable
  damage (24), diesel recalibration. Gas/wind/Solaris markets published to the clone.
- **Optical/AI-infra coverage stream (11 companies):** analyst reports + cheatsheets +
  measurables across the InP optical supply chain — substrate (Sumitomo Electric #1564) →
  epiwafer (IntelliEPI #1562) → laser chips (YJ Semiconductor #1567, MACOM #1569) → assembly
  (Fabrinet) → modules/test (Viavi #1586, EZconn #1591, JPT #1588) → materials (Mitsui
  #1599, Elite Material #1601) → OCS (Luster #1603); consolidated 5-company PR #1572
  (+2,635). ~1,000+ lines per package; dark-theme HTML analyst reports with Mermaid
  supply-chain maps and scenario tables. Early PRs closed because the measurable workflow
  moved from repo-YAML to a DB/API path mid-internship — the work re-shipped in the new
  shape; two later single-company PRs merged.
- **Local unmerged branches:** `mikhail/smm-history-backfill` (3-year SMM daily backfill,
  1,345 rows, ex-VAT, idempotent MERGE, validated equal on the 55-day overlap with the live
  scraper); `feat/alpaca-price-momentum-v2` (empty-universe validation fix atop the strategy
  framework migration); qwen3 die branches (§4).

---

## 6. Research repos

### Thurstone Models — "When Better Choice Models Do Not Matter" (flagship)
- **What:** pre-registered empirical audit + methods paper (Poddar & Brodhead) — does
  replacing Bradley-Terry (Chatbot Arena's ranking model) with a richer **Thurstonian
  lattice** link improve rankings? A rigorous **no**, and a reusable method for knowing that
  *before* building. 39 of 46 commits (2026-07-09 → 07-17); 42 pipeline scripts, fitting
  library, ~30 pre-registration/findings logs, ~90 result tables, 1,202-line LaTeX paper,
  4 pytest suites.
- **Data:** 1,799,991 LMSYS Arena battles / 129 models (2023–24; 1,670,250 deduped) + a
  135,634-battle / 53-model 2025 replication set. Pipeline reproduces the published
  leaderboard to **MAE 0.18 Elo, ρ = 0.99997**.
- **Models/inference:** BT logistic σ(g); lattice link (discretized noise, width u entangles
  tie mass with decisive steepness — slope 0.284–0.337 vs logistic 0.25); Davidson ties as
  the structural contrast. Direct MLE (L-BFGS-B, analytic gradients), production-equivalent
  pseudo-likelihood + native trinomial modes, gpt-4-0613 gauge anchor, profile likelihood
  for u.
- **Theorems:** **Thm 1 (tie orthogonality)** — Davidson's conditional decisive link is
  exactly logistic for every tie parameter. **Thm 2 (approximation-limited regime)** — the
  **effect ceiling** C = min-over-rescalings E[KL] between links under the empirical gap
  distribution is a sharp upper bound on the richer family's advantage, computable **before
  fitting**.
- **Results:** ceiling ≤ **0.23×** the practical-relevance threshold; RQ1 rank stability
  NULL (sample size matters ~26× more than link choice); RQ2 additivity indistinguishable;
  RQ3 held-out calibration equivalent (all 13 windows in-band; 2025 legs replicate); RQ4
  ties inconclusive. **Error budget:** model choice bounded at a fraction of one practical
  unit while estimation, cold-start coverage, and drift (recalibration worth 12.4×
  threshold) dwarf it by 1–2 orders of magnitude.
- **The retraction (integrity showcase):** an exciting "cold-start shrinkage" effect (+5.79)
  was a **spline-interpolation artifact** — corrected +0.017; self-caught, retracted,
  documented (Appendix A); real mechanism identified (explicit ridge, θ̂→0 over λ∈[1,3]).
- **The company bridge:** the same lattice machinery extended to **multiray global
  calibration** (per-bot latent Z∈R², per-condition rays, alternating least squares) —
  now the rating engine for **GSMC's bot fleet (97 resolved multi-bot races, 11 bots) and
  163 platform traders**; a held-out dimensionality test showed the 2nd skill dimension is
  memorization, not shared structure.

### DIffusion — Diffusion Factor Models (2026-06-15)
- From-scratch PyTorch implementation of arXiv:2504.06566 — a **score-based generative
  diffusion model for asset returns exploiting factor structure** R = βF + ε in the n ≪ d
  regime. OU forward process; score network s_θ(r,t) = α_t·D_t·V·g_ζ(VᵀD_t r, t) − D_t·r
  with **bottleneck width = factor dimension k** (tied orthonormal encoder/decoder = learned
  loadings; linear idiosyncratic skip per the paper's Lemma 1); denoising score-matching
  loss; reverse-SDE Euler-Maruyama sampling with EMA weights. Evaluated by mean/cov recovery,
  PCA subspace recovery, and a **global-min-variance portfolio test vs
  Ledoit-Wolf/OAS/factor/empirical baselines**. Trained on tal's real 19-asset metals panel
  (XAU/XAG/XPT/XPD + MB critical-minerals codes, 2018–2026). Thematically related to
  diffusion_mc but distinct code — generative panels vs terminal-price simulation.

### RMT — random-matrix correlation denoising (2026-07-13)
- Laloux-Cizeau-Bouchaud-Potters **Marchenko-Pastur eigenvalue clipping**: MP edges
  λ± = σ²(1±√q)², q = N/T, with the Bouchaud-Potters market-mode correction σ² = 1 − λ₁/N;
  eigenvalues below λ₊ replaced by their mean (trace-preserving), renormalized, variances
  reapplied. On **466 S&P equities × 2,511 days**: **92.7% of correlation eigenvalues
  indistinguishable from noise** (34 signal factors; λ₁≈162); condition number 5.6×10⁴ →
  1.4×10³. Walk-forward min-variance backtest (504d window, 21d rebalance, 58 rebalances):
  sample-cov Sharpe **0.00** (vol 34.7%, DD −52.8%) vs RMT-clipped **0.78** (vol 11.1%,
  DD −26.2%) vs equal-weight 0.51 — **68% realized-vol reduction**. Survivorship/cost
  caveats documented.

### GSMC — the v1 origin (June 2025)
- The original single-strategy prototype: a **cocoa rainfall-anomaly bot** (West-Africa
  rainfall + USDA FAS signal, LLM estimator, "cinch-factor" Ψ signal with a 3D viz, SQLite
  intention log, no-lookahead backtest with LLM-response caching). The seed that generalized
  into the entire quantbots platform.

---

## 7. Agentic systems

### sandbox-agents — CriticalMineralsBot + framework hardening
The company's local-first agent framework: an LLM in a Docker sandbox (colima) reads a
market snapshot and proposes trade *intents*; a deterministic host-side gate owns the
credential and executes bounded orders (position/order/liquidity/budget/impact caps; the
model has no trading tool — its only output is `intents.json`).

- **CriticalMineralsBot (authored):** expertise profile for lithium/cobalt/tantalum/NdPr —
  resolvability-aware rules (prefer exchange/public-feed price markets over cancel-prone
  operational markets; trade only >0.10 divergence with a concrete stated mispricing reason;
  unit-conversion discipline $/t vs $/kg; cite the data anchor + resolution source);
  universe terms + liquidity floor; single-cell intents (no fabricated grids —
  documented opt-out because the clone lists these metals as plain binaries).
- **Data pipeline (authored):** `scripts/export_metals_data.py` — exports curated real
  SMM/GFEX series (lithium carbonate, NdPr oxide, GFEX term-slope/tightness) from the
  equity-options research dataset into the sandbox's read-only data dir, de-credentialed,
  plus a hand-written supply/demand briefing that leads with the resolvability warning.
- **The anti-hallucination read-back gate (framework contribution, commit 7dccd00):** a live
  dry run caught qwen3-coder:30b quoting a market at **98.2% when the venue price was 5.6%**
  — a phantom 0.95-vs-0.056 gap the sizer would have traded (it sized Ṁ15 before the fix).
  Prompt fixes didn't stop it, so: a required `market_probability` field the model must echo
  verbatim, **rejected deterministically on >0.05 mismatch**, documented in the system
  prompt, covered by tests (hallucinated-price rejected / in-tolerance passes / missing
  rejected).
- **Service fixes (commit 7e450f2):** `install-service` hardcoded max-turns 25 (starving the
  agent, which needs ~45) and never passed cooldown — both made configurable through CLI +
  launchd/systemd generators.
- **Environment:** colima Docker context, Ollama qwen3-coder:30b, live-clone credentials +
  CF Access provisioned; agent account minted (0 positions ever).
- **Status:** end-to-end dry-run works against the live clone; the single `--execute` run
  **correctly declined to trade** (divergence 0.017 < 0.10 threshold). Go-live blockers,
  small and known: launchd PATH can't find the colima `docker` binary (service crash),
  colima not auto-started, exporter script uncommitted.

---

## 8. Data engineering & market creation

- **Source framework** (`sources/`): normalized `Observation(source, entity, ts, value, …)`
  keyed by canonical entity so multiple feeds describe one quantity; ingestion →
  observations cache; signal processing (`processing/signals.py`) persisted daily.
- **Keyless/keyed feeds built:** stooq (prices/FX/equities/softs), FRED (US macro), NOAA
  (ENSO/ONI), World Bank, RSS news, USDA FAS PSD (bulk CSV), USDA NASS QuickStats, CFTC
  DCOT (Socrata), open-meteo growing regions, CPC Atlantic SST, US Drought Monitor, ICE
  certified cocoa stocks, **Ornn Compute Price Index** (keyless GPU rental $/GPU-hr —
  H100/H200/B200/A100; a genuinely-resolving external price feed; ladder backtest showed
  promising tail edge on short history).
- **tal Snowflake access:** sanctioned read-only path from ~/Bots (doppler + dev role);
  9M expectation rows; finding: ticker reference is commodity-only and measurables are
  critical-minerals-heavy → use prediction-market price expectations as the signal.
- **SMM/GFEX paid-data pipeline:** authenticated-browser SMM pulls (36-month cap), GFEX free
  CSVs; committed critical-minerals dataset with documented ex-VAT alignment; 3-year SMM
  backfill branch in tal (1,345 rows, overlap-validated).
- **Real AMM liquidity provision:** `scripts/provide_ag_liquidity.py` — added **Ṁ40k of real
  pool depth to 134 thin ag markets** (100→400 depth), idempotent re-runs; discovered the
  clone rejects dryRun on add-liquidity and fixed the client with a local no-op; subsidy is
  refunded on CANCEL so recoverable. Taker-side coverage by llm_ag_coverage.
- **Market creation:** `create_h100_compute_markets.py` (compute-price markets off the Ornn
  index), `create_zirconium_conditionals.py` (Tier-1 nested conditional triangles from
  zircon price research — Iluka public quarterly as best leg; event-predicates rejected
  because the product rule kills resolvability).
- **Bot account minting:** `scripts/mint_bot_accounts.py` via a custom admin endpoint —
  every functional bot got its own clone account.
- **Research benches:** `diffusion_bench.py`, `backtest_softs.py`, `research_softs.py`,
  `research_pairs.py`, `ornn_backtest.py`, `momentum_ab.py`, `validate_news_leadlag.py`,
  carry/DCOT/SHFE builders.

---

## 9. Headline numbers & presentation notes

**Title-slide numbers**
1. Ṁ227,271 staked · 23,616 trades · 3,463 markets — a live autonomous 24-bot fleet
2. ~50k+ lines contributed to tal, 8 merged PRs
3. 1.8M battles, 2 theorems, 1 paper — the Thurstone audit
4. 93% → the resolvability insight that reshaped every bot's capital allocation (~2× profit
   per mana)
5. $/token = watts × $/kWh ÷ (tok/s × 3600) — local models priced onto the leaderboard by
   electricity

**Three headline anecdotes (one lesson at three layers — verify against ground truth):**
- *Statistical:* the self-caught Thurstone retraction (interpolation artifact).
- *Financial:* the −$17k phantom equity cliff → latched broker-truth circuit breaker.
- *Agentic:* the hallucinated 98.2%-vs-5.6% market price → deterministic read-back gate.

**Honest-numbers appendix (know before Q&A; don't lead with):**
- equity_options paper PnL −$4,866 — staged pipeline did its job; edge isolated to precious
  producers; legacy book in break-even wind-down.
- Personal trader accounts rank mid/low on the platform leaderboard (#14/#26; #77 of 163
  per-mana) — the bots and infrastructure are the product, not manual trading.
- Several large tal PRs closed rather than merged because the measurable workflow moved to a
  DB/API path mid-stream; the work re-shipped in the new shape and later coverage PRs merged.
- Market maker shelved and weak signals gated **deliberately** — measured before shipped.
- sandbox-agents go-live blockers are small and enumerated (launchd docker PATH, colima
  autostart, uncommitted exporter).

**Assets:** slide deck `docs/internship/internship-deck.html` (open in a browser; arrows to
navigate, `f` for fullscreen) · this dossier.
