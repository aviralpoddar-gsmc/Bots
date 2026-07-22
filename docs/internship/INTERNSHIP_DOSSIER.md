# GSMC Internship — Presentation Content

*Aviral Poddar (UMass) — identities: `aviralpoddar-gsmc`, `mikhailtal-design`, `Mikhail Tal` — May–July 2026.*

Ordered exactly as the presentation flows:

1. [Bots](#1-bots) — the full fleet, every bot, incl. consensus & adversarial
2. [Local compute & local bots](#2-local-compute--local-bots) — modelrank, the weighted die, hosting
3. [Measurables](#3-measurables) — markets authored + the stats
4. [Thurstone](#4-thurstone) — the bot ranking system + the research (BT replication → comparison)
5. [Equity options](#5-equity-options) — the Alpaca work
6. [Diffusion factor models](#6-diffusion-factor-models)
7. [Sandbox agents](#7-sandbox-agents) — CriticalMineralsBot
8. [Built but not deployed — negative results](#8-built-but-not-deployed--negative-results)

All figures from repos, git history, live databases, and PR records as of 2026-07-22.

---

## 1. Bots

### 1.0 The platform every bot runs on (context first)

**quantbots** (`~/Bots`, ~19,200 LOC Python, 55 test files): a reusable framework for the
private Manifold clone. Bot authors implement one function —
`estimate(group) → {market_id: fair_probability}` — and the platform supplies everything else:

- **Clone-only API client** — base URL hard-wired (safety invariant), CF-Access + key auth,
  rate limiting, batch bets, dry-run validation, limit-order primitives.
- **Sizing** — the 1/3-push rule under four caps (order size, 33% of pool liquidity, 10% max
  price impact via LMSR approximation, run budget) + a hold band against churn.
- **Portfolio allocation** — greedy knapsack on EV-per-mana with correlation-group and
  cross-run exposure caps.
- **The resolvability core (the signature insight):** measured over **9,578 resolved
  markets, ~93% resolve CANCEL** — a market only settles YES/NO when its named source
  publishes a value (LBMA precious ~100%, exchange-settled 27–40%, production 0.6%, demand
  ~0%). So every bot's allocation is ranked by **realized EV = paper EV × P(resolve)**, from
  a question-text score calibrated to observed decided-rates; conditionals score as the
  product of both legs. Measured effect: **~2× expected profit per mana**; the deepest
  mispricings live in markets that never resolve — chasing raw edge is an adverse-selection
  trap, and this is the immune system for it.
- **Ledger** — append-only trades; resolution = synthetic close; CANCEL closes at cost basis
  (realized 0), making 93% cancellation a non-event in the books.
- **Backtest harness** — Brier vs 0.25 baseline, skill, 10-bucket calibration, PnL under
  real sizing.
- **Ops + dashboard** — unattended launchd daily cycle for the whole fleet; per-bot minted
  clone accounts; Doppler secrets; "Mission Control" React dashboard (leaderboard, equity
  curves, live trade tape over SSE).
- **Origin:** grew out of the June-2025 v1 prototype — a single cocoa rainfall-anomaly bot
  (`~/GSMC`) — generalized into a platform of 32 registered strategies.

**Fleet totals (live DB, 2026-07-22): 23,616 trades · Ṁ227,271 staked · 3,463 distinct
markets · 1,441 resolutions realized · 106,915 markets cached (~62k universe, ~96% untraded
at 0.50).**

### 1.1 Structural / model-free arbitrage bots

#### ladder_arb_1 — monotonicity arbitrage
- **Status:** LIVE since 2026-05-27 · **3,939 trades · Ṁ25,265** (the fleet's biggest trader)
- **Model:** within a (metric, resolution-date) strike ladder, P(value > K) must be
  non-increasing in K. Fits the nearest coherent curve by **weighted isotonic regression via
  PAVA** (pool-adjacent-violators, O(n), pure stdlib) and trades off-curve strikes toward the
  fit. Informative weighting (traded/moved strikes ×5, untraded 0.50s ×1); clamp-pinned
  strikes anchor but are never traded; date-aware grouping keeps expiries separate.
- **Why it matters:** domain-agnostic — no data feed, no opinion about the world, only
  internal coherence. ~50k markets in scope.
- **File:** `src/quantbots/strategies/ladder_arb.py`

#### term_structure_1 — time-axis coherence
- **Status:** LIVE since 2026-05-27 · **1,715 trades · Ṁ4,718**
- **Model:** the orthogonal axis to ladder_arb — hold (metric, threshold) fixed, vary
  resolution date; P(value > K) should trace a smooth curve in time. A **Gaussian kernel
  smoother** (6-month bandwidth) fills stale 0.50 dates from *traded anchors only*, with a
  0.5-prior pseudo-count so it never extrapolates confidently far from data; needs ≥2
  anchors, ≥3 dates.
- **File:** `src/quantbots/strategies/term_structure.py`

#### stockpile_grid_arb_1 — 2-D monotone surface arbitrage
- **Status:** LIVE · 22 trades · Ṁ112 (deliberately small; resolvability ~0.03)
- **Model:** vault-procurement ladders form a 2-D grid (strike × expiry); survival must be
  **monotone-down in strike AND monotone-up in expiry** (procurement is cumulative). Fits the
  nearest 2-D monotone surface by **cyclic isotonic projection** — alternating weighted PAVA
  down strike-lines and up expiry-lines to convergence. Cancel-safe breadth play.
- **File:** `src/quantbots/strategies/stockpile_grid_arb.py`

#### conditional_arb_1 — Fréchet-bound coherence on conditional markets
- **Status:** LIVE (built 2026-06-19) · 28 trades · Ṁ170
- **Model:** for "IF [A]=YES: B" markets pricing c = P(B|A), the law of total probability
  gives **Fréchet bounds: max(0, (a+b−1)/a) ≤ c ≤ min(1, b/a)**; the nested case (quantity
  implies predicate, e.g. zinc≥3400 ⟹ zinc≥2800) is **exact: c = b/a**. Trades only the
  conditional toward the band; abstains if legs are survival-inverted (lets ladder_arb repair
  first) or the predicate < 0.05. Resolvability = product of legs; downside capital-neutral
  (predicate-NO → CANCEL refund).
- **Validation:** 3 zinc triangles fire; rhodium correctly suppressed.
- **Files:** `src/quantbots/strategies/conditional_arb.py`, `tests/test_conditional_arb.py`

#### surface_arb_1 — parametric ladder fitting
- **Status:** LIVE · 225 trades · Ṁ7,296
- **Model:** fits a **normal CDF** to a measurable's strike ladder and trades strikes toward
  the fitted curve — the parametric cousin of ladder_arb; `strategies/ladder.py` parses
  threshold/direction from question text.
- **File:** `src/quantbots/strategies/surface_arb.py`

#### semantic_arb — LLM-linked logical arbitrage
- **Status:** built; targets the duplicate-market problem
- **Model:** cross-market relations the structural bots can't link syntactically. A local
  LLM asserts only relations true **by meaning alone** — equivalent (P(a)=P(b)), negation
  (1−P(b)), implies (P(a)≤P(b)), exclusive (P(a)+P(b)≤1). Prices are **never shown to the
  model**; O(n) candidate blocking by shared-rarest-token + date; self-consistency voting
  over a temperature spread; then **POCS (iterated convex projection)** onto the feasible
  region with partial correction (0.7). Target: ~2,485 verified byte-identical duplicate
  market sets trading at different prices.
- **File:** `src/quantbots/strategies/semantic_arb.py`

#### stockpile_facts_1 — reference lookup on strategic-materials facts
- **Status:** LIVE · 624 trades · Ṁ4,106 (the alpha bot of the 3-bot stockpile set)
- **Model:** answers U.S. strategic-materials *fact* markets from curated public record —
  USGS 2022 Critical Minerals list (50 minerals, baked-in frozenset) → P = 0.93/0.07;
  National Defense Stockpile positions (GAO-24-106959 / CRS R47833) → 0.85/0.12; abstains
  where the record is ambiguous.
- **File:** `src/quantbots/strategies/stockpile_facts.py`

#### stockpile_coherence_1 — buffer-stock/policy coherence
- **Status:** LIVE · 3 trades · Ṁ17
- **Model:** consistency across buffer-stock and policy markets (sibling of the above).
- **File:** `src/quantbots/strategies/stockpile_coherence.py`

### 1.2 Stochastic-process pricing bots

#### diffusion_mc_1 — kernel-smoothed bootstrap Monte Carlo  ★ promoted
- **Status:** LIVE since 2026-06-04 as @DiffusionMcBot, **promoted over commodity_spot_1** ·
  **3,388 trades · Ṁ22,811**
- **Model:** prices P(price > K at T) from a simulated terminal distribution. Default
  process = **kernel-smoothed block bootstrap**: resample ~10-trading-day blocks of demeaned
  (zero-drift) daily log-returns (blocks preserve vol clustering), convolve each day with a
  **variance-preserving Student-t kernel** at Silverman bandwidth h = 0.9·n^(−0.2) so sims
  can exceed any observed move (fixes the bootstrap's bounded-tail hole), compound over
  round(T·252) days. Alternatives: plain bootstrap, fitted Student-t (df 3–15 per
  commodity), Gaussian-jitter hybrid. 10y yfinance calibration; lognormal fallback.
- **Validation (the gate story):** first walk-forward gate said *no edge*; diagnosed the
  artifact (uncapped-from-0.50 backtest), re-ran under the real per-market stake cap, and
  the verdict **reversed**: beats the lognormal on **Brier, PnL, Sharpe AND worst-fold
  across 8 commodities × 5 folds**; edge peaks at 21–63-day horizons.
- **Files:** `src/quantbots/strategies/diffusion_mc.py`, `scripts/diffusion_bench.py`

#### commodity_spot_1 — data-anchored lognormal (retired parent)
- **Status:** RETIRED 2026-06-04, resolve-only (superseded by diffusion_mc) · 2,102 trades ·
  Ṁ14,865 · LIVE from 2026-05-27
- **Model:** zero-drift lognormal on genuine spot-price ladders — P = 1 − Φ(ln(K/S)/σ√T),
  σ = max(annual_vol·√T, min_vol); per-commodity vols (oils 0.39–0.45, copper 0.21, gold
  0.16, silver 0.30); horizon cap 1.25y. Lasting contribution: the **strict unit/currency
  guard** (the "confidently-wrong firewall"), inherited by every subclass — requires the
  quoted unit, rejects foreign-currency quotes (CNY/EUR) and chemical compounds
  (sulfate/oxide/carbonate); feed→market conversion factors (silver cents/oz→$/oz ×0.01,
  copper cents/lb→$/MT ×22.0462).
- **File:** `src/quantbots/strategies/commodity_spot.py`

#### pair_trading_1 — cointegration convergence overlay
- **Status:** LIVE since 2026-06-01 · **2,088 trades · Ṁ13,841** (was silently excluded from
  the cron BOTS array at first — found and fixed)
- **Model:** subclasses commodity_spot; research layer fits OLS hedge β, spread mean/std,
  and **Ornstein-Uhlenbeck half-life** per pair. Expected convergence
  E[s_T − s_0] = (μ − s_0)(1 − e^(−θT)), θ = ln2/half-life, injected as log-drift, damped by
  reversion_capture = 0.5, attributed to one leg (partner = martingale anchor); fires past
  entry_z = 1.5. 11 resolvable metal/energy pairs (GOLD/SILVER, WTI/BRENT, PT/PD, …).
- **Files:** `src/quantbots/strategies/pair_trading.py`, `scripts/research_pairs.py`

#### ensemble_1 — deterministic multi-source fusion
- **Status:** LIVE · 866 trades · Ṁ9,943
- **Model:** linker maps question → entity + threshold; each numeric observation contributes
  P = 1 − Φ(ln(T/V)/σ), fused as a source-weighted average; per-entity vol; plausibility
  guard (max_ratio 20 drops mis-links); horizon-scaled σ.
- **Validation:** vol retune 0.5 → 0.15 = **+41% Brier skill** on the FRED mortgage series.
- **File:** `src/quantbots/strategies/ensemble.py`

#### commodity_1 — soft-commodity futures
- **Status:** LIVE · 829 trades · Ṁ8,455
- **Model:** same lognormal family for ag futures price markets (cotton/sugar/wheat/corn/
  cocoa; Stooq/ICE/CBOT catalog).
- **File:** `src/quantbots/strategies/commodity_futures.py`

#### enso_1 — climate persistence
- **Status:** LIVE · 637 trades · Ṁ5,892
- **Model:** ENSO/Oceanic Niño Index markets (NOAA). **Additive Gaussian persistence** (ONI
  isn't a positive price — no lognormal): P(ONI > T) = 1 − Φ((T − V)/σ), σ = monthly_vol·√months.
- **File:** `src/quantbots/strategies/enso.py`

#### mean_reverter — reference implementation
- **Status:** LIVE (reference) · 6 trades · Ṁ61
- **Model:** fades the market toward an EMA of its own price — the "how to write a bot"
  template.
- **File:** `src/quantbots/strategies/mean_reversion.py`

### 1.3 Fundamental / single-source signal bots

Shared base `SignalDriftStrategy`: each bot draws alpha from exactly **one** external
source, expressed as a bounded annualized log-drift on the shared price anchor, priced
through the lognormal CDF; abstains unless the drift clears `min_drift`.

#### cotton_fundamental_1 — USDA stocks-to-use drift
- **Status:** LIVE · 433 trades · Ṁ9,496
- **Model:** USDA FAS PSD world-ex-China cotton stocks-to-use → ±3–5%/yr drift on ICE
  cotton. **The one fundamental signal that beat zero-drift out of sample** (ex-China SUR
  b = −0.39).
- **Validation:** Brier 0.1482 = **+40.7% skill**, near-perfect calibration; USDA drift
  lowers Brier 0.1537 → 0.1482; direction hit-rates 71–79%.

#### cocoa_fundamental_1 — vol-anchored cocoa
- **Status:** LIVE · 137 trades · Ṁ14,362
- **Model:** vol-anchored zero-drift lognormal (no USDA PSD for cocoa; ICCO only).
  Backtest +40.5% skill (high-biased, documented).

#### coffee_consumption_1 — FAS consumption growth
- **Status:** LIVE · 48 trades · Ṁ8,000
- **Model:** FAS consumption-growth normal CDF. +26.2% backtest skill but ~0% practical
  resolvability — kept small. (The stronger coffee price signal was untradeable — see §8.)

#### fas_balance_1 — balance carry-forward
- **Status:** LIVE · 430 trades · Ṁ4,218
- **Model:** carries FAS balance-sheet figures forward to cover far-dated cotton *quantity*
  markets otherwise stuck at 0.50.

#### cftc_softs_1 — positioning drift
- **Status:** LIVE · 548 trades · Ṁ18,044
- **Model:** CFTC Disaggregated Commitments of Traders positioning → bounded drift.

#### weather_cocoa_1 — growing-region weather anomaly
- **Status:** LIVE (T1 of the ags-weather push, shipped 2026-06-02) · 232 trades · Ṁ15,000
- **Model:** open-meteo growing-region weather anomalies → cocoa drift.

#### nass_cotton_1 — crop-condition index
- **Status:** LIVE (abstains without a NASS key) · 11 trades · Ṁ637
- **Model:** **production-weighted per-state cotton condition index** from USDA NASS
  QuickStats.

#### wasde_event — report-surprise overlay
- **Status:** built, gated (abstains until the next WASDE print)
- **Model:** WASDE cotton ending-stocks revision surprise as an event overlay.

#### news_drift_1 ("007") — news-driven drift
- **Status:** LIVE since 2026-06-09 as @Bot007 · **2,760 trades · Ṁ20,691**
- **Model:** a **local LLM digests commodity news RSS** (Investing.com / OilPrice / Mining)
  into per-commodity signed direction signals ∈ [−1,1], confidence-weighted and
  **recency-decayed (36h half-life)**, applied as a small bounded drift (k = 0.08); abstains
  unless ≥2 fresh directionally-clear headlines. Rejected GDELT as a source; fixed the
  closed-market 403 bug in the shared base en route.
- **Files:** `src/quantbots/strategies/news_drift.py`, `scripts/validate_news_leadlag.py`

### 1.4 LLM forecasting bots

#### llm_forecaster — percentile → CDF (local models)
- **Status:** LIVE (local qwen3) · 118 trades · Ṁ785
- **Model:** **one local-model call per measurable** returns p10/p25/p50/p75/p90; fit a
  normal (μ = p50; σ averaged from the 10–90 span [z = 2.5631] and IQR [z = 1.3490]),
  widened ×1.5 against measured local-model overconfidence (57–71% coverage vs ideal 80%);
  **every strike on the ladder is then read off the analytic CDF** — one call prices the
  whole ladder. Confidence cap 0.80. Model selection is empirical via `llm-bench` (§2).
- **File:** `src/quantbots/strategies/llm.py`

#### llm_ag_coverage — scoped coverage bot
- **Status:** shelved after its coverage push · 1,832 trades · Ṁ14,575
- **Model:** same percentile→CDF engine scoped to the cocoa/coffee/corn/beef "0.50 demo
  sea" — the taker side of the ag liquidity-provision push (Ṁ40k of real AMM depth added to
  134 thin markets, recoverable on CANCEL).

#### mercury_ensemble_1 — Bayesian-mixture calibration (hosted-inference exception)
- **Status:** LIVE since 2026-06-11 as @MercuryEnsembleBot, scoped to resolvable price
  markets · 224 trades · Ṁ1,465 · owner-approved hosted experiment (reverts to local if it
  doesn't beat the local baseline)
- **Model:** samples Mercury (Inception Labs) **N=20× over temperatures 0.4–1.0**; mixes
  per-sample CDFs into a posterior-predictive p̄; **law-of-total-variance decomposition** —
  epistemic = Var[pᵢ] (sampler disagreement), aleatoric = p̄(1−p̄) − epistemic; a
  **direction-agreement gate** abstains when <70% of samples agree on side;
  **disagreement-shrinkage** toward market: confidence = clamp(1 − epistemic/τ),
  estimate = market + (p̄ − market)·confidence (τ = 0.04); quorum ≥12 valid samples.
- **Files:** `src/quantbots/strategies/mercury_ensemble.py`, `_mixture.py`,
  `docs/mercury-ensemble-calibration.md`

### 1.5 Consensus & adversarial bots (the comment society)

Built 2026-07-08 on the AIA Forecaster paper (arXiv:2511.07678) — the Bridgewater
supervisor pattern. Clone context: ~1M comments from 66 tal bots.

#### consensus_1 — extremized comment consensus
- **Status:** LIVE 2026-07-08 · 357 trades · Ṁ2,313 · 108 consensus rows computed
- **Model:** pools each market's bot-bettor crowd (one implied probability per user = their
  latest bet's probAfter; ≥3 forecasters required) into a mean, then **Platt-extremizes**:
  p̂ = σ(√3·logit(p̄)) — the paper's correction for ensembles that hedge toward 0.5 — and
  trades toward the extremized consensus.
- **Files:** `src/quantbots/strategies/comment_consensus.py`,
  `src/quantbots/comments/consensus.py`

#### The adversarial judge — evidence-only comment auditing (pipeline)
- **Status:** LIVE — own launchd loop (`com.quantbots.comments.plist`); verdicts to date:
  **797 sound · 211 unsound (high confidence) · 501 noise**
- **Model:** implements the paper's *negative* result correctly — critiquing forecaster
  *reasoning* is worse than nothing; value comes only from **gathering independent evidence
  and overriding at high confidence**. The judge builds an evidence pack from ingested feeds
  (stooq/LBMA/FRED/NOAA + parsed threshold, unit-converted) and marks a comment `unsound`
  **only if its factual claims contradict the evidence numbers** — never for "bad
  reasoning". Anti-false-positive rules: market price is never evidence; no self-derived
  unit conversions; ≤3-day staleness cap on high-confidence verdicts. All local inference
  (qwen3:32b).
- **Files:** `src/quantbots/comments/{judge,cycle}.py`, `scripts/comment_judge_cycle.sh`

#### adversary_metals_1 — the comment-fade bot
- **Status:** LIVE 2026-07-08 · 9 trades · Ṁ61
- **Model:** turns actionable verdicts (unsound + high-confidence + attached bet) into a
  fair-value tilt **against** the bad commenter's position — but `_fights_anchor` only fades
  bets on the *wrong side of the data anchor*: wrong-reasoning-but-right-conclusion is never
  faded. Shift 0.05 per verdict, cap 0.12.
- **File:** `src/quantbots/strategies/comment_fade.py`

*(Bots built but deliberately not deployed — market maker, Atlantic-SST cocoa,
drought-cotton, cocoa-stocks, dup_arb — are in §8.)*

---

## 2. Local compute & local bots

The "run good models on owned hardware" thread — hosting, benchmarking, and pricing local
models so the fleet can actually use them.

### Hosting & ops
- **Mac Studio model hosting (tal PR #817, MERGED 2026-04-16):** wired a self-hosted Ollama
  Gemma on the Mac Studio into tal's LiteLLM proxy as a bot backend; wrote
  `docs/operations/local-model-hosts.md`; secrets/deploy wiring + safety-check test. First
  appearance of the owned-hardware thread.
- **Hardening (tal PR #863, MERGED 2026-04-21):** robust local-model bot calls (timeouts,
  think-mode off) in `utils/llm.py` / `trading_runner.py`, provider tests.
- **vLLM on Apple Metal (`~/vllm-metal-work`):** scripts serving Qwen3-32B-MLX-4bit on
  :8081 — vLLM ops on Apple silicon.
- **Gemma keep-warm (`~/local-gemma-launchd`):** launchd agents keeping local Gemma-4/Ollama
  warm + a concurrency benchmark.
- **quantbots LLM stack:** OpenAI-compatible client pointed at local endpoints only (the
  local-only rule; mercury is the one sanctioned exception) + an Ollama health watchdog.

### modelrank & the weighted die
- **The die** (`sources/bots/model_pool.py`): each tal bot wakeup pins one model deployment,
  sampled with weight ∝ inverse measured $/call from the modelrank snapshot × a yaml weight,
  clamped to a band — cheap models get more rolls, expensive ones keep an exploration share.
- **The problem:** local Ollama models bill **$0** through the LiteLLM proxy → cost reads
  null → no leaderboard rank, and the die can never reprice them off cold-start weight.
- **The fix — energy pricing (tal PR #1666, MERGED 2026-06-26):** `modelrank/energy.py`
  prices a local token by host electricity:

  **$/token = (watts / 1000) × ($/kWh) ÷ (tokens_per_sec × 3600)**

  Measured on the Mac Studio M3 Ultra: ~68 W wall (powermetrics), ~89 tok/s decode
  (ollama/gemma4), $0.13/kWh — all env-overridable; marginal-only (no capex/idle) so it's
  comparable to a cloud per-token invoice; flows into the same mana-per-dollar column and
  the same die weighting. Site methodology updated (local costs starred as estimates). Also
  added openai/o3 (w=0.05) and gemini-2.5-pro (w=0.1) to the pool.
- **Follow-on (local branches):** `ollama/qwen3:8b` measured (~93 tok/s), energy-priced,
  cold-start weight 0.2, verified strict-JSON + tool-calling against the live host
  (`feat/local-qwen3-bot-pool` / `feat/local-model-energy-pricing`, unmerged).
- **Observability gap filed:** tal issue #871 — persist LiteLLM token usage for bot LLM
  evaluations.

### Making the tal bot fleet cheaper/smarter
- **Compact bot prompts (tal PR #877, MERGED, +1,780/−132):** rewrote
  `trading_strategies.py` (+610) and `research.py` (+345); `context_budgets.py`;
  prompt-diagnostics DB migration; ~447 lines of strategy tests.
- **HMM regime-detection router (tal PR #1256, authored; +3,969, 127 unit tests):**
  a Hidden Markov Model (**Baum-Welch + Viterbi in pure numpy**) over a 6-hour evidence
  corpus with QUIET/SIGNAL/STORM hidden states — **routing-only**: boosts wake-up scores of
  bots whose measurables are in a non-QUIET regime, never trades. Layered false-positive
  defenses; 6 new DB tables. Walk-forward, no leakage: **precision@100 = 76% vs 52%
  legacy**.
- **Lead/lag universe expansion (tal PR #1258, authored; +225):** after each LLM
  classification pass, auto-insert 1-hop neighbours from the measurable lead-lag
  plausibility graph (score ≥ 50) into each bot's market universe, bidirectionally.

### Model selection & reliability (the craft of local models)
- **`llm-bench`:** the empirical model-selection harness — validity / p10-p90 coverage /
  p50 error / latency; drove the qwen3:8b → qwen3:32b and gemma4 choices.
- **Qwythos-9B vs gemma4 A/B:** llm-bench + full dry-run diff → kept gemma4 (Qwythos: half
  coverage, flakier); found and fixed 2 llm.py crash bugs in the process.
- **Reliability engineering:** percentile anchoring to strike scale + scale-sanity guards,
  resilient per-ladder calls, explicit timeouts / num_ctx, qwen3 no-think mode,
  abstain-on-null instead of crashing.
- **Local bots on the clone (cross-ref §1):** llm_forecaster, llm_ag_coverage, news_drift
  (007), semantic_arb, and the comment judge all run on local models exclusively.

---

## 3. Measurables

Authoring the supply-demand ontology → markets layer in tal, plus creating markets directly
on the clone. Gross scale across tal PRs: **~50k+ insertions**.

### Measurable/market packages
- **Ebola / DRC (tal PR #1211, MERGED 2026-05-22, +2,008):** 6 measurables → **12 markets**
  for the 2026 DRC Bundibugyo outbreak — countries affected, border-crossing closures
  (Bunagana at 3 horizons), national emergency, M23/Goma, Alphamin Bisie suspension.
- **Cotton complex (tal PR #1283, authored, +9,514):** 12 measurables → **153 markets
  published to the clone** — ICE Cotton No.2 futures, Cotlook A index, the cotton-polyester
  substitution spread (structural demand destruction), cash-futures squeeze basis, USDA S&D
  balance sheet, long-run fiber share.
- **Deep measurables (tal PR #1273, authored, +22,805/−213 — largest single PR):** Solaris
  digital infrastructure (8 measurables / **71 markets**: Oracle/IBM infra revenue, SRU
  cadence, OpenIndiana, illumos-gate commits), Siemens Energy gas-turbine backlog (24
  markets), onshore wind EIA + DOE repowering (24), AI-datacenter electricity share +
  subsea-cable damage (24), diesel recalibration. Gas/wind/Solaris markets published.
- **Surface UI (tal PR #1390, MERGED):** overlay live clone prices on post-trade refresh;
  plus an indium analyst report + essay.

### Company coverage (analyst reports + cheatsheets + measurables)
**11 companies across the InP optical / AI-infrastructure supply chain** — substrate
(Sumitomo Electric #1564) → epiwafer (IntelliEPI #1562) → laser chips (YJ Semiconductor
#1567, MACOM #1569) → assembly (Fabrinet) → modules/test (Viavi #1586, EZconn #1591,
JPT #1588 **merged**) → materials (Mitsui Kinzoku #1599 **merged**, Elite Material #1601) →
OCS (Luster #1603); consolidated 5-company PR #1572 (+2,635). ~1,000+ lines per package;
dark-theme HTML analyst reports with Mermaid supply-chain maps, scenario tables, and
factual-verification sections. Early PRs closed because the measurable workflow moved from
repo-YAML to a DB/API path mid-internship — the work re-shipped in the new shape.

### Market creation on the clone + data behind it
- **H100 compute markets** (`scripts/create_h100_compute_markets.py`): minted compute-price
  markets off the **Ornn Compute Price Index** — a keyless GPU-rental $/GPU-hr feed
  (H100/H200/B200/A100) that genuinely resolves; ladder backtest showed promising tail edge
  on short history.
- **Zirconium conditionals** (`scripts/create_zirconium_conditionals.py`): Tier-1 nested
  conditional triangles from zircon price research (Iluka public quarterly = best leg;
  event-predicates rejected — the product rule kills their resolvability).
- **Real AMM liquidity provision:** Ṁ40k of pool depth into **134 thin ag markets**
  (100→400 depth), idempotent, refunded on CANCEL; discovered the clone rejects dryRun on
  add-liquidity and fixed the client.
- **SMM history backfill (tal branch, unmerged):** 3-year SMM daily history (lithium
  carbonate + PrNd oxide, **1,345 rows**, ex-VAT ÷1.13), idempotent MERGE, validated equal
  on the 55-day overlap with the live scraper.
- **Sources built for the stats layer:** stooq, FRED, NOAA ENSO, World Bank, RSS, USDA FAS
  PSD, USDA NASS, CFTC DCOT, open-meteo, CPC Atlantic SST, US Drought Monitor, ICE cocoa
  stocks, Ornn — all normalized to `Observation(entity, ts, value)` and cached.
- **tal Snowflake access:** sanctioned read-only path (doppler + dev role); 9M expectation
  rows; finding — ticker reference is commodity-only and measurables are
  critical-minerals-heavy, so prediction-market price expectations are the usable signal.

---

## 4. Thurstone

**The bot ranking system + the research paper** — "When Better Choice Models Do Not Matter"
(Poddar & Brodhead; 39 of 46 commits, 2026-07-09 → 07-17; 42 pipeline scripts, ~30
pre-registration/findings logs, ~90 result tables, 1,202-line LaTeX paper, 4 pytest suites).

### Step 1 — replicate Bradley-Terry (the production baseline)
- **Data:** 1,799,991 LMSYS Chatbot Arena battles / 129 models (Apr 2023–Aug 2024; 1,670,250
  deduped) + a 135,634-battle / 53-model 2025 replication set.
- **Replication quality:** reproduces the published Arena leaderboard to **MAE 0.18 Elo,
  ρ = 0.99997** (0.18–1.01 across 9/10 snapshots; the one outlier diagnosed to an
  hours-old entrant). Critical detail found on the way: production pools `tie` +
  `tie (bothbad)` — dropping both-bad puts BT ~13–19 Elo off; pooled ~0.2.
- **Inference:** direct MLE (L-BFGS-B, analytic gradients), production-equivalent
  pseudo-likelihood + native trinomial modes, gpt-4-0613 gauge anchor, profile likelihood
  for the lattice width; cubic-Hermite log-splines to fix optimizer stalls at kinks.

### Step 2 — run the comparison (BT vs the Thurstonian lattice)
- **Models:** BT logistic σ(g); the **lattice-Thurstone link** (discretized noise; a single
  width parameter u *entangles* tie mass with decisive steepness — slope 0.284–0.337 vs
  logistic's 0.25); **Davidson ties** as the structural contrast.
- **Theorem 1 (tie orthogonality):** Davidson's conditional decisive link is exactly
  logistic for every tie parameter — its tie knob provably never touches decisive
  predictions, unlike the lattice's.
- **Theorem 2 + the effect ceiling:** C = min-over-rescalings E[KL] between the links under
  the empirical gap distribution — an upper bound on the richer family's possible advantage,
  **computable before fitting** (sharp in-family). Result: ceiling ≤ **0.23×** the
  practical-relevance threshold.
- **Empirical confirmation (4 pre-registered RQs):** rank stability NULL (median paired
  Δτ_b = +0.00026; sample size matters **~26×** more than link choice); triple additivity
  indistinguishable; held-out calibration equivalent (all 13 windows in-band; every 2025 leg
  replicates); ties inconclusive (driven by one near-peer month; band *shapes* provably
  differ 1.7× but the data can't discriminate).
- **The error budget:** model choice is bounded at a fraction of one practical unit;
  estimation noise, cold-start coverage (24.8–75.8% of next-month votes unscoreable), and
  drift (recalibration worth **12.4×** threshold) each dwarf it by 1–2 orders of magnitude.
- **The retraction (integrity showcase):** an exciting "cold-start shrinkage" effect
  (+5.79) was a **spline-interpolation artifact** — corrected value +0.017; self-caught,
  retracted, documented (Appendix A); the real protection identified as explicit ridge
  regularization.

### Step 3 — the bot rating system (the company bridge)
- The same vendored lattice machinery, extended to **multiray global calibration**
  (per-bot latent skill vectors Z∈R², per-condition rays V + offsets β, priced through the
  lattice race machinery, fit by alternating least squares) — now the rating engine for
  **GSMC's own bot fleet (97 resolved multi-bot races, 11 bots) and 163 platform traders**
  (pairwise leaderboards in scripts/33–34, multiray in 41–42).
- Honest finding: a held-out dimensionality test showed the **2nd skill dimension is not
  real** (per-condition memorization, not shared structure) — see §8.

---

## 5. Equity options

Fenced package `src/quantbots/equity_options/` trading **real listed options via Alpaca
paper** — owner-approved carve-out with safety as architecture.

### Safety design
- May not import the Manifold code — **enforced by an AST import-scan test**; absent from
  all clone entry points; own `eo` CLI (typer, 860 LOC), own SQLite, own launchd loops.
- Staged ladder dry → paper → gated-live; `execution/live.py` is a **refusing stub** —
  **paper is the ceiling by construction**; ungated kill switch `eo flatten`.

### Models
- **Pricing/edge core:** per-structure edge = e^(−rT)·∫payoff·(f_P − f_Q) — physical
  measure f_P from the same diffusion_mc Monte-Carlo machinery, f_Q the option-implied
  density; structures long call/put → verticals → iron condor.
- **VRP harvesting:** delta-hedged variance-risk premium —
  PnL ≈ ∫½·Γ·S²·(σ²_impl − σ²_real)dt; sells vol only when ATM IV exceeds the diffusion
  forecast.
- **TSMOM v2:** commodity time-series momentum (12-month return skipping the last month),
  **multi-lookback blend (3/6/12m)** + **trend-quality regime filter**
  (strength = |trend|/vol, abstain below threshold), propagated to equities via screened
  betas (shrink 0.7, cap ±0.35). Motivated by a live lesson: the book was systematically
  short a trending-up gold-miner complex.
- **Multi-factor layer:** momentum + FRED macro (real rate for precious, dollar otherwise) +
  carry (point-in-time CSV) + CFTC positioning (**indexed by actionable date — no
  lookahead**) + tal consensus; t-stat-prior fusion weights; every factor must pass
  forward-return IC validation before trading.
- **Producer thesis:** consumer-cost metal→equity **fails** (sector beta); the **producer
  side is strong and right-signed** (lithium→ALB/SQM t≈9–10) → tal-consensus tilt per
  material → bull/bear spreads on curated producers (copper→FCX/SCCO, gold→GDX/NEM/AEM/WPM,
  silver→AG/PAAS, lithium→ALB/SQM, rare-earth→MP), weighted by equity↔metal correlation.

### The incident → the risk system
2026-07-07: Alpaca paper re-marked the whole book to zero overnight (**−$17k phantom equity
cliff**, zero fills); the ledger disagreed with the broker and the bot re-armed into the
crater. Rebuilt around **broker-truth**: full-history reconciliation, ghost-leg settlement,
and a **latched circuit breaker** (trips on equity drop or ledger-open legs absent at
broker; persists until a human `eo resume`). It has since latched itself correctly in
production (2026-07-13).

### The empirical result
Bi-weekly walk-forward gate (Brier-skill ≥ 0.02 AND ≥12 trades): **only precious-metal
producers pass** — AEM +0.107 skill / 0.52 Sharpe, PAAS +0.078/0.45, FNV +0.06/0.34, WPM
+0.054/0.67, NEM +0.05/0.47, GDX +0.024/0.41. **Every energy name fails** (XOM −0.023, CVX
−0.036, COP −0.029, SLB −0.025); base metals marginal. Live ledger: 260 legs, realized
−$3,606 / total −$4,866 on $5,510 premium-at-risk; 87 closed / 2 open; posture = Path-1
break-even wind-down of the legacy book. Framing: paper-stage tuition that bought a
production-grade risk system and a validated, narrow edge.

### Data pipeline
SMM lithium-carbonate + NdPr-oxide (ex-VAT ÷1.13 alignment to tal's SMM_SPOT_DAILY), GFEX
lithium futures, tal Snowflake reader (read-only under doppler, parquet-cached);
`metals_master_daily.csv` (727 rows) with back-cast segments (lithium←ALB R²=0.86,
NdPr←REMX R²=0.71) **explicitly quarantined from backtests** (circularity).

---

## 6. Diffusion factor models

From-scratch PyTorch implementation of **"Diffusion Factor Models: Generating
High-Dimensional Returns with Factor Structure"** (arXiv:2504.06566) — a score-based
generative diffusion model for asset returns exploiting factor structure R = βF + ε in the
small-data regime n ≪ d.

- **Forward process:** Ornstein-Uhlenbeck SDE (α_t = e^(−t/2), h_t = 1 − e^(−t)).
- **Score network (the paper's Eq. 18):** s_θ(r,t) = α_t·D_t·V·g_ζ(VᵀD_t r, t) − D_t·r —
  **bottleneck width = factor dimension k** (default 16): tied orthonormal encoder/decoder
  V (learned loadings β), per-asset idiosyncratic-variance vector, small MLP in the k-dim
  subspace, linear complement-score skip — the paper's Lemma 1 decomposition is what makes
  the small-n regime work.
- **Training/sampling:** denoising score-matching loss (paper-faithful uniform or
  noise-prediction weighting); reverse-SDE Euler-Maruyama with EMA weights.
- **Evaluation:** mean/cov relative error, PCA subspace recovery, and the paper's §7.1
  downstream test — **global minimum-variance portfolios** from the generated-sample
  covariance vs Ledoit-Wolf / OAS / diagonal / factor / empirical baselines.
- **Trained on GSMC's own data:** tal's `pcmf_prices.csv` panel — **19 assets** (XAU/XAG/
  XPD/XPT + MB critical-minerals codes, 2018–2026); saved models `tal_market.pt`,
  `tal_weekly.pt`.
- **Companion covariance work — RMT (`~/RMT`):** Marchenko-Pastur eigenvalue clipping
  (Laloux-Cizeau-Bouchaud-Potters, with the market-mode correction σ² = 1 − λ₁/N) on
  **466 S&P equities × 2,511 days**: **92.7% of correlation eigenvalues indistinguishable
  from noise**; condition number −97.5%; walk-forward min-variance backtest **Sharpe 0.00 →
  0.78**, realized vol −68% (survivorship/cost caveats documented). Together: generative
  covariance (deep) and robust covariance (classical) — two answers to "how should a
  portfolio trust its Σ?"

---

## 7. Sandbox agents

The company's local-first agent framework (`~/sandbox-agents`): an LLM in a Docker sandbox
(colima) reads a market snapshot and proposes trade *intents*; a deterministic host-side
gate owns the credential and executes bounded orders (position/order/liquidity/budget/
impact caps; the model has no trading tool — its only output is `intents.json`).

- **CriticalMineralsBot (authored):** expertise profile for lithium / cobalt / tantalum /
  NdPr — resolvability-aware rules (prefer exchange/public-feed price markets over
  cancel-prone operational markets; trade only >0.10 divergence with a concrete stated
  mispricing reason; unit-conversion discipline $/t vs $/kg; cite the data anchor + the
  market's resolution source); universe terms + liquidity floor; single-cell intents (no
  fabricated grids — a documented opt-out because the clone lists these metals as plain
  binaries).
- **Data pipeline (authored):** `scripts/export_metals_data.py` — exports curated real
  SMM/GFEX series (lithium carbonate, NdPr oxide, GFEX term-slope/tightness) from the
  equity-options research dataset into the sandbox's read-only data dir, de-credentialed,
  plus a hand-written supply/demand briefing that leads with the resolvability warning.
- **The anti-hallucination read-back gate (framework contribution):** a live dry run caught
  qwen3-coder:30b quoting a market at **98.2% when the venue price was 5.6%** — a phantom
  0.95-vs-0.056 gap the sizer would have traded (it sized Ṁ15 before the fix). Prompting
  didn't stop it, so: a required `market_probability` field the model must echo verbatim,
  **rejected deterministically on >0.05 mismatch**, documented in the system prompt,
  covered by tests (hallucinated rejected / in-tolerance passes / missing rejected).
- **Service fixes:** install-service hardcoded max-turns 25 (starving the agent, which needs
  ~45) and never passed cooldown — both made configurable through CLI + launchd/systemd
  generators.
- **Environment:** colima Docker context, Ollama qwen3-coder:30b, live-clone credentials +
  CF Access provisioned; agent account minted.
- **Status:** end-to-end dry-run works against the live clone; the single `--execute` run
  **correctly declined to trade** (divergence 0.017 < the 0.10 threshold) — the safety
  story ends with the system choosing *not* to trade. Known small blockers: launchd PATH
  can't find the colima docker binary, colima autostart, exporter uncommitted.

---

## 8. Built but not deployed — negative results

The closing section: things built properly, measured honestly, and *not* shipped. Same
lesson as everything above — measure before you ship, and say no to your own code.

- **Market maker (market_maker_1)** — fully built and live-verified: two-sided limit book
  at fair ± spread around the diffusion_mc value, **Avellaneda-Stoikov-lite inventory
  skew**, TTL re-quoting (~25h), toxic-flow widening; limit-order client primitives added
  for it. **Shelved on measurement:** canary got **0/5 fills from real counterparties**
  (100% AMM crossings — the clone has no organic two-sided flow) and taker slippage was
  already only ~1.2% of stake. Both value levers near-nil. (`maker.py`, 462 LOC;
  `docs/market-maker.md`; @MarketMakerBot minted, enabled:false.)
- **cocoa_atlantic — Atlantic-Niño SST drift** — built 2026-06-02; sign validated (−1) but
  **weak/nonlinear** → enabled:false. Source verdicts from the same push: open-meteo/CPC
  keep; CHIRPS/tidbits/Kalshi cut.
- **drought_cotton / cocoa_stocks — unconventional-data bots** — built + validated
  2026-06-02 (US Drought Monitor DSCI; ICE certified cocoa stocks); kept enabled:false
  post-validation.
- **Coffee price signal** — the strongest fundamental in the softs complex (SUR elasticity
  b = −0.76, t = −7.5, R² = 0.69) but **zero resolvable coffee price markets** — great
  signal, no venue. Only the small consumption bot shipped.
- **dup_arb** — the ~2,485 exact-duplicate market sets are real and verified, but the
  dedicated strategy was removed per owner decision; the surface remains a semantic_arb
  target.
- **Qwythos-9B** — evaluated as the llm-strategy engine via llm-bench + full dry-run diff:
  half the coverage, flakier than gemma4 → rejected (2 crash bugs fixed as a by-product).
- **GDELT** — evaluated as the news source for 007 → rejected in favor of curated RSS.
- **Zinc conditional-markets branch (tal)** — exploratory branch abandoned once the
  platform's conditional-markets feature shipped elsewhere (PR #1571); the clone-side work
  continued as conditional_arb + zirconium research instead.
- **Zirconium event-predicate conditionals** — researched and rejected: the resolvability
  product rule kills event-predicate legs; only Tier-1 nested *price* triangles were built.
- **Thurstone multiray 2nd dimension** — the multidimensional bot-skill model reproduces
  pairwise rankings where data are dense, but a held-out dimensionality test showed the
  **second skill dimension is memorization, not shared structure** → reported as not real.
- **The Thurstone retraction** — the "+5.79 cold-start shrinkage" effect self-caught as a
  spline-interpolation artifact (true value +0.017), retracted and documented in the paper's
  Appendix A; the real mechanism (explicit ridge) identified.
- **007's edge status** — news_drift is live but its lead/lag edge is honestly marked
  UNPROVEN pending validation (`validate_news_leadlag.py` harness built for exactly that).
- **Equity-options energy names** — every energy underlier failed the walk-forward gate
  (XOM/CVX/COP/SLB negative skill) and is excluded from trading; the same gate that
  *admitted* the precious producers.

**Why this slide matters:** roughly a third of everything built ended in a measured "no" —
and every "no" is documented, reproducible, and saved someone from deploying it later.
