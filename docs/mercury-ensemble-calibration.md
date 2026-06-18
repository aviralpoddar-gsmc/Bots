# Mercury Ensemble — `mercury_ensemble` strategy (Design Spec)

**Date:** 2026-06-10
**Status:** Draft for review
**Repo:** quantbots (`github.com/aviralpoddar-gsmc/Bots`)

## 0. ⚠️ Deliberate exception to the local-compute rule

`CLAUDE.md:61-67` mandates **local inference only** for alpha until bots are
demonstrably profitable, and says to *flag anything requiring hosted inference*.
This strategy **intentionally uses a hosted provider (Mercury / Inception Labs,
`api.inceptionlabs.ai`)** for the forecast itself. This is an explicit,
owner-approved exception for evaluating Mercury as a forecasting engine.

Honest caveats the owner accepted in granting it:
- quantbots runs as a **daily batch** (`scripts/daily_cycle.sh`, launchd), so
  Mercury's latency advantage (2-3s vs ~50s local) buys little here.
- Local inference is **free**; Mercury costs ~$0.01/market-group with caching.
- The ensemble/mixture design below is **engine-agnostic** — it runs identically
  on the local model. If the Mercury experiment doesn't beat the local `llm`
  baseline (§7), revert the engine, keep the strategy.

Until this lands, `CLAUDE.md:61-67` still reads "local only." Updating that
section to record the exception is a follow-up the owner should make
consciously; this spec does not touch it.

## 1. Summary

A new quantbots strategy, `mercury_ensemble`, whose edge is **calibration, not
cleverness**. The reference `llm` strategy (`strategies/llm.py`) asks the model
*once* for a percentile distribution of the underlying quantity, fits a normal
CDF, and reads each strike's probability off it. `mercury_ensemble` does the same
but asks Mercury **N times** with a temperature spread, treats the samples as
draws from a posterior predictive, **mixes** them into one calibrated
probability per strike, **decomposes** uncertainty into aleatoric (the model's
honest spread) and epistemic (how much the samples disagree), and **shrinks the
estimate toward the current market price in proportion to disagreement** — so a
split jury sizes small or abstains.

It implements only the one seam quantbots exposes — `estimate(group) ->
{market_id: prob}` (`strategies/base.py:62`) — and touches nothing else
(`sizing.py`, `portfolio.py`, `runner.py`, ledger all unchanged).

## 2. Goals / Non-Goals

**Goals**
- A registered `mercury_ensemble` strategy reusing all shared infrastructure.
- Posterior-predictive mixture of N Mercury percentile-forecasts with
  aleatoric/epistemic decomposition.
- Disagreement-shrinkage + direction-agreement abstention, expressed entirely
  inside the returned `estimate` (no sizing changes).
- A thin Mercury client variant (hosted, OpenAI-compatible, no Ollama options).
- Calibration grading vs. the local `llm` baseline using `backtest.py` and the
  resolved-outcome ledger.

**Non-Goals (v1)**
- Learned post-hoc calibration map (isotonic/Platt) — deferred to v2; needs this
  bot's own resolved history first.
- Any change to core modules (`sizing.py`, `portfolio.py`, `runner.py`,
  `resolvability.py`).
- Streaming / intraday loops (the runner is daily-batch).

## 3. Context: the seams this plugs into (as built)

- **Strategy interface:** `strategies/base.py:62` — implement
  `estimate(group) -> {market_id: prob}`. Omit a market to abstain. Optional
  `prefilter`, `group`, `correlation_key`, `explain`.
- **Reference single-shot forecaster:** `strategies/llm.py` — percentile prompt
  (`_SYSTEM`, `_ask_percentiles`), normal-CDF fit (`estimate`, lines 147-181),
  `spread_mult` overconfidence correction, `conf_cap` clamp, `norm_cdf` from
  `strategies/_model.py`, ladder helpers from `strategies/ladder.py`
  (`attach_ladder_fields`, `measurable_key`).
- **LLM client:** `llm/client.py:28` `LocalLLM` — OpenAI-compatible, but
  `json_completion` injects `extra_body={"options":{"num_ctx":...}}` (Ollama
  only). Mercury needs a variant without it (§6).
- **Sizing (untouched):** `sizing.py:39` `compute_trade(estimate, current_prob,
  position, liquidity, limits)` — pushes price ⅓ of the way from `current_prob`
  to `estimate`; caps by `max_order_size`, `liquidity_pct`, `max_price_impact`;
  `hold_band` suppresses churn. **No confidence knob** — conviction is the gap.
- **Allocator (untouched):** `portfolio.py:48` ranks signals by realized EV
  (paper EV × resolvability), funds best-first under budget + per-`correlation_key`
  concentration caps.
- **Registry:** `strategies/__init__.py:16` `_REGISTRY` maps name → `"module:Class"`.
- **Config:** `config/bots.yaml` per-bot `strategy` / `limits` / `params`;
  `config.py:26` resolves env vars at call time.
- **Outcomes for grading:** `store/schema.sql` `trade` table carries
  `llm_estimate` + `price_after`; `runner.py:112` `sync_resolutions()` writes
  `RESOLUTION_CLOSE` rows at 1.0/0.0/MKT. `backtest.py:94` replays a strategy →
  Brier / calibration buckets / PnL.

## 4. The math: full Bayesian mixture

For a measurable group (one underlying quantity, many strike/date markets):

1. Draw N samples. Sample *i* returns percentiles `(p10..p90)ᵢ`. Fit a normal
   `(μᵢ, σᵢ)` exactly as `llm.py:155-157` (σ from the 10-90 span and IQR,
   averaged, × `spread_mult`).
2. Per market with threshold *X* and direction *d*, sample *i*'s probability:
   ```
   pᵢ(X) = 1 − Φ((X − μᵢ)/σᵢ)      if d == "exceeds"   else   Φ((X − μᵢ)/σᵢ)
   ```
3. **Posterior-predictive probability (mixture):**
   ```
   p̄(X) = (1/N) · Σᵢ pᵢ(X)
   ```
   A uniform mixture of the per-sample CDFs — Bayesian model averaging over
   Mercury's sampling distribution. (Mixtures of normals have fatter tails when
   samples disagree, so the mixture self-widens — expect to need less
   `spread_mult` than the single-shot `llm` bot; start at 1.0-1.25.)
4. **Uncertainty decomposition (law of total variance):**
   ```
   epistemic(X) = Varᵢ[ pᵢ(X) ]          # sample disagreement — the calibration signal
   aleatoric(X) = p̄(X)·(1 − p̄(X)) − epistemic(X)   # residual irreducible Bernoulli variance
   ```

## 5. Estimate output: shrinkage + abstention (no sizing change)

Because `compute_trade` has no confidence knob, the calibration logic lives
entirely in the number `estimate()` returns. For each market:

**Direction-agreement gate (abstain on a split jury):**
```
side(x) = sign(x − current_prob)
agree   = mean_i [ side(pᵢ(X)) == side(p̄(X)) ]
if agree < direction_agreement_floor:  omit market   # don't trade
```

**Disagreement-shrinkage toward market price:**
```
confidence = clamp(1 − epistemic(X) / τ, 0, 1)
est        = current_prob + (p̄(X) − current_prob) · confidence
est        = clamp(est, 1 − conf_cap, conf_cap)       # same hallucination clamp as llm.py:180
return {market_id: est}
```

High disagreement → `confidence → 0` → `est → current_prob` → ⅓-gap ≈ 0 →
`hold_band` suppresses → no trade. Confident consensus → full gap → full size.
Calibration becomes a property of the single number, and sizing/allocation stay
untouched.

`current_prob` is `market["probability"]` (already in the group dict). Record
`p̄`, `epistemic`, `aleatoric`, `agree`, `confidence`, `n_eff` into
`self._explanations[market_id]` so `explain()` posts them in the trade comment.

## 6. Components & boundaries

| File | Responsibility | Purity |
|---|---|---|
| `src/quantbots/llm/mercury.py` | `MercuryLLM(LocalLLM)` — hosted OpenAI-compatible client to `api.inceptionlabs.ai/v1`; overrides `json_completion` to **drop the Ollama `options` extra_body** and read `MERCURY_API_KEY`. Smallest possible shim; core stays generic. | I/O |
| `src/quantbots/strategies/_mixture.py` | Pure. Percentile sample sets → per-strike `p̄`, `epistemic`, `aleatoric`. Reuses `_model.norm_cdf`. | Pure |
| `src/quantbots/strategies/mercury_ensemble.py` | `MercuryEnsembleStrategy(Strategy)`. Reuses `llm.py`'s prompt + percentile parsing; fans out N `MercuryLLM` calls (temperature spread, bounded concurrency); calls `_mixture`; applies §5 shrinkage/gate; populates `_explanations`; implements `explain()`. | I/O |

Register one line in `strategies/__init__.py:16`:
```python
"mercury_ensemble": "quantbots.strategies.mercury_ensemble:MercuryEnsembleStrategy",
```

Shared logic with `llm.py` (the `_SYSTEM` prompt, `_ask_percentiles`,
`_KEYS`, the normal fit, ladder helpers) is factored to a shared helper rather
than copy-pasted — DRY, and keeps the two strategies' forecasting identical so
the A/B isolates *ensembling*, not prompt drift.

## 7. How we prove it's better (acceptance)

A/B against the local single-shot `llm` strategy on the **same measurable
universe** (the cleanest control: same prompt, same CDF fit, the only variables
are ensembling + the Mercury engine). Grade on resolved outcomes only:

- **Backtest first** (`backtest.py:94`) — Brier, calibration buckets, simulated
  PnL — before any live mana. (`dry_run=True` default per `CLAUDE.md:57`.)
- **Live ledger:** `SELECT llm_estimate, price_after FROM trade WHERE
  trade_type='RESOLUTION_CLOSE'` → Brier / log-loss / reliability curve, vs the
  `llm` bot on overlapping markets.
- **Epistemic-vs-error check:** do high-`epistemic` forecasts have higher
  realized error? If not, shrinkage is noise — rethink it.

**Ship bar:** lower Brier/log-loss than the `llm` baseline on a statistically
meaningful resolved sample, with epistemic correlated to error. The cancellation
reality (§8) means "resolved" is ~7% of markets — grading needs patience or
backtest volume.

## 8. The cancellation reality (must shape expectations)

`CLAUDE.md:32-50`: ~93% of clone markets resolve CANCEL/N-A; only ~7% YES/NO,
concentrated in price markets (LBMA/LME/COMEX). Implications for this bot:
- Calibration is only observable on the ~7% that settle — lean on `backtest.py`
  volume and on price-ladder markets for signal.
- The core already multiplies EV by `resolvability.py`, so capital avoids
  cancel-prone markets; this strategy inherits that for free.
- The shrink-when-uncertain thesis **aligns** with the documented
  adverse-selection trap (deep mispricings cluster in markets that won't
  resolve): abstaining on low-confidence forecasts avoids chasing phantom edge.

## 9. Configuration

`.env`: add `MERCURY_API_KEY` (currently local-only; not yet in Doppler — write
was blocked by read-only perms, tracked separately).

`config/bots.yaml`:
```yaml
- name: mercury_ensemble_1
  strategy: mercury_ensemble
  account_env: MANIFOLD_CLONE_API_KEY
  enabled: false            # backtest before enabling; dry_run until human-confirmed
  limits:
    max_order_size: 50
    hold_band: 0.05
    max_run_budget: 150
    post_comments: true
  params:
    model: mercury-2
    n_samples: 20
    min_quorum: 12
    temperature_lo: 0.4
    temperature_hi: 1.0
    sample_concurrency: 10
    epistemic_tau: 0.04       # shrinkage scale on Var of probability; tune in backtest
    direction_agreement_floor: 0.70
    spread_mult: 1.1          # less than llm's 1.5 — the mixture self-widens
    conf_cap: 0.80
    max_groups: 12
```

**Cost/latency (informational):** Mercury caching ($0.025/M reads, $0 writes)
makes the shared per-group prompt nearly free across the N samples → ~$0.01 per
group; 10-wide parallel → a few seconds per group. Neither is a constraint at
daily-batch cadence.

## 10. Testing

Pure-Python, stubbed-model style matching `tests/test_llm_strategy.py`:
- **`_mixture.py` (pure):** `p̄ ∈ [0,1]`; monotone in strike for "exceeds";
  `epistemic → 0` when samples identical; `aleatoric + epistemic = p̄(1−p̄)`.
- **`mercury_ensemble` (stubbed `MercuryLLM`):** N samples fanned out;
  `min_quorum` abstains when too many fail; direction gate abstains on a split;
  `est → current_prob` as `epistemic → ∞`; `conf_cap` clamp holds.
- **`MercuryLLM`:** does not send Ollama `options`; reads `MERCURY_API_KEY`;
  hosted base URL.
- **Regression:** `tests/test_sizing.py`, `tests/test_portfolio.py` unchanged
  and green (core untouched). Run: `uv run pytest`.

## 11. Risks & mitigations

- **Hosted-rule exception (§0):** documented, owner-approved, reversible — the
  strategy runs on the local engine if Mercury is pulled.
- **Correlated samples (low diversity):** if Mercury collapses to one answer,
  `epistemic` understates true uncertainty. Mitigate with the temperature
  spread; the epistemic-vs-error check (§7) catches a dead signal.
- **Thin resolved sample (cancellation):** lean on backtest volume; grade on
  price-ladder markets.
- **Secrets:** `MERCURY_API_KEY` via env/Doppler only, never committed
  (`CLAUDE.md:69-73`).

## 12. v2 / future

- Learned calibration map (isotonic/Platt) on this bot's resolved history.
- Hybrid ensemble (mostly local + a few Mercury samples) once a winner is known.
- Promote Mercury to a first-class provider in `config/` if it beats local.
