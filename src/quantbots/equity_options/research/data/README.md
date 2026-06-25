# Critical-minerals price dataset (lithium / NdPr)

Research data for the **producer-side** commodity→equity work in `equity_options/research`
(screen.py / metal_matrix.py). Built 2026-06-25. **Internal use only** — the SMM series
are from a paid Shanghai Metals Market subscription; SMM's ToS restricts redistribution.
Do not publish or share outside this private repo.

## Files

### `metals_master_daily.csv` — REAL daily data (use this for backtests)
Daily, 2023-06-26 → 2026-06-25 (727 rows).

| column | meaning | source |
| --- | --- | --- |
| `li_smm_usd_t` | Battery-grade lithium carbonate spot, USD/tonne (**ex-VAT**) | SMM `SMM-Li-LC-001` |
| `ndpr_smm_usd_t` | Praseodymium-neodymium oxide, USD/tonne (**ex-VAT**) | SMM `SMM-RE-OX-001` |
| `gfex_li_active` | Most-liquid GFEX lithium-carbonate futures settle, CNY/t | GFEX (free) |
| `gfex_li_term_slope` | (deferred/front − 1); **negative = backwardation** | derived from GFEX curve |
| `gfex_li_oi` | total open interest across contracts | GFEX (free) |
| `gfex_li_tightness` | `-gfex_li_term_slope` (higher = tighter market) | derived |

### `lithium_extended_monthly.csv`, `ndpr_extended_monthly.csv` — REAL + ESTIMATED (context only)
Monthly. Columns: `date, price_usd_t, source`.
`source` ∈ {`REAL_SMM`, `EST_<proxy>`}. Earliest: lithium 2009-01, NdPr 2010-10.

The `EST_*` rows are a **back-cast**: real SMM monthly log-price regressed on a free ETF
proxy over the 37-month overlap, then projected backward.
Best fits: lithium ← ALB (R²=0.86), NdPr ← REMX (R²=0.71).

## ⚠️ Caveats — do not misuse the estimated rows

1. **Magnitude is compressed.** The back-cast captures shape/timing but understates the
   2022 lithium peak (~$37k estimated vs ~$80k real) because the equity proxy didn't spike
   as hard as the metal. Use `EST_*` for **regime context, not price levels**.
2. **Circularity.** Lithium's best proxy is **ALB** and NdPr's is **REMX** — both equities.
   Never use the estimated series to predict ALB / lithium-miner equities; it is partly
   built from them. **Backtests must use only `metals_master_daily.csv` (real).**
3. To replace estimates with real long history: re-pull the SMM weekly/monthly endpoint
   (the daily view caps at 36 months; weekly/monthly go back further). Needs a fresh
   metal.com login (`/setup-browser-cookies`).

## Already in tal Snowflake — this is a BACKFILL

tal ingests the same source into `SOURCES.SMM_SPOT_DAILY` (keyed by `SMM_SERIES_ID`,
e.g. `SMM-Li-LC-001`, `SMM-RE-OX-001`; joins `MATERIAL_QID` → `PCF.TICKER_REFERENCE`)
and `SOURCES.GFEX_PRICES_RAW` (varieties lc/si/ps/pt/pd). **But tal only started ingesting
~2026-04-03** (≈55 daily obs/series); GFEX from 2025-11-27. This dataset is the **3-year
backfill** (lithium/NdPr from 2023-06; GFEX lc from 2023-07) that predates tal's ingestion.

Validated: over the 55-day overlap, this dataset's SMM prices equal `SMM_SPOT_DAILY` exactly
**once divided by 1.13** — tal stores **ex-VAT**, the browser endpoint returned VAT-inclusive.
**These CSVs have already been converted to ex-VAT (÷1.13)** so they splice seamlessly onto
`SMM_SPOT_DAILY`. Read live data from Snowflake (`equity_options/sources/smm.py`); use this
file only for pre-2026-04 history. (GFEX `gfex_li_active` is CNY/t futures settle — no VAT.)

## Provenance / how to refresh
- SMM daily: authenticated pull from `platform.metal.com/spotoverseascenter/v1/product_info/history/<product_id>`
  (`SMM-Li-LC-001` = product 201102250059; Pr-Nd oxide = 201102250162), `currency_type=2` (USD), `line_type=d`.
- GFEX: free CSVs `gfex.com.cn/gfex/gfexfile/history/LCFUTURES<year>_EN.csv`.
