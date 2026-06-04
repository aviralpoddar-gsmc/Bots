#!/usr/bin/env bash
# Daily trading cycle for the quantbots running on the private Manifold clone.
#
# Order matters: realize PnL on anything that resolved, pull fresh markets +
# prices, then trade, then snapshot. Each step is idempotent and safe to re-run.
# Credentials come from Doppler (project aviral-bots / config dev).
#
# Schedule this once per day via launchd (see scripts/com.quantbots.daily.plist).
# Run BOTS in dry-run by setting QUANTBOTS_LIVE=0; default is live.

set -uo pipefail

# Adjust for your machine, or export QUANTBOTS_REPO. The launchd plist also hardcodes
# this path — edit scripts/com.quantbots.daily.plist to match before installing it.
REPO="${QUANTBOTS_REPO:-/Users/mikhail/Bots}"
VENV="$REPO/.venv/bin/activate"
LIVE_FLAG="--live"
[ "${QUANTBOTS_LIVE:-1}" = "0" ] && LIVE_FLAG=""

# Bots to run each cycle, in priority order. Add new bot names here as they ship.
# diffusion_mc_1 replaced commodity_spot_1 on 2026-06-04 (kernel-smoothed bootstrap pricer,
# dominates the lognormal on the walk-forward gate; prices the same strikes, so commodity_spot_1
# is disabled — do NOT add it back here without removing diffusion_mc_1).
BOTS=("diffusion_mc_1" "ladder_arb_1" "term_structure_1" \
      "stockpile_facts_1" "stockpile_grid_arb_1" "stockpile_coherence_1" "pair_trading_1" \
      "cotton_fundamental_1" "cftc_softs_1" "weather_cocoa_1" "nass_cotton_1" \
      "cocoa_fundamental_1" "coffee_consumption_1" "fas_balance_1" "wasde_cotton_1" \
      "surface_arb_1" "ensemble_1" "commodity_1" "enso_1" "mean_reverter" "llm_forecaster")

cd "$REPO" || exit 1
# shellcheck disable=SC1090
source "$VENV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() { log "+ $*"; doppler run -- "$@"; }

log "=== quantbots daily cycle start (live=${QUANTBOTS_LIVE:-1}) ==="

# 1. Refresh the market cache (new markets, current prices) and external feeds.
#    MUST run before `resolve`: the bulk markets endpoint is ~1800x faster than
#    fetching per-id, so resolve reads cached state instead of doing N API calls
#    per open position. Stale cache here means missed resolutions.
run quantbots refresh --limit 70000 || log "refresh failed (continuing with stale cache)"
run quantbots ingest || log "ingest failed (continuing with stale feeds)"
run quantbots process || log "process failed (continuing without fresh signals)"

# 2. Realize PnL on positions whose markets resolved (per bot). Reads from cache.
for bot in "${BOTS[@]}"; do
  run quantbots resolve --bot "$bot" || log "resolve $bot failed (continuing)"
done

# 3. Trade each bot.
for bot in "${BOTS[@]}"; do
  run quantbots run --bot "$bot" $LIVE_FLAG || log "run $bot failed"
done

# 4. Roll up PnL + leaderboard snapshot.
run quantbots snapshot || log "snapshot failed"

log "=== quantbots daily cycle done ==="
