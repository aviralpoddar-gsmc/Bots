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
BOTS=("commodity_spot_1" "ladder_arb_1" "term_structure_1")

cd "$REPO" || exit 1
# shellcheck disable=SC1090
source "$VENV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() { log "+ $*"; doppler run -- "$@"; }

log "=== quantbots daily cycle start (live=${QUANTBOTS_LIVE:-1}) ==="

# 1. Realize PnL on positions whose markets resolved (per bot).
for bot in "${BOTS[@]}"; do
  run quantbots resolve --bot "$bot" || log "resolve $bot failed (continuing)"
done

# 2. Refresh the market cache (new markets, current prices) and external feeds.
run quantbots refresh --limit 70000 || log "refresh failed (continuing with stale cache)"
run quantbots ingest || log "ingest failed (continuing with stale feeds)"

# 3. Trade each bot.
for bot in "${BOTS[@]}"; do
  run quantbots run --bot "$bot" $LIVE_FLAG || log "run $bot failed"
done

# 4. Roll up PnL + leaderboard snapshot.
run quantbots snapshot || log "snapshot failed"

log "=== quantbots daily cycle done ==="
