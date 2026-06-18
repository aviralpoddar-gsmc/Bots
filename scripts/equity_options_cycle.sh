#!/usr/bin/env bash
# equity_options daily cycle — SEPARATE from scripts/daily_cycle.sh (the play-money
# clone). This trades REAL options via Alpaca; see src/quantbots/equity_options/SAFETY.md.
#
# Order matters and each step is idempotent / safe to re-run:
#   1. safety-check     — fence intact (no manifold import, live refuses) or abort
#   2. reconcile        — sync the ledger to ACTUAL Alpaca fills (broker = truth)
#   3. cancel stale     — drop yesterday's unfilled working limit orders
#   4. manage           — close positions that hit an exit rule (profit/stop/DTE)
#   5. trade            — open new structures where edge clears the gates (skips held names)
#   6. snapshot         — record a PnL/greeks snapshot
#
# Defaults to DRY. Set EO_PAPER=1 (the launchd plist does) to actually submit to the
# Alpaca PAPER account. NEVER trades live — live.py refuses real-money trading.
set -uo pipefail

REPO="${QUANTBOTS_REPO:-/Users/mikhail/Bots}"
cd "$REPO" || exit 1

# Secrets: prefer Doppler if present, else the gitignored .env.
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi

EO="$REPO/.venv/bin/python -m quantbots.equity_options.cli"
PAPER_FLAG=""
[ "${EO_PAPER:-0}" = "1" ] && PAPER_FLAG="--paper"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== equity_options cycle start (paper=${EO_PAPER:-0}) ==="

log "1. safety-check"
$EO safety-check || { log "SAFETY CHECK FAILED — aborting"; exit 1; }

if [ "${EO_PAPER:-0}" = "1" ]; then
  log "2. reconcile fills"
  $EO reconcile || log "reconcile failed (continuing)"

  log "3. cancel stale working orders"
  $EO cancel-orders || log "cancel-orders failed (continuing)"
fi

log "4. manage exits"
$EO manage $PAPER_FLAG || log "manage failed (continuing)"

log "5. refresh validation gate (walk-forward backtest -> data/equity_options_gate.json)"
$EO backtest || log "backtest gate refresh failed (continuing; trade will gate-block stale names)"

log "6. open new structures (only gate-PASSing, non-held names)"
$EO trade $PAPER_FLAG || log "trade failed (continuing)"

log "8. snapshot"
$EO snapshot || log "snapshot failed (continuing)"

log "=== equity_options cycle done ==="
