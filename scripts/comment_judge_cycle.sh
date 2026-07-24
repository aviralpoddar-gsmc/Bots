#!/usr/bin/env bash
# Comment-society cycle — runs 3x/day via launchd (com.quantbots.comments).
# ALL LIVE (user go-ahead 2026-07-08):
#
#   1. ingest  — fresh metal spot anchors from tal Snowflake (SMM + PCF fallback).
#                NO doppler wrap: the tal reader shells into ~/tal with its own
#                doppler scope; an outer `doppler run` from this repo SHADOWS it
#                and breaks Snowflake auth (learned 2026-07-08).
#   2. judge   — evidence-anchored judge cycle (local qwen3:32b) + POST replies
#                to unsound/high comments (capped 7/run ≈ 20/day) as
#                @AdversaryMetalsBot. Judging never bets.
#   3. fade    — comment_fade counter-bets actionable verdicts (LIVE, capped by
#                bots.yaml limits; runner applies sizing/resolvability/comments).
#   4. consensus — comment_consensus trades toward the extremized crowd (LIVE).
#
# Each judgment is ~50s of local LLM time; 50/run x 3 runs/day ≈ 2h/day compute.
set -uo pipefail

REPO="${QUANTBOTS_REPO:-/Users/mikhail/Bots}"
cd "$REPO" || exit 1

PY="$REPO/.venv/bin/python"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== comment cycle start (LIVE) ==="

log "1. ingest tal price anchors (Snowflake, no doppler wrap)"
$PY scripts/ingest_tal_prices.py || log "tal ingest failed (continuing — staleness cap protects the judge)"

log "2. judge cycle + live replies"
doppler run -- $PY -m quantbots.cli judge-comments \
  --bot adversary_metals_1 --max-judgments 50 --max-markets 40 \
  --post-replies --max-replies 7 \
  || log "judge cycle failed"

log "3. comment_fade counter-bets (LIVE)"
doppler run -- $PY -m quantbots.cli run --bot adversary_metals_1 --live \
  || log "comment_fade run failed"

log "4. comment_consensus trades (LIVE)"
doppler run -- $PY -m quantbots.cli run --bot consensus_1 --live \
  || log "consensus run failed"

log "=== comment cycle done ==="
