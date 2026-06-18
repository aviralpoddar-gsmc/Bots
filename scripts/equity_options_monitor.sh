#!/usr/bin/env bash
# Continuous intraday loop for equity_options (REAL options via Alpaca PAPER).
# Monitors the market and manages/hedges/exits + opens GATE-PASSING positions every
# EO_INTERVAL seconds while the market is open; idles when closed. See SAFETY.md.
#
# Defaults to DRY. Set EO_PAPER=1 (the launchd plist does) to submit to Alpaca paper.
set -uo pipefail
REPO="${QUANTBOTS_REPO:-/Users/mikhail/Bots}"
cd "$REPO" || exit 1
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }

PAPER_FLAG=""
[ "${EO_PAPER:-0}" = "1" ] && PAPER_FLAG="--paper"
INTERVAL="${EO_INTERVAL:-300}"

exec "$REPO/.venv/bin/python" -m quantbots.equity_options.cli monitor \
  $PAPER_FLAG --interval "$INTERVAL"
