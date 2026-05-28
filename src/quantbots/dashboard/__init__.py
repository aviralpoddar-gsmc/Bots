"""Local web dashboard for tracking bot performance.

Launched via `quantbots dashboard`. Reads from the live SQLite store and renders
a single-page view with: account overview, leaderboard, per-bot cards (strategy,
PnL, recent trades, top exposures), and a global activity feed. Pure read; never
mutates the ledger.
"""
