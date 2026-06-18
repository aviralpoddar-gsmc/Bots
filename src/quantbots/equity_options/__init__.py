"""equity_options — a FENCED package for trading REAL listed equity options.

This is a deliberate departure from the rest of quantbots, which is hard-wired to
the play-money Manifold clone. Everything here concerns *real* options on *real*
exchanges via a broker (Alpaca), so it is physically isolated:

  - It MUST NOT import `quantbots.manifold` (enforced by tests/test_eo_safety.py).
  - It is NEVER wired into the clone runner, cli, daily_cycle.sh, or the strategy
    registry. It has its own CLI (`eo`), its own SQLite DB, and its own ops loop.
  - Execution is staged dry-run -> paper -> gated-live. `execution/live.py` is a
    refusing stub; real-money trading needs explicit owner approval (see SAFETY.md).

It reuses the *math, ingestion, storage, and backtest seams* of the parent package
(BSM/mixture/diffusion/surface-fit/OLS-beta), never the clone client. See SAFETY.md
and docs in the approved plan.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
