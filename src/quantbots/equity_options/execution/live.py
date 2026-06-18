"""LIVE (real-money) broker — REFUSING STUB.

This file exists so the execution ladder is complete and the gate is testable, but it
**refuses to trade real money** in this build. Constructing it raises unless BOTH:

  1. env  EQUITY_OPTIONS_OWNER_APPROVAL == "1", AND
  2. a committed risk-limits file exists at the configured path.

Even when those are set, `submit()` raises: turning live trading on is a deliberate,
human, owner-only step (add a real implementation here, write the risk-limits file,
and document the carve-out). Automated agents must NOT weaken this. See ../SAFETY.md.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import BrokerClient, OptionOrder, OrderResult

OWNER_APPROVAL_ENV = "EQUITY_OPTIONS_OWNER_APPROVAL"


class LiveTradingRefused(RuntimeError):
    """Raised whenever someone tries to trade real money in this build."""


class LiveBroker(BrokerClient):
    name = "live"

    def __init__(self, *, risk_limits_file: str = "config/equity_options_risk_limits.yaml"):
        if os.environ.get(OWNER_APPROVAL_ENV) != "1":
            raise LiveTradingRefused(
                f"Live trading refused: {OWNER_APPROVAL_ENV} is not set. This build ships "
                "PAPER as the ceiling. See equity_options/SAFETY.md."
            )
        if not Path(risk_limits_file).exists():
            raise LiveTradingRefused(
                f"Live trading refused: risk-limits file {risk_limits_file!r} is missing. "
                "A committed, reviewed risk-limits file is required before any live capital."
            )
        # Even with both gates, this build does not implement live execution.
        raise LiveTradingRefused(
            "Live execution is not implemented in this build (paper is the ceiling). "
            "Implementing it is an explicit, owner-only step — see equity_options/SAFETY.md."
        )

    def submit(self, order: OptionOrder) -> OrderResult:  # pragma: no cover - unreachable
        raise LiveTradingRefused("Live execution is disabled.")

    def account_equity(self) -> float:  # pragma: no cover
        raise LiveTradingRefused("Live execution is disabled.")

    def positions(self) -> list[dict]:  # pragma: no cover
        raise LiveTradingRefused("Live execution is disabled.")


def make_broker(mode: str, **kwargs):
    """Factory honoring the broker mode with the safe default. Never returns a live
    client that can trade (live.py refuses)."""
    from ..config import DRY, LIVE, PAPER
    from .alpaca import AlpacaPaperBroker
    from .base import DryRunBroker

    if mode == PAPER:
        return AlpacaPaperBroker(**{k: v for k, v in kwargs.items() if k in ("key", "secret")})
    if mode == LIVE:
        return LiveBroker(**{k: v for k, v in kwargs.items() if k == "risk_limits_file"})
    if mode == DRY:
        return DryRunBroker()
    raise ValueError(f"unknown broker mode {mode!r}")
