"""Processing layer: raw observations -> clean, normalized SIG_* signals.

Sits between ingestion (sources -> raw observations) and the strategies. Each
strategy reads a single source's SIG_* signal rather than doing ad-hoc math
inline, so the computation is centralized, testable, and backtestable.

Run after ingest:  `quantbots process`  (also wired into scripts/daily_cycle.sh).
"""

from .signals import run_all

__all__ = ["run_all"]
