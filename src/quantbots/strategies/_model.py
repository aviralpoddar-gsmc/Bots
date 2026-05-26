"""Small shared helpers for data-driven strategies (no optional deps)."""

from __future__ import annotations

import math
import time
from typing import Any

_YEAR_SECONDS = 365.25 * 24 * 3600


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def years_to_close(market: dict[str, Any], default: float = 1.0) -> float:
    """Years from now until the market's closeTime (epoch ms). Floored at 0."""
    close = market.get("closeTime")
    if not close:
        return default
    return max((close / 1000.0 - time.time()) / _YEAR_SECONDS, 0.0)
