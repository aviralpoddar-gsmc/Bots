"""Position sizing — ported verbatim from the parent repo's shared/sizing.py.

The philosophy: no Kelly, no separate confidence knob. The *gap* between your
estimate and the market price already encodes conviction, so we simply push the
price a fixed fraction (1/3) of the way toward the estimate and clamp that order
by several caps. Pure stdlib (math only) so it stays trivially testable and has
no optional-extra dependency.
"""

from __future__ import annotations

import math
from typing import Any

# Defaults mirror the parent's config/bot_fleet.yaml. Override per-bot in
# config/bots.yaml under `limits:`.
DEFAULT_LIMITS: dict[str, Any] = {
    "max_order_size": 50,  # hard cap on mana per order
    "liquidity_pct": 0.33,  # never spend more than this fraction of liquidity
    "hold_band": 0.05,  # don't churn when already-held edge is within this
    "max_price_impact": 0.10,  # never move the price by more than this
    "min_order_mana": 5,  # skip orders smaller than this
}


def mana_to_move_price(p_current: float, p_target: float, liquidity: float) -> int:
    """Mana required to move a CPMM price from p_current to p_target.

    Uses the LMSR-style approximation b ≈ liquidity / 4, where the cost to shift
    the log-odds by d_logit is |b * d_logit|.
    """
    p1 = min(max(p_current, 1e-3), 0.999)
    p2 = min(max(p_target, 1e-3), 0.999)
    d_logit = math.log(p2 / (1 - p2)) - math.log(p1 / (1 - p1))
    return max(0, int(abs(liquidity * d_logit / 4.0)))


def compute_trade(
    *,
    estimate: float,
    current_prob: float,
    position: Any | None,
    liquidity: float | None,
    limits: dict[str, Any],
) -> dict[str, Any] | None:
    """Decide whether/what to trade for one market.

    Returns {"direction": "YES"|"NO", "amount": int} or None to skip.

    - Hold band: if we already hold this market and the edge is small, don't churn.
    - Target: push 1/3 of the way from the market to our estimate.
    - The order is the *min* of four caps: target move, max order size, a
      fraction of liquidity, and a max-price-impact move.
    """
    # Hold band: don't churn a flat edge on an existing position.
    if position is not None and abs(estimate - current_prob) <= limits["hold_band"]:
        return None

    target_p = current_prob + (estimate - current_prob) / 3.0  # push 1/3 of the way
    liq = max(liquidity or 0, 100)  # min liquidity floor

    impact_p = current_prob + math.copysign(
        limits["max_price_impact"], target_p - current_prob
    )
    caps = {
        "target": mana_to_move_price(current_prob, target_p, liq),
        "max_order": int(limits["max_order_size"]),
        "liquidity_pct": int(liq * limits["liquidity_pct"]),
        "max_impact": mana_to_move_price(current_prob, impact_p, liq),
    }
    amount = min(caps.values())
    if amount < limits["min_order_mana"]:
        return None

    direction = "YES" if estimate > current_prob else "NO"
    return {"direction": direction, "amount": amount}
