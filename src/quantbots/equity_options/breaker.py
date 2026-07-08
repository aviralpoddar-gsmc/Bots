"""Broker-truth circuit breaker for new entries.

On 2026-07-07 Alpaca paper re-marked the entire options book to zero overnight
(-$17k equity cliff with zero fills), the local ledger disagreed with the broker,
and the bot re-armed into the crater. `eo trade` calls this before opening
anything; a non-None reason means HALT entries (exits/hedges are unaffected).

The reason is also LATCHED into the store (`OptionsStore.trip_halt`) by the
caller, so a trip persists across days until a human runs `eo resume` — a
stateless check would silently re-arm once `last_equity` rolls forward.
"""

from __future__ import annotations


def entry_halt_reason(*, equity: float, last_equity: float,
                      ledger_open_symbols: set[str], broker_open_symbols: set[str],
                      open_order_symbols: set[str], max_drop_pct: float) -> str | None:
    """Return why new entries must halt, or None if the account state is sane.

    Trips on either anomaly:
      1. equity fell more than `max_drop_pct` vs prior close (phantom mark-out,
         mass liquidation, or a real crash — none of which should be traded into
         by a momentum bot on autopilot);
      2. the ledger believes legs are open that the broker does not hold
         (books disagree — reconcile before trusting any position-awareness logic).
         Legs with a WORKING order (`open_order_symbols`) are excluded: a resting
         unfilled limit is ledger-open but not a broker position, and must not
         false-alarm every monitor tick.
    """
    if last_equity > 0:
        drop = (last_equity - equity) / last_equity
        if drop >= max_drop_pct:
            return (f"equity down {drop:.1%} vs prior close "
                    f"({equity:,.0f} vs {last_equity:,.0f}), breaker at {max_drop_pct:.0%}")
    ghosts = ledger_open_symbols - broker_open_symbols - open_order_symbols
    if ghosts:
        sample = ", ".join(sorted(ghosts)[:4])
        return (f"ledger/broker mismatch: {len(ghosts)} ledger-open leg(s) absent at "
                f"broker ({sample}) — run `eo reconcile`")
    return None
