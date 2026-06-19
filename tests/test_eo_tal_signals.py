"""tal multi-agent consensus -> spread candidates (pure functions, no network)."""

import pandas as pd

from quantbots.equity_options.research.tal_signals import (
    MATERIALS,
    _classify,
    material_consensus,
    spread_candidates,
)


def test_classify():
    assert _classify("Will copper spot price exceed 9000 USD/tonne?") == "copper"
    assert _classify("Will WTI crude exceed 80 USD?") in ("crude", "wti")
    assert _classify("Will the weather be nice?") is None


def _mkts(rows):
    return pd.DataFrame(rows, columns=["MARKET_QUESTION", "LATEST_MARKET_PROBABILITY",
                                       "UNIQUE_BETTOR_COUNT", "MANIFOLD_VOLUME"])


def test_consensus_direction_and_candidates():
    # Copper markets priced bullish (P(exceeds) high), gold bearish.
    df = _mkts([
        ["Will copper price exceed 9000 USD/tonne?", 0.75, 50, 9000],
        ["Will copper price exceed 10000 USD/tonne?", 0.70, 40, 8000],
        ["Will copper price exceed 11000 USD/tonne?", 0.65, 30, 7000],
        ["Will gold spot price exceed 4000 USD per troy oz?", 0.30, 50, 9000],
        ["Will gold spot price exceed 4500 USD per troy oz?", 0.25, 40, 8000],
        ["Will gold spot price exceed 5000 USD per troy oz?", 0.20, 30, 7000],
    ])
    cons = material_consensus(df)
    assert cons["copper"].tilt > 0          # bullish copper
    assert cons["gold"].tilt < 0            # bearish gold
    cands = spread_candidates(cons, corr_matrix=None, min_tilt=0.05, min_confidence=0.0)
    bydir = {c.equity: c.direction for c in cands}
    assert bydir.get("FCX") == "bull"       # copper bullish -> bull-call on FCX
    assert bydir.get("GDX") == "bear"       # gold bearish -> bear-put on GDX
    # conviction ranking is finite + sorted desc
    assert cands == sorted(cands, key=lambda c: c.conviction, reverse=True)


def test_low_signal_filtered():
    df = _mkts([["Will copper price exceed 9000 USD/tonne?", 0.51, 50, 9000]])  # tilt ~0
    assert spread_candidates(material_consensus(df), min_tilt=0.05) == []
