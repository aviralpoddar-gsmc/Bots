"""Delta-hedge: nets option delta vs held shares for short-vol positions only."""

from quantbots.equity_options.hedge import VOL_STRUCTURES, compute_hedges
from quantbots.equity_options.occ import build_occ
from datetime import date


class _Store:
    def __init__(self, trades): self._t = trades
    def trades(self): return self._t


class _Broker:
    def __init__(self, positions): self._p = positions
    def positions(self): return self._p


class _Chain:
    def __init__(self, deltas): self._d = deltas  # symbol -> delta
    def get_chain(self, und, **k):
        return [{"symbol": s, "delta": d} for s, d in self._d.items()]


def test_hedges_only_vol_structures():
    fly_sym = build_occ("XOM", date(2026, 9, 18), "call", 110)
    spread_sym = build_occ("GDX", date(2026, 9, 18), "put", 100)
    store = _Store([
        {"underlying": "XOM", "structure": "iron_fly", "status": "filled"},
        {"underlying": "GDX", "structure": "bear_put_spread", "status": "filled"},  # directional, skip
    ])
    broker = _Broker([
        {"symbol": fly_sym, "qty": -1},          # short call -> negative delta contribution
        {"symbol": spread_sym, "qty": 1},
    ])
    chain = _Chain({fly_sym: 0.5, spread_sym: -0.4})
    actions = compute_hedges(broker, chain, store)
    # Only XOM (iron_fly) is hedged; GDX (directional) is left alone.
    assert [a.underlying for a in actions] == ["XOM"]
    a = actions[0]
    # net option delta = qty(-1) * 0.5 * 100 = -50 -> buy 50 shares to flatten
    assert a.net_option_delta == -50 and a.trade_shares == 50 and a.side == "buy"


def test_no_vol_positions_no_hedge():
    store = _Store([{"underlying": "GDX", "structure": "bear_put_spread", "status": "filled"}])
    assert compute_hedges(_Broker([]), _Chain({}), store) == []


def test_vol_structures_set():
    assert "iron_fly" in VOL_STRUCTURES and "bear_put_spread" not in VOL_STRUCTURES
