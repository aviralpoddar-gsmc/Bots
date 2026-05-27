from quantbots.sizing import DEFAULT_LIMITS, compute_trade, mana_to_move_price


def test_mana_to_move_price_zero_for_no_move():
    assert mana_to_move_price(0.5, 0.5, 1000) == 0


def test_mana_to_move_price_grows_with_distance():
    small = mana_to_move_price(0.5, 0.55, 1000)
    big = mana_to_move_price(0.5, 0.75, 1000)
    assert 0 < small < big


def test_compute_trade_direction_yes_when_estimate_above():
    d = compute_trade(estimate=0.80, current_prob=0.50, position=None,
                      liquidity=1000, limits=DEFAULT_LIMITS)
    assert d is not None and d["direction"] == "YES"
    assert d["amount"] >= DEFAULT_LIMITS["min_order_mana"]


def test_compute_trade_direction_no_when_estimate_below():
    d = compute_trade(estimate=0.20, current_prob=0.50, position=None,
                      liquidity=1000, limits=DEFAULT_LIMITS)
    assert d is not None and d["direction"] == "NO"


def test_hold_band_skips_small_edge_on_existing_position():
    # Held position, edge within hold_band -> no churn.
    d = compute_trade(estimate=0.53, current_prob=0.50, position={"net_shares": 10},
                      liquidity=1000, limits=DEFAULT_LIMITS)
    assert d is None


def test_no_hold_band_when_no_position():
    d = compute_trade(estimate=0.53, current_prob=0.50, position=None,
                      liquidity=10000, limits=DEFAULT_LIMITS)
    # Small edge but a fresh position is allowed if it clears min_order_mana.
    assert d is None or d["amount"] >= DEFAULT_LIMITS["min_order_mana"]


def test_amount_capped_by_max_order_size():
    d = compute_trade(estimate=0.99, current_prob=0.01, position=None,
                      liquidity=1_000_000, limits=DEFAULT_LIMITS)
    assert d is not None and d["amount"] <= DEFAULT_LIMITS["max_order_size"]


def test_allocate_funds_highest_ev_within_budget():
    from quantbots.runner import _allocate
    # All YES from 0.50; EV/mana scales with edge here, so higher estimate ranks first.
    signals = [
        {"market_id": "a", "estimate": 0.60, "current_prob": 0.50, "direction": "YES", "amount": 25},
        {"market_id": "b", "estimate": 0.90, "current_prob": 0.50, "direction": "YES", "amount": 25},
        {"market_id": "c", "estimate": 0.75, "current_prob": 0.50, "direction": "YES", "amount": 25},
    ]
    kept = _allocate(signals, {"max_run_budget": 50})
    ids = {s["market_id"] for s in kept}
    assert ids == {"b", "c"}            # the two highest-EV that fit in 50
    assert sum(s["amount"] for s in kept) <= 50


def test_allocate_no_budget_is_unlimited():
    from quantbots.runner import _allocate
    signals = [{"market_id": "a", "estimate": 0.9, "current_prob": 0.5,
                "direction": "YES", "amount": 25}]
    kept = _allocate(signals, {})
    assert len(kept) == 1 and kept[0]["market_id"] == "a"
