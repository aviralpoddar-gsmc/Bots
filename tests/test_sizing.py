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


def test_cap_to_budget_keeps_highest_edge_within_budget():
    from quantbots.runner import _cap_to_budget
    signals = [
        {"market_id": "a", "amount": 25, "edge": 0.10},
        {"market_id": "b", "amount": 25, "edge": 0.30},  # highest edge
        {"market_id": "c", "amount": 25, "edge": 0.20},
    ]
    kept = _cap_to_budget(signals, budget=50)
    ids = {s["market_id"] for s in kept}
    assert ids == {"b", "c"}            # the two highest-edge that fit in 50
    assert sum(s["amount"] for s in kept) <= 50


def test_cap_to_budget_none_is_unlimited():
    from quantbots.runner import _cap_to_budget
    signals = [{"market_id": "a", "amount": 25, "edge": 0.1}]
    assert _cap_to_budget(signals, None) == signals
