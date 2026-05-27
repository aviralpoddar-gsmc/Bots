from quantbots.portfolio import allocate, book_summary, ev_per_mana


def _sig(mid, est, q, direction, amount, group=None):
    s = {"market_id": mid, "estimate": est, "current_prob": q,
         "direction": direction, "amount": amount}
    if group is not None:
        s["group"] = group
    return s


def test_ev_per_mana_rewards_cheap_deep_edge():
    # NO bet at 0.99 we think is 0.01: each mana buys ~100 shares worth ~1 each.
    deep = ev_per_mana(estimate=0.01, current_prob=0.99, direction="NO")
    shallow = ev_per_mana(estimate=0.40, current_prob=0.50, direction="NO")
    assert deep > shallow > 0


def test_ev_sign_matches_direction():
    assert ev_per_mana(0.8, 0.5, "YES") > 0
    assert ev_per_mana(0.2, 0.5, "NO") > 0
    # Betting the wrong way is -EV.
    assert ev_per_mana(0.2, 0.5, "YES") < 0


def test_allocate_orders_by_ev_and_caps_total_budget():
    sigs = [
        _sig("cheap", 0.01, 0.99, "NO", 100),   # huge EV/mana
        _sig("mid", 0.40, 0.50, "NO", 100),     # modest EV
    ]
    funded = allocate(sigs, total_budget=100)
    # Only 100 mana available -> the best-EV order is funded first and fills it.
    assert funded[0]["market_id"] == "cheap"
    assert sum(s["amount"] for s in funded) <= 100


def test_allocate_trims_last_order_to_fit_budget():
    sigs = [_sig("a", 0.9, 0.5, "YES", 80), _sig("b", 0.9, 0.5, "YES", 80)]
    funded = allocate(sigs, total_budget=100)
    assert sum(s["amount"] for s in funded) <= 100


def test_per_group_budget_caps_correlated_exposure():
    # Five correlated gold orders, each wants 100, but the group is capped at 150.
    sigs = [_sig(f"g{i}", 0.05, 0.95, "NO", 100, group="GOLD") for i in range(5)]
    funded = allocate(sigs, total_budget=10_000, per_group_budget=150)
    assert sum(s["amount"] for s in funded) <= 150


def test_min_ev_filters_marginal_orders():
    sigs = [_sig("weak", 0.52, 0.50, "YES", 100)]  # EV/mana = 0.04
    assert allocate(sigs, total_budget=1000, min_ev=0.05) == []
    assert allocate(sigs, total_budget=1000, min_ev=0.0)


def test_unlimited_budget_funds_all_positive_ev():
    sigs = [_sig(f"m{i}", 0.9, 0.5, "YES", 10) for i in range(20)]
    funded = allocate(sigs, total_budget=0)  # 0/None => no overall ceiling
    assert len(funded) == 20


def test_max_group_exposure_accounts_for_existing_position():
    # Group already holds 4500 of a 5000 cap -> only 500 more can be funded.
    sigs = [_sig(f"g{i}", 0.05, 0.95, "NO", 100, group="GOLD") for i in range(20)]
    funded = allocate(
        sigs, total_budget=100_000, max_group_exposure=5000,
        existing_group={"GOLD": 4500},
    )
    assert sum(s["amount"] for s in funded) <= 500


def test_max_total_exposure_accounts_for_existing_position():
    sigs = [_sig(f"m{i}", 0.9, 0.5, "YES", 100, group=f"m{i}") for i in range(20)]
    funded = allocate(sigs, total_budget=100_000, max_total_exposure=1000, existing_total=800)
    assert sum(s["amount"] for s in funded) <= 200


def test_book_summary_aggregates():
    sigs = [_sig("a", 0.9, 0.5, "YES", 30, group="X"),
            _sig("b", 0.1, 0.5, "NO", 30, group="X")]
    funded = allocate(sigs, total_budget=1000)
    summ = book_summary(funded)
    assert summ["orders"] == 2
    assert summ["staked"] == sum(s["amount"] for s in funded)
    assert summ["groups"]["X"] == summ["staked"]
    assert summ["exp_profit"] > 0
