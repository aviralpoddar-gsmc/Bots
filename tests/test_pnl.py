from quantbots.store.pnl import bot_pnl, position_pnl


def _entry(market="m1", direction="YES", amount=10, shares=20, price_after=0.5):
    return {"market_id": market, "direction": direction, "trade_type": "ENTRY",
            "amount": amount, "shares": shares, "price_after": price_after}


def _exit(market="m1", direction="YES", amount=15, shares=20, price_after=0.75,
          ttype="EXIT"):
    return {"market_id": market, "direction": direction, "trade_type": ttype,
            "amount": amount, "shares": shares, "price_after": price_after}


def test_open_position_unrealized_only():
    trades = [_entry(amount=10, shares=20, price_after=0.5)]
    realized, unrealized = position_pnl(trades, current_prob=0.5)
    assert realized == 0.0
    # 20 YES shares marked at 0.5 = 10; cost 10 -> 0 unrealized.
    assert abs(unrealized - 0.0) < 1e-9


def test_open_position_gains_when_price_rises():
    trades = [_entry(amount=10, shares=20, price_after=0.5)]
    _, unrealized = position_pnl(trades, current_prob=0.75)
    # 20 * 0.75 = 15, cost 10 -> +5.
    assert abs(unrealized - 5.0) < 1e-9


def test_no_share_unrealized_uses_one_minus_prob():
    trades = [{"market_id": "m", "direction": "NO", "trade_type": "ENTRY",
               "amount": 10, "shares": 20, "price_after": 0.5}]
    _, unrealized = position_pnl(trades, current_prob=0.25)
    # NO shares worth (1-0.25)=0.75 each: 20*0.75=15, cost 10 -> +5.
    assert abs(unrealized - 5.0) < 1e-9


def test_realized_on_full_exit():
    trades = [_entry(amount=10, shares=20, price_after=0.5),
              _exit(amount=15, shares=20, price_after=0.75)]
    realized, unrealized = position_pnl(trades, current_prob=0.75)
    # proceeds 20*0.75=15, minus entry 10 -> +5 realized; no shares left.
    assert abs(realized - 5.0) < 1e-9
    assert abs(unrealized) < 1e-9


def test_resolution_close_realizes_winning_yes():
    trades = [_entry(amount=10, shares=20, price_after=0.5),
              _exit(amount=20, shares=20, price_after=1.0, ttype="RESOLUTION_CLOSE")]
    realized, unrealized = position_pnl(trades, current_prob=1.0)
    # 20 shares * 1.0 = 20 proceeds - 10 entry = +10.
    assert abs(realized - 10.0) < 1e-9
    assert abs(unrealized) < 1e-9


def test_bot_pnl_aggregates_and_counts():
    trades = [
        _entry(market="m1", amount=10, shares=20, price_after=0.5),  # open
        _entry(market="m2", amount=10, shares=20, price_after=0.5),  # closed below
        _exit(market="m2", amount=20, shares=20, price_after=1.0, ttype="RESOLUTION_CLOSE"),
    ]
    summary = bot_pnl(trades, current_prob=lambda mid: 0.5)
    assert summary["open_positions"] == 1
    assert summary["closed_positions"] == 1
    assert summary["total_invested"] == 20
    assert abs(summary["realized_pnl"] - 10.0) < 1e-9
