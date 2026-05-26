import pytest

from quantbots.store.db import Store


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "t.sqlite") as s:
        yield s


def test_bot_upsert_is_idempotent(store):
    a = store.upsert_bot("b1", "mean_reversion", {"x": 1})
    b = store.upsert_bot("b1", "mean_reversion", {"x": 2})
    assert a == b
    assert store.get_bot("b1")["strategy"] == "mean_reversion"


def test_ledger_drives_open_positions(store):
    bot_id = store.upsert_bot("b1", "mean_reversion")
    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="ENTRY",
                       direction="YES", amount=10, shares=20, price_after=0.5)
    positions = store.open_positions(bot_id)
    assert "m1" in positions
    assert positions["m1"]["net_shares"] == 20
    assert positions["m1"]["status"] == "OPEN"


def test_full_exit_closes_position(store):
    bot_id = store.upsert_bot("b1", "mean_reversion")
    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="ENTRY",
                       direction="YES", amount=10, shares=20, price_after=0.5)
    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="RESOLUTION_CLOSE",
                       direction="YES", amount=20, shares=20, price_after=1.0)
    assert store.open_positions(bot_id) == {}


def test_market_cache_roundtrip_and_snapshot(store):
    bot_id = store.upsert_bot("b1", "mean_reversion")
    store.upsert_markets([
        {"id": "m1", "question": "Q?", "probability": 0.6, "totalLiquidity": 500,
         "isResolved": False},
    ])
    assert len(store.load_open_markets()) == 1
    assert store.current_prob("m1") == 0.6

    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="ENTRY",
                       direction="YES", amount=10, shares=20, price_after=0.5)
    summary = store.write_snapshot(bot_id)
    assert summary["open_positions"] == 1
    board = store.leaderboard()
    assert board and board[0]["name"] == "b1"
