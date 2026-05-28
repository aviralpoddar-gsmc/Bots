"""Runner execution tests: throttle detection + retry-with-backoff (no double-bet)."""

import time

import pytest

from quantbots.config import BotConfig
from quantbots.runner import _is_throttle, run_bot, sync_resolutions
from quantbots.sizing import DEFAULT_LIMITS
from quantbots.store.db import Store
from quantbots.store.pnl import position_pnl
from quantbots.store.trades import trades_for_bot
from quantbots.strategies.base import Strategy


def test_is_throttle_detects_transient_server_messages():
    assert _is_throttle({"error": "High volume of requests (0 requests in queue)"})
    assert _is_throttle("please try again later")
    assert _is_throttle(Exception("rate limit exceeded"))
    assert not _is_throttle({"betId": "x"})
    assert not _is_throttle("insufficient balance")


class _FixedStrategy(Strategy):
    """Bets every market to 0.99 so the runner always produces a signal."""
    name = "fixed"

    def estimate(self, group):
        return {m["id"]: 0.99 for m in group}


class _FlakyClient:
    """batch_bet throttles the first `fail_sweeps` times it sees a contract, then
    fills. Lets us assert retries happen and nothing is double-recorded."""

    def __init__(self, fail_sweeps=2):
        self.fail_sweeps = fail_sweeps
        self.seen: dict[str, int] = {}
        self.fills: dict[str, int] = {}

    def batch_bet(self, bets):
        out = []
        for b in bets:
            cid = b["contractId"]
            self.seen[cid] = self.seen.get(cid, 0) + 1
            if self.seen[cid] <= self.fail_sweeps:
                out.append({"contractId": cid, "error": "High volume of requests"})
            else:
                self.fills[cid] = self.fills.get(cid, 0) + 1
                out.append({"contractId": cid, "betId": f"bet-{cid}-{self.fills[cid]}",
                            "amount": b["amount"], "shares": b["amount"], "probAfter": 0.9})
        return out


def _bot():
    limits = {**DEFAULT_LIMITS, "max_order_size": 25, "min_order_mana": 1}
    return BotConfig(name="t", strategy="fixed", account_env="X", enabled=True,
                     limits=limits, params={})


def _markets(n):
    return [{"id": f"m{i}", "question": f"Will X exceed {i}?", "probability": 0.5,
             "totalLiquidity": 100, "isResolved": False,
             "closeTime": time.time() * 1000 + 30 * 24 * 3600 * 1000} for i in range(n)]


def test_run_bot_retries_throttled_bets_until_filled(tmp_path, monkeypatch):
    monkeypatch.setattr("quantbots.runner.RETRY_BACKOFF", 0.0)  # no real sleeping
    with Store(tmp_path / "t.sqlite") as store:
        store.upsert_markets(_markets(5))
        client = _FlakyClient(fail_sweeps=2)  # throttle twice, fill on the 3rd sweep
        res = run_bot(bot=_bot(), client=client, store=store,
                      strategy=_FixedStrategy(), dry_run=False)
        # All 5 eventually placed, each exactly once (no double-bet on retry).
        assert res.orders_placed == 5
        assert all(v == 1 for v in client.fills.values())
        assert len(client.fills) == 5
        assert not res.errors


def test_run_bot_gives_up_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr("quantbots.runner.RETRY_BACKOFF", 0.0)
    with Store(tmp_path / "t.sqlite") as store:
        store.upsert_markets(_markets(3))
        client = _FlakyClient(fail_sweeps=99)  # always throttle
        res = run_bot(bot=_bot(), client=client, store=store,
                      strategy=_FixedStrategy(), dry_run=False)
        assert res.orders_placed == 0
        assert len(res.errors) == 3
        assert all("throttled after" in e for e in res.errors)


# --- sync_resolutions: cache-driven path + CANCEL handling ----------------

class _RecordingClient:
    """Tracks get_market calls so tests can assert cache wasn't bypassed."""
    def __init__(self, markets=None, raises=None):
        self._markets = markets or {}
        self._raises = raises or set()
        self.get_calls: list[str] = []

    def get_market(self, market_id):
        self.get_calls.append(market_id)
        if market_id in self._raises:
            raise TimeoutError(f"simulated timeout on {market_id}")
        return self._markets[market_id]


def _entry(store, bot_id, market_id, amount, shares, direction="YES"):
    store.record_trade(bot_id=bot_id, market_id=market_id, trade_type="ENTRY",
                       direction=direction, amount=amount, shares=shares,
                       price_after=amount / shares)


@pytest.fixture
def store_with_bot(tmp_path):
    with Store(tmp_path / "t.sqlite") as s:
        bot_id = s.upsert_bot("t", "fixed")
        yield s, bot_id


def test_resolve_yes_closes_position_at_one(store_with_bot):
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=10, shares=20)
    store.upsert_markets([{"id": "m1", "isResolved": True, "resolution": "YES"}])
    client = _RecordingClient()
    n = sync_resolutions(client, store, bot_id)
    assert n == 1
    assert client.get_calls == []  # cache hit -> no API
    assert store.open_positions(bot_id) == {}
    # PnL: 20 shares * 1.0 - 10 cost = +10
    r, u = position_pnl(trades_for_bot(store.conn, bot_id), current_prob=None)
    assert r == pytest.approx(10.0) and u == 0.0


def test_resolve_no_closes_yes_position_at_zero(store_with_bot):
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=10, shares=20, direction="YES")
    store.upsert_markets([{"id": "m1", "isResolved": True, "resolution": "NO"}])
    n = sync_resolutions(_RecordingClient(), store, bot_id)
    assert n == 1
    # Lost the full stake: PnL = -10
    r, _ = position_pnl(trades_for_bot(store.conn, bot_id), current_prob=None)
    assert r == pytest.approx(-10.0)


def test_resolve_cancel_refunds_at_cost_basis(store_with_bot):
    """CANCEL on the clone refunds the stake — realized PnL must be 0."""
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=15, shares=30, direction="YES")  # 0.5/share
    store.upsert_markets([{"id": "m1", "isResolved": True, "resolution": "CANCEL"}])
    n = sync_resolutions(_RecordingClient(), store, bot_id)
    assert n == 1
    assert store.open_positions(bot_id) == {}
    r, _ = position_pnl(trades_for_bot(store.conn, bot_id), current_prob=None)
    assert r == pytest.approx(0.0)


def test_resolve_cancel_refunds_after_partial_exit(store_with_bot):
    """Partial exit before CANCEL: refund only the remaining net_shares at cost."""
    store, bot_id = store_with_bot
    # Entry: 20 mana for 40 shares @ 0.5/share. Then sell 10 shares @ 0.6.
    _entry(store, bot_id, "m1", amount=20, shares=40, direction="YES")
    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="PARTIAL_EXIT",
                       direction="YES", amount=6, shares=10, price_after=0.6)
    store.upsert_markets([{"id": "m1", "isResolved": True, "resolution": "CANCEL"}])
    n = sync_resolutions(_RecordingClient(), store, bot_id)
    assert n == 1
    # Realized PnL = exit proceeds - cost basis of exited shares = 6 - 10*0.5 = +1
    # CANCEL contributes 0 -> total realized = +1
    r, _ = position_pnl(trades_for_bot(store.conn, bot_id), current_prob=None)
    assert r == pytest.approx(1.0)


def test_resolve_unresolved_market_is_noop(store_with_bot):
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=10, shares=20)
    store.upsert_markets([{"id": "m1", "isResolved": False}])
    assert sync_resolutions(_RecordingClient(), store, bot_id) == 0
    assert "m1" in store.open_positions(bot_id)


def test_resolve_cache_miss_falls_back_to_api(store_with_bot):
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=10, shares=20)
    # Note: no upsert_markets() -> cache miss
    client = _RecordingClient(markets={
        "m1": {"id": "m1", "isResolved": True, "resolution": "YES"},
    })
    assert sync_resolutions(client, store, bot_id) == 1
    assert client.get_calls == ["m1"]  # fallback fired


def test_resolve_cache_miss_with_api_failure_skips(store_with_bot):
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=10, shares=20)
    _entry(store, bot_id, "m2", amount=10, shares=20)
    # m1: cache miss + API timeout (skip). m2: cache hit + resolved (close).
    store.upsert_markets([{"id": "m2", "isResolved": True, "resolution": "YES"}])
    client = _RecordingClient(markets={}, raises={"m1"})
    n = sync_resolutions(client, store, bot_id)
    assert n == 1  # m2 closed, m1 skipped (one bad market must not abort the walk)
    assert "m1" in store.open_positions(bot_id)
    assert "m2" not in store.open_positions(bot_id)


def test_resolve_is_idempotent(store_with_bot):
    """Running twice doesn't double-close: after first run, position is no longer OPEN."""
    store, bot_id = store_with_bot
    _entry(store, bot_id, "m1", amount=10, shares=20)
    store.upsert_markets([{"id": "m1", "isResolved": True, "resolution": "CANCEL"}])
    assert sync_resolutions(_RecordingClient(), store, bot_id) == 1
    assert sync_resolutions(_RecordingClient(), store, bot_id) == 0
