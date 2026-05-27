"""Runner execution tests: throttle detection + retry-with-backoff (no double-bet)."""

import time

from quantbots.config import BotConfig
from quantbots.runner import _is_throttle, run_bot
from quantbots.sizing import DEFAULT_LIMITS
from quantbots.store.db import Store
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
