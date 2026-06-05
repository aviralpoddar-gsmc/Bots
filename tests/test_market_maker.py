"""Tests for the market-maker: quote construction, the fill-reconcile loop
(idempotency is the critical property), source delegation, and the dry-run/live
execution path. No network — a FakeClient stands in for the clone."""

import time

import pytest

from quantbots.config import BotConfig
from quantbots.maker import (
    _clamp_prob,
    _ledger_state,
    _quote_size,
    build_maker_strategy,
    build_quotes,
    reconcile_fills,
    run_maker,
)
from quantbots.store.db import Store
from quantbots.strategies.commodity_spot import CommoditySpotStrategy
from quantbots.strategies.market_maker import MarketMakerStrategy


# --- fixtures -----------------------------------------------------------------

class FakeStrat:
    """Minimal stand-in for build_quotes (half_spread + inventory_cap + group)."""

    def __init__(self, half=0.04, cap=200.0):
        self._half = half
        self.inventory_cap = cap

    def half_spread(self, market_id=None):
        return self._half

    def correlation_key(self, m):
        return m["id"]


class StubMakerStrat:
    """Full maker-strategy interface for run_maker, pricing markets from a dict."""

    inventory_cap = 200.0
    quote_ttl_hours = 25.0
    max_markets = 10

    def __init__(self, fair):
        self.fair = fair

    def bind(self, obs):
        pass

    def prefilter(self, markets):
        return markets

    def group(self, markets):
        return [[m] for m in markets]

    def estimate(self, group):
        return {m["id"]: self.fair[m["id"]] for m in group if m["id"] in self.fair}

    def correlation_key(self, m):
        return m["id"]

    def half_spread(self, market_id=None):
        return 0.04


class FakeClient:
    def __init__(self, fail_cancel=None):
        self.me = {"id": "uid1"}
        self.bets = {}  # contractId -> [bet dict]
        self.cancelled = []
        self.placed = []
        self.dry = []
        self.fail_cancel = set(fail_cancel or ())  # bet ids whose cancel raises
        self._n = 0

    def get_me(self):
        return self.me

    def get_bets(self, **p):
        cid = p.get("contractId")
        if cid is not None:
            return list(self.bets.get(cid, []))
        out = []  # userId-only query: all bets across markets
        for bs in self.bets.values():
            out += bs
        return out

    def get_open_limit_orders(self, market_id=None, user_id=None):
        out = []
        for cid, bs in self.bets.items():
            if market_id and cid != market_id:
                continue
            out += [b for b in bs if not b.get("isFilled") and not b.get("isCancelled")]
        return out

    def cancel_bet(self, bet_id):
        if bet_id in self.fail_cancel:
            raise RuntimeError(f"cancel boom for {bet_id}")
        for bs in self.bets.values():
            for b in bs:
                if b["id"] == bet_id:
                    b["isCancelled"] = True
        self.cancelled.append(bet_id)
        return {"id": bet_id, "isCancelled": True}

    def batch_bet(self, bets):
        resp = []
        for b in bets:
            self._n += 1
            bid = f"b{self._n}"
            rec = {
                "id": bid, "betId": bid, "contractId": b["contractId"], "outcome": b["outcome"],
                "limitProb": b["limitProb"], "orderAmount": b["amount"], "amount": 0.0,
                "shares": 0.0, "isFilled": False, "isCancelled": False, "fills": [],
            }
            self.bets.setdefault(b["contractId"], []).append(rec)
            self.placed.append(rec)
            resp.append({"betId": bid, "contractId": b["contractId"]})
        return resp

    def place_bet(self, market_id, outcome, amount, limit_prob=None,
                  expires_millis_after=None, expires_at=None, dry_run=False):
        self.dry.append((market_id, outcome, amount, limit_prob, expires_millis_after, dry_run))
        return {"betId": "dry-run", "amount": 0, "shares": 0, "isFilled": False}

    # test helper
    def fill(self, bet_id, shares, amount):
        for bs in self.bets.values():
            for b in bs:
                if b["id"] == bet_id:
                    b["shares"] = shares
                    b["amount"] = amount
                    b["isFilled"] = True


def _market(mid, q="Will Gold spot price exceed $5,181/ozt on Dec 31?", liq=400):
    return {"id": mid, "question": q, "probability": 0.5, "totalLiquidity": liq,
            "isResolved": False, "closeTime": time.time() * 1000 + 1e10}


def _cfg(**limits):
    base = {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5,
            "max_run_budget": 5000, "dry_run_sample": 25}
    base.update(limits)
    return BotConfig(name="mm_test", strategy="market_maker", enabled=False, limits=base, params={})


# --- pure helpers -------------------------------------------------------------

def test_clamp_prob_whole_percent_and_band():
    assert _clamp_prob(0.544) == 0.54
    assert _clamp_prob(-0.2) == 0.01
    assert _clamp_prob(1.5) == 0.99


def test_quote_size_liquidity_capped():
    assert _quote_size(400, {"max_order_size": 50, "liquidity_pct": 0.25}) == 50  # 100 capped to 50
    assert _quote_size(120, {"max_order_size": 50, "liquidity_pct": 0.25}) == 30  # 0.25*120


# --- build_quotes -------------------------------------------------------------

def test_build_quotes_two_sided_spread():
    q = build_quotes(FakeStrat(half=0.04), [_market("m1")], {"m1": 0.54}, {},
                     {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                     min_resolv=0.0, budget=1e9)
    assert len(q) == 1
    assert q[0].bid == 0.50 and q[0].ask == 0.58 and q[0].sides == ("bid", "ask")


def test_build_quotes_skips_crossed_at_clamp():
    # 0.50 ± 0.004 both round to 0.50 -> crossed -> skipped.
    q = build_quotes(FakeStrat(half=0.004), [_market("m1")], {"m1": 0.50}, {},
                     {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                     min_resolv=0.0, budget=1e9)
    assert q == []


def test_build_quotes_skips_near_boundary_squashed_leg():
    # fair 0.99: bid 0.95 (4pt) but ask clamps to 0.99 (0pt) -> below min spread -> skip.
    s = FakeStrat(half=0.04)
    s.min_half_spread = 0.02
    for f in (0.99, 0.995, 0.01):
        q = build_quotes(s, [_market("m1")], {"m1": f}, {},
                         {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                         min_resolv=0.0, budget=1e9)
        assert q == [], f"f={f} should be skipped (a leg quotes inside min_spread)"


def test_build_quotes_inventory_cap_one_sided():
    # Net long YES past cap -> only the selling (ask) side.
    long_yes = build_quotes(FakeStrat(half=0.04, cap=100.0), [_market("m1")], {"m1": 0.54},
                            {"m1": 150.0}, {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                            min_resolv=0.0, budget=1e9)
    assert long_yes[0].sides == ("ask",)
    # Net long NO past cap -> only the buying (bid) side.
    long_no = build_quotes(FakeStrat(half=0.04, cap=100.0), [_market("m1")], {"m1": 0.54},
                           {"m1": -150.0}, {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                           min_resolv=0.0, budget=1e9)
    assert long_no[0].sides == ("bid",)


def test_build_quotes_resolvability_gate():
    m = _market("m1", q="Will copper production exceed 2000 kt on June 30, 2026?")
    # A production market scores ~0 resolvability; a high gate drops it.
    assert build_quotes(FakeStrat(), [m], {"m1": 0.54}, {},
                        {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                        min_resolv=0.5, budget=1e9) == []


def test_build_quotes_budget_stops():
    markets = [_market(f"m{i}") for i in range(5)]
    fair = {m["id"]: 0.54 for m in markets}
    # budget only covers 2 legs (size 50 each) -> 1 two-sided quote.
    q = build_quotes(FakeStrat(half=0.04), markets, fair, {},
                     {"max_order_size": 50, "liquidity_pct": 0.25, "min_order_mana": 5},
                     min_resolv=0.0, budget=100)
    assert len(q) == 1


# --- reconcile loop (the critical idempotency property) -----------------------

def test_reconcile_records_delta_only_and_is_idempotent(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    bot_id = store.upsert_bot("mm_test", "market_maker")
    c = FakeClient()
    # one resting limit that gradually fills
    c.bets["m1"] = [{"id": "x1", "outcome": "NO", "limitProb": 0.58, "shares": 0.0,
                     "amount": 0.0, "probBefore": 0.6, "probAfter": 0.58}]
    rec_sh, rec_amt = {}, {}

    # nothing filled yet
    assert reconcile_fills(c, store, bot_id, "uid1", ["m1"], rec_sh, rec_amt) == 0

    # 3 shares fill -> one ENTRY row
    c.bets["m1"][0]["shares"] = 3.0
    c.bets["m1"][0]["amount"] = 1.5
    assert reconcile_fills(c, store, bot_id, "uid1", ["m1"], rec_sh, rec_amt) == 1
    # re-run with no change -> idempotent (no new row)
    assert reconcile_fills(c, store, bot_id, "uid1", ["m1"], rec_sh, rec_amt) == 0

    # 2 more shares fill -> only the delta recorded
    c.bets["m1"][0]["shares"] = 5.0
    c.bets["m1"][0]["amount"] = 2.5
    assert reconcile_fills(c, store, bot_id, "uid1", ["m1"], rec_sh, rec_amt) == 1

    inv, ish, iamt, filled = _ledger_state(store, bot_id)
    assert inv["m1"] == -5.0  # net NO 5 shares
    assert round(filled, 4) == 2.5
    assert len(store.trades_for_bot(bot_id)) == 2  # two delta rows, total 5 shares


# --- source delegation --------------------------------------------------------

def test_strategy_delegates_to_commodity_spot():
    mm = MarketMakerStrategy(fair_value_source="commodity_spot")
    assert isinstance(mm.source, CommoditySpotStrategy)

    class Obs:
        def latest_observation(self, entity, source=None):
            return {"entity": entity, "value": 7515.3} if entity == "SILVER" else None

    mm.bind(Obs())
    m = _market("s1", q="Will LBMA silver spot price exceed 100 USD per troy oz on Dec 31?")
    direct = CommoditySpotStrategy()
    direct.bind(Obs())
    # Same model; allow microsecond float drift in years_to_close(time.time()).
    assert mm.estimate([m])["s1"] == pytest.approx(direct.estimate([m])["s1"], abs=1e-6)
    assert mm.correlation_key(m) == "SILVER"


# --- end-to-end run_maker -----------------------------------------------------

def test_run_maker_dry_run_no_writes(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    store.upsert_markets([_market("m1")])
    c = FakeClient()
    res = run_maker(bot=_cfg(), client=c, store=store,
                    strategy=StubMakerStrat({"m1": 0.54}), dry_run=True)
    assert len(res.quotes) == 1 and res.reserved_mana == 100  # 2 legs x 50
    assert all(d[5] is True for d in c.dry)  # every validation used dry_run=True
    assert c.placed == [] and c.cancelled == []  # no live writes
    bot_id = store.get_bot("mm_test")["bot_id"]
    assert store.trades_for_bot(bot_id) == []  # ledger untouched


def test_run_maker_live_posts_cancels_and_reconciles(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    store.upsert_markets([_market("m1")])
    c = FakeClient()
    strat = StubMakerStrat({"m1": 0.54})
    cfg = _cfg()

    r1 = run_maker(bot=cfg, client=c, store=store, strategy=strat, dry_run=False)
    assert r1.legs_posted == 2 and len(c.placed) == 2 and r1.fills_recorded == 0

    # the ask (NO) leg crosses and fills before the next cycle
    ask = next(b for b in c.placed if b["outcome"] == "NO")
    c.fill(ask["id"], shares=3.0, amount=1.0)

    r2 = run_maker(bot=cfg, client=c, store=store, strategy=strat, dry_run=False)
    assert r2.fills_recorded == 1          # the fill got recorded
    assert r2.cancelled >= 1               # the still-resting bid got cancelled before re-quoting

    r3 = run_maker(bot=cfg, client=c, store=store, strategy=strat, dry_run=False)
    assert r3.fills_recorded == 0          # idempotent — no double count

    bot_id = store.get_bot("mm_test")["bot_id"]
    entries = store.trades_for_bot(bot_id)
    assert len(entries) == 1 and entries[0]["shares"] == 3.0 and entries[0]["direction"] == "NO"


def test_run_maker_reconciles_orphaned_market_fill(tmp_path):
    # A market we hold an order in but DON'T quote this cycle must still have its
    # fills reconciled (H1: reconcile spans open-orders + inventory + recent bets).
    store = Store(tmp_path / "t.sqlite")
    store.upsert_markets([_market("m1")])  # only m1 is priced/quoted
    c = FakeClient()
    # an order on orphaned m2 that fully filled between cycles (gone from open-limit)
    c.bets["m2"] = [{"id": "o1", "contractId": "m2", "outcome": "NO", "limitProb": 0.6,
                     "shares": 4.0, "amount": 2.0, "isFilled": True, "isCancelled": False,
                     "probBefore": 0.62, "probAfter": 0.6}]
    run_maker(bot=_cfg(), client=c, store=store, strategy=StubMakerStrat({"m1": 0.54}), dry_run=False)
    bot_id = store.get_bot("mm_test")["bot_id"]
    entries = store.trades_for_bot(bot_id)
    assert any(e["market_id"] == "m2" and e["shares"] == 4.0 for e in entries)


def test_run_maker_cancel_failure_does_not_repost(tmp_path):
    # If cancelling m1's stale order fails, m1 must NOT be reposted (H2: no
    # double-stack); other markets still quote.
    store = Store(tmp_path / "t.sqlite")
    store.upsert_markets([_market("m1"), _market("m2")])
    c = FakeClient(fail_cancel={"a1"})
    c.bets["m1"] = [{"id": "a1", "contractId": "m1", "outcome": "YES", "limitProb": 0.5,
                     "shares": 0.0, "amount": 0.0, "isFilled": False, "isCancelled": False}]
    c.bets["m2"] = [{"id": "a2", "contractId": "m2", "outcome": "YES", "limitProb": 0.5,
                     "shares": 0.0, "amount": 0.0, "isFilled": False, "isCancelled": False}]
    r = run_maker(bot=_cfg(), client=c, store=store,
                  strategy=StubMakerStrat({"m1": 0.54, "m2": 0.54}), dry_run=False)
    posted = {b["contractId"] for b in c.placed}
    assert "m1" not in posted and "m2" in posted
    assert any("cancel m1" in e for e in r.errors)


def test_build_maker_strategy_wraps_plain_source():
    # maker mode: any bot with maker:true wraps its OWN strategy as the source,
    # taking maker knobs from limits (no separate wrapper bot in config).
    cfg = BotConfig(name="cs_mm", strategy="commodity_spot", maker=True,
                    limits={"half_spread": 0.05, "min_half_spread": 0.03,
                            "inventory_cap": 120, "quote_ttl_hours": 12, "max_markets": 7},
                    params={})
    mm = build_maker_strategy(cfg)
    assert isinstance(mm.source, CommoditySpotStrategy)
    assert mm.half_spread() == 0.05 and mm.min_half_spread == 0.03
    assert mm.inventory_cap == 120 and mm.quote_ttl_hours == 12 and mm.max_markets == 7


def test_build_maker_strategy_explicit_wrapper():
    # the explicit Phase 2 wrapper still works: strategy: market_maker + params.
    cfg = BotConfig(name="mm", strategy="market_maker", maker=False,
                    params={"fair_value_source": "commodity_spot", "base_half_spread": 0.03})
    mm = build_maker_strategy(cfg)
    assert isinstance(mm.source, CommoditySpotStrategy) and mm.half_spread() == 0.03


def test_config_market_maker_1_is_maker_mode():
    # the shipped market_maker_1 is maker-mode over diffusion_mc (no wrapper).
    # Re-anchored 2026-06-05 from commodity_spot (retired) to diffusion_mc (the live
    # pricer that superseded it) — same markets/matcher, better fair values.
    from quantbots.config import load_bot
    cfg = load_bot("market_maker_1")
    assert cfg.maker is True and cfg.strategy == "diffusion_mc"
    assert cfg.limits.get("half_spread") and cfg.limits.get("max_markets")


def test_open_position_legs_keeps_both_sides(tmp_path):
    # H3: a market with both a YES and NO fill must surface as two distinct legs
    # (open_positions would collapse them, breaking two-sided resolution).
    store = Store(tmp_path / "t.sqlite")
    bot_id = store.upsert_bot("mm_test", "market_maker")
    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="ENTRY", direction="YES",
                       amount=5, shares=8, platform_bet_id="y1")
    store.record_trade(bot_id=bot_id, market_id="m1", trade_type="ENTRY", direction="NO",
                       amount=4, shares=6, platform_bet_id="n1")
    legs = store.open_position_legs(bot_id)
    assert set(legs) == {("m1", "YES"), ("m1", "NO")}
    assert len(store.open_positions(bot_id)) == 1  # market-keyed view collapses them
