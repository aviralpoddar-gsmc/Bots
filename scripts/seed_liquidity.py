"""Seed CPMM pool liquidity into a bot's target markets so they have proper depth.

The clone's commodity markets are created thin (totalLiquidity ~Ṁ100), so a taker
bot can only deploy a few mana before hitting its price-impact cap. This deepens
each of a bot's prefiltered markets up to a target liquidity, which (a) gives the
markets real depth and (b) lets the bot then take meaningful directional positions.

The subsidy is an LP provision — returned at resolution (refunded on CANCEL), not
a directional bet. Run via doppler so the bot's own key + CF headers are present.

Usage:
    doppler run -- .venv/bin/python scripts/seed_liquidity.py --bot cotton_fundamental_1 \
        --target 2000 --max-add 1500 [--execute]

Without --execute it only reports what it WOULD add (dry preview).
"""
from __future__ import annotations

import argparse

from quantbots.config import load_bot
from quantbots.manifold.client import ManifoldClient
from quantbots.store.db import Store
from quantbots.strategies import get_strategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True)
    ap.add_argument("--target", type=int, default=2000, help="Deepen each market up to this totalLiquidity")
    ap.add_argument("--max-add", type=int, default=1500, help="Cap mana added to any single market")
    ap.add_argument("--execute", action="store_true", help="Actually add liquidity (else preview)")
    args = ap.parse_args()

    cfg = load_bot(args.bot)
    if not cfg.api_key:
        raise SystemExit(f"No key in env var {cfg.account_env!r} (run via doppler).")
    client = ManifoldClient(api_key=cfg.api_key)
    strat = get_strategy(cfg.strategy, **cfg.params)

    with Store() as store:
        strat.bind(store)
        markets = strat.prefilter(store.load_open_markets())

    print(f"{args.bot}: {len(markets)} target markets; target depth Ṁ{args.target}, cap Ṁ{args.max_add}/mkt")
    total_add = 0
    added = 0
    for m in markets:
        mid = m["id"]
        liq = m.get("totalLiquidity") or 0
        need = args.target - liq
        if need < 50:  # already deep enough
            continue
        add = min(int(need), args.max_add)
        total_add += add
        q = m.get("question", "")[:55]
        if args.execute:
            try:
                client.add_liquidity(mid, add)
                added += 1
                print(f"  +Ṁ{add:<5} (liq {liq:.0f}->{liq+add:.0f})  {q}")
            except Exception as exc:
                print(f"  FAIL {mid}: {exc}")
        else:
            print(f"  would add Ṁ{add:<5} (liq {liq:.0f})  {q}")
    verb = "added to" if args.execute else "would add to"
    print(f"\nTotal Ṁ{total_add:,} {verb} {added if args.execute else len(markets)} markets"
          f"{'' if args.execute else ' (preview — re-run with --execute)'}")


if __name__ == "__main__":
    main()
