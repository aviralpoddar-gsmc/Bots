#!/usr/bin/env python
"""Phase 1 connection smoke test — prove auth works and you can move a market.

    python scripts/manual_trade.py --slug <open-market-slug>            # dry-run only
    python scripts/manual_trade.py --slug <open-market-slug> --execute  # real 10-mana bet

Requires MANIFOLD_CLONE_API_KEY, CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET in env.
"""

from __future__ import annotations

import argparse

from quantbots.manifold import ManifoldClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True, help="Slug of an open market to trade")
    ap.add_argument("--amount", type=int, default=10, help="Mana to bet")
    ap.add_argument("--outcome", default="YES", choices=["YES", "NO"])
    ap.add_argument("--execute", action="store_true", help="Place a REAL bet (else dry-run)")
    args = ap.parse_args()

    client = ManifoldClient()

    me = client.get_me()  # proves API key + Cloudflare Access both work
    print(f"Authenticated as @{me['username']} (balance Ṁ{me['balance']})")

    m = client.get_market_by_slug(args.slug)
    print(f"Market: {m['question']!r}  prob={m['probability']:.3f}")

    # Conservative manual sizing: push 1/4 toward a small move past the current price.
    gap = 0.02 if args.outcome == "YES" else -0.02
    limit_prob = round(min(max(m["probability"] + gap, 0.01), 0.99), 2)

    dry = client.place_bet(m["id"], args.outcome, args.amount, limit_prob=limit_prob, dry_run=True)
    print(f"Dry-run OK: would fill ~{dry.get('shares')} shares")

    if not args.execute:
        print("(dry-run only; pass --execute to place a real bet)")
        return

    res = client.place_bet(m["id"], args.outcome, args.amount, limit_prob=limit_prob)
    print(f"Placed bet {res['betId']}: {res['shares']} shares, prob {res['probBefore']:.3f} -> {res['probAfter']:.3f}")
    print(f"New balance: Ṁ{client.get_me()['balance']}")


if __name__ == "__main__":
    main()
