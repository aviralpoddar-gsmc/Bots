#!/usr/bin/env python
"""Provide REAL AMM liquidity (pool depth) to the thin ag markets — the proper
"liquidity provision" the cosmetic taker coverage bot can't do.

WHY
---
`llm_ag_coverage` is a TAKER: it moves PRICE off 0.50 and builds coherent ladders,
but it does NOT deepen the pool. These ag markets are thin (totalLiquidity ~100), so
any real bet moves price a lot and the fleet's own taker bots choke after ~Ṁ2k.
`POST /market/:id/add-liquidity` subsidizes the CPMM pool directly: +Ṁ50 takes a
market from totalLiquidity 100 → 150 (verified). The subsidy is returned to the
provider at resolution and REFUNDED on CANCEL — and ~93% of these markets cancel —
so the capital is recoverable; the cost is mostly opportunity cost (tied-up mana).

WHAT IT DOES
------------
Tops each in-scope ag market UP TO a target depth, adding at most a per-market cap,
skipping markets already at/above target, until a total budget is exhausted
(neediest-first). Scope = the SAME universe as the llm_ag_coverage bot (its
include/exclude terms, loaded from config/bots.yaml) so price markets other bots own
are excluded.

SAFETY
------
- Clone-only (the client hard-codes the clone host).
- PREVIEW by default — prints the full plan + total cost and writes NOTHING.
  Re-run with --execute to actually add liquidity. (The clone has no server-side
  dry-run for add-liquidity, so preview is computed locally.)
- --budget caps total mana; --per-market caps per-market; never exceeds account balance.
- Appends every provision to data/ag_liquidity_log.json for auditing.

USAGE
    # preview the default plan (no writes):
    doppler run -- .venv/bin/python scripts/provide_ag_liquidity.py
    # deeper target / bigger budget preview:
    doppler run -- .venv/bin/python scripts/provide_ag_liquidity.py --target 400 --budget 40000
    # actually provide the liquidity:
    doppler run -- .venv/bin/python scripts/provide_ag_liquidity.py --target 400 --budget 40000 --execute
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from quantbots.config import load_bot
from quantbots.manifold.client import ManifoldClient
from quantbots.store.db import Store

LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "ag_liquidity_log.json"


def _liq(m: dict) -> float:
    return float(m.get("totalLiquidity") or m.get("total_liquidity") or 0.0)


def in_scope(question: str, include_re: re.Pattern | None, exclude_re: re.Pattern | None) -> bool:
    if not question:
        return False
    if include_re and not include_re.search(question):
        return False
    if exclude_re and exclude_re.search(question):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="llm_ag_coverage", help="bot whose scope (include/exclude) + account to use")
    ap.add_argument("--target", type=float, default=400.0, help="top each market UP TO this totalLiquidity")
    ap.add_argument("--per-market", type=float, default=300.0, help="max mana to add to any one market")
    ap.add_argument("--min-add", type=float, default=25.0, help="skip markets needing less than this")
    ap.add_argument("--budget", type=float, default=30000.0, help="total mana cap across all markets")
    ap.add_argument("--reserve", type=float, default=2000.0, help="keep at least this much account balance")
    ap.add_argument("--max-markets", type=int, default=100000, help="cap number of markets (default: all)")
    ap.add_argument("--no-skip-logged", action="store_true",
                    help="re-fund markets already in the liquidity log (default: skip them — "
                         "the local cache doesn't reflect prior adds, so skipping prevents double-funding)")
    ap.add_argument("--execute", action="store_true", help="actually add liquidity (else preview only)")
    args = ap.parse_args()

    # Markets already funded in a prior run. add-liquidity does NOT update the local
    # market_cache, so totalLiquidity here is the PRE-funding value — without this skip
    # a re-run would top the already-deep markets again. Idempotent re-runs by default.
    already: set[str] = set()
    if not args.no_skip_logged and LOG_PATH.exists():
        already = {e["market_id"] for e in json.loads(LOG_PATH.read_text())}

    cfg = load_bot(args.bot)
    inc = cfg.params.get("include_terms")
    exc = cfg.params.get("exclude_terms")
    include_re = re.compile("|".join(inc), re.I) if inc else None
    exclude_re = re.compile("|".join(exc), re.I) if exc else None

    client = ManifoldClient(cfg.api_key)
    balance = float(client.get_me().get("balance", 0.0))

    with Store() as st:
        markets = st.load_open_markets()

    # in-scope, neediest (thinnest) first, excluding already-funded markets
    scope = [m for m in markets
             if in_scope(m.get("question", "") or "", include_re, exclude_re)
             and m["id"] not in already]
    scope.sort(key=_liq)

    spend_cap = min(args.budget, max(balance - args.reserve, 0.0))
    plan: list[tuple[str, str, float, float, float]] = []  # (id, question, current, add, new)
    total = 0.0
    for m in scope:
        if len(plan) >= args.max_markets or total >= spend_cap:
            break
        cur = _liq(m)
        add = min(args.per_market, max(args.target - cur, 0.0))
        if add < args.min_add:
            continue
        add = float(int(add))  # integer mana
        if total + add > spend_cap:
            add = float(int(spend_cap - total))
            if add < args.min_add:
                continue
        plan.append((m["id"], (m.get("question") or "")[:60], cur, add, cur + add))
        total += add

    print(f"bot={args.bot}  account balance Ṁ{balance:,.0f}  (reserve Ṁ{args.reserve:,.0f})")
    if already:
        print(f"skipping {len(already)} already-funded markets (from {LOG_PATH.name})")
    print(f"scope: {len(scope)} in-scope ag markets | target depth {args.target:g}, "
          f"per-market ≤{args.per_market:g}, budget Ṁ{args.budget:,.0f} → spend cap Ṁ{spend_cap:,.0f}")
    print(f"PLAN: provide liquidity to {len(plan)} markets, total Ṁ{total:,.0f}\n")
    # show the thinnest 12 and a tail sample
    for mid, q, cur, add, new in plan[:12]:
        print(f"  {q:62}  {cur:6.0f} +{add:5.0f} → {new:6.0f}")
    if len(plan) > 12:
        print(f"  … and {len(plan) - 12} more")

    if not args.execute:
        print("\n[PREVIEW] nothing written. Re-run with --execute to provide this liquidity.")
        return

    if not plan:
        print("\nNothing to do.")
        return

    print(f"\n[EXECUTE] providing Ṁ{total:,.0f} of liquidity across {len(plan)} markets…")
    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    ok = 0
    added = 0.0
    errors = 0
    for i, (mid, q, cur, add, new) in enumerate(plan):
        try:
            r = client.add_liquidity(mid, add)
            ok += 1
            added += add
            log.append({"ts": int(time.time()), "bot": args.bot, "market_id": mid,
                        "question": q, "amount": add, "lp_id": r.get("id") if isinstance(r, dict) else None})
        except Exception as e:  # noqa: BLE001 - one bad market must not abort the run
            errors += 1
            print(f"  ✗ {mid} {q[:40]}: {type(e).__name__} {str(e)[:80]}", file=sys.stderr)
        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{len(plan)} done (Ṁ{added:,.0f} added, {errors} errors)")
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\nDONE: provided Ṁ{added:,.0f} liquidity to {ok} markets, {errors} errors. "
          f"Logged → {LOG_PATH.name}. Balance now Ṁ{client.get_me().get('balance', 0):,.0f}")


if __name__ == "__main__":
    main()
