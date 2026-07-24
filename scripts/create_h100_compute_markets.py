#!/usr/bin/env python3
"""Mint a threshold-ladder of H100 SXM GPU compute-price markets on the clone.

OPERATOR ACTION — not part of the bot framework (runner/sizing never create
markets). Creates a ladder of BINARY "Will the Ornn H100 SXM compute price index
exceed $T per GPU-hour by <date>?" markets, anchored to the live Ornn index.

WHY THESE RESOLVE (the whole point): a GPU rental price is a genuine,
externally-published price with a NAMED source (Ornn OCPI, on the Bloomberg
Terminal), so it settles YES/NO rather than joining the ~93% that CANCEL. The
description names the exact source + rule so resolution is unambiguous. See
[[ornn-compute-index]] and [[resolvability-filter]].

SEEDING: each strike launches at the model's lognormal fair value (vol from the
SKU's own daily returns), NOT a flat 0.50 — so the ladder is internally coherent
at launch instead of dumping every strike at an independence prior. Use
`--seed-prob 0.5` to instead reproduce the untraded-0.50 condition the backtest
edge assumed (for a bot canary).

RESOLUTION PATH: the clone has no oracle — settlement is an operator/creator
action (or a tal measurable wired to the Ornn feed). This script only MINTS; it
does not resolve. The description carries the machine-checkable rule so either a
human or a future tal measurable can settle it against api.ornnai.com.

SAFETY: dry-run by default (prints the plan, no network writes). `--live` creates
for real and additionally requires `--yes`. Creating costs the bot account mana
(one liquidity-tier per market). Run live under Doppler so secrets are present:

    uv run python scripts/create_h100_compute_markets.py                 # dry-run
    doppler run -- uv run python scripts/create_h100_compute_markets.py --live --yes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone

from quantbots.manifold.client import ManifoldClient
from quantbots.sources.ornn import OrnnComputeSource
from quantbots.strategies._model import norm_cdf

ENTITY = "GPU_H100_SXM"
GPU = "H100 SXM"
GPU_URL_NAME = "H100%20SXM"  # api path-encoded
_TRADING_DAYS = 252.0
_YEAR_SECONDS = 365.25 * 24 * 3600
# Strike ladder as fractions of the live spot. Tails included on purpose — those
# are the far-OTM strikes the realized edge lives in (see backtest).
_DEFAULT_FRACS = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)


def _annual_vol(series: list[tuple[str, float]]) -> float:
    rets = [
        math.log(series[i][1] / series[i - 1][1])
        for i in range(1, len(series))
        if series[i - 1][1] > 0 and series[i][1] > 0
    ]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(_TRADING_DAYS)


def _fair_prob(spot: float, threshold: float, annual_vol: float, years: float,
               min_vol: float = 0.05) -> float:
    """Zero-drift lognormal P(index_at_close > threshold)."""
    sigma = max(annual_vol * math.sqrt(max(years, 0.0)), min_vol)
    surv = 1.0 - norm_cdf(math.log(threshold / spot) / sigma)
    return min(max(surv, 0.01), 0.99)


def _description(threshold: float, close_date: str, spot: float, spot_ts: str) -> str:
    return (
        f"**Resolves YES** if the Ornn Compute Price Index (OCPI) for **{GPU}** — "
        f"the GPU rental price in USD per GPU-hour — has an index value **strictly "
        f"greater than ${threshold:.2f}/GPU-hour** on the settlement date "
        f"**{close_date}** (the last value published on or before that date). "
        f"Otherwise resolves **NO**.\n\n"
        f"**Source of record:** Ornn OCPI, {GPU} `index_value`.\n"
        f"- Index page: https://index.ornn.com\n"
        f"- Machine-readable: "
        f"https://api.ornnai.com/api/gpu/{GPU_URL_NAME}/index-history\n\n"
        f"Spot at creation: **${spot:.2f}/GPU-hour** (feed @ {spot_ts}).\n\n"
        f"_Settles against an externally-published price; not affiliated with Ornn._"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--close-date", default="2026-12-31", help="settlement date YYYY-MM-DD")
    ap.add_argument("--fracs", default="", help="comma-sep strike fractions of spot")
    ap.add_argument("--seed-prob", default="fair",
                    help="'fair' (lognormal, default) or a flat probability e.g. 0.5")
    ap.add_argument("--liquidity-tier", type=int, default=100,
                    choices=ManifoldClient.LIQUIDITY_TIERS)
    ap.add_argument("--live", action="store_true", help="actually create (default: dry-run)")
    ap.add_argument("--yes", action="store_true", help="skip the confirm prompt (with --live)")
    args = ap.parse_args()

    fracs = (
        tuple(float(x) for x in args.fracs.split(",") if x.strip())
        if args.fracs else _DEFAULT_FRACS
    )

    # Live spot + vol from the Ornn index.
    obs = OrnnComputeSource(gpus={ENTITY: GPU}).fetch()
    series = sorted(((o.ts, o.value) for o in obs if o.value is not None), key=lambda x: x[0])
    if len(series) < 5:
        print(f"insufficient Ornn history for {GPU} ({len(series)} pts)")
        return 1
    spot_ts, spot = series[-1]
    vol = _annual_vol(series)

    dt = datetime.strptime(args.close_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=0, tzinfo=timezone.utc)
    close_ms = int(dt.timestamp() * 1000)
    years = (close_ms / 1000 - time.time()) / _YEAR_SECONDS
    if years <= 0:
        print(f"close date {args.close_date} is not in the future")
        return 1

    plans = []
    for frac in fracs:
        threshold = round(spot * frac, 2)
        if threshold <= 0:
            continue
        if args.seed_prob == "fair":
            seed = _fair_prob(spot, threshold, vol, years)
        else:
            seed = min(max(float(args.seed_prob), 0.01), 0.99)
        question = (
            f"Will the {GPU} GPU compute price index (Ornn OCPI) exceed "
            f"${threshold:.2f}/GPU-hour on {args.close_date}?"
        )
        plans.append({"frac": frac, "threshold": threshold, "seed": seed, "question": question})

    print(f"H100 SXM compute-price ladder — spot ${spot:.2f}/GPU-hr (@ {spot_ts}), "
          f"vol {vol:.1%}/yr, close {args.close_date} (T={years:.2f}y), "
          f"seed={args.seed_prob}, tier {args.liquidity_tier}\n")
    print(f"{'frac':>5} {'strike $/hr':>11} {'seed':>6}  question")
    for p in plans:
        print(f"{p['frac']:>5.2f} {p['threshold']:>11.2f} {p['seed']:>6.2f}  {p['question']}")
    print(f"\n{len(plans)} markets. Est. cost ≈ {len(plans) * args.liquidity_tier} mana "
          f"(tier {args.liquidity_tier} each).")

    if not args.live:
        print("\nDRY-RUN — nothing created. Re-run with `--live --yes` (under Doppler) to mint.")
        return 0

    if not args.yes:
        if input("\nType CREATE to mint these on the clone: ").strip() != "CREATE":
            print("aborted.")
            return 1

    client = ManifoldClient()  # clone-only; reads MANIFOLD_CLONE_API_KEY + CF headers
    me = client.get_me()
    print(f"\nCreating as @{me.get('username')} (balance {me.get('balance')})...\n")

    # Idempotency: skip questions that already exist verbatim.
    try:
        existing = {m.get("question") for m in client.search_markets(f"{GPU} compute price index", 200)}
    except Exception:  # noqa: BLE001
        existing = set()

    created, skipped, errors = [], [], []
    for p in plans:
        if p["question"] in existing:
            skipped.append(p["question"])
            print(f"  SKIP exists: {p['question'][:70]}")
            continue
        try:
            m = client.create_binary_market(
                p["question"],
                close_time_ms=close_ms,
                initial_prob=p["seed"],
                description=_description(p["threshold"], args.close_date, spot, spot_ts),
                liquidity_tier=args.liquidity_tier,
            )
            created.append(m)
            print(f"  CREATED {m.get('url') or m.get('id')}  (seed {p['seed']:.2f})")
        except Exception as e:  # noqa: BLE001
            errors.append((p["question"], str(e)))
            print(f"  ERROR {p['question'][:60]}: {e}")

    print(f"\nDONE — created {len(created)}, skipped {len(skipped)}, errors {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
