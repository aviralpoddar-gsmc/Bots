#!/usr/bin/env python3
"""Create the Tier-1 zirconium conditional ("B if A") markets on the clone.

Operator action only — mirrors the zinc price-band triangle flow (no code path in
the bot framework touches this). Each triangle is a NESTED price band on ONE metric
and ONE settlement date: predicate A = lower strike, quantity B = higher strike, so
B implies A and the coherent fair value is exactly P(B)/P(A). We create one
conditional per (A, B) pair and seed it at that coherent b/a — NOT the platform's
default independence seed P(B), which would launch the market underpriced.

Buildable legs (both resolvable, see docs / [[zirconium-conditionals]]):
  - Iluka quarterly zircon price  (BEST: public quarterly $/t in Iluka's review)
  - Zircon sand CIF China         (annual/USGS-aligned settlement)

Resolution path (handled by tal, not here): predicate NO -> CANCEL (refund);
predicate YES -> copy the quantity outcome.

  ── HOW IT WORKS ─────────────────────────────────────────────────────────────
  The tal endpoint POST /markets/conditional takes tal INTEGER market ids. We map
  clone-string-id -> tal-int-id from data/tal/zircon_leg_ids.json (dumped once from
  MARKET.PREDICTION_MARKET) and read LIVE leg probabilities from the local Bots
  market cache. Pairs are formed locally; only the create call hits prod (via the
  canonical `doppler run --config analyst` + gcloud identity-token auth — see
  ~/tal/.claude/skills/prod-api-access.md).

  ── PREREQUISITES ────────────────────────────────────────────────────────────
  1. data/tal/zircon_leg_ids.json present (regenerate from tal if legs change)
  2. fresh-ish market cache (quantbots refresh) for live leg probabilities
  3. for --live only: `gcloud auth login` (interactive) + run from ~/tal

  ── USAGE ────────────────────────────────────────────────────────────────────
  python create_zirconium_conditionals.py          # dry-run: compose + fair values
  python create_zirconium_conditionals.py --live    # actually create on prod
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEG_MAP = REPO / "data" / "tal" / "zircon_leg_ids.json"

# Strike ladders → every nested (lower=predicate, higher=quantity) pair is a triangle.
ILUKA_STRIKES = [1450, 1550, 1700]
SAND_STRIKES = [1300, 1800, 2500]
# Zircon sand: restrict to annual/USGS-aligned settlement dates (the quarterly
# Chinese spot is only published publicly once a year). Iluka is public quarterly,
# so all its quarters qualify.
SAND_DATES_OK = {"December 31, 2026", "June 30, 2027"}

_ILUKA = re.compile(r"Iluka's zircon price for the quarter ending (.+?) exceed (\d+) USD/t", re.I)
_SAND = re.compile(r"zircon sand price CIF China exceed (\d+) USD/t on (.+?)\?", re.I)


def classify(question: str):
    """Return (metric_family, settlement_key, strike) or None for a leg question."""
    m = _ILUKA.search(question)
    if m:
        return ("Iluka", m.group(1).strip(), int(m.group(2)))
    m = _SAND.search(question)
    if m:
        return ("Sand", m.group(2).strip(), int(m.group(1)))
    return None


def _api_post(path: str, body: dict) -> dict:
    payload = json.dumps(body).replace("'", "'\\''")
    inner = (
        f'curl -s -X POST "$VITE_API_URL{path}" '
        f'-H "Authorization: Bearer $(gcloud auth print-identity-token)" '
        f"-H 'Content-Type: application/json' -d '{payload}'"
    )
    cmd = ["doppler", "run", "--config", "analyst", "--", "bash", "-c", inner]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err = res.stderr.strip()
        if "auth login" in err or "Reauthentication" in err:
            raise SystemExit(
                "BLOCKED: gcloud auth expired. Run `gcloud auth login` (interactive), "
                "then re-run from ~/tal."
            )
        raise RuntimeError(f"create failed: {err[:300]}")
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"non-JSON from create: {res.stdout.strip()[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually create (default: dry-run)")
    ap.add_argument("--family", choices=["Iluka", "Sand"], default=None,
                    help="restrict to one metric family (default: both)")
    args = ap.parse_args()

    if not LEG_MAP.exists():
        print(f"missing {LEG_MAP} — regenerate the clone->tal id map from tal first.")
        return 1
    legs = json.loads(LEG_MAP.read_text())
    talid_by_clone = {l["clone_id"]: l["tal_id"] for l in legs}

    # Live leg probabilities from the local cache.
    sys.path.insert(0, str(REPO / "src"))
    from quantbots.store.db import Store

    cache = {m["id"]: m for m in Store().load_open_markets()}

    # Bucket legs by (family, settlement): {strike: {clone_id, tal_id, prob, question}}.
    buckets: dict[tuple[str, str], dict[int, dict]] = {}
    for leg in legs:
        c = classify(leg["question"])
        if not c:
            continue
        family, settle, strike = c
        if family == "Sand" and settle not in SAND_DATES_OK:
            continue
        mkt = cache.get(leg["clone_id"])
        if not mkt:
            continue
        buckets.setdefault((family, settle), {})[strike] = {
            "tal_id": leg["tal_id"], "clone_id": leg["clone_id"],
            "prob": mkt.get("probability"), "question": leg["question"],
        }

    plans: list[dict] = []
    for (family, settle), by_strike in sorted(buckets.items()):
        if args.family and family != args.family:
            continue
        strikes = sorted(by_strike)
        for i, lo in enumerate(strikes):
            for hi in strikes[i + 1:]:
                a = by_strike[lo]["prob"]   # P(>= lower)  = predicate
                b = by_strike[hi]["prob"]   # P(>= higher) = quantity
                label = f"{family} {settle}: >={lo} -> >={hi}"
                if not a or not b or a < 0.05:
                    print(f"  SKIP {label} — degenerate a={a} b={b}")
                    continue
                if b > a:  # survival-inverted ladder; let ladder_arb fix legs first
                    print(f"  SKIP {label} — legs inverted (b={b:.2f} > a={a:.2f})")
                    continue
                seed = round(min(max(b / a, 0.01), 0.99), 3)
                plans.append({
                    "label": label,
                    "predicate_id": by_strike[lo]["tal_id"],
                    "quantity_id": by_strike[hi]["tal_id"],
                    "predicate_q": by_strike[lo]["question"],
                    "quantity_q": by_strike[hi]["question"],
                    "a": a, "b": b, "seed": seed,
                })

    print(f"\n{len(plans)} nested triangles ready:\n")
    for p in plans:
        print(f"  [{p['label']}]")
        print(f"     IF [{p['predicate_q']}]")
        print(f"        = YES: {p['quantity_q']}")
        print(f"     a=P(A)={p['a']:.3f}  b=P(B)={p['b']:.3f}  ->  coherent seed b/a={p['seed']:.3f}")
        print(f"     predicate_id={p['predicate_id']}  quantity_id={p['quantity_id']}\n")

    if not args.live:
        print("DRY-RUN — no markets created. Re-run with --live (from ~/tal, gcloud authed).")
        return 0

    print("Creating LIVE...\n")
    created, skipped, errors = [], [], []
    for p in plans:
        body = {
            "predicate_market_id": int(p["predicate_id"]),
            "quantity_market_id": int(p["quantity_id"]),
            "initial_probability": p["seed"],
            "dry_run": False,
        }
        try:
            res = _api_post("/markets/conditional", body)
        except RuntimeError as e:
            errors.append((p["label"], str(e)))
            print(f"  ERROR [{p['label']}]: {e}")
            continue
        for m in res.get("created", []):
            created.append(m)
            print(f"  CREATED #{m.get('id')} {(m.get('market_question') or '')[:80]}")
        skipped.extend(res.get("skipped_existing", []))
        errors.extend(res.get("errors", []))

    print(f"\nDONE — created {len(created)}, skipped_existing {len(skipped)}, errors {len(errors)}")
    for e in errors:
        print(f"  ! {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
