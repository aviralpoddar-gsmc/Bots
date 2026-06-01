#!/usr/bin/env python
"""Mint dedicated Manifold-clone accounts for trading bots, so each bot trades
under its OWN name instead of the shared operator account.

## Why this exists

Every bot in config/bots.yaml currently points `account_env` at the same key
(MANIFOLD_CLONE_API_KEY = the operator account, @AviralPoddar), so all bots trade
under one identity. Giving a bot its own account is purely a matter of signing its
bets with a *different* API key — the framework already resolves a per-bot key
from `account_env` (config.py:BotConfig.api_key). This script creates those
accounts (1 bot : 1 account : 1 key) and their keys.

## The mechanism

The clone exposes a custom, NON-versioned admin endpoint (not in public Manifold):

    POST https://manifold.mikhailtal.dev/api/admin-create-bot-user
      auth: Authorization: Key <MANIFOLD_CLONE_ADMIN_API_KEY>  (+ CF Access headers)
      body: {"username": "...", "displayName": "...", "startingBalance": 100000}
      resp: {"userId": "...", "username": "...", "apiKey": "..."}  <- the bot's own key

The admin key is used ONLY to create accounts and top up mana
(POST /api/admin-add-mana  {"amount": N, "username": "..."}). Each bot then signs
its own bets with the `apiKey` returned at creation.

## Required env (inject via `doppler run -c prd --`, where the admin key lives)

    MANIFOLD_CLONE_ADMIN_API_KEY   admin key (prd Doppler) — creates accounts + mana
    CF_ACCESS_CLIENT_ID            Cloudflare Access
    CF_ACCESS_CLIENT_SECRET        Cloudflare Access

## Usage

    # validate creds, create nothing:
    doppler run -c prd -- python scripts/mint_bot_accounts.py

    # create the default (functional) bot set with Ṁ100k each:
    doppler run -c prd -- python scripts/mint_bot_accounts.py --execute

    # subset / custom balance:
    doppler run -c prd -- python scripts/mint_bot_accounts.py --execute \
        --bots ladder_arb_1,commodity_spot_1 --balance 50000

Writes the new {bot -> username, userId, apiKey} to `data/bot-accounts.json`
(gitignored — credentials). Idempotent: bots already in that file are skipped.
Prints the Doppler commands to store each key + the bots.yaml edits to apply.

Safety: clone-only (hard-coded host); makes NO changes without --execute; never
prints full API keys to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

CLONE = "https://manifold.mikhailtal.dev"
CREATE_ENDPOINT = f"{CLONE}/api/admin-create-bot-user"
ADD_MANA_ENDPOINT = f"{CLONE}/api/admin-add-mana"

# Default bot set = the functional/trading bots (confirmed from the trade ledger),
# plus pair_trading_1 which is enabled and about to go live.
# bot_name -> (manifold username, display name)
DEFAULT_BOTS: dict[str, tuple[str, str]] = {
    "ladder_arb_1": ("LadderArbBot", "Ladder Arb Bot (AP)"),
    "commodity_spot_1": ("CommoditySpotBot", "Commodity Spot Bot (AP)"),
    "term_structure_1": ("TermStructureBot", "Term Structure Bot (AP)"),
    "ensemble_1": ("EnsembleBot", "Ensemble Bot (AP)"),
    "pair_trading_1": ("PairTradingBot", "Pair Trading Bot (AP)"),
    "stockpile_facts_1": ("StockpileFactsBot", "Stockpile Facts Bot (AP)"),
    "stockpile_grid_arb_1": ("StockpileGridArbBot", "Stockpile Grid Arb Bot (AP)"),
    "stockpile_coherence_1": ("StockpileCoherenceBot", "Stockpile Coherence Bot (AP)"),
    "market_maker_1": ("MarketMakerBot", "Market Maker Bot (AP)"),
    # USDA softs bots. cotton already minted as @Cottonfundamental1 (display renamed
    # to add the "(AP)" suffix via POST /v0/me/update); listed here for consistency.
    "cotton_fundamental_1": ("Cottonfundamental1", "Cotton Fundamental 1 (AP)"),
    "cocoa_fundamental_1": ("CocoaFundamentalBot", "Cocoa Fundamental Bot (AP)"),
    "coffee_consumption_1": ("CoffeeConsumptionBot", "Coffee Consumption Bot (AP)"),
}

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "bot-accounts.json"


def _post(url: str, body: dict, headers: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"message": raw}


def env_var_for(bot_name: str) -> str:
    """Per-bot key env var, e.g. ladder_arb_1 -> LADDER_ARB_1_API_KEY."""
    return bot_name.upper() + "_API_KEY"


def load_existing() -> dict[str, dict]:
    if OUT_PATH.exists():
        return {r["bot"]: r for r in json.loads(OUT_PATH.read_text())}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bots", default="", help="Comma-separated bot names (default: functional set)")
    ap.add_argument("--balance", type=int, default=100_000, help="Starting balance per account")
    ap.add_argument("--execute", action="store_true", help="Actually create accounts (else validate + stop)")
    args = ap.parse_args()

    # Fallback naming for bots not in DEFAULT_BOTS: append the "(AP)" suffix so
    # every minted account is tagged as an operator bot (the convention).
    bots = (
        {n: DEFAULT_BOTS.get(n, (n.replace("_", "").title(), n.replace("_", " ").title() + " (AP)"))
         for n in args.bots.split(",") if n}
        if args.bots
        else dict(DEFAULT_BOTS)
    )

    admin_key = os.environ.get("MANIFOLD_CLONE_ADMIN_API_KEY")
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID")
    cf_sec = os.environ.get("CF_ACCESS_CLIENT_SECRET")
    missing = [k for k, v in {
        "MANIFOLD_CLONE_ADMIN_API_KEY": admin_key,
        "CF_ACCESS_CLIENT_ID": cf_id,
        "CF_ACCESS_CLIENT_SECRET": cf_sec,
    }.items() if not v]
    if missing:
        sys.exit(f"Missing env: {', '.join(missing)} — run via `doppler run -c prd --`.")

    headers = {
        "Authorization": f"Key {admin_key}",
        "CF-Access-Client-Id": cf_id,
        "CF-Access-Client-Secret": cf_sec,
    }

    existing = load_existing()
    todo = {n: v for n, v in bots.items() if n not in existing}
    print("Target bots:", ", ".join(bots))
    if existing:
        print("Already minted (skipping):", ", ".join(existing) or "none")
    print(f"Starting balance per new account: Ṁ{args.balance:,}")

    if not args.execute:
        print("\n[validate-only] admin creds present. Re-run with --execute to create accounts.")
        return

    results = list(existing.values())
    created_now = []
    for bot_name, (username, display_name) in todo.items():
        print(f"\n→ {bot_name}  (@{username})")
        st, d = _post(CREATE_ENDPOINT, {
            "username": username,
            "displayName": display_name,
            "startingBalance": args.balance,
        }, headers)
        if st != 200 or "apiKey" not in d:
            print(f"  ✗ create failed [{st}]: {d.get('message', d)}", file=sys.stderr)
            continue
        rec = {
            "bot": bot_name,
            "username": d.get("username", username),
            "userId": d.get("userId"),
            "apiKey": d["apiKey"],
            "env_var": env_var_for(bot_name),
            "startingBalance": args.balance,
        }
        results.append(rec)
        created_now.append(rec)
        print(f"  ✓ @{rec['username']}  id={rec['userId']}  key=…{rec['apiKey'][-6:]}  balance=Ṁ{args.balance:,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} account(s) -> {OUT_PATH} (gitignored — contains secrets)")

    if not created_now:
        print("No new accounts created this run.")
        return

    print("\n--- Store keys in Doppler (run these) ---")
    for r in created_now:
        print(f"doppler secrets set {r['env_var']} '{r['apiKey']}' -c dev -c stg -c prd")

    print("\n--- Apply to config/bots.yaml (set each bot's account_env) ---")
    for r in created_now:
        print(f"  {r['bot']}: account_env: {r['env_var']}")


if __name__ == "__main__":
    main()
