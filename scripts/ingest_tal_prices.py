#!/usr/bin/env python
"""Ingest fresh metal spot anchors from tal Snowflake into the observations cache.

The keyless stooq quote feed died (404s since ~June 2026), leaving the clone's
spot anchors a month stale. tal's warehouse has fresh ones: SMM spot prices
(SOURCES.SMM_SPOT_DAILY — updated daily, already in USD market units) with the
PCF benchmark series (XAU/XAG/XPT/XPD, USD/oz) as fallback. This feeds the
comment-judge cycle (comments/judge.py) and anything else reading `observations`.

Values are stored in MARKET units under source `tal_smm` / `tal_pcf` — consumers
apply NO feed-unit factor (unlike stooq's cents/oz silver and cents/lb copper).

Usage:
    .venv/bin/python scripts/ingest_tal_prices.py
(NO doppler wrap: the tal reader shells into ~/tal with its own doppler scope;
an outer `doppler run` from this repo's project shadows it and breaks auth.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantbots.equity_options.sources import tal_snowflake as tal  # noqa: E402
from quantbots.store.db import Store  # noqa: E402

# entity -> (SMM category, preferred product names in order). Premia/rod/wire
# series are excluded — we want the benchmark cathode/ingot price. NB: SMM's own
# data spells silver "Sliver Ingot" — that typo is IN THE DATA, keep it.
_SMM_PREFERENCE = {
    "GOLD": ("gold", ["Gold (99.99%)", "Gold (99.95%)"]),
    "SILVER": ("silver", ["SMM 1# Sliver Ingot", "SMM 2# Sliver Ingot",
                          "Sliver Ingot (98-99.89%)"]),
    "COPPER": ("copper", ["Changjiang 1# Copper Cathode", "SMM Shandong 1# Copper Cathode",
                          "SMM Guangdong Standard-Grade Copper Cathode"]),
    "PLATINUM": ("platinum", ["Platinum (99.95%)"]),
    "PALLADIUM": ("palladium", ["Palladium (99.95%)"]),
}
_PCF_FALLBACK = {"GOLD": "XAU", "SILVER": "XAG", "PLATINUM": "XPT", "PALLADIUM": "XPD"}
_EXPECTED_UNIT = {"GOLD": "USD/oz", "SILVER": "USD/oz", "PLATINUM": "USD/oz",
                  "PALLADIUM": "USD/oz", "COPPER": "USD/tonne"}
_TROY_OZ_PER_KG = 32.1507
# Deterministic feed-unit -> market-unit conversions (code, never the LLM).
_UNIT_CONVERT = {("USD/kg", "USD/oz"): 1.0 / _TROY_OZ_PER_KG}


def _smm_latest(entity: str) -> dict | None:
    category, preferred = _SMM_PREFERENCE[entity]
    rows = tal.query(f"""
        SELECT PRODUCT_NAME, OBSERVATION_DATE, PRICE_AVERAGE, UNIT
        FROM SOURCES.SMM_SPOT_DAILY
        WHERE CATEGORY = '{category}' AND PRICE_AVERAGE IS NOT NULL
          AND PRODUCT_NAME NOT ILIKE '%premium%' AND PRODUCT_NAME NOT ILIKE '%rod%'
          AND PRODUCT_NAME NOT ILIKE '%wire%'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY PRODUCT_NAME ORDER BY OBSERVATION_DATE DESC) = 1
        ORDER BY OBSERVATION_DATE DESC""")
    market_unit = _EXPECTED_UNIT[entity]
    usable = []
    for r in rows:
        if r.get("UNIT") == market_unit:
            usable.append(r)
        elif (r.get("UNIT"), market_unit) in _UNIT_CONVERT:  # e.g. USD/kg silver -> USD/oz
            factor = _UNIT_CONVERT[(r["UNIT"], market_unit)]
            usable.append({**r, "PRICE_AVERAGE": round(float(r["PRICE_AVERAGE"]) * factor, 4),
                           "UNIT": market_unit, "_converted_from": r["UNIT"]})
    if not usable:
        return None
    for name in preferred:  # deterministic benchmark choice
        for r in usable:
            if r["PRODUCT_NAME"] == name:
                return r
    return usable[0]  # newest of whatever benchmark-ish series exists


def _pcf_latest(entity: str) -> dict | None:
    ticker = _PCF_FALLBACK.get(entity)
    if not ticker:
        return None
    rows = tal.query(f"SELECT DATE, PRICE FROM PCF.PRICES_DAILY WHERE TICKER = '{ticker}' "
                     f"ORDER BY DATE DESC LIMIT 1")
    return rows[0] if rows else None


def main() -> int:
    obs = []
    for entity in _SMM_PREFERENCE:
        row = _smm_latest(entity)
        if row:
            obs.append({"source": "tal_smm", "entity": entity,
                        "ts": f"{row['OBSERVATION_DATE']}T00:00:00",
                        "value": float(row["PRICE_AVERAGE"]),
                        "payload": {"product": row["PRODUCT_NAME"], "unit": row["UNIT"]}})
            print(f"{entity:10} tal_smm  {row['PRICE_AVERAGE']:>12,.2f} {row['UNIT']:9} "
                  f"{row['OBSERVATION_DATE']}  ({row['PRODUCT_NAME']})")
            continue
        row = _pcf_latest(entity)
        if row:
            obs.append({"source": "tal_pcf", "entity": entity,
                        "ts": f"{row['DATE']}T00:00:00", "value": float(row["PRICE"]),
                        "payload": {"ticker": _PCF_FALLBACK[entity], "unit": "USD/oz"}})
            print(f"{entity:10} tal_pcf  {row['PRICE']:>12,.2f} USD/oz    {row['DATE']}")
        else:
            print(f"{entity:10} NO SOURCE — skipped")
    with Store() as store:
        n = store.upsert_observations(obs)
    print(f"wrote {n} observations")
    return 0 if obs else 1


if __name__ == "__main__":
    raise SystemExit(main())
