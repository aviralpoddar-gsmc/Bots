#!/usr/bin/env python
"""Lead/lag validation for the 007 news bot — the go/no-go gate before live capital.

THE QUESTION: does SIG_<COM>_NEWS (today's digested-news consensus) LEAD the commodity's
price, or just lag/echo a move the market already made? News sentiment is low-Sharpe and
the live price absorbs public headlines within minutes, so 007 only earns its keep if the
signal has forward predictive content. This measures exactly that.

METHOD (honest, simple, no look-ahead):
- Pull the accumulated SIG_<COM>_NEWS daily series from the store (built forward by the
  daily `process` step — so this is only meaningful after ~2+ weeks of accumulation).
- Pull the commodity's daily price history (yfinance via research.data_fetch).
- For each signal observation at day t with value s_t, compute the FORWARD log return of
  the price over the next h trading days (t -> t+h), for h in {1,3,5,10}.
- Report, per commodity and pooled:
    * sign hit-rate: P(sign(forward_return) == sign(s_t))  vs the 50% null,
    * Spearman corr(s_t, forward_return),
    * the same for the CONTEMPORANEOUS/PAST return (t-h -> t) — if the signal correlates
      with PAST returns but not FUTURE ones, it LAGS (echoes), which is the kill signal.
- VERDICT: leads (forward hit-rate > ~55% AND > past hit-rate), lags, or inconclusive
  (too little data / no edge). This is descriptive evidence for a human go/no-go, not a
  backtest of PnL.

USAGE (after data has accumulated):
    doppler run -- .venv/bin/python scripts/validate_news_leadlag.py
    doppler run -- .venv/bin/python scripts/validate_news_leadlag.py --min-obs 10 --horizons 1,3,5,10
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from quantbots.llm.news_extractor import COMMODITY_TO_ENTITY
from quantbots.research.data_fetch import DEFAULT_UNIVERSE, fetch_yf_history
from quantbots.store.db import Store

# SIG_<COM>_NEWS short keys -> the research yfinance universe key (price history).
# COM short is the price_entity with CME_/_OIL stripped (matches the strategy + signal).
COM_TO_YF = {
    "GOLD": "GOLD", "SILVER": "SILVER", "PLATINUM": "PLATINUM", "PALLADIUM": "PALLADIUM",
    "COPPER": "COPPER", "WTI": "WTI_OIL", "BRENT": "BRENT_OIL", "GASOLINE": "GASOLINE",
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _price_on_or_before(dates: np.ndarray, closes: np.ndarray, day: datetime):
    """Index of the last price row on/before `day`, or None."""
    import numpy as _np
    target = _np.datetime64(day.date())
    idx = _np.searchsorted(dates, target, side="right") - 1
    return int(idx) if idx >= 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,3,5,10", help="forward trading-day horizons")
    ap.add_argument("--min-obs", type=int, default=8, help="min signal obs/commodity to judge")
    ap.add_argument("--period", default="1y")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    with Store() as st:
        pooled = {h: {"fwd": [], "sig": [], "past": []} for h in horizons}
        any_com = False
        for com, yf_key in COM_TO_YF.items():
            sigs = st.load_observations(entity=f"SIG_{com}_NEWS", source="signal", limit=500)
            sigs = [s for s in sigs if s.get("value") is not None]
            if len(sigs) < args.min_obs:
                continue
            any_com = True
            tkr = DEFAULT_UNIVERSE.get(yf_key)
            try:
                df = fetch_yf_history(tkr, period=args.period)
                dates = df.index.to_numpy().astype("datetime64[D]")
                closes = df["Close"].astype(float).to_numpy()
            except Exception as e:  # noqa: BLE001
                print(f"{com}: price history failed ({e})")
                continue
            print(f"\n=== {com}  ({len(sigs)} signal obs) ===")
            for h in horizons:
                rows = []  # (s_t, fwd_ret, past_ret)
                for s in sigs:
                    try:
                        day = datetime.fromisoformat(s["ts"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                    i = _price_on_or_before(dates, closes, day)
                    if i is None or i + h >= len(closes) or i - h < 0:
                        continue
                    fwd = float(np.log(closes[i + h] / closes[i]))
                    past = float(np.log(closes[i] / closes[i - h]))
                    rows.append((float(s["value"]), fwd, past))
                if len(rows) < args.min_obs:
                    print(f"  h={h:>2}: only {len(rows)} paired obs — skip")
                    continue
                sig = np.array([r[0] for r in rows]); fwd = np.array([r[1] for r in rows])
                past = np.array([r[2] for r in rows])
                hit = float(np.mean(np.sign(sig) == np.sign(fwd)))
                hit_past = float(np.mean(np.sign(sig) == np.sign(past)))
                print(f"  h={h:>2}: n={len(rows)}  fwd hit={hit:.0%}  past hit={hit_past:.0%}  "
                      f"spearman(fwd)={_spearman(sig, fwd):+.2f}  spearman(past)={_spearman(sig, past):+.2f}")
                pooled[h]["fwd"] += list(fwd); pooled[h]["sig"] += list(sig); pooled[h]["past"] += list(past)

        if not any_com:
            print("No commodity has >= %d SIG_<COM>_NEWS observations yet — let it accumulate "
                  "(the daily `process` step builds the series). Re-run in ~2 weeks." % args.min_obs)
            return

        print("\n" + "=" * 60 + "\nPOOLED (all commodities)")
        verdict_leads = []
        for h in horizons:
            sig = np.array(pooled[h]["sig"]); fwd = np.array(pooled[h]["fwd"]); past = np.array(pooled[h]["past"])
            if len(sig) < args.min_obs:
                print(f"  h={h}: n={len(sig)} — too few")
                continue
            hit = float(np.mean(np.sign(sig) == np.sign(fwd)))
            hit_past = float(np.mean(np.sign(sig) == np.sign(past)))
            leads = hit > 0.55 and hit > hit_past
            verdict_leads.append(leads)
            print(f"  h={h:>2}: n={len(sig)}  FWD hit={hit:.0%}  past hit={hit_past:.0%}  "
                  f"spearman(fwd)={_spearman(sig, fwd):+.2f}  -> {'LEADS' if leads else 'no lead'}")
        print("\nGO/NO-GO: " + (
            "GO — signal shows forward lead at >=1 horizon. Consider a small live canary."
            if any(verdict_leads) else
            "NO-GO (yet) — no clear forward lead; signal likely lags/echoes price. Keep paper."))


if __name__ == "__main__":
    main()
