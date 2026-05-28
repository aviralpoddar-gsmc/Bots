#!/usr/bin/env python
"""Generate the pair-trading research report.

Usage:
    python scripts/research_pairs.py [--period 3y] [--lookback 750] [--out-dir DIR]

Writes:
    data/research/pairs_YYYYMMDD/report.md
    data/research/pairs_YYYYMMDD/charts/*.png
"""

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

from quantbots.research.report import generate_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="3y", help="yfinance lookback period (e.g. '1y', '3y', '5y')")
    p.add_argument("--lookback", type=int, default=750, help="business-day window after alignment")
    p.add_argument("--out-dir", default="", help="override the default data/research/pairs_<date>/")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    # yfinance produces a flood of pandas warnings on edge cases; mute them.
    warnings.filterwarnings("ignore")
    out = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "data" / "research"
        / f"pairs_{datetime.now():%Y%m%d}"
    )
    report = generate_report(out, period=args.period, lookback_days=args.lookback)
    print(f"\nreport: {report}")
    print(f"charts: {report.parent / 'charts'}")


if __name__ == "__main__":
    main()
