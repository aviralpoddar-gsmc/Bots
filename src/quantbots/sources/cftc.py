"""CFTC Commitments of Traders — managed-money positioning (keyless Socrata API).

The CFTC publishes weekly COT data with no API key via Socrata. We read the
disaggregated futures-only report (dataset 72hh-3qpy) and track **managed-money
net positioning** per commodity — the speculative crowd. Extreme net length/short
(vs its own history) is a well-known mean-reversion signal: when specs are
max-long, the marginal buyer is exhausted and price tends to revert.

Emits, per commodity, the latest weekly values:
- ``CFTC_<COM>_MM_NET``     managed-money (long - short) contracts
- ``CFTC_<COM>_MM_NETPCT``  that net as a fraction of open interest
- ``CFTC_<COM>_OI``         total open interest

`fetch_history(commodity)` returns the full (date, net_pct) series so the
processing layer can z-score the latest reading.

Configure in config/sources.yaml:

    - name: cftc
      params:
        commodities: [cotton, cocoa, coffee]
"""

from __future__ import annotations

import logging

import requests

from .base import Observation, Source

logger = logging.getLogger(__name__)

_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
# commodity -> CFTC contract_market_name (disaggregated report)
_CONTRACT = {
    "cotton": "COTTON NO. 2",
    "cocoa": "COCOA",
    "coffee": "COFFEE C",
}


def _rows(contract: str, limit: int = 400) -> list[dict]:
    params = {
        "$where": f"contract_market_name='{contract}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(limit),
    }
    resp = requests.get(_URL, params=params, timeout=45)
    resp.raise_for_status()
    return resp.json()


def _net_pct(r: dict) -> tuple[str, float, float, float] | None:
    """(date, net_contracts, net_pct_of_OI, open_interest) from one COT row."""
    try:
        oi = float(r["open_interest_all"])
        net = float(r["m_money_positions_long_all"]) - float(r["m_money_positions_short_all"])
        date = r["report_date_as_yyyy_mm_dd"][:10]
    except (KeyError, ValueError, TypeError):
        return None
    if oi <= 0:
        return None
    return date, net, net / oi, oi


def fetch_history(commodity: str) -> list[tuple[str, float]]:
    """Full (date, managed-money net % of OI) history, oldest first."""
    contract = _CONTRACT.get(commodity)
    if not contract:
        return []
    out = []
    for r in _rows(contract):
        parsed = _net_pct(r)
        if parsed:
            out.append((parsed[0], parsed[2]))
    return sorted(out)


class CftcSource(Source):
    name = "cftc"

    def fetch(self) -> list[Observation]:
        commodities = self.params.get("commodities") or list(_CONTRACT)
        out: list[Observation] = []
        for c in commodities:
            contract = _CONTRACT.get(c)
            if not contract:
                continue
            try:
                rows = _rows(contract, limit=1)
            except Exception as exc:
                logger.warning("cftc %s: %s", c, exc)
                continue
            if not rows:
                continue
            parsed = _net_pct(rows[0])
            if not parsed:
                continue
            date, net, netpct, oi = parsed
            U = c.upper()
            ts = f"{date}T00:00:00"
            out.append(Observation(self.name, f"CFTC_{U}_MM_NET", ts, value=net,
                                   payload={"contract": contract}))
            out.append(Observation(self.name, f"CFTC_{U}_MM_NETPCT", ts, value=netpct,
                                   payload={"contract": contract}))
            out.append(Observation(self.name, f"CFTC_{U}_OI", ts, value=oi,
                                   payload={"contract": contract}))
        return out
