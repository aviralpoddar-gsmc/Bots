"""Alpaca options market data: the live chain (quotes + greeks + IV + OI) and
historical bars for the backtest.

Two Alpaca hosts are involved, both read-only and paper-safe:
  - DATA host  /v1beta1/options/snapshots/{underlying}  -> latest quote + greeks + IV
  - DATA host  /v1beta1/options/bars                     -> historical OHLC bars
  - TRADING host /v2/options/contracts                   -> strike/expiry/type + open_interest

`get_chain` merges the snapshot (quote/greeks) with the contracts metadata (OI) by
OCC symbol and returns a list of plain dicts the rest of the pipeline consumes. Each
row: {symbol, underlying, strike, expiry(date), kind, bid, ask, mid, iv, delta,
gamma, vega, theta, rho, open_interest, dte}.

NOTE: Alpaca options history begins ~Feb 2024, so any backtest window is bounded
there; the backtest logs this so coverage is never overstated.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

from .._alpaca_http import DATA_URL, PAPER_TRADING_URL, AlpacaHTTP
from ..occ import parse_occ

logger = logging.getLogger(__name__)


class ChainClient:
    def __init__(self, *, key: str | None = None, secret: str | None = None):
        self._data = AlpacaHTTP(DATA_URL, key=key, secret=secret)
        self._trading = AlpacaHTTP(PAPER_TRADING_URL, key=key, secret=secret)

    # --- contracts metadata (strike/expiry/type + open interest) -------------

    def _contracts(self, underlying: str, *, expiration_lte: str | None = None,
                   expiration_gte: str | None = None, limit: int = 10000) -> dict[str, dict]:
        """{occ_symbol: {open_interest, close_price}} from the trading host."""
        out: dict[str, dict] = {}
        params: dict = {"underlying_symbols": underlying.upper(), "limit": 1000,
                        "status": "active"}
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        page_token = None
        while len(out) < limit:
            if page_token:
                params["page_token"] = page_token
            resp = self._trading.get("/v2/options/contracts", params) or {}
            for c in resp.get("option_contracts", []):
                oi = c.get("open_interest")
                out[c["symbol"]] = {
                    "open_interest": int(oi) if oi is not None else None,
                    "close_price": float(c["close_price"]) if c.get("close_price") else None,
                }
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return out

    # --- snapshots (quotes + greeks + IV) ------------------------------------

    def _snapshots(self, underlying: str, *, feed: str = "indicative") -> dict[str, dict]:
        """{occ_symbol: snapshot} from the data host (latest quote + greeks + IV)."""
        out: dict[str, dict] = {}
        params: dict = {"feed": feed, "limit": 1000}
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._data.get(f"/v1beta1/options/snapshots/{underlying.upper()}", params) or {}
            out.update(resp.get("snapshots", {}) or {})
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return out

    # --- the merged chain ----------------------------------------------------

    def get_chain(self, underlying: str, *, min_dte: int = 0, max_dte: int = 400) -> list[dict]:
        """Merged option chain for an underlying, filtered to a DTE window."""
        today = datetime.now(UTC).date()
        exp_lte = (date.fromordinal(today.toordinal() + max_dte)).isoformat()
        exp_gte = (date.fromordinal(today.toordinal() + min_dte)).isoformat()
        contracts = self._contracts(underlying, expiration_lte=exp_lte, expiration_gte=exp_gte)
        snaps = self._snapshots(underlying)
        rows: list[dict] = []
        for sym, snap in snaps.items():
            try:
                occ = parse_occ(sym)
            except ValueError:
                continue
            dte = (occ.expiry - today).days
            if dte < min_dte or dte > max_dte:
                continue
            quote = snap.get("latestQuote") or {}
            bid, ask = quote.get("bp"), quote.get("ap")
            mid = (0.5 * (bid + ask)) if (bid and ask) else None
            g = snap.get("greeks") or {}
            meta = contracts.get(sym, {})
            rows.append({
                "symbol": sym, "underlying": occ.underlying, "strike": occ.strike,
                "expiry": occ.expiry, "kind": occ.kind, "dte": dte,
                "bid": bid, "ask": ask, "mid": mid,
                "iv": snap.get("impliedVolatility"),
                "delta": g.get("delta"), "gamma": g.get("gamma"), "vega": g.get("vega"),
                "theta": g.get("theta"), "rho": g.get("rho"),
                "open_interest": meta.get("open_interest"),
            })
        rows.sort(key=lambda r: (r["expiry"], r["strike"], r["kind"]))
        return rows

    # --- historical chain reconstruction (backtest) --------------------------

    def historical_chain(self, underlying: str, *, as_of, expiry, spot: float, r: float,
                         half_spread: float = 0.04) -> list[dict]:
        """Reconstruct a tradeable chain AS OF a past date from historical bars.

        Alpaca has no historical chain snapshot, so we generate the standard strike grid
        + the target monthly expiry, fetch each contract's daily bar on `as_of`, and keep
        the ones that actually traded (a bar exists). The bar close is the mark; bid/ask
        are modeled with `half_spread` (no historical NBBO in bars); IV is inverted from
        the close. open_interest is unknown historically, so left None (the OI floor is
        skipped — a printed bar already implies the contract traded).
        """
        from ..backtest import strike_grid
        from ..occ import build_occ
        from ..pricing.greeks import implied_vol

        strikes = strike_grid(spot)
        symbols = [build_occ(underlying, expiry, kind, k)
                   for k in strikes for kind in ("call", "put")]
        bars = self.get_bars(symbols, start=as_of.isoformat(),
                             end=(as_of + timedelta(days=1)).isoformat())
        T = max((expiry - as_of).days, 0) / 365.25
        dte = (expiry - as_of).days
        rows: list[dict] = []
        for sym, blist in bars.items():
            close = next((b.get("c") for b in (blist or []) if b.get("c")), None)
            if not close or close <= 0:
                continue
            occ = parse_occ(sym)
            mid = float(close)
            iv = implied_vol(mid, spot, occ.strike, T, r, occ.kind) if T > 0 else None
            rows.append({
                "symbol": sym, "underlying": occ.underlying, "strike": occ.strike,
                "expiry": occ.expiry, "kind": occ.kind, "dte": dte,
                "bid": mid * (1 - half_spread), "ask": mid * (1 + half_spread), "mid": mid,
                "iv": iv, "delta": None, "gamma": None, "vega": None, "theta": None,
                "open_interest": None,
            })
        rows.sort(key=lambda x: (x["strike"], x["kind"]))
        return rows

    # --- historical bars (backtest) ------------------------------------------

    def get_bars(self, symbols: list[str], *, start: str, end: str,
                 timeframe: str = "1Day") -> dict[str, list[dict]]:
        """{symbol: [bar, ...]} historical option bars for the given OCC symbols."""
        out: dict[str, list[dict]] = {}
        # Alpaca caps the symbols-per-request; chunk to be safe.
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            params = {"symbols": ",".join(chunk), "timeframe": timeframe,
                      "start": start, "end": end, "limit": 10000}
            page_token = None
            while True:
                if page_token:
                    params["page_token"] = page_token
                resp = self._data.get("/v1beta1/options/bars", params) or {}
                for sym, bars in (resp.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(bars)
                page_token = resp.get("next_page_token")
                if not page_token:
                    break
        return out
