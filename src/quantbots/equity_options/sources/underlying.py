"""Underlying market data: equity/commodity OHLCV (yfinance) + the risk-free curve
(FRED), reusing `research.data_fetch` (the parent's cached fetchers) verbatim.

No Alpaca here — these are the free, keyless feeds that drive the f_P forecast.
`risk_free_rate` reads the 3-month T-bill (FRED DGS3MO) as the continuous-compounding
short rate; `dividend_yield` is best-effort from yfinance .info (0.0 if unavailable).
"""

from __future__ import annotations

import logging

from ...research.data_fetch import fetch_fred_history, fetch_yf_history

logger = logging.getLogger(__name__)

_RATE_SERIES = "DGS3MO"   # 3-month Treasury, percent


def history(ticker: str, *, period: str = "10y"):
    """Cached daily OHLCV DataFrame (Date-indexed, 'Close' column)."""
    return fetch_yf_history(ticker, period=period)


def spot(ticker: str, *, period: str = "1mo") -> float | None:
    """Latest close for a ticker, or None if unavailable."""
    df = fetch_yf_history(ticker, period=period)
    if df.empty or "Close" not in df.columns:
        return None
    closes = df["Close"].dropna()
    return float(closes.iloc[-1]) if len(closes) else None


def risk_free_rate(*, default: float = 0.04) -> float:
    """Continuous short rate from FRED 3M T-bill (decimal). Falls back to `default`."""
    try:
        df = fetch_fred_history(_RATE_SERIES)
        if not df.empty:
            pct = float(df["value"].dropna().iloc[-1])
            return pct / 100.0
    except Exception as e:  # noqa: BLE001 - FRED outage must not break pricing
        logger.warning("risk_free_rate: FRED %s failed (%s); using %.3f", _RATE_SERIES, e, default)
    return default


def dividend_yield(ticker: str, *, default: float = 0.0) -> float:
    """Best-effort trailing dividend yield (decimal) from yfinance; default if absent."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        dy = info.get("dividendYield")
        if dy is not None:
            # yfinance returns either a fraction (0.02) or a percent (2.0) by version.
            return float(dy) / 100.0 if dy > 1.0 else float(dy)
    except Exception as e:  # noqa: BLE001
        logger.debug("dividend_yield: %s lookup failed (%s)", ticker, e)
    return default
