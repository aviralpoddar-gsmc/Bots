"""OCC option-symbol build/parse (the 21-char format Alpaca uses).

Format:  ROOT (1-6 chars, left-justified) + YYMMDD + C/P + STRIKE*1000 (8 digits).
Example: AAPL  240119C00100000  ->  AAPL, 2024-01-19, call, $100.00
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class OccSymbol:
    underlying: str
    expiry: date
    kind: str          # "call" | "put"
    strike: float

    @property
    def symbol(self) -> str:
        return build_occ(self.underlying, self.expiry, self.kind, self.strike)


def build_occ(underlying: str, expiry: date, kind: str, strike: float) -> str:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be call|put, got {kind!r}")
    root = underlying.upper()
    if not 1 <= len(root) <= 6:
        raise ValueError(f"underlying root must be 1-6 chars, got {underlying!r}")
    cp = "C" if kind == "call" else "P"
    strike_int = round(strike * 1000)
    if strike_int <= 0 or strike_int > 99_999_999:
        raise ValueError(f"strike out of OCC range: {strike}")
    return f"{root}{expiry:%y%m%d}{cp}{strike_int:08d}"


def parse_occ(symbol: str) -> OccSymbol:
    symbol = symbol.strip().upper()
    if len(symbol) < 16:
        raise ValueError(f"not an OCC symbol: {symbol!r}")
    # The fixed tail is 6(date)+1(C/P)+8(strike) = 15 chars; the rest is the root.
    tail = symbol[-15:]
    root = symbol[:-15]
    if not root:
        raise ValueError(f"no underlying root in {symbol!r}")
    expiry = datetime.strptime(tail[:6], "%y%m%d").date()
    cp = tail[6]
    if cp not in ("C", "P"):
        raise ValueError(f"bad call/put flag in {symbol!r}")
    strike = int(tail[7:]) / 1000.0
    return OccSymbol(underlying=root, expiry=expiry, kind="call" if cp == "C" else "put",
                     strike=strike)
