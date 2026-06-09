"""007 — the news-driven commodity bot (single source = digested commodity news).

Reads ONE signal family — ``SIG_<COM>_NEWS`` (a recency-decayed, confidence-weighted
MEAN SIGNED direction in [-1, 1], built by a LOCAL-LLM digestion of commodity news
headlines; see processing.compute_news_signal + llm/news_extractor.py) — and expresses
it as a small bounded price DRIFT on the live commodity anchor, then prices threshold
markets with the shared lognormal CDF. Bullish news tilt (raw > 0) -> upward drift;
bearish (raw < 0) -> downward.

It ABSTAINS unless the news actually says something: it requires a minimum number of
in-window qualifying headlines (``min_items``) AND a minimum conviction (``|raw|`` >=
``min_conviction``), on top of the base class's ``drift_cap`` / ``min_drift`` gates.
Honest framing: news sentiment is low-Sharpe and decays fast (hence small ``k``, a short
``max_horizon_years``, and the 36h half-life in the signal), and only commodity PRICE
markets resolve — so this is a disciplined, abstain-heavy TILT on the resolving price
commodities, not a forecast. The edge is the pipeline discipline, not a secret feed.
"""

from __future__ import annotations

import re
from typing import Any

from ._signal_base import SignalDriftStrategy, obs_payload


class NewsDriftStrategy(SignalDriftStrategy):
    name = "news_drift"
    description = (
        "Single-source (digested commodity NEWS) bot: a local-LLM reads commodity "
        "headlines, attributes a signed price direction per commodity, and the bot "
        "applies a small bounded drift to the live price anchor — bullish news → up, "
        "bearish → down — pricing the resolving commodity price-threshold markets. "
        "Abstains unless the news is fresh, plural, and directionally clear."
    )
    # (word-boundary regex, price_entity, annual_vol). Word boundaries avoid the
    # "tin inside scrutiny" substring trap. Only commodities with a live price anchor
    # in the Stooq feed are listed (coffee has no anchor → excluded).
    CATALOG = [
        (re.compile(r"\bgold\b", re.I), "GOLD", 0.16),
        (re.compile(r"\bsilver\b", re.I), "SILVER", 0.30),
        (re.compile(r"\bplatinum\b", re.I), "PLATINUM", 0.22),
        (re.compile(r"\bpalladium\b", re.I), "PALLADIUM", 0.30),
        (re.compile(r"\bcopper\b", re.I), "COPPER", 0.21),
        (re.compile(r"\bbrent\b", re.I), "BRENT_OIL", 0.39),
        (re.compile(r"\bwti\b|west texas", re.I), "WTI_OIL", 0.40),
        (re.compile(r"\bgasoline\b|\brbob\b", re.I), "GASOLINE", 0.45),
        (re.compile(r"natural gas|henry hub|\bnatgas\b", re.I), "NATGAS", 0.55),
        (re.compile(r"\bcotton\b", re.I), "CME_COTTON", 0.24),
        (re.compile(r"\bcocoa\b", re.I), "CME_COCOA", 0.40),
        (re.compile(r"\bcorn\b", re.I), "CME_CORN", 0.30),
        (re.compile(r"\bsugar\b", re.I), "CME_SUGAR", 0.30),
        (re.compile(r"\bwheat\b", re.I), "CME_WHEAT", 0.30),
    ]

    def __init__(self, k: float = 0.08, min_conviction: float = 0.25,
                 min_items: int = 2, **params: Any):
        super().__init__(**params)
        self.k = k                          # drift per unit of signed news consensus
        self.min_conviction = min_conviction  # |raw| floor: ignore near-neutral news
        self.min_items = int(min_items)     # need >= this many qualifying headlines

    def signal_drift(self, spot: float, price_entity: str, T: float):
        com = price_entity.replace("CME_", "").replace("_OIL", "")
        o = self._obs.latest_observation(f"SIG_{com}_NEWS")
        if not o or o.get("value") is None:
            return None
        raw = float(o["value"])
        pay = obs_payload(o)
        n = int(pay.get("n_items", 0))
        if n < self.min_items or abs(raw) < self.min_conviction:
            return None  # not enough fresh, directionally-clear news -> abstain
        mu = self.k * raw
        n_pos, n_neg = pay.get("n_pos", 0), pay.get("n_neg", 0)
        reason = (
            f"news consensus {raw:+.2f} over {n} headlines "
            f"({n_pos}↑/{n_neg}↓, half-life {pay.get('halflife_h', 36)}h) "
            f"→ {'bullish' if raw > 0 else 'bearish'} tilt"
        )
        return mu, {"news_raw": raw, "n_items": n, "top_headlines": pay.get("top_headlines"),
                    "reason": reason}
