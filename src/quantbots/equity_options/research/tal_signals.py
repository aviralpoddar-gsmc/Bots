"""Multi-agent tal-market consensus → tradeable directional spread candidates.

Pipeline:
  1. Pull actively-traded (multi-bettor) tal price markets — real crowd/agent consensus,
     not the stale 0.50/1.0 untraded markets.
  2. Classify each by the material it references (keyword on the question).
  3. Per material, aggregate a CONSENSUS TILT = bettor-weighted mean of (P(exceeds) − 0.5)
     across its price-threshold markets. >0 means the crowd systematically prices the
     material's price ABOVE the quoted thresholds → bullish; <0 → bearish. Confidence =
     total bettors × log-volume × number of markets.
  4. Map the material to optionable equities (curated producer map) and, where we have a
     futures feed, weight by the equity↔metal return correlation (the metal matrix).
  5. Emit ranked spread candidates: bullish consensus → bull-call debit spread on the
     producer, bearish → bear-put spread, conviction = |tilt|·confidence·|corr|.

These are CANDIDATES — they must still clear the walk-forward gate before live trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Material (keyword in market question) -> optionable producer equities + futures feed
# (feed = a DEFAULT_UNIVERSE key when we can validate with a return correlation; else None).
MATERIALS: dict[str, dict] = {
    "copper":     {"equities": ["FCX", "SCCO", "COPX", "TECK"], "feed": "COPPER"},
    "gold":       {"equities": ["GDX", "GDXJ", "NEM", "AEM", "WPM", "FNV"], "feed": "GOLD"},
    "silver":     {"equities": ["AG", "HL", "PAAS", "SIL"], "feed": "SILVER"},
    "platinum":   {"equities": [], "feed": "PLATINUM"},
    "palladium":  {"equities": [], "feed": "PALLADIUM"},
    "nickel":     {"equities": ["VALE", "BHP"], "feed": None},
    "aluminum":   {"equities": ["AA", "CENX"], "feed": None},
    "aluminium":  {"equities": ["AA", "CENX"], "feed": None},
    "cobalt":     {"equities": ["VALE", "BHP"], "feed": None},
    "lithium":    {"equities": ["ALB", "SQM", "LAC"], "feed": None},
    "uranium":    {"equities": ["CCJ", "URA"], "feed": None},
    "iron ore":   {"equities": ["VALE", "RIO", "BHP"], "feed": None},
    "rare earth": {"equities": ["MP"], "feed": None},
    "neodymium":  {"equities": ["MP"], "feed": None},
    "ndpr":       {"equities": ["MP"], "feed": None},
    "crude":      {"equities": ["XOM", "CVX", "COP", "OXY", "SLB"], "feed": "WTI_OIL"},
    "brent":      {"equities": ["XOM", "CVX", "COP", "OXY"], "feed": "BRENT_OIL"},
    "wti":        {"equities": ["XOM", "CVX", "COP", "OXY"], "feed": "WTI_OIL"},
    "natural gas": {"equities": ["EQT", "AR", "RRC", "LNG"], "feed": "NATGAS"},
}


@dataclass
class MaterialConsensus:
    material: str
    tilt: float            # bettor-weighted mean(P-0.5) across the material's markets; sign = direction
    n_markets: int
    total_bettors: int
    total_volume: float
    confidence: float      # 0..1


@dataclass
class SpreadCandidate:
    equity: str
    material: str
    direction: str         # "bull" -> bull_call_spread, "bear" -> bear_put_spread
    consensus_tilt: float
    confidence: float
    correlation: float | None   # equity↔metal return corr (None if no futures feed)
    conviction: float           # |tilt| · confidence · |corr or 0.5|
    n_markets: int = 0


def _classify(question: str) -> str | None:
    q = question.lower()
    for kw in MATERIALS:
        if kw in q:
            return kw
    return None


def material_consensus(markets_df) -> dict[str, MaterialConsensus]:
    """Aggregate the multi-agent consensus tilt per material."""
    import pandas as pd
    if markets_df is None or len(markets_df) == 0:
        return {}
    agg: dict[str, dict] = {}
    for _, m in markets_df.iterrows():
        kw = _classify(str(m.get("MARKET_QUESTION", "")))
        if kw is None:
            continue
        p = float(m["LATEST_MARKET_PROBABILITY"])
        bettors = float(m.get("UNIQUE_BETTOR_COUNT") or 0)
        vol = float(m.get("MANIFOLD_VOLUME") or 0)
        w = bettors  # weight the tilt by participation
        a = agg.setdefault(kw, {"wsum": 0.0, "w": 0.0, "n": 0, "bettors": 0.0, "vol": 0.0})
        a["wsum"] += w * (p - 0.5); a["w"] += w
        a["n"] += 1; a["bettors"] += bettors; a["vol"] += vol
    out: dict[str, MaterialConsensus] = {}
    for kw, a in agg.items():
        if a["w"] <= 0:
            continue
        tilt = a["wsum"] / a["w"]                          # in [-0.5, 0.5]
        # confidence rises with breadth (markets), participation (bettors), volume.
        conf = min(1.0, (math.log1p(a["bettors"]) / 8.0) * min(a["n"] / 5.0, 1.0)
                   * min(math.log1p(a["vol"]) / 12.0, 1.0) * 3.0)
        out[kw] = MaterialConsensus(material=kw, tilt=tilt, n_markets=a["n"],
                                    total_bettors=int(a["bettors"]), total_volume=a["vol"],
                                    confidence=round(conf, 3))
    return out


def spread_candidates(consensus: dict[str, MaterialConsensus], *, corr_matrix=None,
                      min_tilt: float = 0.05, min_confidence: float = 0.15) -> list[SpreadCandidate]:
    """Cross the per-material consensus with optionable producers (+ correlation) into
    ranked bull/bear spread candidates."""
    out: list[SpreadCandidate] = []
    for kw, c in consensus.items():
        if abs(c.tilt) < min_tilt or c.confidence < min_confidence:
            continue
        spec = MATERIALS.get(kw, {})
        feed = spec.get("feed")
        direction = "bull" if c.tilt > 0 else "bear"
        for eq in spec.get("equities", []):
            corr = None
            if corr_matrix is not None and feed and eq in corr_matrix.index and feed in corr_matrix.columns:
                v = corr_matrix.loc[eq, feed]
                corr = float(v) if v == v else None  # NaN guard
            corr_w = abs(corr) if corr is not None else 0.5   # 0.5 default when no feed to validate
            conviction = abs(c.tilt) * c.confidence * corr_w
            out.append(SpreadCandidate(
                equity=eq, material=kw, direction=direction, consensus_tilt=round(c.tilt, 3),
                confidence=c.confidence, correlation=round(corr, 2) if corr is not None else None,
                conviction=round(conviction, 4), n_markets=c.n_markets))
    out.sort(key=lambda s: s.conviction, reverse=True)
    return out
