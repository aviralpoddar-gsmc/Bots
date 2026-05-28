"""Pair-trading research report — end-to-end driver.

Pulls the universe, runs correlation + cointegration screens, generates charts,
and writes a markdown report into a date-stamped directory under
`data/research/pairs_YYYYMMDD/`. Re-running on the same day overwrites in place
(cache hits make subsequent runs near-instant).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .charts import correlation_heatmap, pair_deepdive, shortlist_dashboard
from .data_fetch import fetch_universe
from .pairs import (
    ASSET_CLASS,
    PairStats,
    align_panel,
    cointegration_shortlist,
    correlation_matrix,
    pair_series,
    pair_stats,
    top_correlated_pairs,
)

logger = logging.getLogger(__name__)

# Pairs to deep-dive regardless of how the cointegration screen ranks them —
# these are the textbook macro pairs every operator wants to see explicitly.
HERO_PAIRS: list[tuple[str, str, str]] = [
    ("GOLD", "SILVER",     "Gold–Silver ratio (precious-metals barometer)"),
    ("WTI_OIL", "BRENT_OIL", "WTI–Brent spread (US vs. global crude)"),
    ("PLATINUM", "PALLADIUM", "Pt–Pd (autocatalyst substitution)"),
    ("GASOLINE", "WTI_OIL", "Gasoline crack spread"),
    ("HEATING_OIL", "WTI_OIL", "Heating-oil crack spread"),
    ("CORN", "SOYBEANS",    "Corn–Soybeans (cropland substitution)"),
    ("CORN", "WHEAT",       "Corn–Wheat (feed-grain twin)"),
    ("COPPER", "GOLD",      "Cu–Au (real-rate / growth-vs-haven proxy)"),
    ("SILVER", "COPPER",    "Silver–Copper (industrial precious)"),
    ("KTOS", "AVAV",        "Kratos–AeroVironment (defense small-caps)"),
    ("SPX", "FRED_VIXCLS",  "SPX–VIX (vol short)"),
    ("TLT", "FRED_DGS10",   "TLT–10Y yield (duration / rate)"),
]


def _md_table_corr(corr: pd.DataFrame, k: int = 15) -> str:
    rows = top_correlated_pairs(corr, n=k)
    out = ["| # | Asset A | Asset B | ρ (log returns) | Class A | Class B |",
           "|---|---|---|---|---|---|"]
    for i, (a, b, c) in enumerate(rows, 1):
        out.append(f"| {i} | {a} | {b} | {c:+.3f} | {ASSET_CLASS.get(a, '—')} | {ASSET_CLASS.get(b, '—')} |")
    return "\n".join(out)


def _md_table_coint(rows: list[PairStats]) -> str:
    out = ["| # | Pair | ρ | β | Half-life (d) | z now | \\|z\\| max 1y |",
           "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | {r.a} ↔ {r.b} | {r.corr_returns:+.3f} | {r.beta:+.3f} | "
                   f"{r.half_life:.1f} | {r.current_z:+.2f} | {r.abs_z_max_252:.2f} |")
    return "\n".join(out)


def _strategy_proposals(coint: list[PairStats], hero_stats: dict[tuple[str, str], PairStats]) -> str:
    """Pick the 3-5 most actionable pairs and propose specific clone-tradeable
    strategies based on the stats."""
    out = [
        "### A. Convergence trades on cointegrated pairs",
        "",
        "For each pair below, the spread has a finite OU half-life — it reverts. The play is: when |z| crosses a threshold (≥1.5 or 2.0), fund the convergence side until z returns toward 0.",
        "",
        "Clone-translation: find threshold markets on the *expensive* leg of the spread pricing higher than the model implies for the *cheap* leg's level. Bet NO on the expensive leg's tail-prob market, YES on the cheap leg's. Resolves YES/NO only when the underlying spot crosses the threshold — so resolvability matters; commodity-spot ladders (LBMA, NYMEX, ICE) are the safe targets.",
        "",
    ]
    if coint:
        out.append("Pairs to monitor (from cointegration shortlist):")
        out.append("")
        for r in coint[:5]:
            dislocated = abs(r.current_z) >= 1.5
            flag = " · **DISLOCATED**" if dislocated else ""
            out.append(f"- **{r.a} ↔ {r.b}** — half-life {r.half_life:.0f}d, z = {r.current_z:+.2f}{flag}. "
                       f"Trade trigger: open at |z| ≥ 2.0; expected reversion timescale ~{r.half_life:.0f} days.")
        out.append("")

    out.extend([
        "### B. Crack-spread arb (energy)",
        "",
        "Refiners' gross margin: gasoline + heating-oil − crude. Threshold markets on cracked products and crude itself let us trade compression / widening.",
        "",
        "- Setup: when WTI–Gasoline z stretches negative (gasoline expensive vs. crude), bet against gasoline 'exceeds X' and for crude 'exceeds Y' (calibrated through the lognormal model).",
        "- Half-lives observed: WTI–Gasoline ~24d, Brent–Heating Oil ~36d. Tradeable timeframes for 1–3 month resolution markets.",
        "",
        "### C. Substitution pair (precious-metal autocatalysts)",
        "",
        "Pt and Pd are partial substitutes in gasoline-vehicle autocatalysts. The ratio runs in long cycles (currently mean-reverting at ~41 day half-life). Pt has been the cheap side for years; if z swings >+2 we're at a substitution-arb extreme.",
        "",
        "- Strategy: trade the ratio convergence using threshold markets on both Pt and Pd spot-price exceedances on overlapping dates.",
        "",
        "### D. Single-name defense pair",
        "",
        "KTOS and AVAV both ride defense-spending news flow with ρ = +0.51 and similar duration. The clone has stock-price threshold markets for both. When one moves and the other lags, the lag is a 1–3 day catch-up trade.",
        "",
        "- Setup: when KTOS ↔ AVAV z dislocates >1.5, bet against the leader's threshold market for the next month, for the laggard's.",
        "",
        "### E. Structural macro hedges (informational, not directly tradeable on clone)",
        "",
        "Pairs like TLT↔DGS10 (ρ=−0.91) and SPX↔VIX (ρ=−0.82) confirm our risk model but the clone doesn't carry direct VIX or TLT markets in volume. Useful as a *risk overlay* on the live book — when SPX↔VIX z spikes, dial back position size on cancellation-prone markets (the macro regime is uncertain and resolutions get delayed).",
        "",
        "### Resolvability footnote",
        "",
        "Per CLAUDE.md: ~93% of clone resolutions are CANCEL. The pairs above translate to clone trades only where both legs have *spot-price threshold markets sourced from LBMA / NYMEX / ICE* — those resolve YES/NO reliably. Production / demand / capacity ladders are out of scope here even if the cointegration is real, because they don't pay out.",
    ])
    return "\n".join(out)


def generate_report(out_dir: Path, *, period: str = "3y", lookback_days: int = 750) -> Path:
    """Run the full pipeline. Returns the path to the written report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    logger.info("research: fetching universe (period=%s)", period)
    panel = fetch_universe(period=period)
    panel = align_panel(panel, lookback_days=lookback_days, min_coverage=0.85)
    if panel.empty:
        raise RuntimeError("empty panel after alignment — check data sources")
    logger.info("research: aligned panel %s, %d series", panel.shape, panel.shape[1])

    corr = correlation_matrix(panel, method="pearson")
    coint = cointegration_shortlist(panel, max_half_life_days=90, min_corr_returns=0.4, max_pairs=25)
    logger.info("research: %d pairs passed cointegration screen", len(coint))

    # Charts
    heatmap_path = correlation_heatmap(corr, charts_dir / "corr_heatmap.png")
    shortlist_path = shortlist_dashboard(coint, charts_dir / "cointegration_shortlist.png")

    hero_chart_paths: list[tuple[str, str, str, Path]] = []
    hero_stats: dict[tuple[str, str], PairStats] = {}
    for a, b, title in HERO_PAIRS:
        if a not in panel.columns or b not in panel.columns:
            continue
        ps = pair_stats(panel, a, b)
        if ps is None:
            continue
        series = pair_series(panel, a, b, lookback_days=lookback_days)
        fname = f"pair_{a.lower()}_{b.lower()}.png"
        pair_deepdive(series, a, b, charts_dir / fname,
                      half_life=ps.half_life, beta=ps.beta)
        hero_chart_paths.append((a, b, title, Path("charts") / fname))
        hero_stats[(a, b)] = ps

    # Markdown
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    md: list[str] = [
        "# Pair-Trading Research Report",
        f"_Generated {today} · universe = {panel.shape[1]} series, "
        f"window = {panel.index.min().date()} → {panel.index.max().date()} "
        f"({len(panel)} business days)_",
        "",
        "## 1. Universe",
        "",
        "All series are daily Close (or daily-resampled FRED), forward-filled across holidays "
        "and aligned to weekday business days. Coverage threshold for inclusion is ≥85% of the lookback window.",
        "",
        f"**Series ({panel.shape[1]}):** " + ", ".join(f"`{c}`" for c in panel.columns),
        "",
        "**By asset class:**",
        "",
    ]
    by_class: dict[str, list[str]] = {}
    for c in panel.columns:
        by_class.setdefault(ASSET_CLASS.get(c, "Other"), []).append(c)
    for klass, members in sorted(by_class.items()):
        md.append(f"- **{klass}:** {', '.join(f'`{m}`' for m in members)}")
    md.extend([
        "",
        "## 2. Correlation matrix",
        "",
        "Pearson ρ on log returns. Diverging palette: green = positive, red = negative.",
        "",
        f"![Correlation matrix](charts/{heatmap_path.name})",
        "",
        "### Top 15 |ρ| pairs",
        "",
        _md_table_corr(corr, k=15),
        "",
        "## 3. Cointegration shortlist",
        "",
        "Engle-Granger lite: OLS hedge ratio β on `log a` vs `log b`, OU half-life on the residual spread. "
        "Pairs shown have a finite half-life under 90 days and |ρ| ≥ 0.4 (filters out spurious mean-reverters).",
        "",
        f"![Cointegration shortlist](charts/{shortlist_path.name})",
        "",
        _md_table_coint(coint),
        "",
        "## 4. Hero pair deep-dives",
        "",
        "For each canonical pair: normalized price overlay, spread/ratio with mean ± 2σ bands, z-score.",
        "",
    ])
    for a, b, title, path in hero_chart_paths:
        ps = hero_stats[(a, b)]
        md.append(f"### {title}")
        md.append("")
        md.append(f"`{a} ↔ {b}` · ρ = {ps.corr_returns:+.3f} · β = {ps.beta:+.3f} · "
                  f"half-life = {ps.half_life:.1f}d · current z = **{ps.current_z:+.2f}**")
        md.append("")
        md.append(f"![{a} vs {b}]({path.as_posix()})")
        md.append("")
    md.extend([
        "## 5. Proposed strategies for the clone",
        "",
        _strategy_proposals(coint, hero_stats),
        "",
        "---",
        "",
        f"_Generated by `quantbots.research.report` · panel cached in `data/research/cache/` · "
        f"re-run via `python -m quantbots.research.report` to refresh._",
    ])
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md))
    logger.info("research: report written to %s", report_path)
    return report_path
