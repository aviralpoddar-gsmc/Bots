"""Matplotlib chart helpers — Mission Control aesthetic.

Strict palette discipline (per DESIGN.md):
- bg: #08090C, surface: #0E1015, text: #E8ECEF, hairlines: #1E222B
- signal accent: cyan #00D9FF
- semantic: green #00C896 (positive), red #FF5C5C (negative), amber #FFB740
- never rainbow palettes; never decorative gradients

All exports write a single PNG to the given path and return the absolute path.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ----- design tokens (mirror DESIGN.md) ----------------------------------------
BG       = "#08090C"
SURFACE  = "#0E1015"
HAIR     = "#1E222B"
HAIR2    = "#2A2F3A"
TEXT     = "#E8ECEF"
TEXT2    = "#B8BEC8"
TEXT3    = "#7A828F"
SIGNAL   = "#00D9FF"
SIGNAL_D = "#0099B8"
POS      = "#00C896"
NEG      = "#FF5C5C"
WARN     = "#FFB740"

DIVERGING = LinearSegmentedColormap.from_list(
    "qb_diverging",
    [(0.00, NEG), (0.5, HAIR), (1.00, POS)],
    N=256,
)


def _apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "savefig.facecolor": BG, "savefig.edgecolor": "none",
        "text.color": TEXT, "axes.labelcolor": TEXT2,
        "xtick.color": TEXT3, "ytick.color": TEXT3,
        "axes.edgecolor": HAIR, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": HAIR, "grid.linewidth": 0.5, "grid.alpha": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "monospace",
        "font.monospace": ["Geist Mono", "JetBrains Mono", "SF Mono", "Menlo", "Consolas", "monospace"],
        "font.sans-serif": ["Geist", "Inter", "system-ui", "sans-serif"],
        "font.size": 10.5,
        "axes.titleweight": "600", "axes.titlesize": 12,
        "axes.titlepad": 14,
        "axes.titlelocation": "left",
    })


def _label_arrow(ax, x: float, y: float, text: str, color: str = TEXT3) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=color, va="center", ha="left")


# -----------------------------------------------------------------------------
# Chart: correlation heatmap
# -----------------------------------------------------------------------------

def correlation_heatmap(corr: pd.DataFrame, out: Path, *, title: str = "Correlation matrix · log returns") -> Path:
    """Symmetric square heatmap of correlation coefficients."""
    _apply_style()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.42), max(7, n * 0.42)))

    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8.5, color=TEXT2)
    ax.set_yticklabels(corr.index, fontsize=8.5, color=TEXT2)

    # Overlay correlation values on each cell
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if np.isnan(v) or i == j:
                continue
            txt = f"{v:+.2f}"
            # text color contrasts with cell intensity
            tc = TEXT if abs(v) < 0.5 else BG
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=tc)

    ax.set_title(title, color=TEXT, fontweight="600", fontsize=13, pad=14, loc="left")
    ax.tick_params(length=0)

    # Colorbar in a thin strip
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_tick_params(color=TEXT3, labelcolor=TEXT3, length=2)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Chart: per-pair deep dive
# -----------------------------------------------------------------------------

def pair_deepdive(series: pd.DataFrame, a: str, b: str, out: Path,
                  *, half_life: float | None = None, beta: float | None = None) -> Path:
    """3-panel chart: normalized prices, ratio with mean band, z-score."""
    _apply_style()
    if series.empty:
        # Stub chart so the report has a placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, f"No data for {a} ↔ {b}", color=TEXT3, ha="center")
        fig.savefig(out, dpi=160); plt.close(fig); return out

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1.5, 1.2], "hspace": 0.14})
    idx = series.index

    # --- panel 1: normalized price overlay
    ax = axes[0]
    ax.plot(idx, series["norm_a"], color=SIGNAL, linewidth=1.6, label=a)
    ax.plot(idx, series["norm_b"], color=WARN, linewidth=1.4, label=b, linestyle="-")
    ax.set_title(f"{a}  ↔  {b}    ·    normalized to 100 at window start",
                 fontsize=12, fontweight="600", color=TEXT)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=TEXT2)
    ax.set_ylabel("indexed (100 = start)", color=TEXT3)

    # --- panel 2: ratio with mean + ±2σ
    ax = axes[1]
    ratio = series["ratio"]
    mu = float(ratio.mean()); sd = float(ratio.std())
    ax.plot(idx, ratio, color=TEXT, linewidth=1.4)
    ax.axhline(mu, color=SIGNAL_D, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.axhline(mu + 2 * sd, color=TEXT3, linewidth=0.6, linestyle=":", alpha=0.8)
    ax.axhline(mu - 2 * sd, color=TEXT3, linewidth=0.6, linestyle=":", alpha=0.8)
    ax.set_ylabel(f"ratio  {a}/{b}", color=TEXT3)
    last = float(ratio.iloc[-1])
    _label_arrow(ax, idx[-1], mu, f"mean = {mu:.3f}", SIGNAL_D)
    last_color = POS if abs(last - mu) < 2 * sd else WARN
    _label_arrow(ax, idx[-1], last, f"last = {last:.3f}", last_color)

    # --- panel 3: z-score with ±2 bands
    ax = axes[2]
    z = series["z"]
    ax.plot(idx, z, color=TEXT, linewidth=1.2)
    ax.fill_between(idx, -1, 1, color=SIGNAL, alpha=0.06)
    ax.axhline(0, color=TEXT3, linewidth=0.4)
    ax.axhline(2, color=WARN, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.axhline(-2, color=WARN, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_ylabel("z-score", color=TEXT3)
    ax.set_ylim(min(-3, float(z.min()) - 0.3), max(3, float(z.max()) + 0.3))

    # caption strip with the headline stats
    cap_bits = []
    if beta is not None:
        cap_bits.append(f"β = {beta:+.3f}")
    if half_life is not None and math.isfinite(half_life):
        cap_bits.append(f"OU half-life = {half_life:.1f}d")
    cap_bits.append(f"current z = {float(z.iloc[-1]):+.2f}")
    cap = "    ".join(cap_bits)
    fig.text(0.012, 0.005, cap, color=TEXT3, fontsize=10)

    for ax in axes:
        ax.tick_params(length=2, color=TEXT3)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(HAIR)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Chart: cointegration shortlist as a horizontal-bar dashboard
# -----------------------------------------------------------------------------

def shortlist_dashboard(rows: list, out: Path, *, title: str = "Cointegration shortlist") -> Path:
    """Horizontal bars showing half-life (shorter = better) for each shortlisted pair.

    `rows` is a list of PairStats. Coloured by |current_z| — flags pairs that
    are currently dislocated as actionable.
    """
    _apply_style()
    n = len(rows)
    if n == 0:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No pairs passed the screen.", color=TEXT3, ha="center")
        fig.savefig(out, dpi=160); plt.close(fig); return out

    fig, ax = plt.subplots(figsize=(11, max(3, 0.42 * n + 1.5)))
    labels = [f"{r.a}  ↔  {r.b}" for r in rows]
    hl = [r.half_life for r in rows]
    z = [abs(r.current_z) for r in rows]

    # Color bars by |z| — dislocated pairs (|z|≥1.5) glow cyan/warn.
    def _color(zv: float) -> str:
        if zv >= 2.0: return WARN
        if zv >= 1.5: return SIGNAL
        return TEXT3

    bar_colors = [_color(zv) for zv in z]
    ypos = np.arange(n)
    ax.barh(ypos, hl, color=bar_colors, height=0.7, edgecolor="none")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10, color=TEXT2)
    ax.invert_yaxis()
    ax.set_xlabel("OU half-life (days)  · shorter = stronger mean reversion", color=TEXT3, fontsize=10)
    ax.set_title(title, color=TEXT, fontweight="600", fontsize=13, pad=14, loc="left")

    # Annotate each bar with current z
    for i, r in enumerate(rows):
        ax.text(r.half_life + 0.6, i, f"z = {r.current_z:+.2f}",
                va="center", fontsize=9, color=TEXT3)

    ax.tick_params(length=2, color=TEXT3)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out
