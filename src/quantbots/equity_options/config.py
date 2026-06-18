"""Config loading for the equity_options package.

Reads `config/equity_options.yaml`: the universe (equity tickers + their commodity
and the OLS beta windows), risk limits, the broker mode, and which option
structures the selector may use. Secrets (Alpaca keys) are NEVER in the yaml — only
read from env at call time, exactly like the parent `config.py` resolves
`account_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(
    os.environ.get("EQUITY_OPTIONS_CONFIG", _REPO_ROOT / "config" / "equity_options.yaml")
)
# `eo screen` writes the statistically-discovered universe here; load_config prefers it
# over the inline `underlyings:` when present (so the screen drives what we trade).
DISCOVERED_UNIVERSE = _REPO_ROOT / "config" / "equity_options_universe.yaml"

# Broker modes, in increasing order of danger. See SAFETY.md.
DRY = "dry"
PAPER = "paper"
LIVE = "live"

# Risk-limit defaults (per-position + portfolio). Conservative; override in yaml.
DEFAULT_RISK_LIMITS: dict[str, Any] = {
    "max_premium_per_trade": 250.0,    # $ premium at risk on one position
    "max_premium_per_underlying": 500.0,
    "max_total_premium": 2000.0,       # $ across the whole book
    "kelly_fraction": 0.25,            # fractional Kelly (1/4)
    "min_premium": 20.0,               # skip tickets smaller than this
    "min_open_interest": 100,          # liquidity floor per contract
    "max_rel_spread": 0.15,            # max (ask-bid)/mid to trade a contract
    "min_dte": 14,                     # don't trade < 2 weeks to expiry (theta)
    "max_dte": 120,                    # don't trade > ~4 months out
    "min_edge_return": 0.05,          # require model edge >= 5% of capital-at-risk
    "edge_hurdle": 1.10,              # (legacy; unused by selection — kept for compatibility)
    # Portfolio greek budgets (absolute, in $-equivalents at 1% / 1-vol-pt / 1-day).
    "max_net_vega": 500.0,
    "max_net_theta": 100.0,
    "max_gross_gamma": 200.0,
}


@dataclass
class Underlying:
    """One tradable name: an equity ticker mapped to its driving commodity."""

    ticker: str                       # e.g. "FCX"
    commodity: str                    # entity key in research.data_fetch.DEFAULT_UNIVERSE, e.g. "COPPER"
    market_ticker: str = "SPY"        # market factor proxy for the beta regression
    name: str = ""
    beta_lookback_days: int = 504     # ~2y of daily returns for the OLS beta
    enabled: bool = True


# Backtest go/no-go gate (overridable via config `gate:`). A name may only be traded
# live once its walk-forward backtest clears this MEANINGFUL bar (not just >0).
DEFAULT_GATE: dict[str, Any] = {
    "required": True,          # if True, `eo trade` only enters names that PASS a fresh gate
    "min_trades": 12,
    "min_brier_skill": 0.02,
    "min_sharpe": 0.25,
    "max_age_days": 14,        # a gate result older than this is stale -> treated as not passed
}


# Exit / position-management rules (overridable via config `manage:`).
DEFAULT_MANAGE_RULES: dict[str, Any] = {
    "profit_target_frac": 0.60,   # take 60% of a vertical's max profit
    "stop_loss_frac": 0.60,       # cut at 60% of premium lost
    "min_hold_dte": 10,           # close inside 10 DTE regardless
    "assignment_dte": 21,         # short-leg-ITM assignment guard within this DTE
}


@dataclass
class EquityOptionsConfig:
    underlyings: list[Underlying] = field(default_factory=list)
    broker: str = DRY                 # dry | paper | live
    structures: list[str] = field(default_factory=lambda: ["long_call", "long_put"])
    risk_limits: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RISK_LIMITS))
    forecast: dict[str, Any] = field(default_factory=dict)   # LLM/tal seam toggles + diffusion params
    manage: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MANAGE_RULES))
    gate: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_GATE))
    risk_limits_file: str = "config/equity_options_risk_limits.yaml"  # gate for LIVE

    # --- broker secrets, resolved from env at call time (never stored) -------

    @property
    def alpaca_key(self) -> str | None:
        return os.environ.get("ALPACA_API_KEY")

    @property
    def alpaca_secret(self) -> str | None:
        return os.environ.get("ALPACA_SECRET_KEY")

    def enabled_underlyings(self) -> list[Underlying]:
        return [u for u in self.underlyings if u.enabled]

    def find(self, ticker: str) -> Underlying | None:
        return next((u for u in self.underlyings if u.ticker == ticker.upper()), None)


def _merge_risk(raw: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_RISK_LIMITS)
    merged.update(raw or {})
    return merged


def _merge_manage(raw: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_MANAGE_RULES)
    merged.update(raw or {})
    return merged


def _merge_gate(raw: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_GATE)
    merged.update(raw or {})
    return merged


def _parse_underlyings(entries: list[dict]) -> list[Underlying]:
    return [
        Underlying(
            ticker=str(e["ticker"]).upper(),
            commodity=str(e["commodity"]).upper(),
            market_ticker=str(e.get("market_ticker", "SPY")).upper(),
            name=e.get("name", ""),
            beta_lookback_days=int(e.get("beta_lookback_days", 504)),
            enabled=e.get("enabled", True),
        )
        for e in entries
    ]


def load_config(path: Path | str = DEFAULT_CONFIG) -> EquityOptionsConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    # Prefer the screen-discovered universe when present.
    if DISCOVERED_UNIVERSE.exists():
        disc = yaml.safe_load(DISCOVERED_UNIVERSE.read_text()) or {}
        underlyings = _parse_underlyings(disc.get("underlyings", []))
        if not underlyings:  # empty discovered file -> fall back to inline
            underlyings = _parse_underlyings(data.get("underlyings", []))
    else:
        underlyings = _parse_underlyings(data.get("underlyings", []))
    broker = str(data.get("broker", DRY)).lower()
    if broker not in (DRY, PAPER, LIVE):
        raise ValueError(f"broker must be one of dry|paper|live, got {broker!r}")
    return EquityOptionsConfig(
        underlyings=underlyings,
        broker=broker,
        structures=list(data.get("structures", ["long_call", "long_put"])),
        risk_limits=_merge_risk(data.get("risk_limits")),
        forecast=dict(data.get("forecast", {})),
        manage=_merge_manage(data.get("manage")),
        gate=_merge_gate(data.get("gate")),
        risk_limits_file=data.get("risk_limits_file", "config/equity_options_risk_limits.yaml"),
    )
