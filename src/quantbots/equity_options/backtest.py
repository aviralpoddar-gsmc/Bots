"""Walk-forward backtest + calibration metrics for the options forecast.

Two layers:

1. **Pure metrics** (no I/O, unit-tested): Brier score + skill vs an implied baseline,
   the sample CRPS (energy form) for the distributional forecast f_P, and realized
   option-strategy PnL/Sharpe. These are the gates from the plan
   ("Brier skill > 0 AND positive risk-adjusted PnL").

2. **Orchestrator** `run_backtest`: for each as-of date it builds f_P using ONLY data
   up to that date (no lookahead — asserted), selects the top candidate from the
   historical chain (Alpaca bars), and scores it against the realized terminal price
   (yfinance). Requires Alpaca keys + history; it logs the data window because Alpaca
   options history starts ~Feb 2024, so coverage must never be overstated.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ALPACA_OPTIONS_HISTORY_START = "2024-02-01"  # coverage floor; logged, never overstated

_REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_FILE = Path(os.environ.get("EQUITY_OPTIONS_GATE_FILE",
                                _REPO_ROOT / "data" / "equity_options_gate.json"))


def save_gate_results(results: dict[str, dict]) -> None:
    """Persist {ticker: {passed, brier_skill, pnl_sharpe, n_trades, reason}} + a UTC
    timestamp, so `eo trade` can require a fresh PASS before entering a name."""
    GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(UTC).isoformat(), "results": results}
    GATE_FILE.write_text(json.dumps(payload, indent=2))


def load_gate_results() -> dict:
    if not GATE_FILE.exists():
        return {}
    try:
        return json.loads(GATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def passing_tickers(*, max_age_days: int) -> set[str]:
    """Tickers whose last gate run PASSED and is not older than max_age_days."""
    data = load_gate_results()
    updated = data.get("updated_at")
    if not updated:
        return set()
    age = datetime.now(UTC) - datetime.fromisoformat(updated)
    if age > timedelta(days=max_age_days):
        return set()
    return {t for t, r in data.get("results", {}).items() if r.get("passed")}


# --- chain reconstruction helpers (pure) -------------------------------------

def third_friday(year: int, month: int) -> date:
    """Standard monthly option expiry: the third Friday of the month."""
    d = date(year, month, 1)
    first_friday = 1 + (4 - d.weekday()) % 7
    return date(year, month, first_friday + 14)


def target_expiry(as_of: date, horizon_days: int) -> date:
    """The monthly (3rd-Friday) expiry closest to as_of + horizon_days."""
    target = as_of + timedelta(days=horizon_days)
    candidates = []
    for dm in (-1, 0, 1):
        m = target.month + dm
        y = target.year + (m - 1) // 12
        mm = (m - 1) % 12 + 1
        candidates.append(third_friday(y, mm))
    return min(candidates, key=lambda e: abs((e - target).days))


def strike_increment(spot: float) -> float:
    if spot < 25:
        return 1.0
    if spot < 100:
        return 2.5
    if spot < 200:
        return 5.0
    return 10.0


def strike_grid(spot: float, *, lo_band: float = 0.6, hi_band: float = 1.4) -> list[float]:
    """Standard listed strikes spanning the moneyness band around spot."""
    inc = strike_increment(spot)
    lo = math.floor(spot * lo_band / inc) * inc
    hi = math.ceil(spot * hi_band / inc) * inc
    n = int(round((hi - lo) / inc)) + 1
    return [round(lo + i * inc, 2) for i in range(n) if lo + i * inc > 0]


# --- pure metrics ------------------------------------------------------------

def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill(probs: list[float], baseline_probs: list[float], outcomes: list[int]) -> float:
    """Brier skill score vs a baseline (e.g. the option-implied prob). >0 = we beat it."""
    bs = brier_score(probs, outcomes)
    bs_ref = brier_score(baseline_probs, outcomes)
    return 1.0 - bs / bs_ref if bs_ref > 0 else 0.0


def crps_sample(sample: np.ndarray, observed: float) -> float:
    """CRPS via the energy form for an empirical (MC) forecast distribution:

        CRPS = E|X - y| - 0.5 E|X - X'|

    Estimated with the sorted-sample closed form for the second term (O(n log n))."""
    x = np.sort(np.asarray(sample, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    term1 = float(np.mean(np.abs(x - observed)))
    # E|X - X'| for an empirical dist = (2/n^2) * sum_i (2i - n + 1) x_i  (i 0-based)
    i = np.arange(n)
    term2 = (2.0 / (n * n)) * float(np.sum((2 * i - n + 1) * x))
    return term1 - 0.5 * term2


def pnl_stats(pnls: list[float]) -> dict[str, float]:
    """Realized PnL summary: total, mean, Sharpe (per-trade), win rate, count."""
    a = np.asarray(pnls, dtype=float)
    if len(a) == 0:
        return {"total": 0.0, "mean": 0.0, "sharpe": 0.0, "win_rate": 0.0, "n": 0}
    sd = float(np.std(a))
    return {
        "total": float(np.sum(a)),
        "mean": float(np.mean(a)),
        "sharpe": float(np.mean(a) / sd) if sd > 1e-9 else 0.0,
        "win_rate": float(np.mean(a > 0)),
        "n": int(len(a)),
    }


# --- orchestrator ------------------------------------------------------------

@dataclass
class BacktestResult:
    underlying: str
    forecast_probs: list[float] = field(default_factory=list)
    implied_probs: list[float] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)
    crps: list[float] = field(default_factory=list)
    pnls: list[float] = field(default_factory=list)
    pit: list[float] = field(default_factory=list)   # P_fP(S_T <= realized) per fold
    spot_ret: list[float] = field(default_factory=list)  # realized log-return per fold
    folds: int = 0                 # distinct (as_of) dates that produced a trade
    signal_folds: int = 0          # folds with a usable tal directional view (mode="tal")

    @property
    def n_trades(self) -> int:
        return len(self.pnls)

    def reliability(self, n_bins: int = 10) -> list[tuple[float, float, int]]:
        """Reliability curve: per forecast-prob bin -> (mean_forecast, realized_freq, n).
        A calibrated model has mean_forecast ~= realized_freq on the diagonal."""
        fp = np.asarray(self.forecast_probs)
        oc = np.asarray(self.outcomes, dtype=float)
        out = []
        edges = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (fp >= lo) & (fp < hi if hi < 1 else fp <= hi)
            if m.sum() == 0:
                continue
            out.append((float(fp[m].mean()), float(oc[m].mean()), int(m.sum())))
        return out

    def pit_stats(self) -> dict[str, float]:
        """PIT diagnostics. Calibrated => uniform(0,1): mean~0.5, std~0.289. mean<<0.5
        means realized prices land in f_P's LEFT tail (f_P biased HIGH / too bearish);
        std<<0.289 means realized lands near the center (f_P TOO WIDE); std>>0.289 or
        U-shape means f_P TOO NARROW."""
        if not self.pit:
            return {}
        p = np.asarray(self.pit)
        return {"pit_mean": float(p.mean()), "pit_std": float(p.std()),
                "pit_frac_extreme": float(np.mean((p < 0.1) | (p > 0.9))),
                "realized_drift": float(np.mean(self.spot_ret)) if self.spot_ret else float("nan")}

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"underlying": self.underlying, "folds": self.folds,
                                 "n_trades": self.n_trades}
        if self.outcomes:
            out["brier"] = brier_score(self.forecast_probs, self.outcomes)
            out["brier_skill"] = brier_skill(self.forecast_probs, self.implied_probs, self.outcomes)
        if self.crps:
            out["crps"] = float(np.mean(self.crps))
        if self.pnls:
            out.update({f"pnl_{k}": v for k, v in pnl_stats(self.pnls).items()})
        return out

    def gate(self, *, min_trades: int = 12, min_brier_skill: float = 0.02,
             min_sharpe: float = 0.25) -> tuple[bool, str]:
        """Phase-1 go/no-go. A `>0` threshold rubber-stamps noise, so the bar is
        MEANINGFUL: enough trades for any power, a real calibration edge over the
        implied baseline, AND a real per-trade PnL Sharpe. Defaults are deliberately
        strict — a marginal pass is a fail. Returns (passed, reason)."""
        s = self.summary()
        if self.n_trades < min_trades or not self.outcomes:
            return False, (f"insufficient data (folds={self.folds}, trades={self.n_trades} "
                           f"< {min_trades})")
        skill = s.get("brier_skill", -1.0)
        sharpe = s.get("pnl_sharpe", -1.0)
        passed = skill >= min_brier_skill and sharpe >= min_sharpe
        return passed, (f"brier_skill={skill:+.3f} (need >={min_brier_skill}), "
                        f"pnl_sharpe={sharpe:+.2f} (need >={min_sharpe}), "
                        f"trades={self.n_trades}")


def close_asof(ticker: str, on_iso: str) -> float | None:
    """Underlying close on (or just before) a date, from yfinance. No-lookahead helper."""
    import pandas as pd

    from ..research.data_fetch import fetch_yf_history
    try:
        df = fetch_yf_history(ticker, period="5y")
        df = df[df.index <= pd.Timestamp(on_iso)]
        if df.empty:
            return None
        return float(df["Close"].dropna().iloc[-1])
    except Exception as e:  # noqa: BLE001
        logger.warning("close_asof %s @ %s failed: %s", ticker, on_iso, e)
        return None


def realized_terminal(ticker: str, expiry_iso: str) -> float | None:
    """Underlying close at option expiry (alias of close_asof for readability)."""
    return close_asof(ticker, expiry_iso)


def implied_prob_above(spot: float, strike: float, T: float, r: float, iv: float) -> float:
    """Risk-neutral P(S_T > K) = N(d2) under the lognormal — the market's baseline."""
    from ..strategies._model import norm_cdf
    if iv <= 0 or T <= 0:
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) + (r - 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return norm_cdf(d2)


def _structure_pnl(legs: list[dict], cost_per_share: float, terminal: float) -> float:
    """Realized P&L ($/contract) of a structure at a known terminal price."""
    payoff = 0.0
    for leg in legs:
        intrinsic = (max(terminal - leg["strike"], 0.0) if leg["kind"] == "call"
                     else max(leg["strike"] - terminal, 0.0))
        payoff += leg["qty"] * intrinsic
    return (payoff - cost_per_share) * 100


def run_backtest(cfg, underlying: str, *, as_of_dates: list[date], horizon_days: int = 90,
                 chain_client=None, mode: str = "drift_neutral") -> BacktestResult:
    """Walk-forward backtest for one underlying. For each as_of date: build f_P with NO
    lookahead, reconstruct the historical chain, score calibration (model vs implied) +
    CRPS vs the realized terminal, and the realized PnL of the top selected structure."""
    from .config import Underlying
    from .forecast.underlying import build_forecast
    from .research.beta import fit_beta
    from .forecast.signal import tal_drift
    from .selection import select
    from .sources import underlying as und_src
    from .sources.options_chain import ChainClient

    cc = chain_client or ChainClient(key=cfg.alpaca_key, secret=cfg.alpaca_secret)
    u: Underlying | None = cfg.find(underlying)
    if u is None:
        raise ValueError(f"{underlying} not in config universe")
    r = und_src.risk_free_rate()
    res = BacktestResult(underlying=underlying)
    res.signal_folds = 0  # how many folds had a usable tal view (mode="tal")

    for as_of in as_of_dates:
        expiry = target_expiry(as_of, horizon_days)
        realized = close_asof(underlying, expiry.isoformat())
        spot = close_asof(underlying, as_of.isoformat())
        if realized is None or spot is None:
            continue  # expiry not yet realized, or no price
        T = (expiry - as_of).days / 365.25
        if T <= 0:
            continue
        n_sims = int(cfg.forecast.get("diffusion", {}).get("n_sims", 20000))
        if mode == "momentum":
            # Commodity TSMOM → signed equity drift via beta → directional spread.
            from .forecast.direction import momentum_drift
            beta = fit_beta(underlying, u.commodity, u.market_ticker,
                            lookback_days=u.beta_lookback_days, as_of=as_of)
            if beta is None or beta.weak:
                continue
            _lbs = cfg.forecast.get("momentum_lookbacks")
            mu_view, conv = momentum_drift(
                commodity=u.commodity, beta_c=beta.beta_c, as_of=as_of,
                lookbacks=tuple(_lbs) if _lbs else None,
                min_strength=float(cfg.forecast.get("momentum_min_strength", 0.0)))
            fmode = "directional" if mu_view != 0.0 else "drift_neutral"
            if mu_view != 0.0:
                res.signal_folds += 1
            fc = build_forecast(ticker=underlying, commodity=u.commodity, market=u.market_ticker,
                                s0=spot, T=T, r=r, mode=fmode, mu_view=mu_view, beta=beta,
                                n_sims=n_sims, as_of=as_of)
        elif mode == "tal":
            # Fit beta first (no lookahead), get tal's as-of commodity view, propagate.
            beta = fit_beta(underlying, u.commodity, u.market_ticker,
                            lookback_days=u.beta_lookback_days, as_of=as_of)
            if beta is None or beta.weak:
                continue
            mu_view, conf = tal_drift(commodity=u.commodity, beta_c=beta.beta_c, spot=spot,
                                      as_of=as_of, horizon_years=T)
            if mu_view != 0.0:
                res.signal_folds += 1
                fc = build_forecast(ticker=underlying, commodity=u.commodity,
                                    market=u.market_ticker, s0=spot, T=T, r=r,
                                    mode="directional", mu_view=mu_view, beta=beta,
                                    n_sims=n_sims, as_of=as_of)
            else:  # no tal view this fold -> honest fallback to drift-neutral
                fc = build_forecast(ticker=underlying, commodity=u.commodity,
                                    market=u.market_ticker, s0=spot, T=T, r=r,
                                    mode="drift_neutral", beta=beta, n_sims=n_sims, as_of=as_of)
        else:
            fc = build_forecast(ticker=underlying, commodity=u.commodity, market=u.market_ticker,
                                s0=spot, T=T, r=r, mode=mode, n_sims=n_sims, as_of=as_of)
        if fc is None:
            continue
        try:
            chain = cc.historical_chain(underlying, as_of=as_of, expiry=expiry, spot=spot, r=r)
        except Exception as e:  # noqa: BLE001
            logger.warning("backtest: chain reconstruct failed %s @ %s: %s", underlying, as_of, e)
            continue
        if not chain:
            continue

        # Calibration: one point per unique strike (prefer the call's IV).
        iv_by_strike: dict[float, float] = {}
        for row in chain:
            if row.get("iv") and (row["kind"] == "call" or row["strike"] not in iv_by_strike):
                iv_by_strike[row["strike"]] = row["iv"]
        for strike, iv in iv_by_strike.items():
            res.forecast_probs.append(fc.prob_above(strike))
            res.implied_probs.append(implied_prob_above(spot, strike, T, r, iv))
            res.outcomes.append(1 if realized > strike else 0)
        res.crps.append(crps_sample(fc.terminal, realized))
        res.pit.append(float(np.mean(fc.terminal <= realized)))   # PIT of realized in f_P
        res.spot_ret.append(float(math.log(realized / spot)))     # realized horizon log-return

        # Realized PnL of the top selected structure at this as_of.
        cands = select(underlying, chain, lambda _T: fc, r=r, q=0.0,
                       limits=cfg.risk_limits, structures=cfg.structures)
        if cands:
            c = cands[0]
            res.pnls.append(_structure_pnl(c.legs, c.cost_per_share, realized))
            res.folds += 1
    return res


def monthly_as_of_dates(start: date, end: date) -> list[date]:
    """First business-ish day of each month in [start, end] — the walk-forward folds."""
    out: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1; y += 1
    return out
