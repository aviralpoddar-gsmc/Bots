"""The fence: equity_options must not import manifold, and live trading must refuse.

These are the non-negotiable safety invariants from equity_options/SAFETY.md.
"""

import ast
from pathlib import Path

import pytest

import quantbots.equity_options as eo
from quantbots.equity_options.config import DRY, PAPER
from quantbots.equity_options.execution.base import DryRunBroker
from quantbots.equity_options.execution.live import LiveBroker, LiveTradingRefused, make_broker

PKG = Path(eo.__file__).resolve().parent


def _imports(py: Path) -> list[str]:
    mods: list[str] = []
    for node in ast.walk(ast.parse(py.read_text())):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level and not node.module:
            mods.append("." * node.level)
    return mods


def test_no_manifold_import_anywhere():
    offenders = []
    for py in PKG.rglob("*.py"):
        for m in _imports(py):
            if "manifold" in m:
                offenders.append(f"{py.relative_to(PKG.parent)} -> {m}")
    assert not offenders, f"equity_options must not import manifold: {offenders}"


def test_live_broker_refuses_without_approval(monkeypatch):
    monkeypatch.delenv("EQUITY_OPTIONS_OWNER_APPROVAL", raising=False)
    with pytest.raises(LiveTradingRefused):
        LiveBroker()


def test_live_broker_refuses_even_with_approval_no_limits(monkeypatch, tmp_path):
    # Even with the approval flag, a missing risk-limits file must refuse.
    monkeypatch.setenv("EQUITY_OPTIONS_OWNER_APPROVAL", "1")
    with pytest.raises(LiveTradingRefused):
        LiveBroker(risk_limits_file=str(tmp_path / "nope.yaml"))


def test_make_broker_modes(monkeypatch):
    assert isinstance(make_broker(DRY), DryRunBroker)
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    from quantbots.equity_options.execution.alpaca import AlpacaPaperBroker
    assert isinstance(make_broker(PAPER), AlpacaPaperBroker)
