"""Loading bot definitions from config/bots.yaml and resolving secrets from env.

A bot's API key is NEVER stored in the yaml — only the *name* of the env var that
holds it (`account_env`), resolved at load time. Limits fall back to
`sizing.DEFAULT_LIMITS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .sizing import DEFAULT_LIMITS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(os.environ.get("QUANTBOTS_CONFIG", _REPO_ROOT / "config" / "bots.yaml"))


@dataclass
class BotConfig:
    name: str
    strategy: str
    account_env: str = "MANIFOLD_CLONE_API_KEY"
    enabled: bool = True
    limits: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        """The bot account's key, resolved from its env var at call time."""
        return os.environ.get(self.account_env)


def _merge_limits(raw: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_LIMITS)
    merged.update(raw or {})
    return merged


def load_bots(path: Path | str = DEFAULT_CONFIG) -> list[BotConfig]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    bots = []
    for entry in data.get("bots", []):
        bots.append(
            BotConfig(
                name=entry["name"],
                strategy=entry["strategy"],
                account_env=entry.get("account_env", "MANIFOLD_CLONE_API_KEY"),
                enabled=entry.get("enabled", True),
                limits=_merge_limits(entry.get("limits")),
                params=entry.get("params", {}),
            )
        )
    return bots


def load_bot(name: str, path: Path | str = DEFAULT_CONFIG) -> BotConfig:
    for bot in load_bots(path):
        if bot.name == name:
            return bot
    raise KeyError(f"No bot named {name!r} in {path}")
