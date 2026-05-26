"""The Source interface — one feed of external information.

Mirrors the Strategy pattern: lots of small, independent modules, each one a
`Source` subclass implementing `fetch() -> list[Observation]`. Sources only
*ingest and normalize*; they never reason or trade. Their output is cached in the
store's `observations` table and later consumed by strategies.

An Observation is the normalized unit:
- `value` for numeric feeds (a price, an index level, a macro print),
- `text`  for text feeds (a news headline),
- `entity` is the canonical key for *what* was observed, so different sources can
  describe the same underlying quantity (e.g. several feeds all keyed "WTI_OIL").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    source: str
    entity: str
    ts: str  # ISO-8601 of when the observation refers to
    value: float | None = None
    text: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entity": self.entity,
            "ts": self.ts,
            "value": self.value,
            "text": self.text,
            "payload": self.payload,
        }


class Source(ABC):
    #: Stable identifier, also the key in the source REGISTRY.
    name: str = "base"

    def __init__(self, **params: Any):
        #: Free-form params from config/sources.yaml (symbols, feed URLs, ...).
        self.params = params

    @abstractmethod
    def fetch(self) -> list[Observation]:
        """Pull the latest data and return it as normalized Observations.

        Must be side-effect free besides the network read — the caller persists
        the result. Raise on hard failures; return [] for "nothing new".
        """
