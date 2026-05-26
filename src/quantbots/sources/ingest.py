"""Ingestion runner: fetch each configured source and cache its observations.

Decoupled from trading on purpose — run this on its own schedule (cron / loop) so
data collection cadence is independent of when bots trade, and rate limits are
respected per source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import SourceConfig, load_sources
from ..store.db import Store
from . import get_source

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    by_source: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_source.values())


def ingest(store: Store, configs: list[SourceConfig] | None = None,
           only: str | None = None) -> IngestResult:
    configs = configs or load_sources()
    result = IngestResult()
    for cfg in configs:
        if not cfg.enabled or (only and cfg.name != only):
            continue
        try:
            source = get_source(cfg.name, **cfg.params)
            observations = source.fetch()
            n = store.upsert_observations(observations)
            result.by_source[cfg.name] = result.by_source.get(cfg.name, 0) + n
            logger.info("ingested %d observations from %s", n, cfg.name)
        except Exception as e:  # noqa: BLE001 - isolate one source's failure
            result.errors[cfg.name] = str(e)
            logger.warning("source %s failed: %s", cfg.name, e)
    return result
