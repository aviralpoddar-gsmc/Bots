"""RSS / Atom news headlines — keyless text feed.

Ingests headlines as text Observations (one per item). Turning that text into a
numeric trading signal is a later step (a local-LLM extractor); this source just
collects and caches the raw text, keyed by feed. Stdlib only (urllib + ElementTree),
so no extra dependency.

Configure in config/sources.yaml:

    - name: rss
      params:
        feeds:
          - { entity: REUTERS_COMMODITIES, url: "https://www.reuters.com/.../rss" }
          - { entity: EIA_NEWS, url: "https://www.eia.gov/rss/todayinenergy.xml" }
        max_items: 30
"""

from __future__ import annotations

import urllib.request
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

from .base import Observation, Source

# Atom namespace (RSS 2.0 uses no namespace on its elements).
_ATOM = "{http://www.w3.org/2005/Atom}"


def _text(el: ET.Element | None) -> str | None:
    return el.text.strip() if el is not None and el.text else None


class RSSSource(Source):
    name = "rss"

    def _fetch_one(self, entity: str, url: str, max_items: int) -> list[Observation]:
        req = urllib.request.Request(url, headers={"User-Agent": "quantbots/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read())

        out: list[Observation] = []
        # RSS 2.0: channel/item with <title>,<pubDate>. Atom: <entry> with <title>,<updated>.
        items = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
        now = datetime.now(UTC).isoformat()
        for item in items[:max_items]:
            title = _text(item.find("title")) or _text(item.find(f"{_ATOM}title"))
            if not title:
                continue
            pub = (
                _text(item.find("pubDate"))
                or _text(item.find(f"{_ATOM}updated"))
                or now
            )
            link = _text(item.find("link")) or ""
            out.append(
                Observation(
                    source=self.name,
                    entity=entity,
                    ts=pub,
                    text=title,
                    payload={"link": link, "feed": url},
                )
            )
        return out

    def fetch(self) -> list[Observation]:
        feeds = self.params.get("feeds") or []
        max_items = int(self.params.get("max_items", 30))
        out: list[Observation] = []
        for feed in feeds:
            try:
                out.extend(self._fetch_one(feed["entity"], feed["url"], max_items))
            except Exception:  # noqa: BLE001 - one bad feed shouldn't sink ingest
                continue
        return out
