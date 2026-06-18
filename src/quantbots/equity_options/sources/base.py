"""Re-export the parent Observation dataclass so the options store speaks the same
normalized unit as the rest of quantbots. (Importing sources.base is fine — the
fence only forbids importing `manifold`.)"""

from __future__ import annotations

from ...sources.base import Observation

__all__ = ["Observation"]
