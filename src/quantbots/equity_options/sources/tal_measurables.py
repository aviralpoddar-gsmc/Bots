"""tal coverage-package measurables — STUB SEAM (off by default).

The approved design fuses tal's numeric measurables + LLM-extracted text into the
forecast. That channel is **auth-blocked** today (tal /measurables/batch is local-only;
gcloud + gh scope are interactive-auth blockers — see memory `doppler-tal-scope` /
`optical-coverage-consolidated`). So this is a deliberate seam: the interface exists,
the forecast can fuse it the moment access lands, but it raises clearly until then and
is never required (config `forecast.tal_enabled` defaults to false).
"""

from __future__ import annotations

from .base import Observation


class TalMeasurablesSource:
    """Placeholder for the tal measurables feed. Enable via config once access lands."""

    name = "tal_measurables"

    def __init__(self, **params: object):
        self.params = params

    def fetch(self, tickers: list[str] | None = None) -> list[Observation]:  # noqa: ARG002
        raise NotImplementedError(
            "tal measurables access is not wired up (auth-blocked). Resolve tal "
            "/measurables/batch access (see memory doppler-tal-scope), then implement "
            "fetch() here and set forecast.tal_enabled: true in config/equity_options.yaml."
        )
