"""Hosted OpenAI-compatible client for Mercury (Inception Labs).

⚠️ HOSTED INFERENCE — a sanctioned exception to the local-only rule
(`CLAUDE.md`), used ONLY by the `mercury_ensemble` strategy. See
`docs/mercury-ensemble-calibration.md` §0.

Subclasses `LocalLLM` for the messages plumbing but overrides `json_completion`
to drop the Ollama-specific `options` extra_body, which Inception rejects.
"""

from __future__ import annotations

import os

from .client import LocalLLM

MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_MERCURY_MODEL = "mercury-2"


class MercuryLLM(LocalLLM):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        super().__init__(
            model=model or DEFAULT_MERCURY_MODEL,
            base_url=MERCURY_BASE_URL,
            api_key=api_key or os.environ["INCEPTION_API_KEY"],
            timeout=timeout,
        )
        # The ensemble fires N concurrent calls per group and trips Mercury's rate
        # limit; the SDK default of 2 retries is too few. Retry generously — the SDK
        # backs off exponentially and honours Retry-After on its own.
        self.client = self.client.with_options(max_retries=8)

    def json_completion(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
