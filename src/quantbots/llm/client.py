"""OpenAI-compatible client pointed at a LOCAL model endpoint.

LOCAL COMPUTE ONLY: the default base URL is localhost Ollama. Do not point this
at a hosted inference provider (OpenAI/Anthropic/Gemini cloud) — that's against
the project constraint until bots are demonstrably profitable. Ollama speaks the
OpenAI protocol natively, and a LiteLLM proxy (also local) can sit in front for
multi-model routing.

Requires the `llm` extra (`openai`).
"""

from __future__ import annotations

import os

# Default to local Ollama. QUANTBOTS_LLM_BASE_URL may override to another LOCAL
# host (e.g. a Mac Studio on the LAN) or a local LiteLLM proxy.
DEFAULT_BASE_URL = os.environ.get("QUANTBOTS_LLM_BASE_URL", "http://localhost:11434/v1")
# qwen3:32b won our local forecasting benchmark (best calibration: 71% coverage
# vs ideal ~80%; comparable accuracy to 8b). ~50s/call on M3 Ultra.
DEFAULT_MODEL = os.environ.get("QUANTBOTS_LLM_MODEL", "qwen3:32b")

# Ollama's context window defaults to 2048 and SILENTLY truncates, which breaks
# JSON mode (you get empty/partial JSON). Always set this high.
DEFAULT_NUM_CTX = 32768


class LocalLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
    ):
        from openai import OpenAI  # imported lazily so the extra is optional

        self.model = model or DEFAULT_MODEL
        self.num_ctx = num_ctx
        self.client = OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key or os.environ.get("QUANTBOTS_LLM_API_KEY", "ollama"),
        )

    def json_completion(self, system: str, user: str, temperature: float = 0.0) -> str:
        """One JSON-mode chat completion. Returns the raw content string."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            extra_body={"options": {"num_ctx": self.num_ctx}},  # critical for Ollama
        )
        return resp.choices[0].message.content or ""
