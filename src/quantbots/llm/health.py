"""Ollama health probe + watchdog.

TAL learned this the hard way: Ollama can "wedge" — `/api/tags` still returns 200
while `/api/generate` returns "server busy, maximum pending requests exceeded" and
stays stuck. So a real health check must hit a generate endpoint, not just tags.

Recommended server env (set where Ollama runs):
    OLLAMA_NUM_PARALLEL=4
    OLLAMA_MAX_QUEUE=32

Stdlib only (urllib) so importing it costs nothing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# The Ollama native host (note: NOT the /v1 OpenAI path — these are raw endpoints).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def is_healthy(model: str | None = None, timeout: float = 30.0) -> bool:
    """True only if the server can actually generate (not just list tags)."""
    model = model or os.environ.get("QUANTBOTS_LLM_MODEL", "gemma2")
    body = json.dumps({"model": model, "prompt": "ping", "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read())
            return bool(payload.get("response"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def wait_until_healthy(model: str | None = None, attempts: int = 3, delay: float = 5.0) -> bool:
    """Probe a few times; returns False if still wedged. Wire a kill/restart of
    the Ollama process to a False result in your deploy supervisor."""
    import time

    for _ in range(attempts):
        if is_healthy(model):
            return True
        time.sleep(delay)
    return False
