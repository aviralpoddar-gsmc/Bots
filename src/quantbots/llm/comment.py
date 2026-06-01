"""Local-LLM trade-comment writer (qwen3 via Ollama, local only).

Turns a bot's STRUCTURED numbers (the strategy's explanation dict) into a concise,
human-readable rationale for the trade comment. The LLM is given ONLY the bot's own
figures and is instructed to rephrase, not invent — the trade *decision* is fully
deterministic; the LLM only writes prose. Any failure (LLM down, timeout) returns
None so the caller falls back to the deterministic explain() text.

Default model is qwen3:8b (fast, ~seconds) — comments aren't decisions, so we trade
the 32b's extra calibration for latency.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.S)

_SYSTEM = (
    "You are a quant trading bot writing a one-paragraph rationale for a bet you "
    "just placed on a prediction market. Use ONLY the numbers provided — never "
    "invent data, prices, or facts. Be specific and concise (2-4 sentences). State "
    "the data signal, the resulting fair value vs market, and why the bet direction "
    "follows. No preamble, no disclaimers, no markdown headers."
)


# Hosted comment backends (COMMENTS ONLY — alpha/data-processing stays local per
# CLAUDE.md). Maps a model name -> (base_url, env var holding the API key).
_HOSTED = {
    "mercury-2": ("https://api.inceptionlabs.ai/v1", "INCEPTION_API_KEY"),
}


def _complete(model: str, system: str, user: str) -> str:
    """One chat completion. Routes hosted models (e.g. mercury-2) to their cloud
    endpoint; everything else goes to the LOCAL Ollama client."""
    if model in _HOSTED:
        import os
        from openai import OpenAI
        base_url, key_env = _HOSTED[model]
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"{key_env} not set")
        client = OpenAI(base_url=base_url, api_key=key)
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    from .client import LocalLLM  # local Ollama (default)
    return LocalLLM(model=model).text_completion(user, temperature=0.2, system=system)


def generate_comment(*, bot: str, question: str, direction: str, amount: float,
                     detail: dict, model: str = "qwen3:8b") -> str | None:
    """Return an LLM-written rationale, or None to fall back to deterministic text."""
    facts = "; ".join(f"{k}={v}" for k, v in detail.items() if k != "reason" and v is not None)
    reason = detail.get("reason", "")
    user = (
        f"Bot: {bot}\nMarket: {question}\nBet: {direction} Ṁ{int(amount)}\n"
        f"Signal: {reason}\nNumbers: {facts}\n\n"
        "Write the rationale paragraph."
    )
    try:
        out = _complete(model, _SYSTEM, user)
    except Exception as e:  # LLM unavailable / timeout -> deterministic fallback
        logger.info("llm comment unavailable (%s); using deterministic explain", e)
        return None
    out = _THINK.sub("", out).strip()
    return out or None
