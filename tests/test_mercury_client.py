"""MercuryLLM is a hosted OpenAI-compatible client for Inception's Mercury.
It must point at the Inception endpoint and must NOT send Ollama's `options`
extra_body (which the local client injects and Mercury would reject)."""

from types import SimpleNamespace

from quantbots.llm.mercury import MercuryLLM


def _reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_points_at_inception_endpoint():
    llm = MercuryLLM(api_key="test-key")
    assert "inceptionlabs.ai" in str(llm.client.base_url)


def test_client_retries_rate_limits_with_backoff():
    # Hosted Mercury rate-limits the ensemble burst; the SDK default of 2 retries
    # is too few. The client must retry 429s generously (SDK backs off on its own).
    llm = MercuryLLM(api_key="test-key")
    assert llm.client.max_retries >= 6


def test_json_completion_omits_ollama_options():
    llm = MercuryLLM(model="mercury-2", api_key="test-key")
    recorded: dict = {}

    class FakeCompletions:
        def create(self, **kw):
            recorded.update(kw)
            return _reply('{"p50": 1}')

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    out = llm.json_completion(system="s", user="u", temperature=0.7)

    assert out == '{"p50": 1}'
    assert "extra_body" not in recorded  # no Ollama num_ctx options
    assert recorded["response_format"] == {"type": "json_object"}
    assert recorded["temperature"] == 0.7
    assert recorded["model"] == "mercury-2"
