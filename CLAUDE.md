# Bots — Arbitrage Bots for Internal Manifold Clone

## What's being built here

Arbitrage bots that trade on the company's internal Manifold-clone prediction market platform. The goal is to make the markets more efficient.

## Platform context

- The company runs a Manifold clone with **~43,000 markets** (including commodity prices and other event markets).
- **60–70 bots** are already running and training on the platform.
- The platform source/code lives separately at `/Users/mikhail/manifold/` (not in this repo).

## Hard constraint: local compute only

**All training and inference must use locally-running models** — LLMs, VLMs, whatever architecture fits the task. No hosted/online inference APIs (OpenAI, Anthropic, Gemini, etc.) until the bots are demonstrably profitable.

- Once profitability is achieved, online compute can be reconsidered.
- Until then: assume Ollama, llama.cpp, vLLM, local fine-tuning, local RL, etc.
- When suggesting tools, models, or architectures, **flag anything that requires hosted inference** so it can be ruled out.

## Scope notes

- Improving the local LLMs used to train the bots is explicitly in-scope (fine-tuning, distillation, RL, prompt/agent design, etc.) — not just bot trading logic on top of off-the-shelf models.
- Any local model family is fair game as long as it runs on local hardware.
