#!/usr/bin/env python3
"""
LiteLLM Provider -- instant access to 100+ models via litellm.

Offline-safe: the demo below injects a fake `litellm` module so it runs
without litellm installed or any API key. A real call is shown at the end,
guarded behind GROQ_API_KEY.

Requires (for live use): pip install selectools[litellm]
Run: python examples/101_litellm_provider.py
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

print("=== LiteLLM Provider Example ===\n")

print(
    """
from selectools import Agent, AgentConfig
from selectools.providers import LiteLLMProvider

# Any litellm "provider/model" identifier works:
provider = LiteLLMProvider(model="deepseek/deepseek-chat")
provider = LiteLLMProvider(model="groq/llama-3.1-70b")
provider = LiteLLMProvider(model="bedrock/anthropic.claude-3-sonnet")

# Optional: explicit key, gateway override, extra litellm kwargs
provider = LiteLLMProvider(
    model="groq/llama-3.1-70b",
    api_key="gsk_...",               # else litellm reads GROQ_API_KEY etc.
    api_base="https://my-proxy/v1",  # optional proxy/gateway
    drop_params=True,                # forwarded to every litellm call
)

# Use like any other provider (set AgentConfig.model to the same id)
agent = Agent(
    tools=[my_tool],
    provider=provider,
    config=AgentConfig(model="groq/llama-3.1-70b"),
)
result = agent.run("Hello from the long tail!")
"""
)

# ----------------------------------------------------------------------
# Offline demo: fake litellm module returning an OpenAI-shaped response
# ----------------------------------------------------------------------

print("--- Offline demo (mocked litellm) ---")

fake_litellm = types.ModuleType("litellm")


def _completion(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello! (routed through litellm)", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20),
    )


fake_litellm.completion = _completion  # type: ignore[attr-defined]
fake_litellm.acompletion = _completion  # type: ignore[attr-defined]
fake_litellm.cost_per_token = lambda **kw: (0.000012, 0.000008)  # type: ignore[attr-defined]
sys.modules.setdefault("litellm", fake_litellm)

from selectools.providers import LiteLLMProvider  # noqa: E402
from selectools.types import Message, Role  # noqa: E402

provider = LiteLLMProvider(model="groq/llama-3.1-70b")
message, usage = provider.complete(
    model="groq/llama-3.1-70b",
    system_prompt="You are a helpful assistant.",
    messages=[Message(role=Role.USER, content="Say hello.")],
)
print(f"Response: {message.content}")
print(f"Usage: {usage.total_tokens} tokens, ${usage.cost_usd:.6f} ({usage.provider})")

# ----------------------------------------------------------------------
# Live call (only runs when a real litellm install + API key are present)
# ----------------------------------------------------------------------

_mock_active = sys.modules["litellm"] is fake_litellm

if os.getenv("GROQ_API_KEY") and not _mock_active:
    print("\n--- Live call (GROQ_API_KEY detected) ---")
    live = LiteLLMProvider(model="groq/llama-3.1-70b")
    live_msg, live_usage = live.complete(
        model="groq/llama-3.1-70b",
        system_prompt="You are a helpful assistant.",
        messages=[Message(role=Role.USER, content="Say hello in five words.")],
    )
    print(f"Response: {live_msg.content}")
    print(f"Cost: ${live_usage.cost_usd:.6f}")
else:
    print("\nSet GROQ_API_KEY (and pip install selectools[litellm]) for a live call.")

print("Done!")
