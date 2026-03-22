"""Pre-execution token estimation for agent runs.

Provides approximate token counts for system prompts, tool schemas,
and messages before calling the LLM. Uses tiktoken for OpenAI models
when available; falls back to a character-based heuristic for other
providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .tools.base import Tool
    from .types import Message


@dataclass
class TokenEstimate:
    """Breakdown of estimated token usage for the first iteration of an agent run.

    Attributes:
        system_tokens: Tokens in the system prompt.
        message_tokens: Tokens in conversation messages.
        tool_schema_tokens: Tokens in the tool JSON schemas.
        total_tokens: Sum of the above.
        context_window: Model's context window size (from model registry).
        remaining_tokens: context_window - total_tokens (may be negative).
        model: Model used for the estimate.
        method: Estimation method used ("tiktoken" or "heuristic").
    """

    system_tokens: int
    message_tokens: int
    tool_schema_tokens: int
    total_tokens: int
    context_window: int
    remaining_tokens: int
    model: str
    method: str


def _heuristic_count(text: str) -> int:
    """Estimate tokens using a character-based heuristic (~4 chars per token)."""
    return max(1, len(text) // 4) if text else 0


def _tiktoken_count(text: str, model: str) -> Optional[int]:
    """Try to count tokens using tiktoken. Returns None if unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model — try cl100k_base (GPT-4 family)
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    return len(enc.encode(text))


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate the token count for a string.

    Uses tiktoken for OpenAI-compatible models when installed;
    falls back to a ``len(text) // 4`` heuristic otherwise.

    Args:
        text: The text to estimate.
        model: Model name for tokenizer selection.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    exact = _tiktoken_count(text, model)
    if exact is not None:
        return exact
    return _heuristic_count(text)


def estimate_run_tokens(
    messages: List[Message],
    tools: List[Tool],
    system_prompt: str = "",
    model: str = "gpt-4o",
) -> TokenEstimate:
    """Estimate first-iteration token usage with a breakdown.

    Args:
        messages: Conversation history messages.
        tools: List of Tool objects whose schemas will be sent.
        system_prompt: The system prompt string.
        model: Model name for tokenizer and context window lookup.

    Returns:
        TokenEstimate with per-component breakdown.
    """
    # Determine estimation method
    method = "heuristic"
    try:
        import tiktoken  # noqa: F401

        method = "tiktoken"
    except ImportError:
        pass

    # System prompt tokens
    system_tokens = estimate_tokens(system_prompt, model)

    # Message tokens (content + role overhead ~4 tokens per message)
    message_tokens = 0
    for msg in messages:
        message_tokens += estimate_tokens(msg.content or "", model) + 4

    # Tool schema tokens
    tool_schema_tokens = 0
    for tool in tools:
        schema_json = json.dumps(tool.schema(), separators=(",", ":"))
        tool_schema_tokens += estimate_tokens(schema_json, model)

    total_tokens = system_tokens + message_tokens + tool_schema_tokens

    # Context window from model registry
    context_window = 0
    try:
        from .models import MODELS_BY_ID

        info = MODELS_BY_ID.get(model)
        if info:
            context_window = info.context_window
    except Exception:  # noqa: BLE001 # nosec B110
        pass

    return TokenEstimate(
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_schema_tokens=tool_schema_tokens,
        total_tokens=total_tokens,
        context_window=context_window,
        remaining_tokens=context_window - total_tokens if context_window else 0,
        model=model,
        method=method,
    )


__all__ = ["TokenEstimate", "estimate_tokens", "estimate_run_tokens"]
