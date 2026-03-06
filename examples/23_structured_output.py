#!/usr/bin/env python3
"""
Structured Output — Get typed, validated responses from the LLM.

Demonstrates:
  1. Pydantic BaseModel as response_format
  2. Dict JSON Schema as response_format
  3. Auto-retry on validation failure
  4. Using result.parsed for typed access

No API key needed — uses a mock provider that returns JSON.

Prerequisites: pip install selectools pydantic
Run: python examples/23_structured_output.py
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.types import AgentResult
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Mock provider that returns JSON responses
# ---------------------------------------------------------------------------


class JSONProvider:
    """Provider that returns a predetermined JSON response."""

    name = "json-mock"
    supports_streaming = False
    supports_async = True

    def __init__(self, json_text: str) -> None:
        self.json_text = json_text
        self.call_count = 0

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        self.call_count += 1
        return (
            Message(role=Role.ASSISTANT, content=self.json_text),
            UsageStats(100, 50, 150, 0.001, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class RetryProvider:
    """Provider that returns invalid JSON first, then valid JSON on retry."""

    name = "retry-mock"
    supports_streaming = False
    supports_async = True

    def __init__(self, invalid_response: str, valid_response: str) -> None:
        self.responses = [invalid_response, valid_response]
        self.call_count = 0

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return (
            Message(role=Role.ASSISTANT, content=self.responses[idx]),
            UsageStats(100, 50, 150, 0.001, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------


class TicketClassification(BaseModel):
    intent: Literal["billing", "support", "sales", "cancel"]
    confidence: float
    priority: Literal["low", "medium", "high"]
    summary: str


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float
    keywords: List[str]


# ---------------------------------------------------------------------------
# Tools (not used for structured output, but required by Agent)
# ---------------------------------------------------------------------------


@tool(description="Placeholder tool for classification")
def classify(text: str) -> str:
    return f"Classified: {text}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("  Structured Output Demo")
    print("=" * 70)

    tools = [classify]

    # --- Step 1: Pydantic BaseModel as response_format ---
    print("\n--- Step 1: Pydantic BaseModel as response_format ---\n")

    json_response = '{"intent": "cancel", "confidence": 0.95, "priority": "high", "summary": "Customer wants to cancel subscription"}'
    provider = JSONProvider(json_response)
    agent = Agent(
        tools=tools,
        provider=provider,
        config=AgentConfig(max_iterations=1),
    )

    result = agent.ask(
        "I want to cancel my subscription immediately",
        response_format=TicketClassification,
    )

    print(f"  result.parsed = {result.parsed}")
    print(f"  type(result.parsed) = {type(result.parsed).__name__}")
    print(f"  result.parsed.intent = {result.parsed.intent}")
    print(f"  result.parsed.confidence = {result.parsed.confidence}")
    print(f"  result.parsed.priority = {result.parsed.priority}")
    print(f"  result.content (raw) = {result.content[:60]}...")
    assert isinstance(result.parsed, TicketClassification)
    assert result.parsed.intent == "cancel"
    print("\n  PASS: Pydantic model validated and accessible via result.parsed\n")

    # --- Step 2: Dict JSON Schema as response_format ---
    print("--- Step 2: Dict JSON Schema as response_format ---\n")

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "score": {"type": "number"},
        },
        "required": ["sentiment", "score"],
    }

    json_response_2 = '{"sentiment": "positive", "score": 0.87}'
    provider_2 = JSONProvider(json_response_2)
    agent_2 = Agent(
        tools=tools,
        provider=provider_2,
        config=AgentConfig(max_iterations=1),
    )

    result_2 = agent_2.ask("I love this product!", response_format=schema)

    print(f"  result.parsed = {result_2.parsed}")
    print(f"  type(result.parsed) = {type(result_2.parsed).__name__}")
    assert isinstance(result_2.parsed, dict)
    assert result_2.parsed["sentiment"] == "positive"
    print("\n  PASS: Dict schema returns a plain dict\n")

    # --- Step 3: Auto-retry on validation failure ---
    print("--- Step 3: Auto-retry on validation failure ---\n")

    invalid_json = "Sure, here's the classification: not valid json"
    valid_json = '{"intent": "billing", "confidence": 0.80, "priority": "medium", "summary": "Billing inquiry"}'

    retry_provider = RetryProvider(invalid_json, valid_json)
    agent_3 = Agent(
        tools=tools,
        provider=retry_provider,
        config=AgentConfig(max_iterations=3),
    )

    result_3 = agent_3.ask(
        "Why was I charged twice?",
        response_format=TicketClassification,
    )

    print(f"  Provider was called {retry_provider.call_count} times")
    print(f"  result.parsed = {result_3.parsed}")
    assert isinstance(result_3.parsed, TicketClassification)
    assert result_3.parsed.intent == "billing"
    assert retry_provider.call_count == 2
    print("\n  PASS: First call failed, auto-retried, second call succeeded\n")

    # --- Step 4: Structured output with fenced code block ---
    print("--- Step 4: JSON inside fenced code block ---\n")

    fenced = (
        '```json\n{"sentiment": "negative", "score": 0.2, "keywords": ["broken", "terrible"]}\n```'
    )
    provider_4 = JSONProvider(fenced)
    agent_4 = Agent(
        tools=tools,
        provider=provider_4,
        config=AgentConfig(max_iterations=1),
    )

    result_4 = agent_4.ask("This product is terrible", response_format=SentimentResult)

    print(f"  result.parsed = {result_4.parsed}")
    assert isinstance(result_4.parsed, SentimentResult)
    assert result_4.parsed.sentiment == "negative"
    assert "broken" in result_4.parsed.keywords
    print("\n  PASS: JSON extracted from fenced code block\n")

    print("=" * 70)
    print("  All structured output tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
