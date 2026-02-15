from __future__ import annotations

import uuid
from typing import Any, List, Optional, Tuple

from selectools import Agent, AgentConfig
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool as tool_decorator
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats


class MockNativeToolProvider(LocalProvider):
    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        print(f"DEBUG: complete called with {len(messages)} messages")
        print(f"DEBUG: messages type: {type(messages)}")
        if len(messages) > 0:
            print(f"DEBUG: last message type: {type(messages[-1])}")
            print(f"DEBUG: last message value: {messages[-1]}")

        last_msg = messages[-1]

        # If user asks for weather tool, simulate tool call
        if last_msg.role == Role.USER and "weather" in last_msg.content:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="get_weather",
                            parameters={"location": "London"},
                            id=f"call_{uuid.uuid4().hex}",
                        )
                    ],
                ),
                UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="mock",
                    provider="mock",
                ),
            )

        # If last message is tool result, answer
        if last_msg.role == Role.TOOL:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content=f"The weather is {last_msg.content}",
                ),
                UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="mock",
                    provider="mock",
                ),
            )

        return (
            Message(role=Role.ASSISTANT, content="I don't know."),
            UsageStats(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                cost_usd=0.0,
                model="mock",
                provider="mock",
            ),
        )


@tool_decorator(description="Get weather for location")
def get_weather(location: str) -> str:
    return "Sunny"


def test_native_tool_calling() -> None:
    print("Starting native tool calling test...")
    provider = MockNativeToolProvider()

    agent = Agent(config=AgentConfig(), provider=provider, tools=[get_weather])

    try:
        response = agent.run([Message(role=Role.USER, content="What is the weather in London?")])
        print(f"Final Response: {response}")
        assert "Sunny" in response.message.content
        print("Test PASSED!")
    except Exception as e:
        print(f"Test FAILED with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_native_tool_calling()
