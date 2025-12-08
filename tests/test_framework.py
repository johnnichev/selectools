"""
Test Suite for the selectools library.

These tests avoid real API calls by using fake providers.
"""

from __future__ import annotations

import json
import os
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import selectools
from agent import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.cli import _default_tools, build_parser, run_agent
from selectools.memory import ConversationMemory
from selectools.parser import ToolCallParser
from selectools.providers.anthropic_provider import AnthropicProvider
from selectools.providers.base import ProviderError
from selectools.providers.gemini_provider import GeminiProvider
from selectools.providers.stubs import LocalProvider
from selectools.tools import ToolRegistry


class FakeProvider:
    """Minimal provider stub that returns queued responses."""

    name = "fake"
    supports_streaming = False
    supports_async = False

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def complete(
        self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000, timeout=None
    ):  # noqa: D401
        from selectools import UsageStats

        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            model=model or "fake",
            provider="fake",
        )
        return response, usage


class FakeStreamingProvider(FakeProvider):
    """Provider that yields streaming chunks before returning a final string."""

    supports_streaming = True

    def __init__(self, stream_chunks, final_response):
        super().__init__(responses=[final_response])
        self.stream_chunks = stream_chunks

    def stream(
        self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000, timeout=None
    ):
        for chunk in self.stream_chunks:
            yield chunk


def test_role_enum():
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"
    assert Role.TOOL.value == "tool"


def test_message_creation_and_image_encoding():
    image_path = Path(__file__).resolve().parents[1] / "assets" / "environment.png"
    try:
        msg = Message(role=Role.USER, content="What's in this image?", image_path=str(image_path))
        assert msg.image_base64 is not None
    except FileNotFoundError:
        # Skip if asset is missing in CI
        pass

    msg = Message(role=Role.USER, content="Hello")
    formatted = msg.to_dict()
    assert formatted["role"] == "user"
    assert formatted["content"] == "Hello"


def test_tool_schema_and_validation():
    param = ToolParameter(name="query", param_type=str, description="Search query", required=True)
    tool = Tool(
        name="search",
        description="Search the web",
        parameters=[param],
        function=lambda query: query,
    )

    schema = tool.schema()
    assert schema["name"] == "search"
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["query"]["type"] == "string"

    # Valid params should not raise
    tool.validate({"query": "python"})

    # Invalid params should raise ToolValidationError
    from selectools.exceptions import ToolValidationError

    try:
        tool.validate({})
        assert False, "Should have raised ToolValidationError"
    except ToolValidationError as e:
        assert "Missing required parameter" in str(e)


def test_conversation_memory_basic():
    """Test basic ConversationMemory operations."""
    memory = ConversationMemory(max_messages=5)

    # Test empty memory
    assert len(memory) == 0
    assert memory.get_history() == []

    # Add single message
    msg1 = Message(role=Role.USER, content="Hello")
    memory.add(msg1)
    assert len(memory) == 1
    assert memory.get_history()[0].content == "Hello"

    # Add multiple messages
    msg2 = Message(role=Role.ASSISTANT, content="Hi there")
    msg3 = Message(role=Role.USER, content="How are you?")
    memory.add_many([msg2, msg3])
    assert len(memory) == 3

    # Test get_recent
    recent = memory.get_recent(2)
    assert len(recent) == 2
    assert recent[0].content == "Hi there"
    assert recent[1].content == "How are you?"

    # Test clear
    memory.clear()
    assert len(memory) == 0


def test_conversation_memory_max_messages():
    """Test that ConversationMemory enforces max_messages limit."""
    memory = ConversationMemory(max_messages=3)

    # Add more messages than the limit
    messages = [
        Message(role=Role.USER, content="Message 1"),
        Message(role=Role.ASSISTANT, content="Message 2"),
        Message(role=Role.USER, content="Message 3"),
        Message(role=Role.ASSISTANT, content="Message 4"),
        Message(role=Role.USER, content="Message 5"),
    ]
    memory.add_many(messages)

    # Should only keep the last 3 messages
    assert len(memory) == 3
    history = memory.get_history()
    assert history[0].content == "Message 3"
    assert history[1].content == "Message 4"
    assert history[2].content == "Message 5"


def test_conversation_memory_with_agent():
    """Test Agent integration with ConversationMemory."""
    memory = ConversationMemory(max_messages=10)
    tool = Tool(name="echo", description="Echo the input", parameters=[], function=lambda: "echoed")
    provider = FakeProvider(responses=["Done with that"])
    agent = Agent(tools=[tool], provider=provider, memory=memory)

    # First turn
    response1 = agent.run([Message(role=Role.USER, content="Hello")])
    assert response1.content == "Done with that"
    # Should have USER + ASSISTANT in memory
    assert len(memory) == 2
    assert memory.get_history()[0].role == Role.USER
    assert memory.get_history()[0].content == "Hello"
    assert memory.get_history()[1].role == Role.ASSISTANT

    # Second turn - memory should persist
    response2 = agent.run([Message(role=Role.USER, content="Hi again")])
    assert len(memory) == 4  # 2 previous + 2 new
    history = memory.get_history()
    assert history[0].content == "Hello"
    assert history[2].content == "Hi again"


def test_conversation_memory_persistence_across_turns():
    """Test that memory persists across multiple agent turns."""
    memory = ConversationMemory(max_messages=20)
    tool = Tool(name="counter", description="Count", parameters=[], function=lambda: "counted")
    provider = FakeProvider(responses=["Response 1", "Response 2", "Response 3"])
    agent = Agent(tools=[tool], provider=provider, memory=memory)

    # Multiple turns
    agent.run([Message(role=Role.USER, content="Turn 1")])
    agent.run([Message(role=Role.USER, content="Turn 2")])
    agent.run([Message(role=Role.USER, content="Turn 3")])

    # Should have all 6 messages (3 USER + 3 ASSISTANT)
    assert len(memory) == 6
    history = memory.get_history()
    assert history[0].content == "Turn 1"
    assert history[2].content == "Turn 2"
    assert history[4].content == "Turn 3"


def test_conversation_memory_to_dict():
    """Test ConversationMemory serialization."""
    memory = ConversationMemory(max_messages=5)
    memory.add(Message(role=Role.USER, content="Test"))

    data = memory.to_dict()
    assert data["max_messages"] == 5
    assert data["message_count"] == 1
    assert len(data["messages"]) == 1
    assert data["messages"][0]["content"] == "Test"


def test_conversation_memory_without_memory():
    """Test that Agent works without memory (backward compatibility)."""
    tool = Tool(name="test", description="Test", parameters=[], function=lambda: "ok")
    provider = FakeProvider(responses=["Done"])
    agent = Agent(tools=[tool], provider=provider)  # No memory

    # Should work as before
    response = agent.run([Message(role=Role.USER, content="Test")])
    assert response.content == "Done"
    assert agent.memory is None


def test_tool_decorator_and_registry_infer_schema():
    registry = ToolRegistry()

    @registry.tool(description="Greets a user")
    def greet(name: str, excited: bool = False) -> str:
        return f"Hello {name}" + ("!" if excited else "")

    tool_obj = registry.get("greet")
    assert tool_obj is not None
    names = {p.name: p for p in tool_obj.parameters}
    assert names["name"].required is True
    assert names["excited"].required is False


def test_parser_handles_fenced_blocks():
    parser = ToolCallParser()
    response = """
    Here you go:
    ```
    TOOL_CALL: {"tool_name": "search", "parameters": {"query": "docs"}}
    ```
    """
    result = parser.parse(response)
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search"
    assert result.tool_call.parameters["query"] == "docs"


def test_parser_handles_multiple_candidates_and_mixed_text():
    parser = ToolCallParser()
    response = """
    Some explanation text.
    TOOL_CALL: {"tool_name": "first", "parameters": {"x": 1}}
    More text.
    ```
    TOOL_CALL: {"tool_name": "second", "parameters": {"y": 2}}
    ```
    trailing text.
    """
    result = parser.parse(response)
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "first"


def test_parser_respects_size_limit():
    oversized_json = "{" + '"tool_name":"big","parameters":' + '"' + ("x" * 200) + '"' + "}"
    parser = ToolCallParser(max_payload_chars=10)
    result = parser.parse(f"TOOL_CALL: {oversized_json}")
    assert result.tool_call is None


def test_parser_handles_mixed_text_json():
    parser = ToolCallParser()
    response = "Random text TOOL_CALL: {'tool':'echo','params':{'text':'hi'}} trailing text"
    result = parser.parse(response)
    assert result.tool_call is not None
    assert result.tool_call.parameters["text"] == "hi"


def test_agent_executes_tool_and_returns_final_message():
    def add(a: int, b: int) -> str:
        return json.dumps({"sum": a + b})

    add_tool = Tool(
        name="add",
        description="Add two integers",
        parameters=[
            ToolParameter(name="a", param_type=int, description="first"),
            ToolParameter(name="b", param_type=int, description="second"),
        ],
        function=add,
    )

    provider = FakeProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "add", "parameters": {"a": 2, "b": 3}}',
            "The sum is 5.",
        ]
    )

    agent = Agent(
        tools=[add_tool],
        provider=provider,
        config=AgentConfig(max_iterations=3, model="fake-model"),
    )

    result = agent.run([Message(role=Role.USER, content="Add 2 and 3")])
    assert provider.calls == 2
    assert "5" in result.content


def test_agent_streaming_handler_and_fallback():
    captured = []

    def handler(chunk: str):
        captured.append(chunk)

    provider = FakeStreamingProvider(
        stream_chunks=["Hello", " ", "world"], final_response="Hello world"
    )
    noop_tool = Tool(
        name="noop",
        description="no-op",
        parameters=[],
        function=lambda: "",
    )
    agent = Agent(
        tools=[noop_tool],
        provider=provider,
        config=AgentConfig(max_iterations=1, model="fake-model", stream=True),
    )
    result = agent.run([Message(role=Role.USER, content="hi")], stream_handler=handler)
    assert result.content == "Hello world"
    assert "".join(captured) == "Hello world"


def test_agent_retries_on_provider_error():
    class FlakyProvider(FakeProvider):
        def complete(
            self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000, timeout=None
        ):  # noqa: D401
            if self.calls == 0:
                self.calls += 1
                raise ProviderError("temporary failure")
            return super().complete(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

    provider = FlakyProvider(responses=["All good"])
    noop_tool = Tool(
        name="noop",
        description="no-op",
        parameters=[],
        function=lambda: "",
    )
    agent = Agent(
        tools=[noop_tool],
        provider=provider,
        config=AgentConfig(max_iterations=1, model="fake", max_retries=1, retry_backoff_seconds=0),
    )
    result = agent.run([Message(role=Role.USER, content="retry?")])
    assert "All good" in result.content
    assert provider.calls == 2


def test_local_provider_streams_tokens():
    provider = LocalProvider()
    chunks = list(
        provider.stream(
            model="local",
            system_prompt="system",
            messages=[Message(role=Role.USER, content="hi")],
        )
    )
    assert any("hi" in chunk for chunk in chunks)


def test_anthropic_and_gemini_require_api_keys():
    from selectools.exceptions import ProviderConfigurationError

    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    try:
        AnthropicProvider()
        assert False, "AnthropicProvider should require an API key"
    except (ProviderError, ProviderConfigurationError):
        pass

    try:
        GeminiProvider()
        assert False, "GeminiProvider should require an API key"
    except (ProviderError, ProviderConfigurationError):
        pass


def test_anthropic_provider_with_mocked_client():
    """Test Anthropic provider using a mocked anthropic package."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    fake_resp_block = types.SimpleNamespace(text="hello anthropic")

    class FakeUsage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 5

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            if kwargs.get("stream"):
                event = types.SimpleNamespace(
                    type="content_block_delta", delta=types.SimpleNamespace(text="hello anthropic")
                )
                return [event]
            return types.SimpleNamespace(content=[fake_resp_block], usage=FakeUsage())

    class FakeAnthropicClient:
        def __init__(self, api_key=None, base_url=None):
            self.messages = FakeMessages()

    class FakeAsyncAnthropicClient:
        def __init__(self, api_key=None, base_url=None):
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(
        Anthropic=FakeAnthropicClient, AsyncAnthropic=FakeAsyncAnthropicClient
    )
    sys.modules["anthropic"] = fake_module
    try:
        provider = AnthropicProvider()
        result, usage = provider.complete(
            model="claude-mock",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert "hello anthropic" in result

        streamed = "".join(
            provider.stream(
                model="claude-mock",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert "hello anthropic" in streamed
    finally:
        sys.modules.pop("anthropic", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_gemini_provider_with_mocked_client():
    """Test Gemini provider using a mocked google-genai package."""
    os.environ["GEMINI_API_KEY"] = "test-key"

    class FakeStreamChunk:
        def __init__(self, text):
            self.text = text

    class FakeUsageMetadata:
        def __init__(self):
            self.prompt_token_count = 10
            self.candidates_token_count = 5

    class FakeResponse:
        def __init__(self, text):
            self.text = text
            self.usage_metadata = FakeUsageMetadata()

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            return FakeResponse("gemini-response")

        def generate_content_stream(self, model, contents, config=None):
            yield FakeStreamChunk("stream-1")
            yield FakeStreamChunk("stream-2")

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    # Mock types module
    class FakeTypes:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                pass

        class Part:
            def __init__(self, text=None, inline_data=None):
                self.text = text

        class Content:
            def __init__(self, role=None, parts=None):
                self.role = role
                self.parts = parts

        class Blob:
            def __init__(self, mime_type=None, data=None):
                pass

    # Create mock module structure
    google_pkg = types.ModuleType("google")
    genai_pkg = types.ModuleType("google.genai")
    genai_types_pkg = types.ModuleType("google.genai.types")

    genai_pkg.Client = FakeClient
    genai_types_pkg.GenerateContentConfig = FakeTypes.GenerateContentConfig
    genai_types_pkg.Part = FakeTypes.Part
    genai_types_pkg.Content = FakeTypes.Content
    genai_types_pkg.Blob = FakeTypes.Blob
    genai_pkg.types = genai_types_pkg

    google_pkg.genai = genai_pkg
    sys.modules["google"] = google_pkg
    sys.modules["google.genai"] = genai_pkg
    sys.modules["google.genai.types"] = genai_types_pkg

    try:
        provider = GeminiProvider()
        result, usage = provider.complete(
            model="gemini-mock",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert "gemini-response" in result

        stream_result = "".join(
            provider.stream(
                model="gemini-mock",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert "stream-1" in stream_result
    finally:
        sys.modules.pop("google", None)
        sys.modules.pop("google.genai", None)
        sys.modules.pop("google.genai.types", None)
        os.environ.pop("GEMINI_API_KEY", None)


def test_cli_streaming_with_local_provider():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--provider",
            "local",
            "--model",
            "local",
            "--prompt",
            "hello world",
            "--stream",
            "--max-iterations",
            "1",
            "--max-tokens",
            "10",
        ]
    )
    tools: Dict[str, Tool] = _default_tools()
    buf = StringIO()
    with redirect_stdout(buf):
        run_agent(args, tools or {})  # run_agent fills tools default internally if empty
    output = buf.getvalue()
    assert "local provider" in output.lower()


@pytest.mark.asyncio
async def test_async_agent_basic():
    """Test basic async agent execution."""
    import asyncio

    @selectools.tool(description="Async echo tool")
    async def async_echo(text: str) -> str:
        await asyncio.sleep(0.01)  # Simulate async work
        return f"async_echoed: {text}"

    agent = Agent(tools=[async_echo], provider=LocalProvider())

    response = await agent.arun([Message(role=Role.USER, content="Test async")])
    assert response.role == Role.ASSISTANT
    assert "async_echoed" in response.content or "local provider" in response.content.lower()


@pytest.mark.asyncio
async def test_async_tool_execution():
    """Test that both sync and async tools work with async agent."""
    import asyncio

    def sync_func(x: int) -> str:
        return f"sync:{x}"

    async def async_func(x: int) -> str:
        await asyncio.sleep(0.01)
        return f"async:{x}"

    sync_tool = Tool(
        name="sync",
        description="Sync tool",
        parameters=[ToolParameter(name="x", param_type=int, description="Number")],
        function=sync_func,
    )

    async_tool = Tool(
        name="async",
        description="Async tool",
        parameters=[ToolParameter(name="x", param_type=int, description="Number")],
        function=async_func,
    )

    # Test individual tool execution
    assert sync_tool.is_async is False
    assert async_tool.is_async is True

    # Test async execution of both
    sync_result = await sync_tool.aexecute({"x": 5})
    assert sync_result == "sync:5"

    async_result = await async_tool.aexecute({"x": 10})
    assert async_result == "async:10"


@pytest.mark.asyncio
async def test_async_with_memory():
    """Test async agent with conversation memory."""
    from selectools.memory import ConversationMemory

    memory = ConversationMemory(max_messages=10)

    def simple_tool(x: int) -> str:
        return str(x * 2)

    tool = Tool(
        name="double",
        description="Double a number",
        parameters=[ToolParameter(name="x", param_type=int, description="Number")],
        function=simple_tool,
    )

    agent = Agent(tools=[tool], provider=LocalProvider(), memory=memory)

    # First turn
    response1 = await agent.arun([Message(role=Role.USER, content="Hello 1")])
    assert len(memory) >= 1

    # Second turn
    response2 = await agent.arun([Message(role=Role.USER, content="Hello 2")])
    assert len(memory) >= 2


@pytest.mark.asyncio
async def test_async_provider_fallback():
    """Test that agent falls back to sync when provider doesn't support async."""
    # LocalProvider doesn't have async support
    provider = LocalProvider()
    assert not getattr(provider, "supports_async", False)

    def simple_tool() -> str:
        return "done"

    tool = Tool(
        name="test",
        description="Test tool",
        parameters=[],
        function=simple_tool,
    )

    agent = Agent(tools=[tool], provider=provider)
    response = await agent.arun([Message(role=Role.USER, content="test")])
    assert response.role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_async_multiple_iterations():
    """Test async agent with multiple tool call iterations."""
    call_count = 0

    @selectools.tool(description="Counter tool")
    async def counter() -> str:
        nonlocal call_count
        call_count += 1
        return f"call_{call_count}"

    agent = Agent(tools=[counter], provider=LocalProvider(), config=AgentConfig(max_iterations=3))

    response = await agent.arun([Message(role=Role.USER, content="Count")])
    assert response.role == Role.ASSISTANT


def run_async_test(test_func):
    """Helper to run async tests."""
    import asyncio

    asyncio.run(test_func())


if __name__ == "__main__":
    # Simple runner for environments without pytest
    all_tests = [
        test_role_enum,
        test_message_creation_and_image_encoding,
        test_tool_schema_and_validation,
        test_conversation_memory_basic,
        test_conversation_memory_max_messages,
        test_conversation_memory_with_agent,
        test_conversation_memory_persistence_across_turns,
        test_conversation_memory_to_dict,
        test_conversation_memory_without_memory,
        test_tool_decorator_and_registry_infer_schema,
        test_parser_handles_fenced_blocks,
        test_parser_handles_multiple_candidates_and_mixed_text,
        test_parser_respects_size_limit,
        test_parser_handles_mixed_text_json,
        test_agent_executes_tool_and_returns_final_message,
        test_agent_streaming_handler_and_fallback,
        test_agent_retries_on_provider_error,
        test_local_provider_streams_tokens,
        test_anthropic_and_gemini_require_api_keys,
        test_anthropic_provider_with_mocked_client,
        test_gemini_provider_with_mocked_client,
        test_cli_streaming_with_local_provider,
    ]

    # Async tests run separately
    async_tests = [
        test_async_agent_basic,
        test_async_tool_execution,
        test_async_with_memory,
        test_async_provider_fallback,
        test_async_multiple_iterations,
    ]

    failures = 0
    for test in all_tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as exc:  # noqa: BLE001
            failures += 1
            print(f"✗ {test.__name__}: {exc}")

    # Run async tests
    for test in async_tests:
        try:
            run_async_test(test)
            print(f"✓ {test.__name__}")
        except AssertionError as exc:  # noqa: BLE001
            failures += 1
            print(f"✗ {test.__name__}: {exc}")

    if failures:
        raise SystemExit(1)
