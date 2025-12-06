"""
Test Suite for the toolcalling library.

These tests avoid real API calls by using fake providers.
"""

from __future__ import annotations

import json
import os
import sys
import types
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from toolcalling.tools import ToolRegistry
from toolcalling.examples.bbox import BBOX_MOCK_ENV, detect_bounding_box_impl
from toolcalling.parser import ToolCallParser
from toolcalling.providers.base import ProviderError
from toolcalling.providers.stubs import AnthropicProvider, GeminiProvider, LocalProvider
from toolcalling.cli import build_parser, run_agent, _default_tools


class FakeProvider:
    """Minimal provider stub that returns queued responses."""

    name = "fake"
    supports_streaming = False

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def complete(self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000, timeout=None):  # noqa: D401
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class FakeStreamingProvider(FakeProvider):
    """Provider that yields streaming chunks before returning a final string."""

    supports_streaming = True

    def __init__(self, stream_chunks, final_response):
        super().__init__(responses=[final_response])
        self.stream_chunks = stream_chunks

    def stream(self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000, timeout=None):
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
    tool = Tool(name="search", description="Search the web", parameters=[param], function=lambda query: query)

    schema = tool.schema()
    assert schema["name"] == "search"
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["query"]["type"] == "string"

    is_valid, error = tool.validate({"query": "python"})
    assert is_valid is True and error is None

    is_valid, error = tool.validate({})
    assert is_valid is False and "Missing required parameter" in error


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

    provider = FakeStreamingProvider(stream_chunks=["Hello", " ", "world"], final_response="Hello world")
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
        def complete(self, *, model, system_prompt, messages, temperature=0.0, max_tokens=1000, timeout=None):  # noqa: D401
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
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
    try:
        AnthropicProvider()
        assert False, "AnthropicProvider should require an API key"
    except ProviderError:
        pass

    try:
        GeminiProvider()
        assert False, "GeminiProvider should require an API key"
    except ProviderError:
        pass


def test_anthropic_provider_with_mocked_client():
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    fake_resp_block = types.SimpleNamespace(text="hello anthropic")

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            if kwargs.get("stream"):
                event = types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(text="hello anthropic"))
                return [event]
            return types.SimpleNamespace(content=[fake_resp_block])

    class FakeAnthropicClient:
        def __init__(self, api_key=None, base_url=None):
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
    sys.modules["anthropic"] = fake_module
    try:
        provider = AnthropicProvider()
        result = provider.complete(
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
    os.environ["GEMINI_API_KEY"] = "test-key"

    class FakeStreamChunk:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeGenerativeModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt_parts, temperature, max_output_tokens, request_options=None, stream=False):
            if stream:
                return [FakeStreamChunk("stream-1"), FakeStreamChunk("stream-2")]
            return FakeResponse("gemini-response")

    fake_module = types.SimpleNamespace(
        configure=lambda api_key=None: None,
        GenerativeModel=FakeGenerativeModel,
    )

    google_pkg = types.ModuleType("google")
    genai_pkg = types.ModuleType("google.generativeai")
    genai_pkg.configure = fake_module.configure
    genai_pkg.GenerativeModel = fake_module.GenerativeModel
    sys.modules["google"] = google_pkg
    sys.modules["google.generativeai"] = genai_pkg

    try:
        provider = GeminiProvider()
        result = provider.complete(
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
        sys.modules.pop("google.generativeai", None)
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


def test_bounding_box_uses_mock_when_configured():
    mock_path = PROJECT_ROOT / "tests" / "fixtures" / "bbox_mock.json"
    image_path = PROJECT_ROOT / "assets" / "environment.png"
    os.environ[BBOX_MOCK_ENV] = str(mock_path)
    try:
        result = detect_bounding_box_impl(target_object="dog", image_path=str(image_path))
        payload = json.loads(result)
        assert payload["success"] is True
        assert payload["coordinates"]["normalized"]["x_min"] == 0.15
    finally:
        os.environ.pop(BBOX_MOCK_ENV, None)
        output = image_path.parent / f"{image_path.stem}_with_bbox.png"
        if output.exists():
            try:
                output.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    # Simple runner for environments without pytest
    all_tests = [
        test_role_enum,
        test_message_creation_and_image_encoding,
        test_tool_schema_and_validation,
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
        test_bounding_box_uses_mock_when_configured,
    ]
    failures = 0
    for test in all_tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as exc:  # noqa: BLE001
            failures += 1
            print(f"✗ {test.__name__}: {exc}")
    if failures:
        raise SystemExit(1)
