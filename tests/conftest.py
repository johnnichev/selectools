"""
Pytest configuration for selectools tests.

This file configures pytest with custom markers, command-line options,
and shared mock provider classes + fixtures for use across all test files.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pytest

from selectools.providers.base import ProviderError
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests with real API calls",
    )


def pytest_configure(config: Any) -> None:
    """Register custom markers and load environment."""
    # Load .env file for E2E tests
    from selectools.env import load_default_env

    load_default_env()

    # Register markers
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires real API keys, use --run-e2e to run)"
    )
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI API key")
    config.addinivalue_line("markers", "anthropic: mark test as requiring Anthropic API key")
    config.addinivalue_line("markers", "gemini: mark test as requiring Gemini API key")
    config.addinivalue_line("markers", "ollama: mark test as requiring Ollama running locally")


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Skip e2e tests unless --run-e2e is passed."""
    if config.getoption("--run-e2e"):
        # --run-e2e given: do not skip e2e tests
        return

    skip_e2e = pytest.mark.skip(reason="Need --run-e2e option to run end-to-end tests")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_usage(model: str = "fake", provider: str = "fake") -> UsageStats:
    """Return a minimal UsageStats for mock providers."""
    return UsageStats(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.0001,
        model=model,
        provider=provider,
    )


def _to_message(response: Union[str, Message]) -> Message:
    """Ensure a response value is a Message object."""
    if isinstance(response, str):
        return Message(role=Role.ASSISTANT, content=response)
    return response


# ---------------------------------------------------------------------------
# Shared mock provider classes
# ---------------------------------------------------------------------------


class SharedFakeProvider:
    """Configurable fake provider that returns queued responses.

    Accepts a list of responses that can be:
    - ``str`` — automatically wrapped in a ``Message(role=ASSISTANT, content=...)``
    - ``Message`` — used as-is (allows setting ``tool_calls``, etc.)
    - ``(Message, UsageStats)`` tuple — used verbatim

    When more calls are made than responses configured, the provider cycles
    back to the beginning of the response list.

    Attributes:
        name: Provider name (default ``"fake"``).
        supports_streaming: Whether streaming is supported (default ``True``).
        supports_async: Whether async methods are supported (default ``True``).
        calls: Number of times ``complete`` / ``acomplete`` has been called.
        last_messages: The ``messages`` argument from the most recent call.
        last_system_prompt: The ``system_prompt`` from the most recent call.
        last_tools: The ``tools`` argument from the most recent call.
    """

    name: str = "fake"
    supports_streaming: bool = True
    supports_async: bool = True

    def __init__(
        self,
        responses: Optional[Sequence[Union[str, Message, Tuple[Message, UsageStats]]]] = None,
        *,
        name: str = "fake",
        supports_streaming: bool = True,
        supports_async: bool = True,
    ) -> None:
        self._responses: List[Union[str, Message, Tuple[Message, UsageStats]]] = list(
            responses or ["response"]
        )
        self.name = name
        self.supports_streaming = supports_streaming
        self.supports_async = supports_async
        self.calls: int = 0
        self.last_messages: List[Message] = []
        self.last_system_prompt: str = ""
        self.last_tools: Optional[List[Any]] = None

    # -- internal helpers ---------------------------------------------------

    def _next(self, model: str) -> Tuple[Message, UsageStats]:
        idx = self.calls % len(self._responses)
        raw = self._responses[idx]
        self.calls += 1
        if isinstance(raw, tuple):
            return raw
        return _to_message(raw), _default_usage(model=model, provider=self.name)

    def _record(
        self,
        messages: List[Message],
        system_prompt: str,
        tools: Optional[List[Any]],
    ) -> None:
        self.last_messages = list(messages)
        self.last_system_prompt = system_prompt
        self.last_tools = tools

    # -- Provider protocol --------------------------------------------------

    def complete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self._record(messages, system_prompt, tools)
        return self._next(model)

    async def acomplete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self._record(messages, system_prompt, tools)
        return self._next(model)

    def stream(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Iterable[str]:
        self._record(messages, system_prompt, tools)
        msg, _ = self._next(model)
        if msg.content:
            yield msg.content

    async def astream(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        self._record(messages, system_prompt, tools)
        msg, _ = self._next(model)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                yield tc  # type: ignore[misc]
        elif msg.content:
            yield msg.content  # type: ignore[misc]


class SharedRecordingProvider:
    """Wraps any provider and records all call arguments for assertion.

    Every call to ``complete``, ``acomplete``, ``stream``, or ``astream``
    is appended to the corresponding ``*_calls`` list as a dict of the
    keyword arguments received. The wrapped provider's result is returned
    unchanged.

    If no wrapped provider is given, a ``SharedFakeProvider`` with default
    responses is used automatically.

    Attributes:
        name: Provider name (default ``"recording"``).
        supports_streaming: Mirrors the wrapped provider.
        supports_async: Mirrors the wrapped provider.
        complete_calls: List of kwargs dicts from ``complete()`` calls.
        acomplete_calls: List of kwargs dicts from ``acomplete()`` calls.
        stream_calls: List of kwargs dicts from ``stream()`` calls.
        astream_calls: List of kwargs dicts from ``astream()`` calls.
    """

    name: str = "recording"
    supports_streaming: bool = True
    supports_async: bool = True

    def __init__(
        self,
        wrapped: Optional[Any] = None,
        *,
        name: str = "recording",
    ) -> None:
        self._wrapped = wrapped or SharedFakeProvider()
        self.name = name
        self.supports_streaming = getattr(self._wrapped, "supports_streaming", True)
        self.supports_async = getattr(self._wrapped, "supports_async", True)
        self.complete_calls: List[Dict[str, Any]] = []
        self.acomplete_calls: List[Dict[str, Any]] = []
        self.stream_calls: List[Dict[str, Any]] = []
        self.astream_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _capture(
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Any]],
        temperature: float,
        max_tokens: int,
        timeout: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

    def complete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self.complete_calls.append(
            self._capture(model, system_prompt, messages, tools, temperature, max_tokens, timeout)
        )
        return self._wrapped.complete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def acomplete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self.acomplete_calls.append(
            self._capture(model, system_prompt, messages, tools, temperature, max_tokens, timeout)
        )
        return await self._wrapped.acomplete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def stream(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Iterable[str]:
        self.stream_calls.append(
            self._capture(model, system_prompt, messages, tools, temperature, max_tokens, timeout)
        )
        return self._wrapped.stream(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def astream(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        self.astream_calls.append(
            self._capture(model, system_prompt, messages, tools, temperature, max_tokens, timeout)
        )
        async for chunk in self._wrapped.astream(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        ):
            yield chunk


class SharedToolCallProvider:
    """Provider that returns specific ToolCall objects in responses.

    Configure with a list of ``(tool_calls, text)`` tuples. On the *n*-th
    call, the provider returns the *n*-th entry (a ``Message`` carrying
    the given ``tool_calls`` list and ``text``). When all entries are
    exhausted, the provider returns a plain text ``"Done"`` response.

    This is useful for testing the agent tool-execution pipeline without
    having to construct full ``Message`` objects in every test.

    Attributes:
        name: Provider name (default ``"tool-call"``).
        supports_streaming: Always ``False``.
        supports_async: Always ``True``.
        calls: Number of calls made so far.
    """

    name: str = "tool-call"
    supports_streaming: bool = False
    supports_async: bool = True

    def __init__(
        self,
        responses: Optional[Sequence[Tuple[List[ToolCall], str]]] = None,
        *,
        name: str = "tool-call",
    ) -> None:
        self._responses: List[Tuple[List[ToolCall], str]] = list(responses or [])
        self.name = name
        self.calls: int = 0

    def _next(self, model: str) -> Tuple[Message, UsageStats]:
        usage = _default_usage(model=model, provider=self.name)
        if self.calls < len(self._responses):
            tool_calls, text = self._responses[self.calls]
            self.calls += 1
            return (
                Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls),
                usage,
            )
        self.calls += 1
        return Message(role=Role.ASSISTANT, content="Done"), usage

    def complete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        return self._next(model)

    async def acomplete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        return self._next(model)


class SharedErrorProvider:
    """Provider that raises exceptions on demand.

    Supports two modes:

    1. **Always fail** (default): every call raises the configured exception.
    2. **Pattern mode**: pass a list of ``True``/``False`` values via
       ``success_pattern``. ``True`` means the call succeeds (using
       ``fallback_response``), ``False`` means the call raises.
       The pattern cycles when exhausted.

    Attributes:
        name: Provider name (default ``"error"``).
        supports_streaming: Always ``False``.
        supports_async: Always ``True``.
        calls: Number of calls made so far.
    """

    name: str = "error"
    supports_streaming: bool = False
    supports_async: bool = True

    def __init__(
        self,
        exception: Optional[Exception] = None,
        *,
        success_pattern: Optional[Sequence[bool]] = None,
        fallback_response: str = "OK",
        name: str = "error",
    ) -> None:
        self._exception = exception or ProviderError("mock provider error")
        self._pattern: Optional[List[bool]] = list(success_pattern) if success_pattern else None
        self._fallback = fallback_response
        self.name = name
        self.calls: int = 0

    def _maybe_raise(self, model: str) -> Tuple[Message, UsageStats]:
        idx = self.calls
        self.calls += 1
        if self._pattern is not None:
            should_succeed = self._pattern[idx % len(self._pattern)]
            if should_succeed:
                return (
                    Message(role=Role.ASSISTANT, content=self._fallback),
                    _default_usage(model=model, provider=self.name),
                )
        raise self._exception

    def complete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        return self._maybe_raise(model)

    async def acomplete(
        self,
        *,
        model: str = "fake-model",
        system_prompt: str = "",
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        return self._maybe_raise(model)


# ---------------------------------------------------------------------------
# Pytest fixtures (factory pattern — each returns a callable)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_provider() -> Callable[..., SharedFakeProvider]:
    """Factory fixture that creates ``SharedFakeProvider`` instances.

    Usage::

        def test_something(fake_provider):
            provider = fake_provider(["Hello", "World"])
            assert provider.complete(messages=[...])
    """

    def _factory(
        responses: Optional[Sequence[Union[str, Message, Tuple[Message, UsageStats]]]] = None,
        **kwargs: Any,
    ) -> SharedFakeProvider:
        return SharedFakeProvider(responses=responses, **kwargs)

    return _factory


@pytest.fixture()
def recording_provider() -> Callable[..., SharedRecordingProvider]:
    """Factory fixture that creates ``SharedRecordingProvider`` instances.

    Usage::

        def test_something(recording_provider):
            provider = recording_provider()  # wraps a default FakeProvider
            agent = Agent(provider=provider, ...)
            agent.run(...)
            assert len(provider.complete_calls) == 1
    """

    def _factory(
        wrapped: Optional[Any] = None,
        **kwargs: Any,
    ) -> SharedRecordingProvider:
        return SharedRecordingProvider(wrapped=wrapped, **kwargs)

    return _factory


@pytest.fixture()
def tool_call_provider() -> Callable[..., SharedToolCallProvider]:
    """Factory fixture that creates ``SharedToolCallProvider`` instances.

    Usage::

        def test_something(tool_call_provider):
            tc = ToolCall(tool_name="search", parameters={"q": "hello"}, id="tc1")
            provider = tool_call_provider([([tc], "")])
            # First call returns tool call, second returns "Done"
    """

    def _factory(
        responses: Optional[Sequence[Tuple[List[ToolCall], str]]] = None,
        **kwargs: Any,
    ) -> SharedToolCallProvider:
        return SharedToolCallProvider(responses=responses, **kwargs)

    return _factory


@pytest.fixture()
def error_provider() -> Callable[..., SharedErrorProvider]:
    """Factory fixture that creates ``SharedErrorProvider`` instances.

    Usage::

        def test_something(error_provider):
            provider = error_provider(ProviderError("rate limit"))
            with pytest.raises(ProviderError):
                provider.complete(messages=[...])

        def test_flaky(error_provider):
            # Fails first call, succeeds second, repeats
            provider = error_provider(success_pattern=[False, True])
    """

    def _factory(
        exception: Optional[Exception] = None,
        **kwargs: Any,
    ) -> SharedErrorProvider:
        return SharedErrorProvider(exception=exception, **kwargs)

    return _factory
