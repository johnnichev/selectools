---
name: test
description: Write tests following selectools patterns — mocks, recording, regression, integration
argument-hint: <module-or-feature-to-test>
---

# Test Writing

Write tests for: $ARGUMENTS

## Live Project State

- Current tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`

## Test Organization

```
tests/
    test_<module>.py          # Unit tests per source module
    agent/                    # Agent core, observer, batch, regression
        test_regression.py    # ALL regression tests go here
    providers/                # Provider-specific tests
    rag/                      # RAG pipeline, chunking, stores
    integration/              # Cross-module integration tests
    tools/                    # Tool system tests
```

## Mock Provider Pattern

Always include `tools=None` in the signature — missing it silently hides bugs:

```python
from selectools.types import Message, Role

class FakeProvider:
    name = "fake"
    supports_streaming = True
    supports_async = True

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        return Message(role=Role.ASSISTANT, content="response")

    async def acomplete(self, *, model, system_prompt, messages, tools=None, **kw):
        return Message(role=Role.ASSISTANT, content="response")

    def stream(self, *, model, system_prompt, messages, tools=None, **kw):
        yield "response"

    async def astream(self, *, model, system_prompt, messages, tools=None, **kw):
        yield "response"
```

## Recording Provider Pattern

Use to verify exact args passed to provider methods:

```python
class RecordingProvider(FakeProvider):
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(("complete", kwargs))
        return Message(role=Role.ASSISTANT, content="ok")
```

Then assert: `assert provider.calls[0][1]["tools"] is not None`

## Tool-Returning Provider

For testing tool call flows:

```python
class ToolCallingProvider:
    call_count = 0

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        self.call_count += 1
        if self.call_count == 1:
            return Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="tc1", name="my_tool", arguments={"arg": "val"})],
            )
        return Message(role=Role.ASSISTANT, content="Final answer")
```

## Regression Tests

Every bug fix gets a test in `tests/agent/test_regression.py`:

```python
class TestSpecificBugDescription:
    """Regression: [describe what broke].

    Fixed in v0.XX.X. The bug was caused by [root cause].
    """

    def test_the_specific_scenario(self):
        # Reproduce the exact conditions that triggered the bug
        ...
```

## Key Assertions

- **Model counts**: `assert len(MODELS) == 146` — update when models change
- **Observer events**: Verify `run_id` is passed to all events
- **Streaming**: Verify `ToolCall` objects are yielded (use `isinstance(chunk, ToolCall)`)
- **Policy**: Verify `deny` actually blocks execution
- **Guardrails**: Verify `block` raises error, `rewrite` modifies content

## Integration Test Pattern

Test features working together through the agent:

```python
def test_feature_through_agent():
    provider = FakeProvider()
    agent = Agent(
        tools=[my_tool],
        provider=provider,
        config=AgentConfig(feature_option=True),
    )
    result = agent.run("test prompt")
    assert result.content == "expected"
    assert result.trace.steps  # Verify trace recorded
```

## Running Tests

```bash
pytest tests/ -x -q                    # All tests, stop on first failure
pytest tests/ -k "not e2e" -x -q       # Skip E2E (no API keys needed)
pytest tests/agent/ -x -q              # Just agent tests
pytest tests/test_specific.py -x -q    # Single file
```

ALL tests must pass before any commit. No exceptions.
