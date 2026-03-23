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
    conftest.py               # SharedFakeProvider, fixtures, helpers
    test_<module>.py           # Unit tests per source module
    agent/                     # Agent core, observer, batch, regression
        test_regression.py     # ALL regression tests go here
    providers/                 # Provider-specific tests
    rag/                       # RAG pipeline, chunking, stores
    integration/               # Cross-module integration tests
    tools/                     # Tool system tests
```

## SharedFakeProvider (from conftest.py)

Use the `fake_provider` fixture — it returns a factory. Responses can be:
- `str` — auto-wrapped as `Message(role=ASSISTANT, content=...)`
- `Message` — used as-is (for tool_calls, etc.)
- `(Message, UsageStats)` tuple — controls token/cost tracking

```python
def test_example(self, fake_provider):
    # Simple text response
    provider = fake_provider(responses=["Hello"])

    # Tool call response
    provider = fake_provider(responses=[
        Message(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(tool_name="search", parameters={"q": "test"})
        ]),
        "Final answer",
    ])

    # Response with specific usage stats (for budget/cost tests)
    provider = fake_provider(responses=[
        (Message(role=Role.ASSISTANT, content="answer"),
         UsageStats(prompt_tokens=100, completion_tokens=50,
                    total_tokens=150, cost_usd=0.01,
                    model="test", provider="test")),
    ])
```

**Important:** Agent requires at least one tool. Use a dummy:
```python
_DUMMY = Tool(name="noop", description="noop", parameters=[], function=lambda: "ok")
```

## Recording Provider Pattern

Use to verify exact args passed to provider methods:

```python
class RecordingProvider:
    name = "recording"
    supports_streaming = False
    supports_async = False

    def __init__(self):
        self.last_messages = []
        self.last_system_prompt = ""
        self.last_tools = None

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        self.last_messages = list(messages)
        self.last_system_prompt = system_prompt
        self.last_tools = tools
        return Message(role=Role.ASSISTANT, content="ok"), UsageStats(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            cost_usd=0.0001, model=model, provider="recording",
        )
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

## Testing v0.17.3+ Features

### Budget tests
```python
def test_budget_stops_agent(self, fake_provider):
    provider = fake_provider(responses=[
        _tool_resp("noop", total_tokens=500),
        _resp("done"),
    ])
    agent = Agent(tools=[_DUMMY], provider=provider,
        config=AgentConfig(max_iterations=10, max_total_tokens=400))
    result = agent.run("test")
    assert "budget exceeded" in result.content.lower()
    assert any(s.type == StepType.BUDGET_EXCEEDED for s in result.trace.steps)
```

### Cancellation tests
```python
from selectools.cancellation import CancellationToken

token = CancellationToken()
token.cancel()  # Pre-cancel
agent = Agent(tools=[_DUMMY], provider=provider,
    config=AgentConfig(cancellation_token=token))
result = agent.run("test")
assert "cancelled" in result.content.lower()
```

### Observer tests
```python
events = []
class MyObserver(AgentObserver):
    def on_budget_exceeded(self, run_id, reason, tokens_used, cost_used):
        events.append({"event": "budget_exceeded", "reason": reason})

config = AgentConfig(observers=[MyObserver()])
```

### Approval gate tests
```python
danger = Tool(name="danger", description="dangerous", parameters=[],
    function=lambda: "boom", requires_approval=True)
config = AgentConfig(confirm_action=lambda name, args, reason: False)
# Tool should be denied
```

## Key Assertions

- **Model counts**: `assert len(MODELS) == 146` — update when models change
- **StepType counts**: `assert len(StepType) == 16` — update when types added
- **Observer events**: Verify `run_id` is passed to all events
- **Streaming**: Verify `ToolCall` objects are yielded (use `isinstance(chunk, ToolCall)`)
- **Policy**: Verify `deny` actually blocks execution
- **Guardrails**: Verify `block` raises error, `rewrite` modifies content
- **Budget**: Verify `BUDGET_EXCEEDED` trace step on budget hit
- **Cancellation**: Verify `CANCELLED` trace step on cancel

## Running Tests

```bash
pytest tests/ -x -q                    # All tests, stop on first failure
pytest tests/ -k "not e2e" -x -q       # Skip E2E (no API keys needed)
pytest tests/agent/ -x -q              # Just agent tests
pytest tests/test_specific.py -x -q    # Single file
```

ALL tests must pass before any commit. No exceptions.
