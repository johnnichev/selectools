# tests/ — Testing Rules

## Running Tests

```bash
pytest tests/ -x -q                    # All tests, stop on first failure
pytest tests/ -k "not e2e" -x -q       # Skip E2E (no API keys needed)
pytest tests/providers/test_models.py   # Specific test file
```

## Key Conventions

- **File naming**: `test_<module>.py` — mirrors source module names
- **No real API calls** in unit tests — always mock providers
- **E2E tests**: mark with `@pytest.mark.e2e`, skipped in CI
- **Regression tests**: go in `tests/agent/test_regression.py` — every bug fix gets one

## Agent Setup in Tests

- `Agent()` requires at least one tool as the first argument
- Use `_DUMMY` tool (a no-op `@tool()` function) when tools are irrelevant to the test
- Provider responses use `(Message, UsageStats)` tuples for controlled usage tracking

## Mock Patterns

- **`LocalProvider(responses=[...])`** — returns canned responses without API keys
- **`RecordingProvider`** — captures exact args passed to `complete()`/`stream()` for assertion
- Both live in `selectools.providers.stubs`

## Provider Mocking Example

```python
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role

provider = LocalProvider(responses=["Hello!"])
agent = Agent([dummy_tool], provider=provider, model="local")
result = agent.run("Hi")
```

## Gotchas

- Update model count tests in `tests/providers/test_models.py` when adding/removing models
- Integration tests that touch the agent loop go in `tests/integration/`
- Never use `datetime.utcnow()` — use `datetime.now(timezone.utc)`
- Test both sync (`run`) and async (`arun`/`astream`) paths for agent features
