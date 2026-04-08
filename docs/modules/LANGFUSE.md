---
description: "Langfuse observer — send agent traces, generations, and spans to Langfuse Cloud or self-hosted"
tags:
  - observability
  - langfuse
  - tracing
---

# Langfuse Observer

**Import:** `from selectools.observe import LangfuseObserver`
**Stability:** beta
**Added in:** v0.21.0

`LangfuseObserver` ships selectools traces to [Langfuse](https://langfuse.com), an
open-source LLM observability platform. Each agent run becomes a Langfuse trace,
each LLM call becomes a generation (with input/output/tokens/cost), and each tool
call becomes a span. Works with both Langfuse Cloud and self-hosted instances.

```python title="langfuse_quick.py"
import os
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.observe import LangfuseObserver

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
# os.environ["LANGFUSE_HOST"] = "https://my-langfuse.example.com"  # self-hosted

@tool()
def search(query: str) -> str:
    return f"Results for {query}"

agent = Agent(
    tools=[search],
    provider=OpenAIProvider(),
    config=AgentConfig(observers=[LangfuseObserver()]),
)

result = agent.run("Find articles about Python")
# View the trace in your Langfuse dashboard
```

!!! tip "See Also"
    - [OpenTelemetry](OTEL.md) - Alternative observer for OTLP backends
    - [Trace Store](TRACE_STORE.md) - Persist traces locally as JSONL or SQLite

---

## Install

```bash
pip install "selectools[observe]"
```

The `[observe]` extras include `langfuse>=2.0.0`.

---

## Constructor

```python
LangfuseObserver(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `public_key` | Langfuse public key. Falls back to `LANGFUSE_PUBLIC_KEY` env var. |
| `secret_key` | Langfuse secret key. Falls back to `LANGFUSE_SECRET_KEY` env var. |
| `host` | Langfuse host URL. Defaults to Langfuse Cloud. Set this to point at a self-hosted instance. Falls back to `LANGFUSE_HOST` env var. |

The observer auto-flushes after every `run_end`, so traces are visible in your
Langfuse dashboard within seconds of an agent finishing.

---

## What Gets Recorded

| Selectools event | Langfuse object | Fields |
|---|---|---|
| `on_run_start` | Trace | `id=run_id`, `name="agent.run"`, input messages |
| `on_llm_start` | Generation | `model`, `input` (messages) |
| `on_llm_end` | Generation update | `output`, `usage.input/output/total`, `cost_usd` |
| `on_tool_start` | Span | `name=tool_name`, `input=tool_args` |
| `on_tool_end` | Span update | `output`, `duration_ms` |
| `on_run_end` | Trace update | `output`, total tokens, total cost |

---

## Self-Hosted Langfuse

```python
observer = LangfuseObserver(
    public_key="pk-lf-local-...",
    secret_key="sk-lf-local-...",
    host="https://langfuse.internal.example.com",
)
```

Or via env vars:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://langfuse.internal.example.com"
```

---

## API Reference

| Symbol | Description |
|---|---|
| `LangfuseObserver(public_key, secret_key, host)` | Observer for `agent.run()` / `agent.stream()` |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 88 | [`88_langfuse_observer.py`](https://github.com/johnnichev/selectools/blob/main/examples/88_langfuse_observer.py) | Langfuse trace + generation + span hierarchy |
