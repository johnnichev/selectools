---
description: "OpenTelemetry observer — emit GenAI semantic-convention spans for agent runs, LLM calls, and tool executions"
tags:
  - observability
  - opentelemetry
  - tracing
---

# OpenTelemetry Observer

**Import:** `from selectools.observe import OTelObserver`
**Stability:** beta
**Added in:** v0.21.0

`OTelObserver` maps the 45 selectools observer events to OpenTelemetry spans,
following the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
Once attached, every agent run, LLM call, and tool execution becomes a span you
can ship to Jaeger, Tempo, Honeycomb, Datadog, Grafana, or any other OTLP-capable
backend.

```python title="otel_quick.py"
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.observe import OTelObserver

# 1. Configure your OTel SDK once at process start
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# 2. Attach the observer
@tool()
def search(query: str) -> str:
    return f"Results for {query}"

agent = Agent(
    tools=[search],
    provider=OpenAIProvider(),
    config=AgentConfig(observers=[OTelObserver()]),
)

result = agent.run("Find articles about Python")
# Spans now flow to your OTel exporter
```

!!! tip "See Also"
    - [Langfuse](LANGFUSE.md) - Alternative observer focused on LLM tracing
    - [Trace Store](TRACE_STORE.md) - Persist agent traces to disk or SQLite
    - [Audit](AUDIT.md) - JSONL audit logs

---

## Install

```bash
pip install "selectools[observe]"
```

The `[observe]` extras include `opentelemetry-api>=1.20.0`. **selectools does not
ship `opentelemetry-sdk` or any exporters** — bring your own. Common choices:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp     # OTLP
pip install opentelemetry-sdk opentelemetry-exporter-jaeger   # Jaeger
```

This separation lets you reuse whatever exporter the rest of your stack already
uses without selectools pinning a transitive dependency.

---

## Span Hierarchy

Each agent run becomes a span tree:

```
agent.run                              ← root span
├── gen_ai.llm.call                    ← per LLM round-trip
│   └── gen_ai.tool.execution          ← per tool call
├── gen_ai.llm.call
└── ...
```

| Span name | Attributes |
|---|---|
| `agent.run` | `gen_ai.system="selectools"`, `gen_ai.usage.total_tokens`, `gen_ai.usage.cost_usd` |
| `gen_ai.llm.call` | `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |
| `gen_ai.tool.execution` | `gen_ai.tool.name`, `gen_ai.tool.duration_ms`, `gen_ai.tool.success` |

---

## Constructor

```python
OTelObserver(tracer_name: str = "selectools")
```

| Parameter | Description |
|---|---|
| `tracer_name` | Name passed to `trace.get_tracer()`. Use this to scope spans by service in multi-app processes. |

---

## Async

For `agent.arun()` / `agent.astream()` use the async variant:

```python
from selectools.observe.otel import AsyncOTelObserver
agent = Agent(..., config=AgentConfig(observers=[AsyncOTelObserver()]))
```

---

## API Reference

| Symbol | Description |
|---|---|
| `OTelObserver(tracer_name)` | Sync observer for `agent.run()` / `agent.stream()` |
| `AsyncOTelObserver(tracer_name)` | Async observer for `agent.arun()` / `agent.astream()` |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 87 | [`87_otel_observer.py`](https://github.com/johnnichev/selectools/blob/main/examples/87_otel_observer.py) | Wire selectools traces into an OTLP exporter |
