# Pipeline Module

**Added in:** v0.18.0 (type-safe contracts, `retry()`, `cache_step()` added in v0.18.x)
**Package:** `src/selectools/pipeline.py`
**Classes:** `Pipeline`, `Step`, `StepResult`
**Functions:** `step()`, `parallel()`, `branch()`, `retry()`, `cache_step()`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [@step Decorator](#step-decorator)
4. [| Operator](#pipe-operator)
5. [parallel()](#parallel)
6. [branch()](#branch)
7. [retry()](#retry)
8. [cache_step()](#cache_step)
9. [Streaming](#streaming)
10. [Type-Safe Step Contracts](#type-safe-step-contracts)
11. [Pipeline as AgentGraph Node](#pipeline-as-agentgraph-node)
12. [Error Handling](#error-handling)
13. [API Reference](#api-reference)
14. [Examples](#examples)

---

## Overview

The **pipeline** module provides composable data pipelines built from plain Python functions. Steps are connected with the `|` operator and execute in sequence, with each step receiving the previous step's output.

### The Anti-LCEL

This module exists because LangChain's LCEL (LangChain Expression Language) is the wrong abstraction. Pipelines should be plain functions, not magic Runnables with invisible state.

| | selectools Pipeline | LangChain LCEL |
|---|---|---|
| **Steps** | Plain Python functions | `Runnable` subclasses |
| **Composition** | `step_a \| step_b` (thin sugar) | `chain = prompt \| model \| parser` (deep magic) |
| **Debugging** | `print()` works, breakpoints work | Custom tracing required |
| **Type checking** | Standard type hints, validated at build time | No static checking |
| **Dependencies** | Zero | langchain-core, plus per-component packages |
| **Tracing** | Auto-traced with duration and status | Requires LangSmith (paid) |

### Design Philosophy

- **Steps are plain functions.** A `@step`-decorated function is still callable as `f(x)`. The decorator adds `|`, retry, and tracing -- nothing else.
- **`|` is thin sugar.** It creates a `Pipeline` that calls each function in order. No Pregel, no compilation, no runtime magic.
- **Every step auto-traces.** Each step records its name, duration, and status. No external tracing service required.
- **Type contracts are opt-in.** Annotate your functions with type hints and the pipeline validates adjacent step compatibility at build time.

---

## Quick Start

```python
from selectools import step, Pipeline, parallel, branch

@step
def summarize(text: str) -> str:
    return agent.run(f"Summarize: {text}").content

@step
def translate(text: str) -> str:
    return agent.run(f"Translate to Spanish: {text}").content

# Compose with |
pipeline = summarize | translate
result = pipeline.run("Long article about quantum computing...")

print(result.output)       # Spanish summary
print(result.steps_run)    # 2
print(result.trace)        # [{"step": "summarize", ...}, {"step": "translate", ...}]
```

---

## @step Decorator

The `@step` decorator wraps a plain function as a composable `Step`. The wrapped function remains directly callable -- the decorator only adds composition (`|`), retry logic, and tracing.

### Basic Usage

```python
from selectools import step

@step
def clean(text: str) -> str:
    return text.strip().lower()

# Still callable as a normal function
result = clean("  Hello World  ")  # "hello world"
```

### With Options

```python
@step(name="custom_name", retry=3, on_error="skip")
def flaky_api_call(query: str) -> str:
    return external_api.search(query)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `Optional[str]` | Function name | Override the step name in traces. |
| `retry` | `int` | `0` | Number of retry attempts on failure. |
| `on_error` | `str` | `"raise"` | Error handling: `"raise"` or `"skip"`. |

### Async Steps

Async functions work transparently:

```python
@step
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

Async steps are awaited during `arun()` and run via `asyncio.run()` during sync `run()`.

---

## | Operator {: #pipe-operator }

The pipe operator creates a `Pipeline` from two or more steps. Each step receives the previous step's output as its first argument.

```python
pipeline = step_a | step_b | step_c
```

This is equivalent to:

```python
pipeline = Pipeline(steps=[step_a, step_b, step_c])
```

### Composing with Plain Functions

Undecorated callables are auto-wrapped as `Step` instances:

```python
@step
def clean(text: str) -> str:
    return text.strip()

# str.upper is auto-wrapped
pipeline = clean | str.upper
result = pipeline.run("  hello  ")  # "HELLO"
```

### Composing Pipelines

Pipelines can be composed with other pipelines or steps:

```python
preprocess = clean | normalize
postprocess = format_output | validate

full = preprocess | translate | postprocess
```

---

## parallel()

Run multiple steps concurrently on the same input. Returns a dict mapping step names to their results.

```python
from selectools import parallel

@step
def search_web(query: str) -> str:
    return web_api.search(query)

@step
def search_docs(query: str) -> str:
    return doc_store.search(query)

@step
def search_db(query: str) -> str:
    return database.query(query)

# Fan out to all three, then merge
research = parallel(search_web, search_docs, search_db)
result = research("quantum computing")
# result == {"search_web": "...", "search_docs": "...", "search_db": "..."}
```

### In a Pipeline

```python
@step
def merge(results: dict) -> str:
    return "\n".join(results.values())

pipeline = parallel(search_web, search_docs) | merge | summarize
result = pipeline.run("quantum computing")
```

### Async Execution

When any step in the group is async, `parallel()` uses `asyncio.gather` for true concurrent execution during `arun()`. In sync `run()`, steps execute sequentially.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `*steps_or_fns` | `Union[Step, Callable]` | Steps or callables to run in parallel. |

Returns: `Step` -- a step whose output is `Dict[str, Any]` keyed by step names.

---

## branch()

Route input to one of several named steps based on a classifier function.

```python
from selectools import branch

@step
def classify(text: str) -> str:
    if "bug" in text.lower():
        return "technical"
    return "general"

@step
def technical_review(text: str) -> str:
    return agent.run(f"Technical review: {text}").content

@step
def general_response(text: str) -> str:
    return agent.run(f"Respond to: {text}").content

pipeline = classify | branch(
    technical=technical_review,
    general=general_response,
)
result = pipeline.run("There's a bug in the login page")
```

### With Custom Router

```python
pipeline = branch(
    router=lambda x: x["category"],
    technical=code_review,
    creative=copyedit,
    default=passthrough,
)
```

### Routing Logic

1. If `router` is provided, it is called with the input and must return a branch name (string).
2. If no router, the input itself is used as the branch key (must be a `str`, or a `dict` with a `"branch"` key).
3. If the key matches no branch, the `default` branch is used.
4. If no `default` exists, a `KeyError` is raised.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `router` | `Optional[Callable]` | Function that takes input and returns a branch name. |
| `**named_steps` | `Union[Step, Callable]` | Named branches. Key = branch name, value = step. |

Returns: `Step`

---

## retry()

Wrap a step with retry logic. A convenience wrapper that sets the retry count without modifying the original step.

```python
from selectools import retry

@step
def flaky_call(text: str) -> str:
    return unreliable_api.process(text)

# Retry up to 3 times on failure (4 total attempts)
pipeline = preprocess | retry(flaky_call, attempts=3) | postprocess
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `step_or_fn` | `Union[Step, Callable]` | (required) | Step or callable to wrap. |
| `attempts` | `int` | `3` | Number of retry attempts. |

Returns: `Step` with retry configured.

---

## cache_step()

Wrap a step with LRU + TTL result caching. Same input produces the cached output without re-executing the function.

```python
from selectools import cache_step

@step
def expensive_embedding(text: str) -> list:
    return embedding_model.embed(text)

# Cache results for 10 minutes, max 500 entries
pipeline = preprocess | cache_step(expensive_embedding, ttl=600, max_size=500) | classify
```

### Cache Behavior

- **Key:** String representation of the input value.
- **Eviction:** LRU (oldest entries evicted when `max_size` is reached).
- **TTL:** Entries expire after `ttl` seconds.
- **Scope:** Cache is per-step instance (not shared across pipeline copies).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `step_or_fn` | `Union[Step, Callable]` | (required) | Step or callable to wrap. |
| `ttl` | `int` | `300` | Cache time-to-live in seconds. |
| `max_size` | `int` | `1000` | Maximum cache entries before LRU eviction. |

Returns: `Step` with caching configured.

---

## Streaming

Stream the pipeline's final step output as it is produced. Earlier steps run to completion; only the last step streams.

```python
pipeline = preprocess | summarize | translate

async for chunk in pipeline.astream("Long article..."):
    print(chunk, end="")
```

### How It Works

1. All steps except the last execute normally via `arun()`.
2. The final step's function is inspected:
   - **Async generator:** Chunks are yielded as produced.
   - **Sync generator:** Chunks are yielded as produced.
   - **Regular function:** The complete output is yielded as a single chunk.

### Generator Step Example

```python
@step
def stream_translate(text: str):
    """A generator step that yields chunks."""
    for sentence in text.split(". "):
        yield translate_sentence(sentence) + ". "

pipeline = summarize | stream_translate

async for chunk in pipeline.astream("Long article..."):
    print(chunk, end="", flush=True)
```

---

## Type-Safe Step Contracts

Annotate functions with type hints and the pipeline validates type compatibility between adjacent steps at build time.

### Automatic Inference

Type hints are extracted automatically from function signatures:

```python
@step
def parse(raw: str) -> dict:
    return json.loads(raw)

@step
def extract(data: dict) -> list:
    return data.get("items", [])

@step
def count(items: list) -> int:
    return len(items)

# Types are validated at pipeline creation:
# parse (str -> dict) | extract (dict -> list) | count (list -> int)
pipeline = parse | extract | count  # No warnings
```

### Type Mismatch Warning

When adjacent steps have incompatible types, a warning is emitted at pipeline creation:

```python
@step
def to_int(text: str) -> int:
    return int(text)

@step
def join_words(words: list) -> str:
    return " ".join(words)

# Warning: Pipeline type mismatch: 'to_int' outputs int but 'join_words' expects list
pipeline = to_int | join_words
```

Type checking is advisory -- the pipeline still runs. This catches common mistakes without blocking execution.

### Explicit Type Contracts

Override inferred types when needed:

```python
custom_step = Step(
    my_function,
    name="custom",
    input_type=str,
    output_type=dict,
)
```

### Generic Types

Generic types (`Dict[str, Any]`, `List[int]`, etc.) are accepted but not deeply validated -- the system cannot verify generic type parameters at runtime.

---

## Pipeline as AgentGraph Node

Every `Pipeline` implements `__call__(state)`, making it usable as an `AgentGraph` callable node. This bridges the composition and orchestration modules.

```python
from selectools import AgentGraph, step, Pipeline

@step
def preprocess(text: str) -> str:
    return text.strip().lower()

@step
def enrich(text: str) -> str:
    return f"[enriched] {text}"

preprocessing = preprocess | enrich

# Use pipeline as a graph node
graph = AgentGraph()
graph.add_node("preprocess", preprocessing)  # Pipeline as callable node
graph.add_node("agent", my_agent)
graph.add_edge("preprocess", "agent")
graph.add_edge("agent", AgentGraph.END)
graph.set_entry("preprocess")

result = graph.run("  Raw User Input  ")
```

### How the Bridge Works

When a `Pipeline` receives a `GraphState`:

1. Extracts `last_output` from `state.data` (or the last message content as fallback).
2. Runs the pipeline with that input.
3. Writes the pipeline output back to `state.data[STATE_KEY_LAST_OUTPUT]`.
4. Returns the modified state.

---

## Error Handling

### on_error="raise" (Default)

Exceptions propagate immediately. The pipeline stops and the exception is raised to the caller.

```python
@step(on_error="raise")
def strict_step(x):
    raise ValueError("failed")

pipeline = strict_step | next_step
pipeline.run("input")  # Raises ValueError
```

### on_error="skip"

The failed step is skipped and the pipeline continues with the previous output.

```python
@step(on_error="skip")
def optional_step(x):
    raise ValueError("not critical")

pipeline = optional_step | next_step
result = pipeline.run("input")  # next_step receives "input" unchanged
```

### Retry + Skip

Combine retry with skip for maximum resilience:

```python
@step(retry=3, on_error="skip")
def resilient_step(x):
    return unreliable_api.call(x)
```

This retries 3 times, then skips if all attempts fail.

### Trace Inspection

Every step records its status in the trace, including errors and retries:

```python
result = pipeline.run("input")
for entry in result.trace:
    print(f"{entry['step']}: {entry['status']} ({entry['duration_ms']:.1f}ms)")
    if entry.get("error"):
        print(f"  Error: {entry['error']}")
    if entry.get("retry"):
        print(f"  Retry #{entry['retry']}")
```

---

## API Reference

### Step.__init__()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fn` | `Callable` | (required) | The function to wrap. |
| `name` | `Optional[str]` | Function name | Step name for traces. |
| `retry` | `int` | `0` | Retry attempts on failure. |
| `on_error` | `str` | `"raise"` | Error handling: `"raise"` or `"skip"`. |
| `input_type` | `Optional[type]` | Auto-inferred | Expected input type (for contract validation). |
| `output_type` | `Optional[type]` | Auto-inferred | Declared output type (for contract validation). |

### Pipeline.__init__()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `steps` | `Optional[Sequence[Union[Step, Pipeline, Callable]]]` | `None` | Ordered list of steps. |
| `name` | `str` | `"pipeline"` | Pipeline name. |

### Pipeline Methods

| Method | Description |
|---|---|
| `run(input, **kwargs)` | Execute synchronously. Returns `StepResult`. |
| `arun(input, **kwargs)` | Execute asynchronously. Returns `StepResult`. |
| `astream(input, **kwargs)` | Async generator. Yields chunks from the final step. |
| `steps` | Property. Read-only list of steps in the pipeline. |

### StepResult

| Field | Type | Description |
|---|---|---|
| `output` | `Any` | The final output of the pipeline. |
| `trace` | `List[Dict[str, Any]]` | Per-step trace entries with `step`, `duration_ms`, `status`, and optional `error`/`retry`. |
| `steps_run` | `int` | Number of steps that executed successfully. |

### step()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fn` | `Optional[Callable]` | `None` | Function to wrap (when used without parens). |
| `name` | `Optional[str]` | `None` | Override step name. |
| `retry` | `int` | `0` | Retry attempts. |
| `on_error` | `str` | `"raise"` | Error handling strategy. |

Returns: `Step` (or decorator `Callable[[Callable], Step]` when called with arguments).

### parallel()

| Parameter | Type | Description |
|---|---|---|
| `*steps_or_fns` | `Union[Step, Callable]` | Steps to run concurrently. |

Returns: `Step` whose output is `Dict[str, Any]`.

### branch()

| Parameter | Type | Description |
|---|---|---|
| `router` | `Optional[Callable]` | Routing function. |
| `**named_steps` | `Union[Step, Callable]` | Named branch targets. |

Returns: `Step`

### retry()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `step_or_fn` | `Union[Step, Callable]` | (required) | Step to wrap. |
| `attempts` | `int` | `3` | Retry count. |

Returns: `Step`

### cache_step()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `step_or_fn` | `Union[Step, Callable]` | (required) | Step to wrap. |
| `ttl` | `int` | `300` | Cache TTL in seconds. |
| `max_size` | `int` | `1000` | Max cache entries. |

Returns: `Step`

---

## Examples

| Example | File | Description |
|---|---|---|
| 66 | [`66_pipeline_basics.py`](https://github.com/johnnichev/selectools/blob/main/examples/66_pipeline_basics.py) | Step decorator, pipe operator, run/arun |
| 67 | [`67_pipeline_parallel_branch.py`](https://github.com/johnnichev/selectools/blob/main/examples/67_pipeline_parallel_branch.py) | parallel(), branch(), retry(), cache_step() |
| 68 | [`68_pipeline_graph_bridge.py`](https://github.com/johnnichev/selectools/blob/main/examples/68_pipeline_graph_bridge.py) | Using Pipeline as an AgentGraph node |

---

## Further Reading

- [Orchestration Module](ORCHESTRATION.md) -- AgentGraph for multi-agent workflows
- [Agent Module](AGENT.md) -- The Agent class that powers individual steps
- [Streaming Module](STREAMING.md) -- How streaming works under the hood
- [Tool Caching](TOOL_CACHING.md) -- Caching for individual tool calls

---

**Next Steps:** Learn about multi-agent orchestration in the [Orchestration Module](ORCHESTRATION.md).
