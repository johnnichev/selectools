# Quickstart: Your First Agent in 5 Minutes

This guide takes you from zero to a working AI agent, step by step. No API keys needed for the first two steps.

---

## Step 1: Install

```bash
pip install selectools
```

## Step 2: Build Your First Agent (No API Key Needed)

Create a file called `my_agent.py`:

```python
from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider

@tool(description="Look up the price of a product")
def get_price(product: str) -> str:
    prices = {"laptop": "$999", "phone": "$699", "headphones": "$149"}
    return prices.get(product.lower(), f"No price found for {product}")

@tool(description="Check if a product is in stock")
def check_stock(product: str) -> str:
    stock = {"laptop": "In stock (5 left)", "phone": "Out of stock", "headphones": "In stock (20 left)"}
    return stock.get(product.lower(), f"Unknown product: {product}")

agent = Agent(
    tools=[get_price, check_stock],
    provider=LocalProvider(),
    config=AgentConfig(max_iterations=3),
)

result = agent.ask("What is the price of a laptop?")
print(result.content)
```

Run it:

```bash
python my_agent.py
```

**What just happened:**

1. You defined two tools with the `@tool` decorator — Selectools auto-generates JSON schemas from your type hints
2. You created an agent with `LocalProvider` (a built-in stub that works offline)
3. You asked a question with `agent.ask()` and the agent decided which tool to call

> `LocalProvider` is a testing stub that echoes tool results. It is great for
> learning the API and running tests, but it does not actually call an LLM.
> Step 3 shows you how to connect to a real model.

## Step 3: Connect to a Real LLM

Set your API key and swap the provider:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.models import OpenAI

@tool(description="Look up the price of a product")
def get_price(product: str) -> str:
    prices = {"laptop": "$999", "phone": "$699", "headphones": "$149"}
    return prices.get(product.lower(), f"No price found for {product}")

@tool(description="Check if a product is in stock")
def check_stock(product: str) -> str:
    stock = {"laptop": "In stock (5 left)", "phone": "Out of stock", "headphones": "In stock (20 left)"}
    return stock.get(product.lower(), f"Unknown product: {product}")

agent = Agent(
    tools=[get_price, check_stock],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
    config=AgentConfig(max_iterations=5),
)

result = agent.ask("Is the phone in stock? And how much are headphones?")
print(result.content)
print(f"\nCost: ${agent.total_cost:.6f} | Tokens: {agent.total_tokens}")
```

The only line that changed is the `provider=` argument. Your tools stay identical.

**Other providers work the same way:**

```python
from selectools import AnthropicProvider, GeminiProvider, OllamaProvider

# Anthropic Claude
agent = Agent(tools=[...], provider=AnthropicProvider())

# Google Gemini (free tier available)
agent = Agent(tools=[...], provider=GeminiProvider())

# Ollama (fully local, fully free)
agent = Agent(tools=[...], provider=OllamaProvider())
```

## Step 4: Add Conversation Memory

Make the agent remember previous turns:

```python
from selectools import Agent, AgentConfig, ConversationMemory, OpenAIProvider, tool

@tool(description="Save a note for the user")
def save_note(text: str) -> str:
    return f"Saved note: {text}"

memory = ConversationMemory(max_messages=20)

agent = Agent(
    tools=[save_note],
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=3),
    memory=memory,
)

agent.ask("My name is Alice and I work at Acme Corp")
result = agent.ask("What company do I work at?")
print(result.content)  # Remembers "Acme Corp" from the previous turn
```

## Step 5: Add Document Search (RAG)

Give the agent a knowledge base to search:

```bash
pip install selectools[rag]   # Adds embeddings + vector store support
```

```python
from selectools import OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import Document, RAGAgent, VectorStore

# Create an embedding provider and vector store
embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
store = VectorStore.create("memory", embedder=embedder)

# Load your documents
docs = [
    Document(text="Our return policy allows returns within 30 days of purchase.", metadata={"source": "policy.txt"}),
    Document(text="Shipping takes 3-5 business days for domestic orders.", metadata={"source": "shipping.txt"}),
    Document(text="Premium members get free expedited shipping.", metadata={"source": "membership.txt"}),
]

# Create the agent — chunking, embedding, and tool setup happen automatically
agent = RAGAgent.from_documents(
    documents=docs,
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
    vector_store=store,
)

result = agent.ask("How long does shipping take for premium members?")
print(result.content)
```

## Step 6: Get Structured Output

Get typed, validated results from the LLM:

```python
from pydantic import BaseModel
from typing import Literal

class Classification(BaseModel):
    intent: Literal["billing", "support", "sales"]
    confidence: float

result = agent.ask("I need help with my bill", response_format=Classification)
print(result.parsed)       # Classification(intent="billing", confidence=0.95)
print(result.trace.timeline())  # See what the agent did
print(result.reasoning)    # Why it chose that classification
```

## Step 7: Provider Fallback

Wrap multiple providers in a priority chain. If the primary fails, the next one is tried automatically:

```python
from selectools import Agent, AgentConfig, FallbackProvider, OpenAIProvider, AnthropicProvider
from selectools.providers.stubs import LocalProvider

provider = FallbackProvider(
    providers=[
        OpenAIProvider(),        # Try OpenAI first
        AnthropicProvider(),     # Fall back to Anthropic
        LocalProvider(),         # Last resort (offline)
    ],
    max_failures=3,              # Skip after 3 consecutive failures
    cooldown_seconds=60,         # Skip for 60 seconds
    on_fallback=lambda name, err: print(f"Skipping {name}: {err}"),
)

agent = Agent(tools=[...], provider=provider, config=AgentConfig(max_iterations=5))
result = agent.ask("Hello!")
```

The built-in circuit breaker avoids wasting time on providers that are consistently down.

## Step 8: Tool Policy

Control which tools can run with declarative rules and human-in-the-loop approval:

```python
from selectools import Agent, AgentConfig, tool
from selectools.policy import ToolPolicy

@tool(description="Read a file")
def read_file(path: str) -> str:
    return open(path).read()

@tool(description="Delete a file")
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

policy = ToolPolicy(
    allow=["read_*"],          # Always allowed
    review=["send_*"],         # Needs human approval
    deny=["delete_*"],         # Always blocked
)

def approve(tool_name, tool_args, reason):
    return input(f"Allow {tool_name}({tool_args})? [y/n] ") == "y"

agent = Agent(
    tools=[read_file, delete_file],
    provider=provider,
    config=AgentConfig(
        tool_policy=policy,
        confirm_action=approve,
        approval_timeout=30,
    ),
)
```

## Step 9: Monitor with AgentObserver

For production observability, use `AgentObserver` — a class-based protocol with 15 lifecycle events. Every callback gets a `run_id` for cross-request correlation:

```python
from selectools import Agent, AgentConfig
from selectools.observer import AgentObserver, LoggingObserver

class MyObserver(AgentObserver):
    def on_run_start(self, run_id, messages, system_prompt):
        print(f"[{run_id[:8]}] Starting with {len(messages)} messages")

    def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms):
        print(f"[{run_id[:8]}] {tool_name} took {duration_ms:.0f}ms")

    def on_run_end(self, run_id, result):
        print(f"[{run_id[:8]}] Done — {result.usage.total_tokens} tokens")

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(
        observers=[MyObserver(), LoggingObserver()],
    ),
)

result = agent.ask("Hello!")

# Export execution trace as OpenTelemetry spans
otel_spans = result.trace.to_otel_spans()
```

`LoggingObserver` emits structured JSON to Python's `logging` module — plug it into Datadog, ELK, or any log aggregator.

## Step 10: Add Guardrails

Validate inputs and outputs with a pluggable guardrail pipeline:

```python
from selectools import Agent, AgentConfig
from selectools.guardrails import GuardrailsPipeline, TopicGuardrail, PIIGuardrail, GuardrailAction

guardrails = GuardrailsPipeline(
    input=[
        TopicGuardrail(deny=["politics", "religion"]),
        PIIGuardrail(action=GuardrailAction.REWRITE),  # redact PII
    ],
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(guardrails=guardrails),
)

# PII is automatically redacted before the LLM sees it
result = agent.ask("Look up customer user@example.com")

# Blocked topics raise GuardrailError
from selectools.guardrails import GuardrailError
try:
    agent.ask("Tell me about politics")
except GuardrailError as e:
    print(f"Blocked: {e.reason}")
```

Five built-in guardrails: `TopicGuardrail`, `PIIGuardrail`, `ToxicityGuardrail`, `FormatGuardrail`, `LengthGuardrail`. Or subclass `Guardrail` to write your own.

## Step 11: Audit Logging & Security

Add a JSONL audit trail and prompt injection defence:

```python
from selectools import Agent, AgentConfig, tool
from selectools.audit import AuditLogger, PrivacyLevel

audit = AuditLogger(
    log_dir="./audit",
    privacy=PrivacyLevel.KEYS_ONLY,  # redact argument values
)

@tool(description="Fetch web page", screen_output=True)  # screen for injection
def fetch_page(url: str) -> str:
    import requests
    return requests.get(url).text

agent = Agent(
    tools=[fetch_page],
    provider=provider,
    config=AgentConfig(
        observers=[audit],            # JSONL audit log
        screen_tool_output=True,      # prompt injection screening
        coherence_check=True,         # verify tool calls match intent
        coherence_model="gpt-4o-mini",
    ),
)
```

## What's Next?

You now know the core API. Here is where to go from here:

| Goal | Read |
|---|---|
| Define more complex tools | [Tools Guide](modules/TOOLS.md) |
| Get typed LLM responses | [Agent Guide — Structured Output](modules/AGENT.md#structured-output) |
| See what the agent did | [Agent Guide — Execution Traces](modules/AGENT.md#execution-traces) |
| Switch between providers | [Providers Guide](modules/PROVIDERS.md) |
| Auto-failover between providers | [Providers Guide — Fallback](modules/PROVIDERS.md#fallbackprovider) |
| Classify multiple requests at once | [Agent Guide — Batch Processing](modules/AGENT.md#batch-processing) |
| Control which tools can run | [Agent Guide — Tool Policy](modules/AGENT.md#tool-policy-human-in-the-loop) |
| Monitor with AgentObserver | [Agent Guide — Observer Protocol](modules/AGENT.md#agentobserver-protocol) |
| Export traces to OpenTelemetry | [Agent Guide — OTel Export](modules/AGENT.md#agentobserver-protocol) |
| Stream responses in real time | [Streaming Guide](modules/STREAMING.md) |
| Use hybrid search (keyword + semantic) | [Hybrid Search Guide](modules/HYBRID_SEARCH.md) |
| Load tools from plugin files | [Dynamic Tools Guide](modules/DYNAMIC_TOOLS.md) |
| Cache LLM responses to save money | [Agent Guide — Caching](modules/AGENT.md#response-caching) |
| Browse 146 models with pricing | [Models Guide](modules/MODELS.md) |
| Track costs and token usage | [Usage Guide](modules/USAGE.md) |
| Understand the full architecture | [Architecture](ARCHITECTURE.md) |
| Add input/output guardrails | [Guardrails Guide](modules/GUARDRAILS.md) |
| Add audit logging | [Audit Guide](modules/AUDIT.md) |
| Screen tool outputs for injection | [Security Guide](modules/SECURITY.md) |
| Enable coherence checking | [Security Guide — Coherence](modules/SECURITY.md#coherence-checking) |
| Use 24 pre-built tools | [Toolbox Guide](modules/TOOLBOX.md) |
| Handle errors gracefully | [Exceptions Guide](modules/EXCEPTIONS.md) |
| Look up model pricing at runtime | [Models Guide — Pricing API](modules/MODELS.md#programmatic-pricing-api) |
| Use structured output helpers | [Agent Guide — Structured Helpers](modules/AGENT.md#standalone-helpers) |
| See working examples | [examples/](https://github.com/johnnichev/selectools/tree/main/examples) (32 numbered scripts, 01–32) |

---

**The API in one table:**

| You want to... | Code |
|---|---|
| Ask a question (simple) | `agent.ask("What is X?")` |
| Get typed results | `agent.ask("...", response_format=MyModel)` |
| Send structured messages | `agent.run([Message(role=Role.USER, content="...")])` |
| Ask asynchronously | `await agent.aask("What is X?")` |
| Stream tokens | `async for chunk in agent.astream("What is X?"): ...` |
| Classify a batch | `agent.batch(["msg1", "msg2"], max_concurrency=5)` |
| Check cost | `agent.total_cost`, `agent.get_usage_summary()` |
| See execution trace | `result.trace.timeline()` |
| See reasoning | `result.reasoning` |
| Export to OTel | `result.trace.to_otel_spans()` |
| Add an observer | `AgentConfig(observers=[MyObserver()])` |
| Set tool policy | `AgentConfig(tool_policy=ToolPolicy(allow=["read_*"]))` |
| Add guardrails | `AgentConfig(guardrails=GuardrailsPipeline(input=[...]))` |
| Add audit logging | `AgentConfig(observers=[AuditLogger(log_dir="./audit")])` |
| Screen tool output | `@tool(screen_output=True)` or `AgentConfig(screen_tool_output=True)` |
| Check coherence | `AgentConfig(coherence_check=True, coherence_model="gpt-4o-mini")` |
| Reset state | `agent.reset()` |
| Add a tool at runtime | `agent.add_tool(my_tool)` |
| Remove a tool | `agent.remove_tool("tool_name")` |
