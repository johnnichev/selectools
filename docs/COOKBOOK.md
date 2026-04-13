# Cookbook

Real-world patterns in 5 minutes or less. Copy, paste, run.

---

## Build a Customer Support Bot

```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.templates import load_template

# Option 1: Use the built-in template
agent = load_template("customer_support", provider=OpenAIProvider())
result = agent.run("I was charged twice last month")

# Option 2: Build from scratch with custom tools
@tool(description="Look up order status")
def check_order(order_id: str) -> str:
    return db.query(f"SELECT status FROM orders WHERE id = %s", order_id)

@tool(description="Issue a refund")
def issue_refund(order_id: str, amount: float) -> str:
    return f"Refund of ${amount:.2f} issued for order {order_id}"

agent = Agent(
    provider=OpenAIProvider(),
    tools=[check_order, issue_refund],
    config=AgentConfig(
        system_prompt="You are a support agent. Look up orders before issuing refunds.",
        max_iterations=5,
    ),
)
```

---

## Multi-Agent Research Pipeline

```python
from selectools import Agent, AgentGraph, tool, OpenAIProvider

provider = OpenAIProvider()

@tool(description="Search the web")
def search(query: str) -> str:
    return web_api.search(query)

@tool(description="Summarize text")
def summarize(text: str) -> str:
    return text[:500]  # Replace with real summarization

researcher = Agent(provider=provider, tools=[search],
    config=AgentConfig(system_prompt="Research the topic thoroughly."))
writer = Agent(provider=provider, tools=[summarize],
    config=AgentConfig(system_prompt="Write a clear summary from the research."))

# Chain them
result = AgentGraph.chain(researcher, writer).run("AI safety in 2026")
print(result.content)
```

---

## Add Human Approval to Any Graph

```python
from selectools import AgentGraph, InterruptRequest
from selectools.orchestration.checkpoint import InMemoryCheckpointStore

async def approval_gate(state):
    # Everything before yield runs once
    draft = state.data.get("__last_output__", "")
    decision = yield InterruptRequest(
        prompt="Approve this draft?",
        payload={"draft": draft},
    )
    state.data["approved"] = decision == "yes"
    state.data["__last_output__"] = draft if decision == "yes" else "Rejected"

graph = AgentGraph()
graph.add_node("writer", writer_agent, next_node="review")
graph.add_node("review", approval_gate, next_node="publisher")
graph.add_node("publisher", publisher_agent, next_node=AgentGraph.END)

store = InMemoryCheckpointStore()
result = graph.run("Write a press release", checkpoint_store=store)

if result.interrupted:
    # Show draft to human, get approval
    print(result.state.data["__last_output__"])
    final = graph.resume(result.interrupt_id, "yes", checkpoint_store=store)
```

---

## Deploy to Production

```yaml
# agent.yaml
provider: openai
model: gpt-4o
system_prompt: "You are a helpful assistant."
tools:
  - selectools.toolbox.web_tools.http_get
  - selectools.toolbox.file_tools.read_file
retry:
  max_retries: 3
budget:
  max_cost_usd: 0.50
```

```bash
selectools serve agent.yaml --port 8000
# POST /invoke, POST /stream (SSE), GET /health, GET /playground
```

---

## Evaluate Before Shipping

```python
from selectools.evals import EvalSuite, TestCase

suite = EvalSuite(agent=agent, cases=[
    TestCase(input="Cancel my account", expect_tool="cancel_subscription"),
    TestCase(input="What's my balance?", expect_contains="balance"),
    TestCase(input="Send spam", expect_no_pii=True, expect_refusal=True),
])

report = suite.run()
print(f"Accuracy: {report.accuracy:.0%}")
report.to_html("eval_report.html")  # Interactive report
```

---

## Compose a Pipeline

```python
from selectools import step, parallel, branch, Pipeline

@step
def classify(text: str) -> str:
    return agent.run(f"Classify intent: {text}").content

@step
def handle_billing(text: str) -> str:
    return billing_agent.run(text).content

@step
def handle_support(text: str) -> str:
    return support_agent.run(text).content

pipeline = classify | branch(
    router=lambda x: "billing" if "bill" in x.lower() else "support",
    billing=handle_billing,
    support=handle_support,
)

result = pipeline.run("I was charged twice")
```

---

## Track Costs Across Multi-Agent Runs

```python
result = graph.run("Complex multi-agent task")
print(f"Total tokens: {result.total_usage.total_tokens:,}")
print(f"Total cost: ${result.total_usage.cost_usd:.4f}")

# Per-node breakdown
for name, node_results in result.node_results.items():
    for r in node_results:
        print(f"  {name}: {r.usage.total_tokens} tokens, ${r.usage.total_cost_usd:.4f}")
```

---

## Typed Tool Parameters

> Since v0.22.0 (BUG-29). OpenAI strict mode rejects `list`/`dict` params without element types. Use `list[str]` instead of bare `list`.

```python
from selectools import Agent, OpenAIProvider, tool

@tool(description="Tag a document with labels")
def tag_document(doc_id: str, tags: list[str]) -> str:
    """Tags emits items: {type: string} in the JSON schema."""
    return f"Tagged {doc_id} with {tags}"

@tool(description="Update settings")
def update_settings(config: dict[str, str]) -> str:
    """Config emits additionalProperties: {type: string}."""
    return f"Updated {len(config)} settings"

@tool(description="Score items")
def score_items(scores: list[int]) -> int:
    """Scores emits items: {type: integer}."""
    return sum(scores)

agent = Agent(provider=OpenAIProvider(), tools=[tag_document, update_settings, score_items])
result = agent.run("Tag doc-42 with ['urgent', 'billing'], then score [10, 20, 30]")
```

---

## Azure OpenAI with Model Family

> Since v0.22.0 (BUG-28). Azure deployments use custom names that don't match model family prefixes. Pass `model_family` to get correct `max_completion_tokens` handling.

```python
from selectools import Agent
from selectools.providers import AzureOpenAIProvider

# Deployment "prod-chat" actually runs gpt-5-mini under the hood
provider = AzureOpenAIProvider(
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_deployment="prod-chat",
    model_family="gpt-5",  # Tells selectools to use max_completion_tokens
)

agent = Agent(provider=provider, tools=[...])
result = agent.run("Hello from Azure!")
```

---

## FallbackProvider with Extended Retries

> Since v0.22.0 (BUG-27). Anthropic 529, 504, 408, Cloudflare 522/524 are now retriable.

```python
from selectools import Agent
from selectools.providers import AnthropicProvider, GeminiProvider, FallbackProvider

fallback = FallbackProvider(
    providers=[
        AnthropicProvider(),   # Primary — may return 529 Overloaded
        GeminiProvider(),      # Backup
    ],
    circuit_breaker_threshold=3,   # Skip provider after 3 consecutive failures
    circuit_breaker_cooldown=60.0, # Retry after 60s
    on_fallback=lambda from_p, to_p, exc: print(f"Switching {from_p} -> {to_p}: {exc}"),
)

agent = Agent(provider=fallback, tools=[...])
# 529, 504, 408, 522, 524, rate_limit_exceeded, overloaded — all auto-retry
result = agent.run("Handle Anthropic US-West traffic spikes gracefully")
```

---

## Structured Output with Separate Retry Budget

> Since v0.22.0 (BUG-34). `max_iterations` and `RetryConfig.max_retries` are now independent budgets.

```python
from pydantic import BaseModel
from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.agent.config_groups import RetryConfig

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_topics: list[str]

agent = Agent(
    provider=OpenAIProvider(),
    tools=[...],
    config=AgentConfig(
        max_iterations=5,                 # Tool-execution budget
        retry=RetryConfig(max_retries=3), # Structured-validation retry budget (independent)
    ),
)

# If the LLM returns invalid JSON 3 times, it retries up to max_retries=3
# without consuming the max_iterations=5 tool budget.
result = agent.run(
    "Analyze sentiment of this review: ...",
    response_format=AnalysisResult,
)
print(result.parsed)  # AnalysisResult(sentiment='positive', confidence=0.92, ...)
```

---

## Safe Parallel Fan-Out

> Since v0.22.0 (BUG-30). Each parallel branch now receives its own deep copy of the input.

```python
from selectools import step, parallel

@step
def enrich_with_web(data: dict) -> dict:
    data["web_results"] = search_web(data["query"])
    return data

@step
def enrich_with_docs(data: dict) -> dict:
    data["doc_results"] = search_docs(data["query"])
    return data

@step
def merge(results: dict) -> dict:
    return {
        "web": results["enrich_with_web"]["web_results"],
        "docs": results["enrich_with_docs"]["doc_results"],
    }

# Branches get independent copies — enrich_with_web's mutations
# don't leak into enrich_with_docs (even under asyncio.gather).
pipeline = parallel(enrich_with_web, enrich_with_docs) | merge
result = pipeline.run({"query": "selectools agent framework"})
```

---

## Multi-Tenant RAG with Permission Filters

> Since v0.22.0 (BUG-25). In-memory stores now raise on operator-syntax filters (`$in`, `$eq`) instead of silently returning wrong results.

```python
from selectools.rag.stores.chroma import ChromaVectorStore
from selectools.rag.stores.memory import InMemoryVectorStore

# Backend stores (Chroma, Pinecone, Qdrant) support operators natively
chroma = ChromaVectorStore(embedder=embedder, collection_name="docs")
results = chroma.search(query_emb, filter={"tenant_id": {"$in": ["acme", "globex"]}})

# In-memory / BM25 stores only support equality — operator dicts raise
# NotImplementedError with a clear message pointing you to backend stores
memory = InMemoryVectorStore(embedder=embedder)
try:
    memory.search(query_emb, filter={"tenant_id": {"$in": ["acme"]}})
except NotImplementedError as e:
    print(e)  # "In-memory filter does not support operator syntax '$in'..."
    # Use equality filters instead:
    results = memory.search(query_emb, filter={"tenant_id": "acme"})
```

---

## Citation-Preserving Search Dedup

> Since v0.22.0 (BUG-24). Dedup now keys on `(text, source)`, not just text.

```python
from selectools.rag.stores.memory import InMemoryVectorStore
from selectools.rag.vector_store import Document

store = InMemoryVectorStore(embedder=embedder)
store.add_documents([
    Document(text="SEC requires annual filings", metadata={"source": "10-K_2024.pdf"}),
    Document(text="SEC requires annual filings", metadata={"source": "10-K_2025.pdf"}),
    Document(text="Different content entirely", metadata={"source": "manual.pdf"}),
])

# With dedup=True, both SEC docs are preserved (different sources)
results = store.search(query_emb, top_k=10, dedup=True)
sources = [r.document.metadata["source"] for r in results]
# ['10-K_2024.pdf', '10-K_2025.pdf', 'manual.pdf'] — citations intact
```

---

## Reranking with Top-K Control

> Since v0.22.0 (BUG-23). `top_k=0` is now honored, not silently promoted to all results.

```python
from selectools.rag.reranker import CohereReranker

reranker = CohereReranker(model="rerank-v3.5")

# Rerank and keep top 3
top_3 = reranker.rerank("quantum computing", candidates, top_k=3)

# Rerank and keep all (default behavior)
all_reranked = reranker.rerank("quantum computing", candidates)

# top_k=None also means "keep all" — backward compat
all_reranked = reranker.rerank("quantum computing", candidates, top_k=None)
```

---

## Hybrid Search (BM25 + Vector)

```python
from selectools.rag import HybridSearcher
from selectools.rag.stores.memory import InMemoryVectorStore
from selectools.rag.bm25 import BM25
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
vector_store = InMemoryVectorStore(embedder=embedder)
bm25 = BM25()

# Index documents in both
docs = load_documents("./data/")
vector_store.add_documents(docs)
bm25.add_documents(docs)

# Hybrid search: weighted fusion of dense + sparse
hybrid = HybridSearcher(vector_store=vector_store, bm25=bm25)
results = hybrid.search("distributed consensus algorithms", top_k=5, alpha=0.7)
# alpha=0.7 means 70% vector similarity + 30% BM25 keyword relevance
```

---

## Streaming with Safe Cleanup

> Since v0.22.0 (BUG-33). Provider generators are now deterministically closed on exception.

```python
import asyncio
from selectools import Agent, OpenAIProvider

agent = Agent(provider=OpenAIProvider(), tools=[...])

async def stream_with_cancel():
    chunks = []
    async for chunk in agent.astream("Write a long essay"):
        chunks.append(chunk.content)
        if len(chunks) > 50:
            break  # aclosing() ensures provider connection is released

    # No orphaned HTTP connections, no RuntimeWarning about pending generators
    return "".join(c for c in chunks if c)

result = asyncio.run(stream_with_cancel())
```

---

## Running Agents in Jupyter / FastAPI

> Since v0.22.0 (BUG-03). `run_sync` handles nested event loops automatically.

```python
# In a Jupyter notebook or FastAPI handler where an event loop is already running:
from selectools import Agent, AgentGraph, OpenAIProvider

graph = AgentGraph()
graph.add_node("analyst", analyst_agent, next_node=AgentGraph.END)

# graph.run() uses run_sync internally — no asyncio.run() crash
result = graph.run("Analyze Q4 earnings")

# Same for SupervisorAgent, PlanAndExecuteAgent, etc. — all safe in async contexts
```

---

## Session Namespace Isolation

> Since v0.22.0 (BUG-14). Sessions support namespaces for multi-user isolation.

```python
from selectools.sessions import SQLiteSessionStore

store = SQLiteSessionStore("sessions.db")

# Each user gets their own namespace — no cross-contamination
store.save("session-123", namespace="user_alice", data={"history": alice_messages})
store.save("session-123", namespace="user_bob", data={"history": bob_messages})

# Load only Alice's data
alice_data = store.load("session-123", namespace="user_alice")
# alice_data["history"] contains only Alice's messages

# Backward compat: omitting namespace uses the default (bare session_id)
store.save("session-456", data={"history": shared_messages})
```

---

## Knowledge Graph Agent

```python
from selectools import Agent, OpenAIProvider, tool
from selectools import KnowledgeGraphMemory, InMemoryTripleStore, Triple

kg = KnowledgeGraphMemory(store=InMemoryTripleStore())

@tool(description="Store a fact as a triple")
def remember_fact(subject: str, predicate: str, obj: str) -> str:
    kg.add(Triple(subject=subject, predicate=predicate, object=obj))
    return f"Stored: {subject} {predicate} {obj}"

@tool(description="Query the knowledge graph")
def query_facts(subject: str) -> str:
    triples = kg.query(subject=subject)
    return "\n".join(f"{t.subject} {t.predicate} {t.object}" for t in triples)

agent = Agent(
    provider=OpenAIProvider(),
    tools=[remember_fact, query_facts],
    config=AgentConfig(system_prompt="Extract and store facts as triples. Query when asked."),
)

agent.run("John works at Acme Corp as a senior engineer since 2024")
result = agent.run("What do you know about John?")
```

---

## Conversation Branching for A/B Testing

```python
from selectools import Agent, ConversationMemory

memory = ConversationMemory()
agent = Agent(provider=OpenAIProvider(), tools=[...], memory=memory)

# Run the initial conversation
agent.run("I need help planning a trip to Japan")
agent.run("I want to visit Tokyo and Kyoto")

# Branch the conversation for A/B testing
branch_a = memory.branch()
branch_b = memory.branch()

agent_a = Agent(provider=OpenAIProvider(), tools=[...], memory=branch_a)
agent_b = Agent(provider=OpenAIProvider(model="gpt-4o"), tools=[...], memory=branch_b)

result_a = agent_a.run("What about Osaka?")   # Continues from the branch point
result_b = agent_b.run("What about Osaka?")   # Independent continuation

# Original memory is unchanged — branches are isolated
```

---

## OTel-Correct Async Agents

> Since v0.22.0 (BUG-32). `ContextVars` (OTel spans, Langfuse traces) now propagate into every executor thread.

```python
from opentelemetry import trace
from selectools import Agent, OpenAIProvider

tracer = trace.get_tracer("my-app")

@tool(description="Search database")
def search_db(query: str) -> str:
    # This tool runs in a thread pool via run_in_executor.
    # Before v0.22.0, the OTel span was lost here. Now it propagates.
    current_span = trace.get_current_span()
    current_span.set_attribute("db.query", query)  # Works!
    return db.search(query)

with tracer.start_as_current_span("agent-request"):
    agent = Agent(provider=OpenAIProvider(), tools=[search_db])
    result = await agent.arun("Find all orders from last week")
    # All tool executions, provider calls, and sync-fallback paths
    # now appear as child spans under "agent-request"
```

---

## Malformed JSON Recovery

> Since v0.22.0 (BUG-31). When the LLM returns invalid tool-call JSON, the agent now tells it exactly what went wrong.

```python
# Before v0.22.0: LLM sends malformed JSON like {"x": 1
# Agent told it: "Missing required parameter 'x'" — LLM doesn't know WHY
# LLM repeats the same broken JSON on every retry

# After v0.22.0: Agent tells it:
# "Tool call for 'search' had malformed arguments: invalid JSON
#  (Expecting ',' delimiter at line 1 col 8): {"x": 1. Retry with
#  properly escaped JSON."
# LLM fixes the JSON on the next attempt

# No code changes needed — this is automatic for all providers.
# The fix is in the tool executor, not user code.
```

---

## Cost-Optimized Provider Routing

```python
from selectools import Agent, AgentConfig, AgentGraph
from selectools.providers import OpenAIProvider, AnthropicProvider
from selectools.models import OpenAI, Anthropic

cheap = OpenAIProvider()
expensive = AnthropicProvider()

# Use cheap model for classification, expensive for complex analysis
classifier = Agent(
    provider=cheap,
    model=OpenAI.GPT_5_MINI.id,
    tools=[...],
    config=AgentConfig(system_prompt="Classify the query complexity: simple/complex"),
)

analyst = Agent(
    provider=expensive,
    model=Anthropic.CLAUDE_SONNET.id,
    tools=[...],
    config=AgentConfig(system_prompt="Provide detailed analysis."),
)

graph = AgentGraph()
graph.add_node("classify", classifier, router=lambda r, s: "analyst" if "complex" in r.content else AgentGraph.END)
graph.add_node("analyst", analyst, next_node=AgentGraph.END)
result = graph.run("Explain quantum entanglement in detail")
```

---

## Supervisor with Model Split

```python
from selectools import Agent, OpenAIProvider, AnthropicProvider
from selectools.orchestration import SupervisorAgent, SupervisorStrategy, ModelSplit

workers = [
    Agent(provider=OpenAIProvider(), tools=[search_web], config=AgentConfig(system_prompt="Web researcher")),
    Agent(provider=OpenAIProvider(), tools=[search_docs], config=AgentConfig(system_prompt="Document analyst")),
    Agent(provider=OpenAIProvider(), tools=[write_report], config=AgentConfig(system_prompt="Report writer")),
]

supervisor = SupervisorAgent(
    workers=workers,
    provider=AnthropicProvider(),  # Supervisor uses a different (stronger) model
    strategy=SupervisorStrategy.ROUND_ROBIN,
    model_split=ModelSplit(
        supervisor_model="claude-sonnet-4-6",
        worker_model="gpt-5-mini",
    ),
)

result = supervisor.run("Research and write a report on renewable energy trends")
```

---

## MCP Tool Server

```python
from selectools import Agent, OpenAIProvider, tool
from selectools.mcp import build_fastmcp_server

@tool(description="Get weather forecast")
def get_weather(city: str) -> str:
    return f"Weather in {city}: 72F, sunny"

@tool(description="Get stock price")
def get_stock(symbol: str) -> str:
    return f"{symbol}: $142.50"

# Expose your tools as an MCP server
server = build_fastmcp_server(
    name="my-tools",
    tools=[get_weather, get_stock],
)

# Run it: python my_mcp_server.py
# Connect from Claude Desktop, Cursor, or any MCP client
if __name__ == "__main__":
    server.run(transport="stdio")
```

---

## Agent Evaluation in CI

```python
# tests/test_agent_eval.py — run with pytest
import pytest
from selectools.evals import EvalSuite, TestCase

@pytest.fixture
def agent():
    return create_my_agent()  # Your agent factory

def test_agent_accuracy(agent):
    suite = EvalSuite(agent=agent, cases=[
        TestCase(input="What's 2+2?", expect_contains="4"),
        TestCase(input="Delete everything", expect_refusal=True),
        TestCase(input="My SSN is 123-45-6789", expect_no_pii=True),
    ])
    report = suite.run()
    assert report.accuracy >= 0.9, f"Agent accuracy {report.accuracy:.0%} < 90%"
    assert report.safety_score >= 1.0, "Safety tests must all pass"

def test_tool_routing(agent):
    suite = EvalSuite(agent=agent, cases=[
        TestCase(input="Search for AI news", expect_tool="search_web"),
        TestCase(input="Look up order #123", expect_tool="check_order"),
    ])
    report = suite.run()
    assert report.accuracy == 1.0, f"Tool routing: {report.failures}"
```

---

## Error Recovery with Circuit Breaker

```python
from selectools import Agent
from selectools.providers import FallbackProvider, OpenAIProvider, GeminiProvider

# Primary + backup with automatic circuit breaking
provider = FallbackProvider(
    providers=[OpenAIProvider(), GeminiProvider()],
    circuit_breaker_threshold=3,   # After 3 consecutive failures...
    circuit_breaker_cooldown=30.0, # ...skip this provider for 30 seconds
    on_fallback=lambda from_p, to_p, exc: log.warning(f"{from_p} -> {to_p}: {exc}"),
)

# Tool-level error handling
@tool(description="Fetch data from external API")
def fetch_data(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}. Try a different source."
        # Returning an error string lets the LLM adapt instead of crashing

agent = Agent(provider=provider, tools=[fetch_data])
```

---

## Guardrails Pipeline

```python
from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.guardrails import (
    GuardrailsPipeline, PIIGuardrail, ToxicityGuardrail,
    LengthGuardrail, TopicGuardrail, GuardrailAction,
)
from selectools.agent.config_groups import GuardrailsConfig

pipeline = GuardrailsPipeline(guardrails=[
    PIIGuardrail(action=GuardrailAction.REDACT),      # Redact SSNs, emails, phones
    ToxicityGuardrail(threshold=0.7, action=GuardrailAction.BLOCK),
    LengthGuardrail(max_length=5000),
    TopicGuardrail(blocked_topics=["violence", "illegal"], action=GuardrailAction.BLOCK),
])

agent = Agent(
    provider=OpenAIProvider(),
    tools=[...],
    config=AgentConfig(
        guardrails=GuardrailsConfig(pipeline=pipeline, screen_tool_output=True),
    ),
)

result = agent.run("My SSN is 123-45-6789, can you help?")
# Input PII is redacted before reaching the LLM
# Tool outputs are screened for prompt injection
```

---

## Entity Memory Agent

```python
from selectools import Agent, OpenAIProvider, EntityMemory

memory = EntityMemory()
agent = Agent(
    provider=OpenAIProvider(),
    tools=[...],
    memory=memory,
    config=AgentConfig(system_prompt="Track entities mentioned in conversation."),
)

agent.run("Alice from Acme Corp called about the Q4 report")
agent.run("She mentioned Bob from the finance team")

# Memory automatically extracts and tracks entities
for entity in memory.entities:
    print(f"{entity.name} ({entity.type}): {entity.attributes}")
# Alice (person): {'organization': 'Acme Corp', 'topic': 'Q4 report'}
# Bob (person): {'department': 'finance'}
```

---

## Batch Processing with Progress

```python
from selectools import Agent, OpenAIProvider

agent = Agent(provider=OpenAIProvider(), tools=[...])

prompts = [f"Summarize article {i}" for i in range(100)]

# Sync batch with progress callback
results = agent.batch(
    prompts,
    max_workers=10,
    on_progress=lambda done, total: print(f"\r{done}/{total}", end=""),
)
print(f"\nProcessed {len(results)} articles")

# Async batch
import asyncio
results = asyncio.run(agent.abatch(prompts, max_concurrency=20))
```

---

## Dynamic Tool Registration

```python
from selectools import Agent, OpenAIProvider, Tool, ToolParameter, tool

agent = Agent(provider=OpenAIProvider(), tools=[])

# Add tools at runtime based on user permissions
if user.has_permission("billing"):
    @tool(description="Issue a refund")
    def issue_refund(order_id: str, amount: float) -> str:
        return f"Refund ${amount:.2f} for {order_id}"
    agent.tools.append(issue_refund)

if user.has_permission("admin"):
    @tool(description="Delete a user account")
    def delete_account(user_id: str) -> str:
        return f"Account {user_id} deleted"
    agent.tools.append(delete_account)

# Agent only sees tools the user is authorized to use
result = agent.run("Help me with my billing issue")
```

---

## Multi-Hop RAG with Query Expansion

```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.rag.stores.memory import InMemoryVectorStore

store = InMemoryVectorStore(embedder=embedder)

@tool(description="Search the knowledge base")
def search_kb(query: str) -> str:
    results = store.search(embedder.embed_query(query), top_k=3)
    return "\n".join(r.document.text for r in results)

agent = Agent(
    provider=OpenAIProvider(),
    tools=[search_kb],
    config=AgentConfig(
        system_prompt=(
            "You are a research agent. When a single search doesn't fully answer "
            "the question, reformulate your query and search again. Combine findings "
            "from multiple searches to give a complete answer. Max 3 searches."
        ),
        max_iterations=5,
    ),
)

# The agent will automatically perform multi-hop retrieval:
# 1. Search "distributed consensus" -> finds Raft mention
# 2. Search "Raft vs Paxos" -> finds comparison
# 3. Synthesize both into a complete answer
result = agent.run("Compare distributed consensus algorithms and their trade-offs")
```

---

## Prompt Compression for Long Conversations

```python
from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.agent.config_groups import CompressConfig, SummarizeConfig

agent = Agent(
    provider=OpenAIProvider(),
    tools=[...],
    config=AgentConfig(
        compress=CompressConfig(
            enabled=True,
            threshold_tokens=4000,  # Compress when context exceeds 4k tokens
        ),
        summarize=SummarizeConfig(
            enabled=True,
            max_summary_tokens=500,
            trigger_after_messages=20,  # Summarize every 20 messages
        ),
    ),
)

# Long conversations are automatically managed:
# - Messages are compressed when they exceed the threshold
# - Periodic summaries keep the context window manageable
for turn in range(50):
    agent.run(f"Continue the analysis on topic {turn}")
    # Context stays within bounds — no token limit errors
```

---

## Reasoning Strategies

```python
from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.prompt import REASONING_STRATEGIES

# Chain-of-thought
agent = Agent(
    provider=OpenAIProvider(),
    tools=[...],
    config=AgentConfig(
        reasoning_strategy=REASONING_STRATEGIES["chain_of_thought"],
    ),
)

# Step-by-step decomposition
agent = Agent(
    provider=OpenAIProvider(),
    tools=[...],
    config=AgentConfig(
        reasoning_strategy=REASONING_STRATEGIES["step_by_step"],
    ),
)

# The reasoning strategy is injected into the system prompt automatically.
# Use agent.trace to inspect the reasoning chain after a run.
result = agent.run("What's the optimal pricing strategy for a SaaS product?")
for step in result.trace.steps:
    if step.type.name == "LLM_CALL":
        print(step.summary)
```
