# Usage Tracking Module

**Files:** `src/selectools/usage.py`, `src/selectools/analytics.py`, `src/selectools/pricing.py`
**Classes:** `UsageStats`, `AgentUsage`, `AgentAnalytics`, `ToolMetrics`

## Table of Contents

1. [Overview](#overview)
2. [Usage Statistics](#usage-statistics)
3. [Agent Usage Tracking](#agent-usage-tracking)
4. [Tool Analytics](#tool-analytics)
5. [Pricing System](#pricing-system)
6. [Implementation](#implementation)

---

## Overview

Selectools provides **automatic cost and usage tracking** for:

- Token consumption (prompt, completion, embeddings)
- API costs (per model)
- Per-tool attribution
- Tool usage patterns and success rates
- Iteration-by-iteration breakdown

### Why Track Usage?

1. **Cost Control**: Monitor spending in real-time
2. **Optimization**: Identify expensive operations
3. **Debugging**: Understand token consumption patterns
4. **Analytics**: Track tool effectiveness

---

## Usage Statistics

### UsageStats Dataclass

Tracks a single API call:

```python
@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""

    # RAG support
    embedding_tokens: int = 0
    embedding_cost_usd: float = 0.0
```

### Example

```python
stats = UsageStats(
    prompt_tokens=1500,
    completion_tokens=300,
    total_tokens=1800,
    cost_usd=0.0045,
    model="gpt-4o-mini",
    provider="openai"
)
```

---

## Agent Usage Tracking

### AgentUsage Class

Aggregates statistics across multiple iterations:

```python
@dataclass
class AgentUsage:
    # Cumulative totals
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_embedding_tokens: int = 0
    total_embedding_cost_usd: float = 0.0

    # Per-tool breakdown
    tool_usage: Dict[str, int] = field(default_factory=dict)
    tool_tokens: Dict[str, int] = field(default_factory=dict)

    # Per-iteration history
    iterations: List[UsageStats] = field(default_factory=list)
```

### Automatic Tracking

```python
agent = Agent(tools=[...], provider=provider)

response = agent.run([Message(role=Role.USER, content="Search for Python")])

# Usage automatically tracked
print(f"Total tokens: {agent.total_tokens:,}")
print(f"Total cost: ${agent.total_cost:.6f}")
print(agent.get_usage_summary())
```

### Output

```
============================================================
📊 Usage Summary
============================================================
Total Tokens: 2,543
  - Prompt: 1,890
  - Completion: 653
Total Cost: $0.012345
Iterations: 3

Tool Usage:
  - search: 1 calls, 847 tokens
  - calculate: 2 calls, 1,696 tokens
============================================================
```

### Per-Tool Attribution

```python
print(agent.usage.tool_usage)
# {'search': 1, 'calculate': 2}

print(agent.usage.tool_tokens)
# {'search': 847, 'calculate': 1696}
```

### Iteration Breakdown

```python
for i, iteration in enumerate(agent.usage.iterations):
    print(f"Iteration {i+1}:")
    print(f"  Tokens: {iteration.total_tokens}")
    print(f"  Cost: ${iteration.cost_usd:.6f}")
```

### Reset Usage

```python
# Clear counters for new session
agent.reset_usage()
```

---

## Tool Analytics

### Enable Analytics

```python
from selectools import Agent, AgentConfig

config = AgentConfig(enable_analytics=True)
agent = Agent(tools=[...], provider=provider, config=config)
```

### AgentAnalytics Class

Tracks detailed tool metrics:

```python
@dataclass
class ToolMetrics:
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    total_cost: float = 0.0

    # Streaming metrics
    total_chunks: int = 0
    streaming_calls: int = 0

    # Parameter patterns
    parameter_usage: Dict[str, Dict[Any, int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return (self.successful_calls / self.total_calls * 100.0) if self.total_calls > 0 else 0.0

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.total_calls if self.total_calls > 0 else 0.0
```

### Get Analytics

```python
analytics = agent.get_analytics()

if analytics:
    # Print summary
    print(analytics.summary())

    # Get specific tool metrics
    search_metrics = analytics.get_metrics("search")
    print(f"Search success rate: {search_metrics.success_rate:.1f}%")
    print(f"Avg duration: {search_metrics.avg_duration:.3f}s")

    # Export to file
    analytics.to_json("analytics.json")
    analytics.to_csv("analytics.csv")
```

### Analytics Output

```
============================================================
Tool Usage Analytics
============================================================

search:
  Calls: 15
  Success rate: 93.3% (14/15)
  Failures: 1
  Avg duration: 0.234s
  Total duration: 3.510s
  Total cost: $0.001250
  Common parameters:
    query: "Python tutorials" (8x, 5 unique)

calculate:
  Calls: 23
  Success rate: 100.0% (23/23)
  Avg duration: 0.012s
  Total duration: 0.276s
  Common parameters:
    operation: "add" (12x, 4 unique)

============================================================
```

### Export Analytics

```python
# JSON format
analytics.to_json("analytics.json")
# Output: {"tools": {...}, "summary": {...}}

# CSV format
analytics.to_csv("analytics.csv")
# Columns: tool_name, total_calls, successful_calls, failed_calls,
#          success_rate, avg_duration, total_duration, total_cost
```

---

## Pricing System

### Model Registry Integration

Pricing is derived from the model registry:

```python
from selectools.models import OpenAI, Anthropic, Gemini

# Model info includes pricing
model = OpenAI.GPT_4O
print(f"Prompt: ${model.prompt_cost}/1M tokens")
print(f"Completion: ${model.completion_cost}/1M tokens")
```

### Cost Calculation

```python
from selectools.pricing import calculate_cost

cost = calculate_cost(
    model="gpt-4o",
    prompt_tokens=1000,
    completion_tokens=500
)

# Formula:
# cost = (prompt_tokens / 1M * prompt_cost) + (completion_tokens / 1M * completion_cost)
# cost = (1000 / 1M * 2.50) + (500 / 1M * 10.00)
# cost = 0.0025 + 0.005 = $0.0075
```

### Embedding Cost

```python
from selectools.pricing import calculate_embedding_cost

cost = calculate_embedding_cost(
    model="text-embedding-3-small",
    tokens=1000
)

# Formula:
# cost = tokens / 1M * prompt_cost
# cost = 1000 / 1M * 0.02 = $0.00002
```

### Cost Warnings

```python
config = AgentConfig(
    cost_warning_threshold=0.10  # Warn at $0.10
)

agent = Agent(tools=[...], provider=provider, config=config)

# When threshold exceeded:
# ⚠️  Cost Warning: Total cost $0.125000 exceeds threshold $0.100000
```

---

## Implementation

### Adding Usage Stats

```python
class Agent:
    def __init__(self, ...):
        self.usage = AgentUsage()

    def _call_provider(self, ...):
        # ... call provider ...

        # Track usage
        self.usage.add_usage(usage_stats, tool_name=None)

        # Check warning threshold
        if (
            self.config.cost_warning_threshold
            and self.usage.total_cost_usd > self.config.cost_warning_threshold
        ):
            print(f"⚠️  Cost Warning: Total cost ${self.usage.total_cost_usd:.6f} "
                  f"exceeds threshold ${self.config.cost_warning_threshold:.6f}")
```

### Recording Tool Calls

```python
def run(self, messages):
    # ... agent loop ...

    # Before tool execution
    start_time = time.time()

    try:
        result = self._execute_tool_with_timeout(tool, parameters)
        duration = time.time() - start_time

        # Track analytics
        if self.analytics:
            self.analytics.record_tool_call(
                tool_name=tool.name,
                success=True,
                duration=duration,
                params=parameters,
                cost=0.0,
                chunk_count=chunk_counter["count"]
            )

    except Exception as exc:
        duration = time.time() - start_time

        # Track failure
        if self.analytics:
            self.analytics.record_tool_call(
                tool_name=tool.name,
                success=False,
                duration=duration,
                params=parameters,
                cost=0.0
            )
```

---

## Best Practices

### 1. Monitor Costs in Production

```python
config = AgentConfig(
    cost_warning_threshold=1.00,  # $1 threshold
    verbose=True
)

agent = Agent(..., config=config)

# Check costs periodically
if agent.total_cost > 10.00:
    alert_ops_team(f"High agent costs: ${agent.total_cost}")
```

### 2. Log Usage Metrics

```python
response = agent.run([...])

# Log to monitoring system
metrics.gauge("agent.tokens", agent.total_tokens)
metrics.gauge("agent.cost_usd", agent.total_cost)
metrics.histogram("agent.iterations", len(agent.usage.iterations))
```

### 3. Analyze Tool Performance

```python
config = AgentConfig(enable_analytics=True)
agent = Agent(..., config=config)

# After running
analytics = agent.get_analytics()

for tool_name, metrics in analytics.get_all_metrics().items():
    if metrics.failure_rate > 10.0:
        logger.warning(f"Tool {tool_name} has high failure rate: {metrics.failure_rate:.1f}%")

    if metrics.avg_duration > 5.0:
        logger.warning(f"Tool {tool_name} is slow: {metrics.avg_duration:.2f}s")
```

### 4. Export for Analysis

```python
# Daily analytics export
date = datetime.now().strftime("%Y-%m-%d")
analytics.to_json(f"analytics/{date}.json")
analytics.to_csv(f"analytics/{date}.csv")
```

### 5. Reset Between Sessions

```python
# For chatbots, reset per user session
def new_session(user_id):
    agent.reset_usage()
    if agent.analytics:
        agent.analytics.reset()
```

---

## Testing

```python
def test_usage_tracking():
    agent = Agent(tools=[...], provider=LocalProvider())

    agent.run([Message(role=Role.USER, content="Test")])

    assert agent.total_tokens > 0
    assert len(agent.usage.iterations) > 0

def test_cost_warning():
    captured = []

    def capture_print(*args):
        captured.append(" ".join(str(a) for a in args))

    with patch("builtins.print", capture_print):
        config = AgentConfig(cost_warning_threshold=0.001)
        agent = Agent(..., config=config)
        agent.run([...])

    assert any("Cost Warning" in msg for msg in captured)

def test_analytics():
    config = AgentConfig(enable_analytics=True)
    agent = Agent(..., config=config)

    agent.run([...])

    analytics = agent.get_analytics()
    assert analytics is not None
    assert len(analytics.get_all_metrics()) > 0
```

---

## Cache-Aware Usage Tracking

When response caching is enabled via `AgentConfig(cache=...)`, usage tracking remains accurate even for cached responses.

### How It Works

- **Cache miss**: Provider is called normally; `UsageStats` tracked as usual
- **Cache hit**: The stored `UsageStats` from the original call is replayed via `agent.usage.add_usage()`

This means `agent.total_cost` and `agent.total_tokens` reflect the _logical_ usage (what it would have cost), not just the actual API calls.

### Cache Stats

The cache itself tracks hit/miss/eviction metrics:

```python
from selectools import InMemoryCache

cache = InMemoryCache(max_size=500, default_ttl=600)
config = AgentConfig(cache=cache)
agent = Agent(tools=[...], provider=provider, config=config)

# Run some queries...
agent.run([Message(role=Role.USER, content="Hello")])
agent.reset()
agent.run([Message(role=Role.USER, content="Hello")])  # cache hit

# Cache performance
stats = cache.stats
print(f"Hit rate: {stats.hit_rate:.1%}")    # 50.0%
print(f"Hits: {stats.hits}")                # 1
print(f"Misses: {stats.misses}")            # 1
print(f"Evictions: {stats.evictions}")       # 0
```

### CacheStats Dataclass

```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self.hits / total if total > 0 else 0.0
```

### Monitoring Cache + Usage Together

```python
# Agent usage (logical)
print(f"Tokens used: {agent.total_tokens:,}")
print(f"Cost: ${agent.total_cost:.6f}")

# Cache efficiency
print(f"Cache hit rate: {cache.stats.hit_rate:.1%}")
print(f"API calls saved: {cache.stats.hits}")

# Cost savings estimate
avg_cost_per_call = agent.total_cost / cache.stats.total_requests
savings = cache.stats.hits * avg_cost_per_call
print(f"Estimated savings: ${savings:.6f}")
```

---

## RAG Usage Tracking

When using RAG, both LLM and embedding costs are tracked:

```python
from selectools.rag import RAGAgent, VectorStore
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
store = VectorStore.create("memory", embedder=embedder)

agent = RAGAgent.from_directory("./docs", provider, store)

response = agent.run("What are the features?")

# Usage includes both LLM and embeddings
print(agent.usage)

# Output:
# ============================================================
# 📊 Usage Summary
# ============================================================
# Total Tokens: 5,432
#   - Prompt: 3,210
#   - Completion: 1,200
#   - Embeddings: 1,022
# Total Cost: $0.002150
#   - LLM: $0.002000
#   - Embeddings: $0.000150
# ============================================================
```

---

## Further Reading

- [Agent Module](AGENT.md) - How usage is tracked
- [Agent Module - Caching](AGENT.md#response-caching) - Response caching details
- [Providers Module](PROVIDERS.md) - Usage stat extraction
- [Models Module](MODELS.md) - Pricing information
- [RAG Module](RAG.md) - RAG usage tracking

---

**Next Steps:** Learn about the RAG system in the [RAG Module](RAG.md).
