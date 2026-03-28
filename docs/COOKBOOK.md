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
