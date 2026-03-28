# Coming from LangChain / LangGraph

Side-by-side migration guide. Every example shows the LangChain way and the selectools equivalent.

---

## Tool Calling

**LangChain:**
```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([search])
result = llm_with_tools.invoke("Search for Python tutorials")
```

**selectools:**
```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool

@tool(description="Search the web")
def search(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(tools=[search], provider=OpenAIProvider())
result = agent.run("Search for Python tutorials")
print(result.content)      # The answer
print(result.reasoning)    # Why it chose that tool
print(result.trace)        # Full execution timeline
```

**What's different:** selectools gives you `result.reasoning` and `result.trace` for free. No LangSmith needed.

---

## Multi-Agent Graph

**LangGraph:**
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    text: str

def planner(state): return {"text": "planned"}
def writer(state): return {"text": "written"}
def reviewer(state): return {"text": "reviewed"}

g = StateGraph(State)
g.add_node("planner", planner)
g.add_node("writer", writer)
g.add_node("reviewer", reviewer)
g.add_edge(START, "planner")
g.add_edge("planner", "writer")
g.add_edge("writer", "reviewer")
g.add_edge("reviewer", END)
app = g.compile()
result = app.invoke({"text": "prompt"})
```

**selectools:**
```python
from selectools import AgentGraph

result = AgentGraph.chain(planner, writer, reviewer).run("prompt")
```

**What's different:** No `StateGraph`, no `TypedDict`, no `compile()`. Plain Python.

---

## Conditional Routing

**LangGraph:**
```python
def should_continue(state):
    if state["needs_review"]:
        return "reviewer"
    return END

g.add_conditional_edges("writer", should_continue, {
    "reviewer": "reviewer",
    END: END,
})
```

**selectools:**
```python
graph.add_conditional_edge(
    "writer",
    lambda state: "reviewer" if state.data.get("needs_review") else AgentGraph.END,
)
```

**What's different:** No `path_map` required. The function returns a node name directly.

---

## Human-in-the-Loop

**LangGraph:**
```python
# Node restarts from the top on resume — guard expensive work manually
def review_node(state):
    if "analysis" not in state:
        state["analysis"] = expensive_llm_call(state["draft"])  # runs TWICE without guard
    return Command(goto="human_input")
```

**selectools:**
```python
# Generator pauses at yield, resumes at exact yield point
async def review_node(state):
    analysis = await expensive_llm_call(state.data["draft"])  # runs ONCE
    decision = yield InterruptRequest(prompt="Approve?", payload=analysis)
    state.data["approved"] = decision == "yes"
```

**What's different:** No manual `if key not in state` guards. The generator preserves local variables across pause/resume.

---

## Streaming

**LangChain (LCEL):**
```python
chain = prompt | llm | parser
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="")
```

**selectools:**
```python
async for item in agent.astream("Tell me about AI"):
    if isinstance(item, str):
        print(item, end="")  # Text chunk
    elif isinstance(item, AgentResult):
        print(f"\nDone: {item.iterations} iterations")
```

**What's different:** `astream()` yields both text chunks AND tool calls natively. No separate streaming modes.

---

## Composable Pipelines

**LangChain (LCEL):**
```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | prompt
    | llm
    | parser
)
```

**selectools:**
```python
from selectools import step, parallel, branch

@step
def summarize(text: str) -> str:
    return agent.run(f"Summarize: {text}").content

@step
def translate(text: str) -> str:
    return agent.run(f"Translate: {text}").content

pipeline = summarize | translate
result = pipeline.run("Long article...")
```

**What's different:** Steps are plain functions. No `Runnable` base class, no `RunnablePassthrough`. When it breaks, you get a Python traceback.

---

## Evaluation

**LangChain:** Requires LangSmith (paid SaaS).

**selectools:**
```python
from selectools.evals import EvalSuite, TestCase

suite = EvalSuite(agent=agent, cases=[
    TestCase(input="Cancel account", expect_tool="cancel_sub"),
    TestCase(input="Balance?", expect_contains="balance"),
])
report = suite.run()
report.to_html("report.html")
```

**What's different:** 39 evaluators built into the library. No paid service, no separate install.

---

## Deployment

**LangChain:** `pip install langserve` + FastAPI boilerplate + `add_routes()`.

**selectools:**
```bash
selectools serve agent.yaml
```

That's it. HTTP API + SSE streaming + playground UI. Or in Python:

```python
from selectools.serve import create_app
app = create_app(agent, playground=True)
app.serve(port=8000)
```

---

## Cost Tracking

**LangChain:** Manual. Use callbacks or LangSmith.

**selectools:**
```python
result = agent.run("Search and summarize")
print(f"Cost: ${result.usage.total_cost_usd:.4f}")
print(f"Tokens: {result.usage.total_tokens}")
```

Automatic per-call cost tracking across 152 models with built-in pricing data.

---

## What LangChain Does Better (honest)

- **Ecosystem size** — hundreds of integrations, community answers everywhere
- **LangSmith** — if you want hosted tracing/evals, it's polished
- **Maturity** — battle-tested at thousands of companies
- **LangGraph Platform** — managed deployment with cron, webhooks, SSO

If you need a managed platform or 50+ integrations today, LangChain is the safer bet. If you want a library that stays out of your way and includes everything in one package, give selectools a try.
