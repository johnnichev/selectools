# Migration Guides

Side-by-side comparisons with LangChain, CrewAI, AutoGen, and LlamaIndex. Every example shows the other framework's way and the selectools equivalent.

---

# Coming from LangChain / LangGraph

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

**What's different:** 50 evaluators built into the library. No paid service, no separate install.

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

---
---

# Coming from CrewAI

CrewAI uses role-based agents with a Crew coordinator. Selectools uses graph-based orchestration where any agent can route to any other.

---

## Agent Definition

**CrewAI:**
```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="You are an expert researcher...",
    llm="gpt-4o",
)
writer = Agent(
    role="Writer",
    goal="Write clear content",
    backstory="You are a skilled writer...",
    llm="gpt-4o",
)

task1 = Task(description="Research AI trends", agent=researcher)
task2 = Task(description="Write a report", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**selectools:**
```python
from selectools import Agent, AgentConfig, AgentGraph
from selectools.providers import OpenAIProvider

provider = OpenAIProvider()
researcher = Agent(
    tools=[search],
    provider=provider,
    config=AgentConfig(model="gpt-4o", system_prompt="You are an expert researcher."),
)
writer = Agent(
    tools=[],
    provider=provider,
    config=AgentConfig(model="gpt-4o", system_prompt="You are a skilled writer."),
)

graph = AgentGraph()
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge("START", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "END")
result = graph.run("Research AI trends and write a report")
```

**What's different:** No `role`/`goal`/`backstory` boilerplate. System prompts are plain strings. Graphs give you conditional routing, parallel execution, and HITL that CrewAI's sequential task model doesn't support.

---

## What CrewAI Does Better (honest)

- **Simpler mental model** for sequential task chains (no graph concepts)
- **Role-based prompting** is automatic (role/goal/backstory templating)
- **Enterprise plan** includes hosted orchestration

---
---

# Coming from AutoGen

AutoGen uses conversational agents that chat with each other. Selectools uses directed graphs with explicit routing.

---

## Multi-Agent Chat

**AutoGen:**
```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={"model": "gpt-4o"})
user_proxy = UserProxyAgent("user", code_execution_config={"work_dir": "coding"})

user_proxy.initiate_chat(assistant, message="Write a Python script")
```

**selectools:**
```python
from selectools import Agent, AgentConfig
from selectools.providers import OpenAIProvider

agent = Agent(
    tools=[],
    provider=OpenAIProvider(),
    config=AgentConfig(model="gpt-4o"),
)
result = agent.run("Write a Python script")
print(result.content)
```

**What's different:** selectools doesn't use agent-to-agent chat. Instead, you compose agents into graphs where data flows through explicit edges. This is more predictable than open-ended conversations between agents.

---

## Group Chat (AutoGen) vs AgentGraph (selectools)

**AutoGen:**
```python
from autogen import GroupChat, GroupChatManager

group = GroupChat(agents=[agent1, agent2, agent3], messages=[], max_round=10)
manager = GroupChatManager(groupchat=group, llm_config=config)
user_proxy.initiate_chat(manager, message="Solve this problem")
```

**selectools:**
```python
from selectools.orchestration import SupervisorAgent

supervisor = SupervisorAgent(
    agents={"researcher": agent1, "writer": agent2, "reviewer": agent3},
    strategy="dynamic",  # LLM picks the best agent each step
    provider=provider,
)
result = supervisor.run("Solve this problem")
```

**What's different:** `SupervisorAgent` gives you 4 coordination strategies (plan_and_execute, round_robin, dynamic, magentic) instead of AutoGen's single group chat model. The LLM router in `dynamic` mode is similar to AutoGen's speaker selection but with explicit control.

---

## What AutoGen Does Better (honest)

- **Code execution** is built in (Docker sandboxes)
- **Agent-to-agent conversation** is natural for brainstorming/debate scenarios
- **Microsoft ecosystem** integration

---
---

# Coming from LlamaIndex

LlamaIndex focuses on data indexing and retrieval. Selectools has a built-in RAG pipeline but also covers agent orchestration, evals, and deployment.

---

## RAG Pipeline

**LlamaIndex:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the refund policy?")
```

**selectools:**
```python
from selectools.rag import DocumentLoader, TextSplitter, InMemoryVectorStore
from selectools.embeddings import OpenAIEmbeddings

docs = DocumentLoader.from_directory("data")
chunks = TextSplitter(chunk_size=500).split_documents(docs)
store = InMemoryVectorStore(embeddings=OpenAIEmbeddings())
store.add_documents(chunks)

results = store.search("What is the refund policy?", top_k=5)
```

**What's different:** selectools exposes every step (chunking, embedding, retrieval, reranking) as a composable piece. You can swap BM25 for vector search, add a reranker, or use hybrid search with RRF fusion. LlamaIndex's `VectorStoreIndex` hides these choices.

---

## Hybrid Search

**LlamaIndex:** Requires `BM25Retriever` + `QueryFusionRetriever` with manual setup.

**selectools:**
```python
from selectools.rag import HybridSearcher, BM25Index

searcher = HybridSearcher(
    vector_store=store,
    bm25_index=BM25Index(chunks),
    alpha=0.5,  # balance between BM25 and vector
)
results = searcher.search("refund policy", top_k=10)
```

**What's different:** Hybrid search is a first-class feature, not an afterthought. Built-in RRF fusion and cross-encoder reranking.

---

## What LlamaIndex Does Better (honest)

- **Data connectors** for 100+ sources (Notion, Google Drive, Slack, databases)
- **Advanced indexing** (tree, keyword, knowledge graph indexes)
- **Mature RAG ecosystem** with years of optimization
- **LlamaParse** for complex document parsing (tables, PDFs)

If your primary need is sophisticated document retrieval with many data sources, LlamaIndex is purpose-built for that. If you need agents + RAG + evals + deployment in one package, selectools combines all of these.
