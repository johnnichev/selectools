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

## What's Next?

You now know the core API. Here is where to go from here:

| Goal | Read |
|---|---|
| Define more complex tools | [Tools Guide](modules/TOOLS.md) |
| Switch between providers | [Providers Guide](modules/PROVIDERS.md) |
| Stream responses in real time | [Streaming Guide](modules/STREAMING.md) |
| Use hybrid search (keyword + semantic) | [Hybrid Search Guide](modules/HYBRID_SEARCH.md) |
| Load tools from plugin files | [Dynamic Tools Guide](modules/DYNAMIC_TOOLS.md) |
| Cache LLM responses to save money | [Agent Guide — Caching](modules/AGENT.md#response-caching) |
| Track costs and token usage | [Usage Guide](modules/USAGE.md) |
| Understand the full architecture | [Architecture](ARCHITECTURE.md) |
| See working examples | [examples/](../examples/) (22 numbered scripts, 01–22) |

---

**The API in one table:**

| You want to... | Code |
|---|---|
| Ask a question (simple) | `agent.ask("What is X?")` |
| Send structured messages | `agent.run([Message(role=Role.USER, content="...")])` |
| Ask asynchronously | `await agent.aask("What is X?")` |
| Stream tokens | `async for chunk in agent.astream("What is X?"): ...` |
| Check cost | `agent.total_cost`, `agent.get_usage_summary()` |
| Reset state | `agent.reset()` |
| Add a tool at runtime | `agent.add_tool(my_tool)` |
| Remove a tool | `agent.remove_tool("tool_name")` |
