# Selectools Implementation Documentation

**Version:** 0.11.0
**Last Updated:** February 2026

Welcome to the comprehensive technical documentation for selectools - a production-ready Python framework for building AI agents with tool-calling capabilities and RAG support.

---

## 📚 Documentation Structure

### Main Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Start here for a complete system overview, architecture diagrams, data flows, and design principles.

### Module Documentation

Detailed technical documentation for each module:

1. **[AGENT.md](modules/AGENT.md)** - Agent loop, tool selection, retry logic, streaming, parallel execution, and execution flow
2. **[TOOLS.md](modules/TOOLS.md)** - Tool definition, validation, registry, and streaming
3. **[PARSER.md](modules/PARSER.md)** - TOOL_CALL contract and JSON extraction strategies
4. **[PROMPT.md](modules/PROMPT.md)** - System prompt generation and tool schema formatting
5. **[PROVIDERS.md](modules/PROVIDERS.md)** - LLM provider adapters and message formatting
6. **[MEMORY.md](modules/MEMORY.md)** - Conversation memory management and sliding windows
7. **[USAGE.md](modules/USAGE.md)** - Usage tracking, cost calculation, and analytics
8. **[RAG.md](modules/RAG.md)** - Complete RAG pipeline from documents to answers
9. **[EMBEDDINGS.md](modules/EMBEDDINGS.md)** - Embedding providers and semantic search
10. **[VECTOR_STORES.md](modules/VECTOR_STORES.md)** - Vector database implementations
11. **[MODELS.md](modules/MODELS.md)** - Model registry and pricing system

---

## 🎯 Quick Navigation

### By Role

**For Developers:**

- Start: [ARCHITECTURE.md](ARCHITECTURE.md)
- Building agents: [AGENT.md](modules/AGENT.md)
- Creating tools: [TOOLS.md](modules/TOOLS.md)
- Adding RAG: [RAG.md](modules/RAG.md)

**For Contributors:**

- Adding providers: [PROVIDERS.md](modules/PROVIDERS.md)
- Adding vector stores: [VECTOR_STORES.md](modules/VECTOR_STORES.md)
- Understanding parser: [PARSER.md](modules/PARSER.md)

**For DevOps/Production:**

- Cost tracking: [USAGE.md](modules/USAGE.md)
- Model selection: [MODELS.md](modules/MODELS.md)
- Monitoring: [AGENT.md](modules/AGENT.md#hook-system)

### By Feature

**Tool Calling:**

- [AGENT.md](modules/AGENT.md) - Orchestration
- [TOOLS.md](modules/TOOLS.md) - Definition
- [PARSER.md](modules/PARSER.md) - Parsing
- [PROMPT.md](modules/PROMPT.md) - Prompting

**RAG System:**

- [RAG.md](modules/RAG.md) - Overview
- [EMBEDDINGS.md](modules/EMBEDDINGS.md) - Vector generation
- [VECTOR_STORES.md](modules/VECTOR_STORES.md) - Storage

**Cost Management:**

- [USAGE.md](modules/USAGE.md) - Tracking
- [MODELS.md](modules/MODELS.md) - Pricing

---

## 📊 Documentation Stats

- **Total files:** 12 (1 main + 11 modules)
- **Total lines:** 8,500+ lines
- **Total words:** ~55,000 words
- **ASCII diagrams:** 25+ diagrams
- **Code examples:** 200+ examples

---

## 🔍 Understanding the Flow

### Standard Tool-Calling Flow

```
1. User Query
   ↓
2. AGENT loads history (MEMORY) and calls PROVIDER
   ↓
3. PROVIDER formats prompt (PROMPT) and calls LLM
   ↓
4. PARSER extracts TOOL_CALL from response
   ↓
5. TOOLS validates and executes
   ↓
5b. If multiple tools: PARALLEL execution (asyncio.gather)
   ↓
6. USAGE tracks tokens and costs
   ↓
7. Loop continues or returns final response
```

**Read:**

1. [AGENT.md](modules/AGENT.md) - Main loop
2. [PROVIDERS.md](modules/PROVIDERS.md) - LLM communication
3. [PARSER.md](modules/PARSER.md) - Response parsing
4. [TOOLS.md](modules/TOOLS.md) - Tool execution

### RAG Flow

```
1. Documents
   ↓
2. LOADERS read files/PDFs
   ↓
3. CHUNKING splits into pieces
   ↓
4. EMBEDDINGS generate vectors
   ↓
5. VECTOR_STORES persist
   ↓
6. Query → Search → Context → Answer
```

**Read:**

1. [RAG.md](modules/RAG.md) - Complete pipeline
2. [EMBEDDINGS.md](modules/EMBEDDINGS.md) - Vector generation
3. [VECTOR_STORES.md](modules/VECTOR_STORES.md) - Storage

---

## 🎓 Learning Path

### Beginner

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) - Get the big picture
2. Read [AGENT.md](modules/AGENT.md) - Understand the core loop
3. Read [TOOLS.md](modules/TOOLS.md) - Learn to create tools
4. Build your first agent!

### Intermediate

1. Read [PARSER.md](modules/PARSER.md) - Understand parsing
2. Read [PROVIDERS.md](modules/PROVIDERS.md) - Switch providers
3. Read [MEMORY.md](modules/MEMORY.md) - Add conversations
4. Read [USAGE.md](modules/USAGE.md) - Track costs

### Advanced

1. Read [RAG.md](modules/RAG.md) - Add document search
2. Read [EMBEDDINGS.md](modules/EMBEDDINGS.md) - Choose embeddings
3. Read [VECTOR_STORES.md](modules/VECTOR_STORES.md) - Scale storage
4. Build production RAG systems!

---

## 💡 Key Concepts

### Design Principles

1. **Provider Agnosticism** - Switch LLMs without code changes
2. **Library-First** - Composable, no framework lock-in
3. **Production Hardened** - Retries, timeouts, validation
4. **Developer Friendly** - Type hints, decorators, clear errors
5. **Observable** - Hooks, analytics, usage tracking
6. **Cost Aware** - Automatic tracking and warnings
7. **Performance Optimized** - Parallel tool execution, async-first design

### Core Patterns

- **Agent Loop** - Iterative tool calling until completion
- **Native Tool Calling** - Provider-native function calling APIs
- **Parallel Execution** - Concurrent tool calls via asyncio.gather
- **Tool Calling Contract** - TOOL_CALL with JSON payload
- **Schema Generation** - Automatic from type hints
- **Injected Parameters** - Hide secrets from LLM
- **Streaming** - Progressive results via generators
- **RAG Pipeline** - Load → Chunk → Embed → Store → Search

---

## 🔧 Common Tasks

### Create an Agent

See: [AGENT.md](modules/AGENT.md)

### Define a Tool

See: [TOOLS.md](modules/TOOLS.md)

### Add RAG

See: [RAG.md](modules/RAG.md)

### Switch Providers

See: [PROVIDERS.md](modules/PROVIDERS.md)

### Track Costs

See: [USAGE.md](modules/USAGE.md)

### Choose a Model

See: [MODELS.md](modules/MODELS.md)

---

## 🚀 Next Steps

1. **Read the [ARCHITECTURE.md](ARCHITECTURE.md)** for system overview
2. **Explore module docs** based on your needs
3. **Check the main [README.md](../README.md)** for quick start examples
4. **Review the [ROADMAP.md](../ROADMAP.md)** for upcoming features

---

## 📝 Contributing

Found an error or want to improve the docs?

1. Check the source code in `src/selectools/`
2. Submit issues or PRs on GitHub
3. Follow the patterns established in existing docs

---

**Built with ❤️ for developers who want to understand their tools.**
