---
hide:
  - navigation
  - toc
title: Selectools Documentation
description: Selectools is a production-ready Python library for building AI agents with tool calling, RAG, multi-agent orchestration, 50 built-in evaluators, and a visual drag-drop builder. Free, open source, Apache-2.0. Supports OpenAI, Azure OpenAI, Anthropic, Gemini, Ollama, and 100+ models via LiteLLM.
---

```
┌─┐┌─┐┬  ┌─┐┌─┐┌┬┐┌─┐┌─┐┬  ┌─┐
└─┐├┤ │  ├┤ │   │ │ ││ ││  └─┐
└─┘└─┘┴─┘└─┘└─┘ ┴ └─┘└─┘┴─┘└─┘
```

# Welcome to Selectools

**Production-ready AI agents in plain Python.** Multi-agent graphs, tool calling, RAG, 50 evaluators, and a drag-drop visual builder. All in one `pip install`. No DSL. No SaaS. No vendor lock-in.

```bash
pip install selectools
```

## Get started in 5 minutes

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Install and run your first agent with a tool call in under 5 minutes.

    [:octicons-arrow-right-24: Read the Quickstart](QUICKSTART.md)

-   :material-swap-horizontal:{ .lg .middle } **Migration Guides**

    ---

    Coming from LangChain, CrewAI, AutoGen, or LlamaIndex? Drop-in patterns for each.

    [:octicons-arrow-right-24: Migration Guides](MIGRATION.md)

-   :material-book-open-variant:{ .lg .middle } **Cookbook**

    ---

    Recipes for common patterns: RAG, multi-agent graphs, evals, guardrails, streaming.

    [:octicons-arrow-right-24: Browse the Cookbook](COOKBOOK.md)

-   :material-shape:{ .lg .middle } **Architecture**

    ---

    How the pieces fit together: Agent, Tools, Memory, Sessions, Providers, Observers.

    [:octicons-arrow-right-24: Architecture overview](ARCHITECTURE.md)

</div>

## Explore the modules

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **Core Agent**

    ---

    The `Agent` class — the central orchestrator with tool calling, retries, and streaming.

    [:octicons-arrow-right-24: Agent module](modules/AGENT.md)

-   :material-tools:{ .lg .middle } **Toolbox**

    ---

    56 production-ready tools: web search, code execution, file ops, SQL, calculator, email, PDF, Slack, Notion, Linear, Discord, S3, browser, image gen, more.

    [:octicons-arrow-right-24: Toolbox](modules/TOOLBOX.md)

-   :material-database-search:{ .lg .middle } **RAG Pipeline**

    ---

    Hybrid search (BM25 + vector) with reranking, **7 vector store backends** (In-memory, SQLite, Chroma, Pinecone, FAISS, Qdrant, pgvector), semantic chunking, and CSV / JSON / HTML / URL document loaders.

    [:octicons-arrow-right-24: RAG module](modules/RAG.md)

-   :material-graph-outline:{ .lg .middle } **Multi-Agent**

    ---

    AgentGraph, Supervisor strategies, parallel execution, and 7 prebuilt patterns.

    [:octicons-arrow-right-24: Orchestration](modules/ORCHESTRATION.md)

-   :material-shield-check:{ .lg .middle } **Guardrails**

    ---

    PII redaction, injection defense, toxicity screening, and audit logging built in.

    [:octicons-arrow-right-24: Guardrails](modules/GUARDRAILS.md)

-   :material-chart-line:{ .lg .middle } **Evaluation**

    ---

    50 built-in evaluators (30 deterministic + 20 LLM-as-judge) with HTML reports and JUnit XML.

    [:octicons-arrow-right-24: Eval framework](modules/EVALS.md)

-   :material-api:{ .lg .middle } **Agent-as-API**

    ---

    `AgentAPI` serves any agent as a production REST API: chat, SSE streaming, session CRUD, bearer auth.

    [:octicons-arrow-right-24: Serve module](modules/SERVE.md)

-   :material-lan-connect:{ .lg .middle } **A2A Protocol**

    ---

    Agent-to-agent communication: Agent Card discovery and JSON-RPC 2.0 task messaging.

    [:octicons-arrow-right-24: A2A module](modules/A2A.md)

-   :material-call-split:{ .lg .middle } **Providers & Routing**

    ---

    6 LLM providers (incl. 100+ models via LiteLLM), cost-optimized `RouterProvider`, auto-failover `FallbackProvider`, Anthropic prompt caching.

    [:octicons-arrow-right-24: Providers](modules/PROVIDERS.md)

-   :material-brain:{ .lg .middle } **Memory & Sessions**

    ---

    Tiered `UnifiedMemory`, 4 session backends with cross-session search, Supabase/Redis knowledge persistence.

    [:octicons-arrow-right-24: Unified Memory](modules/UNIFIED_MEMORY.md)

</div>

## Try it without installing

The visual agent builder runs in your browser with zero installation. Drag nodes, connect agents, test with real APIs, export runnable Python.

[:material-cursor-default-click: Open the Builder](https://selectools.dev/builder/){ .md-button .md-button--primary }
[:material-github: View on GitHub](https://github.com/johnnichev/selectools){ .md-button }
