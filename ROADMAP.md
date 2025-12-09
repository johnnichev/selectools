# Selectools Development Roadmap

This document tracks the implementation status of all planned features. See [README.md](README.md#roadmap--future-improvements) for detailed descriptions.

## Priority 1: Quick Wins (v0.4.0 - v0.5.1)

| Feature                   | Status         | Notes                                  |
| ------------------------- | -------------- | -------------------------------------- |
| Conversation Memory       | ✅ Implemented | Simple history management (v0.4)       |
| Async Support             | ✅ Implemented | `Agent.arun()`, async tools (v0.4)     |
| Anthropic Provider (Full) | ✅ Implemented | Real SDK integration with async (v0.4) |
| Gemini Provider (Full)    | ✅ Implemented | Real SDK integration with async (v0.4) |
| Remove Pillow Dependency  | ✅ Implemented | Removed bbox example (v0.4)            |
| Better Error Messages     | ✅ Implemented | PyTorch-style helpful errors (v0.5.0)  |
| Cost Tracking             | ✅ Implemented | Track tokens and API costs (v0.5.0)    |
| Pre-built Tool Library    | ✅ Implemented | 22 tools in 5 categories (v0.5.1)      |

---

## Priority 2: High-Impact Features (v0.5.2 - v0.6.0)

| Feature                         | Status         | Notes                                    |
| ------------------------------- | -------------- | ---------------------------------------- |
| Tool Validation at Registration | ✅ Implemented | Validates tools at registration (v0.5.2) |
| Observability Hooks             | ✅ Implemented | 10 lifecycle hooks for monitoring (v0.5.2)|
| Streaming Tool Results          | 🟡 Planned     | Stream tool output as generated          |
| Parallel Tool Execution         | 🟡 Planned     | Auto-detect independent tools            |
| Tool Composition                | 🟡 Planned     | `@compose` decorator                     |
| Interactive Debug Mode          | 🟡 Planned     | Step-through debugging                   |

---

## Priority 3: Advanced Features (v0.7.0+ - Ongoing)

### Context Management

| Feature                              | Status     | Notes                           |
| ------------------------------------ | ---------- | ------------------------------- |
| Automatic Conversation Summarization | 🟡 Planned | Handle long conversations       |
| Sliding Window with Smart Retention  | 🟡 Planned | Keep important context          |
| Multi-Turn Memory System             | 🟡 Planned | Persistent cross-session memory |

### Tool Capabilities

| Feature                   | Status     | Notes                     |
| ------------------------- | ---------- | ------------------------- |
| Dynamic Tool Loading      | 🟡 Planned | Hot-reload tools          |
| Tool Usage Analytics      | 🟡 Planned | Track performance metrics |
| Tool Marketplace/Registry | 🟡 Planned | Community tool sharing    |

### Provider Enhancements

| Feature                  | Status     | Notes                         |
| ------------------------ | ---------- | ----------------------------- |
| Universal Vision Support | 🟡 Planned | Unified vision API            |
| Provider Auto-Selection  | 🟡 Planned | Automatic fallback chains     |
| Streaming Improvements   | 🟡 Planned | SSE, WebSocket support        |
| Local Model Support      | 🟡 Planned | Ollama, LM Studio integration |

### Production Reliability

| Feature                   | Status     | Notes                                 |
| ------------------------- | ---------- | ------------------------------------- |
| Advanced Error Recovery   | 🟡 Planned | Circuit breaker, graceful degradation |
| Observability & Debugging | 🟡 Planned | OpenTelemetry, execution replay       |
| Rate Limiting & Quotas    | 🟡 Planned | Per-tool and user quotas              |
| Security Hardening        | 🟡 Planned | Sandboxing, audit logging             |

### Developer Experience

| Feature                    | Status     | Notes                               |
| -------------------------- | ---------- | ----------------------------------- |
| Visual Agent Builder       | 🟡 Planned | Web UI for agent design             |
| Enhanced Testing Framework | 🟡 Planned | Snapshot testing, load tests        |
| Documentation Generation   | 🟡 Planned | Auto-generate from tool definitions |
| Type Safety Improvements   | 🟡 Planned | Better type inference               |

### Ecosystem Integration

| Feature                | Status     | Notes                             |
| ---------------------- | ---------- | --------------------------------- |
| Framework Integrations | 🟡 Planned | FastAPI, Flask, LangChain adapter |
| CRM & Business Tools   | 🟡 Planned | HubSpot, Salesforce, etc          |
| Data Source Connectors | 🟡 Planned | SQL, vector DBs, cloud storage    |

### Performance Optimizations

| Feature             | Status     | Notes                              |
| ------------------- | ---------- | ---------------------------------- |
| Caching Layer       | 🟡 Planned | LRU, semantic, distributed caching |
| Batch Processing    | 🟡 Planned | Efficient multi-request handling   |
| Prompt Optimization | 🟡 Planned | Automatic prompt compression       |

---

## Status Legend

- ✅ **Implemented** - Feature is complete and merged
- 🔵 **In Progress** - Actively being worked on
- 🟡 **Planned** - Scheduled for implementation
- 🟠 **Blocked** - Waiting on dependencies or decisions
- ⏸️ **Deferred** - Postponed to later release
- ❌ **Cancelled** - No longer planned

---

## How to Contribute

1. **Pick a feature** from Priority 1 or 2 (great for first-time contributors!)
2. **Comment on the issue** or create one if it doesn't exist
3. **Implement the feature** following [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Submit a PR** with clear description
5. **Update this roadmap** to mark feature as ✅ Implemented

---

## Release Schedule

### v0.4.0 - Quick Wins

**Focus:** Developer experience improvements that close gaps with LangChain

**Completed:**

- ✅ Conversation Memory
- ✅ Async Support (Agent.arun(), async tools, async providers)
- ✅ Anthropic Provider (Full SDK integration)
- ✅ Gemini Provider (Full SDK integration)
- ✅ Removed Pillow dependency

**Remaining:**

- Better Error Messages
- Cost Tracking
- Pre-built Tool Library (at least 3 tools)

### v0.6.0 - High-Impact Features

**Focus:** Performance and observability

**Must-have:**

- ✅ Parallel Tool Execution
- ✅ Observability Hooks

**Nice-to-have:**

- Streaming Tool Results
- Tool Composition
- Interactive Debug Mode

### v0.7.0 - Advanced Features

**Focus:** Advanced context management and ecosystem

**Must-have:**

- ✅ Automatic Conversation Summarization
- ✅ Tool Marketplace (basic version)

**Nice-to-have:**

- Provider Auto-Selection
- Local Model Support
- Framework Integrations

### v1.0.0

**Focus:** Enterprise features and stability

**Must-have:**

- ✅ All Priority 1 & 2 features
- ✅ Comprehensive documentation
- ✅ 90%+ test coverage
- ✅ Security hardening
- ✅ Performance benchmarks

## Last Updated

**Date:** 2025-12-08
**By:** John (v0.4.0 progress update)
**Next Review:** 2025-12-15

**Recent Changes:**

- ✅ Completed Conversation Memory feature
- ✅ Completed full Async Support (Agent.arun, async tools, async providers)
- ✅ Implemented real Anthropic and Gemini providers with async support
- ✅ Removed Pillow dependency, cleaned up codebase
- ✅ Added comprehensive async tests
- ✅ Created async usage examples
