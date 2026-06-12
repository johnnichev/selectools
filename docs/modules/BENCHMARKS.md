# Performance Benchmarks

Measured framework overhead for selectools v0.26.0. These numbers answer one
question: **how much time does selectools add on top of the LLM call?**

All benchmarks use `LocalProvider` (a zero-latency mock), so the timings below
are pure framework overhead. In production, LLM API latency (100–2000ms per
call) dominates; everything on this page is noise by comparison.

## Environment

| | |
|---|---|
| selectools | v0.26.0 |
| Python | 3.9 (CPython) |
| Machine | Apple M4, 24 GB RAM, macOS 26.3 |
| Method | 100 iterations per case, fresh agent/graph instances per iteration |
| Date | 2026-06-12 |

## Framework overhead

| Operation | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| `agent.run()` single iteration | 0.04ms | 0.03ms | 0.04ms | 0.25ms |
| `agent.run()` with tool call | 0.03ms | 0.03ms | 0.04ms | 0.04ms |
| `graph.run()` 1 callable node | 0.32ms | 0.31ms | 0.37ms | 0.72ms |
| `graph.run()` 3 callable nodes | 0.43ms | 0.43ms | 0.48ms | 0.55ms |
| `graph.run()` 1 agent node | 0.27ms | 0.26ms | 0.30ms | 0.34ms |
| `graph.run()` 3 agent nodes | 0.48ms | 0.47ms | 0.52ms | 0.54ms |
| `graph.run()` 3 parallel nodes | 0.51ms | 0.51ms | 0.54ms | 0.57ms |
| `pipeline.run()` 1 step | <0.01ms | <0.01ms | <0.01ms | 0.01ms |
| `pipeline.run()` 3 steps | <0.01ms | <0.01ms | <0.01ms | 0.01ms |
| `pipeline.run()` 10 steps | 0.01ms | 0.01ms | 0.01ms | 0.01ms |
| checkpoint save (InMemory) | 0.01ms | 0.01ms | 0.01ms | 0.37ms |
| checkpoint load (InMemory) | <0.01ms | <0.01ms | <0.01ms | 0.01ms |
| trace store save (InMemory) | <0.01ms | <0.01ms | <0.01ms | <0.01ms |
| trace store load (InMemory) | <0.01ms | <0.01ms | <0.01ms | <0.01ms |

Takeaways:

- An agent turn costs **~0.04ms** of framework time. At a typical 500ms LLM
  round trip, selectools overhead is below 0.01% of wall clock.
- Graph orchestration adds **~0.3ms fixed cost** per run plus roughly
  0.05–0.1ms per node.
- Pipelines, checkpoints, and trace stores are effectively free.

## Comparison: selectools vs LangGraph

Same tasks, same zero-latency mock providers, 200 iterations each
(LangGraph 1.x, `langchain-core` current as of 2026-06-12).

| Task | selectools (mean) | LangGraph (mean) | delta |
|---|---|---|---|
| 3-node linear pipeline | 0.43ms | 0.33ms | LangGraph 0.10ms faster |
| Conditional routing | 0.37ms | 0.28ms | LangGraph 0.09ms faster |
| 3-step pipeline composition | <0.01ms | N/A (LCEL, not compared) | — |

Honest reading: LangGraph's compiled Pregel runtime is about **0.1ms faster
per run** on graph micro-tasks. Both frameworks are sub-millisecond, which is
under 0.1% of a single real LLM call — neither framework's orchestration
overhead will ever be the bottleneck in your application. selectools does not
trade performance for its smaller API; it trades a compile step for a simpler
execution model at a cost of ~0.1ms per graph run.

## Reproduce

```bash
# Framework overhead (no extra deps)
python tests/benchmarks/bench_overhead.py

# Comparison (needs the competitor installed)
pip install langgraph langchain-core
python tests/benchmarks/bench_vs_langchain.py
```

The harness builds **fresh agent/graph instances per iteration** outside the
timed window. Reusing one `Agent` across iterations accumulates conversation
history and inflates later timings — an earlier revision of this harness had
exactly that bug, reporting 6.5ms for the 3-agent-node graph that actually
costs 0.48ms.
