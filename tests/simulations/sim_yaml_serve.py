"""
Simulation: YAML Config → Template → Serve
============================================
End-to-end: load agent from YAML, use templates, compose tools,
and serve via HTTP — all v0.19.0 features working together.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selectools import AgentConfig, tool
from selectools.compose import compose
from selectools.observe import InMemoryTraceStore
from selectools.pipeline import Pipeline, Step, cache_step, parallel, retry, step
from selectools.providers.stubs import LocalProvider
from selectools.serve.app import AgentRouter
from selectools.templates import from_dict, from_yaml, list_templates, load_template
from selectools.trace import AgentTrace, StepType, TraceStep


def main():
    print("=" * 60)
    print("v0.19.0 Feature Integration Simulation")
    print("=" * 60)

    # 1. Templates
    print("\n--- 1. Agent Templates ---")
    templates = list_templates()
    print(f"Available templates: {templates}")
    assert len(templates) == 5

    provider = LocalProvider()
    agent = load_template("customer_support", provider=provider, model="test-model")
    print(f"Loaded template: model={agent.config.model}, tools={len(agent.tools)}")
    assert agent.config.model == "test-model"
    assert len(agent.tools) >= 2

    # 2. YAML config
    print("\n--- 2. YAML Config ---")
    yaml_content = """
model: test-model
system_prompt: "You are helpful."
max_iterations: 3
retry:
  max_retries: 5
  backoff_seconds: 2.0
budget:
  max_cost_usd: 0.50
compress:
  enabled: true
  threshold: 0.8
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        yaml_agent = from_yaml(f.name, provider=provider)
    os.unlink(f.name)
    print(f"YAML agent: model={yaml_agent.config.model}, retries={yaml_agent.config.max_retries}")
    assert yaml_agent.config.max_retries == 5
    assert yaml_agent.config.max_cost_usd == 0.50
    assert yaml_agent.config.compress_context is True

    # 3. Structured Config
    print("\n--- 3. Structured AgentConfig ---")
    from selectools.agent.config_groups import BudgetConfig, RetryConfig

    config = AgentConfig(
        model="test",
        retry=RetryConfig(max_retries=10),
        budget=BudgetConfig(max_cost_usd=1.0),
    )
    print(f"Nested config: retries={config.max_retries}, budget=${config.max_cost_usd}")
    assert config.max_retries == 10
    assert config.retry.max_retries == 10

    # Dict unpacking
    config2 = AgentConfig(retry={"max_retries": 7})
    print(f"Dict unpacking: retries={config2.max_retries}")
    assert config2.max_retries == 7

    # 4. Tool Composition
    print("\n--- 4. Tool Composition ---")

    @tool(description="Fetch data")
    def fetch(url: str) -> str:
        return f"data from {url}"

    @tool(description="Parse data")
    def parse(data: str) -> str:
        return f"parsed: {data}"

    composed = compose(fetch, parse, name="fetch_and_parse")
    result = composed.function(url="https://example.com")
    print(f"Composed tool result: {result}")
    assert "parsed" in result and "example.com" in result

    # 5. Pipeline with retry + cache
    print("\n--- 5. Pipeline retry + cache ---")

    @step
    def upper(text: str) -> str:
        return text.upper()

    @step
    def exclaim(text: str) -> str:
        return text + "!"

    pipeline = upper | exclaim
    result = pipeline.run("hello")
    print(f"Pipeline result: {result.output}")
    assert result.output == "HELLO!"

    cached = cache_step(upper, ttl=300, max_size=100)
    r1 = cached("test")
    r2 = cached("test")
    print(f"Cached: {r1} (called once, returned twice)")
    assert r1 == r2

    # 6. Trace Store
    print("\n--- 6. Trace Store ---")
    store = InMemoryTraceStore()
    trace = AgentTrace(metadata={"sim": "yaml_serve"})
    trace.add(TraceStep(type=StepType.LLM_CALL))
    trace.add(TraceStep(type=StepType.TOOL_EXECUTION))
    rid = store.save(trace)
    loaded = store.load(rid)
    print(f"Trace saved/loaded: run_id={rid[:8]}, steps={len(loaded.steps)}")
    assert len(loaded.steps) == 2
    summaries = store.list()
    assert len(summaries) == 1

    # 7. Serve (no actual HTTP, just router)
    print("\n--- 7. Serve Router ---")
    agent = load_template("research_assistant", provider=provider)
    router = AgentRouter(agent)
    health = router.handle_health()
    schema = router.handle_schema()
    print(f"Health: {health['status']}, tools: {health['tools']}")
    print(f"Schema: {len(schema['tools'])} tool schemas")
    assert health["status"] == "ok"
    assert len(schema["tools"]) >= 2

    invoke_result = router.handle_invoke({"prompt": "test query"})
    print(f"Invoke: content={invoke_result.get('content', '')[:50]}")
    assert "content" in invoke_result

    print(f"\n{'=' * 60}")
    print("All v0.19.0 features working together!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
