#!/usr/bin/env python3
"""
OpenTelemetry Observer -- send agent traces to Datadog, Jaeger, Grafana.

Maps selectools observer events to OTel GenAI semantic convention spans.
Works with any OTel-compatible backend.

Prerequisites: pip install opentelemetry-api opentelemetry-sdk
Run: python examples/87_otel_observer.py
"""

print("=== OpenTelemetry Observer Example ===\n")

print(
    """
from selectools import Agent, AgentConfig
from selectools.providers import OpenAIProvider
from selectools.observe.otel import OTelObserver

# Create the observer
otel = OTelObserver(tracer_name="my-agent-service")

# Attach to your agent
agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    config=AgentConfig(
        model="gpt-4o",
        observers=[otel],  # Traces flow to OTel
    ),
)

# Run as normal -- spans are created automatically
result = agent.run("Search and summarize")

# Spans created:
# - agent.run (root span)
#   - gen_ai.chat (LLM call, with model + token counts)
#   - tool.execute (tool call, with name + duration)
#   - gen_ai.chat (second LLM call)

# Configure your exporter (Jaeger, OTLP, etc.):
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

# Traces appear in Datadog, Grafana, Jaeger, etc.
"""
)

print("Install: pip install opentelemetry-api opentelemetry-sdk")
print("Done!")
