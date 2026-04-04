#!/usr/bin/env python3
"""
Langfuse Observer -- send agent traces to Langfuse for LLM observability.

Langfuse is the most popular open-source LLM observability platform.
Traces include LLM calls, tool executions, costs, and latencies.

Prerequisites: pip install langfuse
Run: python examples/88_langfuse_observer.py
"""

print("=== Langfuse Observer Example ===\n")

print(
    """
from selectools import Agent, AgentConfig
from selectools.providers import OpenAIProvider
from selectools.observe.langfuse import LangfuseObserver

# Option 1: Explicit keys
langfuse = LangfuseObserver(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",  # or self-hosted URL
)

# Option 2: Environment variables
# Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
langfuse = LangfuseObserver()

# Attach to your agent
agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    config=AgentConfig(
        model="gpt-4o",
        observers=[langfuse],
    ),
)

# Run as normal -- traces sent to Langfuse automatically
result = agent.run("Analyze this data")

# Langfuse dashboard shows:
# - Trace timeline with LLM calls and tool executions
# - Token counts and costs per call
# - Model used, latency, input/output preview
# - Error tracking and debugging

# Flush on shutdown
langfuse.shutdown()
"""
)

print("Install: pip install langfuse")
print("Dashboard: https://cloud.langfuse.com (free tier available)")
print("Self-hosted: https://langfuse.com/docs/deployment/self-host")
print("Done!")
