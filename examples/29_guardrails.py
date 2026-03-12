"""
Example 29: Guardrails Engine

Demonstrates input and output guardrails for content validation,
PII redaction, topic blocking, and format enforcement.

Usage:
    python examples/29_guardrails.py

No API key needed — uses LocalProvider.
"""

from selectools import Agent, AgentConfig, tool
from selectools.guardrails import (
    FormatGuardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailsPipeline,
    LengthGuardrail,
    PIIGuardrail,
    TopicGuardrail,
    ToxicityGuardrail,
)
from selectools.providers.stubs import LocalProvider


@tool(description="Search for information")
def search(query: str) -> str:
    return f"Results for: {query}"


# ── 1. Basic topic blocking ─────────────────────────────────────────────

print("=" * 60)
print("1. Topic Blocking")
print("=" * 60)

pipeline = GuardrailsPipeline(
    input=[TopicGuardrail(deny=["politics", "religion"])],
)

agent = Agent(
    tools=[search],
    provider=LocalProvider(),
    config=AgentConfig(guardrails=pipeline, max_iterations=2),
)

result = agent.ask("Tell me about Python programming")
print(f"  Allowed: {result.content[:80]}")

try:
    agent.ask("What do you think about politics?")
except GuardrailError as e:
    print(f"  Blocked: {e.guardrail_name} — {e.reason}")


# ── 2. PII redaction ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. PII Redaction")
print("=" * 60)

pii_guard = PIIGuardrail(action=GuardrailAction.REWRITE)

# Standalone usage
result = pii_guard.check("Email me at user@example.com, SSN 123-45-6789")
print(f"  Original contained PII: {not result.passed}")
print(f"  Redacted:  {result.content}")

# Detect without redacting
matches = pii_guard.detect("Card: 4111-1111-1111-1111, IP: 192.168.1.1")
for m in matches:
    print(f"  Found {m.pii_type}: '{m.value}'")


# ── 3. Toxicity detection ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Toxicity Detection")
print("=" * 60)

tox = ToxicityGuardrail(threshold=0.0)

safe = tox.check("Hello, how are you today?")
print(f"  'Hello, how are you?' → passed={safe.passed}")

unsafe = tox.check("I will attack and harass them")
print(f"  Toxic content → passed={unsafe.passed}, reason={unsafe.reason}")


# ── 4. Length enforcement with truncation ────────────────────────────────

print("\n" + "=" * 60)
print("4. Length Guardrail (Rewrite/Truncate)")
print("=" * 60)

length_guard = LengthGuardrail(max_words=5, action=GuardrailAction.REWRITE)
result = length_guard.check("one two three four five six seven eight")
print(f"  Original: 'one two three four five six seven eight'")
print(f"  Truncated: '{result.content}'")
print(f"  Reason: {result.reason}")


# ── 5. Pipeline chaining ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Chained Pipeline (PII redact → Topic block)")
print("=" * 60)

pipeline = GuardrailsPipeline(
    input=[
        PIIGuardrail(action=GuardrailAction.REWRITE),
        TopicGuardrail(deny=["secret_project"]),
    ],
)

agent = Agent(
    tools=[search],
    provider=LocalProvider(),
    config=AgentConfig(guardrails=pipeline, max_iterations=2),
)

result = agent.ask("Search for user@test.com in our database")
print(f"  PII redacted and allowed: {result.content[:80]}")

try:
    agent.ask("Tell me about the secret_project")
except GuardrailError as e:
    print(f"  Blocked: {e.reason}")


# ── 6. Output guardrails ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. Output Guardrails (Format + Length)")
print("=" * 60)

pipeline = GuardrailsPipeline(
    output=[
        LengthGuardrail(max_chars=200, action=GuardrailAction.REWRITE),
    ],
)

agent = Agent(
    tools=[search],
    provider=LocalProvider(),
    config=AgentConfig(guardrails=pipeline, max_iterations=2),
)

result = agent.ask("Search for something")
print(f"  Response length capped: {len(result.content)} chars")


print("\n✅ All guardrail examples complete!")
