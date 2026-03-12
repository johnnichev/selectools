"""
Example 30: Audit Logging

Demonstrates JSONL audit logging with privacy controls
and daily file rotation.

Usage:
    python examples/30_audit_logging.py

No API key needed — uses LocalProvider.
"""

import json
import os
import tempfile

from selectools import Agent, AgentConfig, tool
from selectools.audit import AuditLogger, PrivacyLevel
from selectools.providers.stubs import LocalProvider


@tool(description="Look up a customer by ID")
def lookup_customer(customer_id: str) -> str:
    return f"Customer {customer_id}: John Doe, john@example.com"


@tool(description="Search the knowledge base")
def search_kb(query: str) -> str:
    return f"Found 3 articles about: {query}"


# Create a temporary directory for audit logs
audit_dir = tempfile.mkdtemp(prefix="selectools_audit_")
print(f"Audit logs will be written to: {audit_dir}\n")


# ── 1. Basic audit logging ──────────────────────────────────────────────

print("=" * 60)
print("1. Basic Audit Logging (KEYS_ONLY privacy)")
print("=" * 60)

audit = AuditLogger(
    log_dir=audit_dir,
    privacy=PrivacyLevel.KEYS_ONLY,
    daily_rotation=True,
)

agent = Agent(
    tools=[lookup_customer, search_kb],
    provider=LocalProvider(),
    config=AgentConfig(observers=[audit], max_iterations=2),
)

agent.ask("Look up customer C-12345")
agent.ask("Search for shipping policy")

# Read and display the log
log_files = sorted(os.listdir(audit_dir))
print(f"\n  Log file: {log_files[0]}")
with open(os.path.join(audit_dir, log_files[0])) as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        print(f"  [{i+1}] {entry['event']:20s} run={entry.get('run_id', '')[:8]}")
        if "tool_name" in entry:
            print(f"      tool={entry['tool_name']}")
        if "tool_args" in entry:
            print(f"      args={entry['tool_args']}")


# ── 2. Privacy levels comparison ─────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Privacy Levels Comparison")
print("=" * 60)

for level in [PrivacyLevel.FULL, PrivacyLevel.KEYS_ONLY, PrivacyLevel.HASHED, PrivacyLevel.NONE]:
    sub_dir = os.path.join(audit_dir, level.value)
    logger = AuditLogger(log_dir=sub_dir, privacy=level, daily_rotation=False)
    logger.on_tool_start("run-1", "call-1", "search_kb", {"query": "secret data", "limit": 10})

    with open(os.path.join(sub_dir, "audit.jsonl")) as f:
        entry = json.loads(f.readline())
    print(f"\n  {level.value:10s} → tool_args = {json.dumps(entry['tool_args'])}")


print(f"\n✅ Audit log examples complete! Files in: {audit_dir}")
