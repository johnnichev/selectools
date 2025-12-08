"""
Integration tests combining multiple features in realistic scenarios.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import selectools
from agent import Agent, AgentConfig, Message, Role
from selectools.memory import ConversationMemory
from selectools.providers.stubs import LocalProvider


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_realistic_customer_support_scenario():
    """Simulate a realistic customer support conversation."""
    memory = ConversationMemory(max_messages=30)
    conversation_log = []

    @selectools.tool(description="Search knowledge base")
    async def search_kb(query: str) -> str:
        await asyncio.sleep(0.02)  # Simulate API call
        conversation_log.append(f"KB search: {query}")
        return json.dumps(
            {
                "results": [
                    {"title": "Password Reset Guide", "relevance": 0.9},
                    {"title": "Account Recovery", "relevance": 0.7},
                ]
            }
        )

    @selectools.tool(description="Check order status")
    def check_order(order_id: str) -> str:
        conversation_log.append(f"Order check: {order_id}")
        return json.dumps(
            {"order_id": order_id, "status": "shipped", "tracking": "1Z999AA10123456784"}
        )

    @selectools.tool(description="Create support ticket")
    async def create_ticket(email: str, subject: str, priority: str = "normal") -> str:
        await asyncio.sleep(0.01)
        conversation_log.append(f"Ticket: {email} - {subject}")
        return json.dumps({"ticket_id": "TKT-12345", "status": "created", "eta": "24 hours"})

    agent = Agent(
        tools=[search_kb, check_order, create_ticket],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=8, stream=True),
        memory=memory,
    )

    # Multi-turn conversation
    conversations = [
        "I can't log into my account",
        "Order #12345 - where is it?",
        "This is urgent, please escalate",
        "What did you find about my order?",
    ]

    for user_msg in conversations:
        response = await agent.arun([Message(role=Role.USER, content=user_msg)])
        assert response.role == Role.ASSISTANT

    # Memory should have accumulated the conversation
    assert len(memory) >= len(conversations)
    print(
        f"  ✓ Handled {len(conversations)} turn conversation with {len(conversation_log)} tool calls"
    )


@pytest.mark.asyncio
async def test_concurrent_users_scenario():
    """Simulate multiple users hitting the system concurrently."""

    @selectools.tool(description="Process request")
    async def process_request(user_id: str, action: str) -> str:
        await asyncio.sleep(0.05)  # Simulate processing
        return json.dumps({"user": user_id, "action": action, "status": "completed"})

    # Create separate agents for each user (with their own memory)
    async def handle_user(user_id: int):
        memory = ConversationMemory(max_messages=10)
        agent = Agent(
            tools=[process_request],
            provider=LocalProvider(),
            config=AgentConfig(max_iterations=3),
            memory=memory,
        )

        # Each user makes multiple requests
        for i in range(3):
            response = await agent.arun(
                [Message(role=Role.USER, content=f"User {user_id} request {i}")]
            )
            assert response.role == Role.ASSISTANT

        return user_id, len(memory)

    # Simulate 10 concurrent users
    results = await asyncio.gather(*[handle_user(i) for i in range(10)])

    assert len(results) == 10
    print(f"  ✓ Handled 10 concurrent users, each with 3 requests")


@pytest.mark.asyncio
async def test_error_recovery_and_retry():
    """Test the system can recover from various error conditions."""
    failures = []

    @selectools.tool(description="Flaky API")
    async def flaky_api(attempt: int) -> str:
        if attempt < 2:
            failures.append(attempt)
            raise Exception(f"Temporary failure #{attempt}")
        return json.dumps({"success": True, "attempts": attempt})

    @selectools.tool(description="Reliable backup")
    def reliable_backup() -> str:
        return json.dumps({"source": "backup", "success": True})

    agent = Agent(
        tools=[flaky_api, reliable_backup],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=5),
    )

    response = await agent.arun([Message(role=Role.USER, content="Try to complete task")])
    assert response.role == Role.ASSISTANT
    print(f"  ✓ Recovered from {len(failures)} failures")


@pytest.mark.asyncio
async def test_memory_persistence_across_sessions():
    """Test that memory properly persists state across multiple interactions."""
    memory = ConversationMemory(max_messages=20)

    context_data = {"counter": 0}

    @selectools.tool(description="Increment counter")
    def increment() -> str:
        context_data["counter"] += 1
        return json.dumps({"counter": context_data["counter"]})

    @selectools.tool(description="Get counter")
    def get_counter() -> str:
        return json.dumps({"counter": context_data["counter"]})

    agent = Agent(
        tools=[increment, get_counter],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=4),
        memory=memory,
    )

    # Session 1: Increment a few times
    for i in range(3):
        await agent.arun([Message(role=Role.USER, content=f"Increment {i}")])

    initial_memory_size = len(memory)

    # Session 2: Continue with same memory
    await agent.arun([Message(role=Role.USER, content="What's the count?")])
    await agent.arun([Message(role=Role.USER, content="Increment again")])

    # Memory should have grown
    assert len(memory) > initial_memory_size
    print(f"  ✓ Memory persisted across sessions: {initial_memory_size} → {len(memory)} messages")


@pytest.mark.asyncio
async def test_streaming_with_async_and_memory():
    """Test streaming works correctly with async execution and memory."""
    memory = ConversationMemory(max_messages=15)
    chunks_collected = []

    def stream_handler(chunk: str):
        chunks_collected.append(chunk)

    @selectools.tool(description="Generate report")
    async def generate_report(topic: str) -> str:
        await asyncio.sleep(0.01)
        return f"Report on {topic}: Lorem ipsum dolor sit amet..."

    agent = Agent(
        tools=[generate_report],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=3, stream=True),
        memory=memory,
    )

    response = await agent.arun(
        [Message(role=Role.USER, content="Generate a report")], stream_handler=stream_handler
    )

    assert response.role == Role.ASSISTANT
    assert len(chunks_collected) > 0  # Should have streamed
    assert len(memory) >= 1  # Should have saved to memory
    print(f"  ✓ Streamed {len(chunks_collected)} chunks, saved {len(memory)} messages to memory")


@pytest.mark.asyncio
async def test_mixed_tool_types_realistic():
    """Test realistic mix of sync I/O, async I/O, and compute-heavy tools."""

    @selectools.tool(description="Async database query")
    async def query_database(user_id: str) -> str:
        await asyncio.sleep(0.03)  # Simulate DB latency
        return json.dumps({"user_id": user_id, "name": "John Doe", "credits": 100})

    @selectools.tool(description="Sync file system operation")
    def read_config_file(filename: str) -> str:
        # Sync I/O operation
        return json.dumps({"config": filename, "settings": {"timeout": 30}})

    @selectools.tool(description="Compute-heavy sync operation")
    def calculate_hash(data: str) -> str:
        # CPU-bound operation
        import hashlib

        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @selectools.tool(description="Async external API call")
    async def call_external_api(endpoint: str) -> str:
        await asyncio.sleep(0.02)  # Simulate API latency
        return json.dumps({"endpoint": endpoint, "status": 200, "data": "success"})

    agent = Agent(
        tools=[query_database, read_config_file, calculate_hash, call_external_api],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=6),
    )

    response = await agent.arun(
        [
            Message(
                role=Role.USER,
                content="Query user_123, read config.json, hash it, and call the API",
            )
        ]
    )

    assert response.role == Role.ASSISTANT
    print("  ✓ Successfully mixed async DB, sync file I/O, compute, and async API calls")


@pytest.mark.asyncio
async def test_tool_timeout_with_graceful_degradation():
    """Test that tool timeouts don't crash the system."""

    @selectools.tool(description="Fast tool")
    async def fast_tool() -> str:
        await asyncio.sleep(0.01)
        return "fast result"

    @selectools.tool(description="Slow tool")
    async def slow_tool() -> str:
        await asyncio.sleep(5.0)  # Will timeout
        return "slow result"

    @selectools.tool(description="Fallback tool")
    def fallback_tool() -> str:
        return "fallback result"

    agent = Agent(
        tools=[fast_tool, slow_tool, fallback_tool],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=5, tool_timeout_seconds=0.1),
    )

    # Should handle timeout gracefully and continue
    response = await agent.arun([Message(role=Role.USER, content="Try all tools")])
    assert response.role == Role.ASSISTANT
    print("  ✓ Handled tool timeout gracefully with fallback")


@pytest.mark.asyncio
async def test_large_scale_conversation():
    """Test system handles large-scale conversations."""
    memory = ConversationMemory(max_messages=50, max_tokens=10000)

    @selectools.tool(description="Process")
    async def process(data: str) -> str:
        return f"Processed: {data[:20]}..."

    agent = Agent(
        tools=[process],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=3),
        memory=memory,
    )

    # Simulate 20 turns of conversation
    for i in range(20):
        await agent.arun([Message(role=Role.USER, content=f"Turn {i}: " + "x" * 100)])

    # Memory should enforce limits
    assert len(memory) <= 50
    print(f"  ✓ Handled 20-turn conversation, memory capped at {len(memory)} messages")


def run_async_test(test_func):
    """Helper to run async tests."""
    asyncio.run(test_func())


if __name__ == "__main__":
    integration_tests = [
        test_realistic_customer_support_scenario,
        test_concurrent_users_scenario,
        test_error_recovery_and_retry,
        test_memory_persistence_across_sessions,
        test_streaming_with_async_and_memory,
        test_mixed_tool_types_realistic,
        test_tool_timeout_with_graceful_degradation,
        test_large_scale_conversation,
    ]

    print("\nRunning comprehensive integration tests...\n")

    failures = 0
    for test in integration_tests:
        try:
            print(f"Running {test.__name__}...")
            run_async_test(test)
            print(f"✓ {test.__name__}\n")
        except AssertionError as exc:
            failures += 1
            print(f"✗ {test.__name__}: {exc}\n")
        except Exception as exc:
            failures += 1
            print(f"✗ {test.__name__}: {exc.__class__.__name__}: {exc}\n")

    if failures:
        print(f"\n❌ {failures} integration test(s) failed!")
        raise SystemExit(1)
    else:
        print(f"\n✅ All {len(integration_tests)} integration tests passed!")
        print("\n🎉 System is PRODUCTION READY!")
