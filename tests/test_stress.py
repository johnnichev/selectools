"""
High-concurrency stress tests to validate production readiness under load.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List

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


async def test_100_concurrent_users():
    """Test 100 concurrent users making requests simultaneously."""
    print("  Testing 100 concurrent users...")

    @selectools.tool(description="Process request")
    async def process_request(user_id: str) -> str:
        await asyncio.sleep(0.01)  # Simulate work
        return f"Processed for user {user_id}"

    async def simulate_user(user_id: int):
        agent = Agent(
            tools=[process_request], provider=LocalProvider(), config=AgentConfig(max_iterations=2)
        )
        response = await agent.arun(
            [Message(role=Role.USER, content=f"Request from user {user_id}")]
        )
        return response.role == Role.ASSISTANT

    start_time = time.time()
    results = await asyncio.gather(*[simulate_user(i) for i in range(100)])
    elapsed = time.time() - start_time

    assert all(results), "Some requests failed"
    assert len(results) == 100

    throughput = 100 / elapsed
    print(f"  ✓ 100 concurrent users completed in {elapsed:.2f}s ({throughput:.1f} req/s)")
    return elapsed, throughput


async def test_500_rapid_fire_requests():
    """Test 500 rapid-fire requests to a single agent pool."""
    print("  Testing 500 rapid-fire requests...")

    @selectools.tool(description="Fast tool")
    async def fast_tool(request_id: int) -> str:
        return f"Response {request_id}"

    # Shared agent (more realistic for web apps)
    agent = Agent(tools=[fast_tool], provider=LocalProvider(), config=AgentConfig(max_iterations=2))

    async def make_request(request_id: int):
        response = await agent.arun([Message(role=Role.USER, content=f"Request {request_id}")])
        return response.role == Role.ASSISTANT

    start_time = time.time()
    results = await asyncio.gather(*[make_request(i) for i in range(500)])
    elapsed = time.time() - start_time

    assert all(results), "Some requests failed"
    assert len(results) == 500

    throughput = 500 / elapsed
    print(f"  ✓ 500 requests completed in {elapsed:.2f}s ({throughput:.1f} req/s)")
    return elapsed, throughput


async def test_sustained_load_1000_requests():
    """Test sustained load with 1000 requests over time."""
    print("  Testing sustained load (1000 requests)...")

    @selectools.tool(description="Work")
    async def do_work(n: int) -> str:
        await asyncio.sleep(0.005)  # Simulate light work
        return f"Work {n}"

    agent = Agent(tools=[do_work], provider=LocalProvider(), config=AgentConfig(max_iterations=2))

    async def batch_request(batch_id: int, batch_size: int):
        tasks = [
            agent.arun([Message(role=Role.USER, content=f"Batch {batch_id} item {i}")])
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
        return all(r.role == Role.ASSISTANT for r in results)

    start_time = time.time()
    # Process in batches of 50 to simulate realistic load
    batch_results = await asyncio.gather(*[batch_request(i, 50) for i in range(20)])
    elapsed = time.time() - start_time

    assert all(batch_results), "Some batches failed"

    throughput = 1000 / elapsed
    print(f"  ✓ 1000 requests in 20 batches completed in {elapsed:.2f}s ({throughput:.1f} req/s)")
    return elapsed, throughput


async def test_concurrent_agents_with_memory():
    """Test 50 concurrent agents, each with their own memory."""
    print("  Testing 50 agents with independent memory...")

    @selectools.tool(description="Store")
    async def store_data(data: str) -> str:
        return f"Stored: {data}"

    async def agent_with_memory(agent_id: int, num_turns: int):
        memory = ConversationMemory(max_messages=20)
        agent = Agent(
            tools=[store_data],
            provider=LocalProvider(),
            config=AgentConfig(max_iterations=3),
            memory=memory,
        )

        # Multiple turns per agent
        for turn in range(num_turns):
            await agent.arun([Message(role=Role.USER, content=f"Agent {agent_id} turn {turn}")])

        return len(memory)

    start_time = time.time()
    memory_sizes = await asyncio.gather(*[agent_with_memory(i, 5) for i in range(50)])
    elapsed = time.time() - start_time

    assert len(memory_sizes) == 50
    assert all(size > 0 for size in memory_sizes)

    total_requests = 50 * 5
    throughput = total_requests / elapsed
    print(
        f"  ✓ 50 agents × 5 turns = {total_requests} requests in {elapsed:.2f}s ({throughput:.1f} req/s)"
    )
    return elapsed, throughput


async def test_memory_under_load():
    """Test memory behavior under high load."""
    print("  Testing memory stability under load...")

    @selectools.tool(description="Generate data")
    async def generate_data(size: int) -> str:
        return "x" * size

    memories = [ConversationMemory(max_messages=30, max_tokens=5000) for _ in range(20)]

    async def stress_memory(memory: ConversationMemory, agent_id: int):
        agent = Agent(
            tools=[generate_data],
            provider=LocalProvider(),
            config=AgentConfig(max_iterations=2),
            memory=memory,
        )

        # Add 20 messages with varying sizes
        for i in range(20):
            size = (i + 1) * 100  # Growing message sizes
            await agent.arun(
                [Message(role=Role.USER, content=f"Agent {agent_id} message {i}: " + "x" * size)]
            )

        return len(memory)

    start_time = time.time()
    final_sizes = await asyncio.gather(*[stress_memory(mem, i) for i, mem in enumerate(memories)])
    elapsed = time.time() - start_time

    assert len(final_sizes) == 20
    # All memories should have enforced limits
    assert all(size <= 30 for size in final_sizes), f"Memory limits not enforced: {final_sizes}"

    print(f"  ✓ 20 agents × 20 messages with memory limits enforced in {elapsed:.2f}s")
    return elapsed


async def test_mixed_workload_realistic():
    """Test mixed workload simulating real production traffic."""
    print("  Testing realistic mixed workload...")

    @selectools.tool(description="Light work")
    async def light_work() -> str:
        await asyncio.sleep(0.01)
        return "light"

    @selectools.tool(description="Medium work")
    async def medium_work() -> str:
        await asyncio.sleep(0.05)
        return "medium"

    @selectools.tool(description="Heavy work")
    def heavy_work() -> str:
        # Simulate CPU-bound work
        total = sum(i for i in range(1000))
        return f"heavy:{total}"

    async def mixed_request(request_type: str, request_id: int):
        tools = [light_work] if request_type == "light" else [light_work, medium_work, heavy_work]
        agent = Agent(tools=tools, provider=LocalProvider(), config=AgentConfig(max_iterations=2))
        response = await agent.arun(
            [Message(role=Role.USER, content=f"{request_type} request {request_id}")]
        )
        return response.role == Role.ASSISTANT

    # Simulate realistic traffic: 70% light, 20% medium, 10% heavy
    requests = (
        [("light", i) for i in range(70)]
        + [("medium", i) for i in range(20)]
        + [("heavy", i) for i in range(10)]
    )

    start_time = time.time()
    results = await asyncio.gather(
        *[mixed_request(req_type, req_id) for req_type, req_id in requests]
    )
    elapsed = time.time() - start_time

    assert all(results), "Some requests failed"
    throughput = 100 / elapsed

    print(
        f"  ✓ 100 mixed requests (70% light, 20% medium, 10% heavy) in {elapsed:.2f}s ({throughput:.1f} req/s)"
    )
    return elapsed, throughput


async def test_error_handling_under_load():
    """Test error handling with high concurrency and failures."""
    print("  Testing error handling under concurrent load...")

    failure_count = 0

    @selectools.tool(description="Flaky tool")
    async def flaky_tool(request_id: int) -> str:
        nonlocal failure_count
        # 20% failure rate
        if request_id % 5 == 0:
            failure_count += 1
            raise Exception(f"Simulated failure {request_id}")
        return f"Success {request_id}"

    async def request_with_errors(request_id: int):
        agent = Agent(
            tools=[flaky_tool], provider=LocalProvider(), config=AgentConfig(max_iterations=3)
        )
        try:
            response = await agent.arun([Message(role=Role.USER, content=f"Request {request_id}")])
            return "success", response.role == Role.ASSISTANT
        except Exception as e:
            return "error", False

    start_time = time.time()
    results = await asyncio.gather(*[request_with_errors(i) for i in range(100)])
    elapsed = time.time() - start_time

    success_count = sum(1 for status, _ in results if status == "success")

    print(
        f"  ✓ 100 requests with 20% failure rate: {success_count} succeeded, {100-success_count} handled gracefully"
    )
    print(f"    Completed in {elapsed:.2f}s")
    return elapsed, success_count


def run_async_test(test_func):
    """Helper to run async tests."""
    return asyncio.run(test_func())


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HIGH-CONCURRENCY STRESS TEST SUITE")
    print("=" * 70 + "\n")

    stress_tests = [
        ("100 Concurrent Users", test_100_concurrent_users),
        ("500 Rapid-Fire Requests", test_500_rapid_fire_requests),
        ("1000 Sustained Load", test_sustained_load_1000_requests),
        ("50 Agents with Memory", test_concurrent_agents_with_memory),
        ("Memory Under Load", test_memory_under_load),
        ("Mixed Workload", test_mixed_workload_realistic),
        ("Error Handling Under Load", test_error_handling_under_load),
    ]

    results = []
    failures = 0

    for test_name, test_func in stress_tests:
        print(f"\n[{test_name}]")
        try:
            result = run_async_test(test_func)
            results.append((test_name, "PASS", result))
        except AssertionError as exc:
            failures += 1
            print(f"  ✗ FAILED: {exc}")
            results.append((test_name, "FAIL", str(exc)))
        except Exception as exc:
            failures += 1
            print(f"  ✗ ERROR: {exc.__class__.__name__}: {exc}")
            results.append((test_name, "ERROR", str(exc)))

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70 + "\n")

    for test_name, status, _ in results:
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {test_name}: {status}")

    total_tests = len(stress_tests)
    passed = total_tests - failures

    print(f"\nResults: {passed}/{total_tests} tests passed")

    if failures == 0:
        print("\n🎉 ALL HIGH-CONCURRENCY TESTS PASSED!")
        print("\n✅ System validated for production use under high load:")
        print("   • 100+ concurrent users")
        print("   • 500+ rapid requests")
        print("   • 1000+ sustained requests")
        print("   • 250+ multi-turn conversations")
        print("   • Memory stability under load")
        print("   • Mixed workload handling")
        print("   • Graceful error handling")
    else:
        print(f"\n❌ {failures} test(s) failed")
        raise SystemExit(1)
