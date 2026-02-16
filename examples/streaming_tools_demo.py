"""
Demonstrate streaming tools for long-running operations.

This example shows how to use streaming tools that yield results progressively,
providing better user experience for long-running operations.

Features demonstrated:
- Basic streaming tools with Generator return type
- Async streaming tools with AsyncGenerator
- Real-time chunk display via on_tool_chunk hook
- Analytics tracking for streaming tools
- Toolbox streaming tools (read_file_stream, process_csv_stream)
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator, Generator

from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers import LocalProvider

# === Define streaming tools ===


@tool(description="Process a large dataset with progress updates", streaming=True)
def process_dataset(size: int) -> Generator[str, None, None]:
    """
    Simulate processing a large dataset, yielding progress as we go.

    Args:
        size: Number of items to process

    Yields:
        Progress updates for each processed item
    """
    yield f"🚀 Starting to process {size} items...\n\n"

    for i in range(size):
        # Simulate some processing time
        time.sleep(0.1)

        # Yield progress update
        if i == 0:
            yield f"[Item {i+1}/{size}] Processing first item\n"
        elif i == size - 1:
            yield f"[Item {i+1}/{size}] Processing final item\n"
        else:
            yield f"[Item {i+1}/{size}] Processing...\n"

        # Yield milestone updates
        if (i + 1) % 5 == 0:
            percent = ((i + 1) / size) * 100
            yield f"\n✅ Milestone: {i+1} items processed ({percent:.0f}% complete)\n\n"

    yield f"\n🎉 Successfully processed all {size} items!\n"


@tool(description="Search logs for a pattern, streaming matches", streaming=True)
def search_logs(pattern: str, max_results: int = 10) -> Generator[str, None, None]:
    """
    Simulate searching through logs and streaming matches as they're found.

    Args:
        pattern: Pattern to search for
        max_results: Maximum number of results to return

    Yields:
        Each matching log line as it's found
    """
    # Simulated log entries
    logs = [
        "2025-01-15 10:23:01 INFO User alice logged in",
        "2025-01-15 10:23:15 ERROR Database connection failed",
        "2025-01-15 10:23:20 WARN Retry attempt 1",
        "2025-01-15 10:23:25 INFO Database connection restored",
        "2025-01-15 10:24:01 ERROR API timeout",
        "2025-01-15 10:24:10 INFO User bob logged in",
        "2025-01-15 10:25:01 ERROR File not found: config.json",
        "2025-01-15 10:25:15 INFO Cache cleared",
        "2025-01-15 10:26:01 ERROR Permission denied",
        "2025-01-15 10:27:01 INFO User alice logged out",
    ]

    yield f"🔍 Searching logs for pattern: '{pattern}'\n"
    yield f"📊 Scanning {len(logs)} log entries...\n\n"

    matches_found = 0
    for _i, log_line in enumerate(logs):
        time.sleep(0.05)  # Simulate search time

        if pattern.lower() in log_line.lower():
            matches_found += 1
            yield f"Match {matches_found}: {log_line}\n"

            if matches_found >= max_results:
                yield f"\n⚠️  Reached maximum of {max_results} results\n"
                break

    if matches_found == 0:
        yield f"\n❌ No matches found for '{pattern}'\n"
    else:
        yield f"\n✅ Found {matches_found} matching log entries\n"


@tool(description="Async streaming data fetcher", streaming=True)
async def fetch_data_async(url: str, chunks: int = 5) -> AsyncGenerator[str, None]:
    """
    Simulate asynchronously fetching data in chunks.

    Args:
        url: URL to fetch from
        chunks: Number of chunks to fetch

    Yields:
        Each chunk of data as it arrives
    """
    yield f"🌐 Fetching data from: {url}\n"
    yield f"📦 Expecting {chunks} chunks...\n\n"

    for i in range(chunks):
        # Simulate async network delay
        await asyncio.sleep(0.2)

        # Yield chunk
        chunk_size = 1024 * (i + 1)
        yield f"Chunk {i+1}/{chunks}: Received {chunk_size} bytes\n"

    yield f"\n✅ Download complete! Total: {sum(1024 * (i+1) for i in range(chunks))} bytes\n"


# === Demo functions ===


def demo_basic_streaming() -> None:
    """Demonstrate basic streaming with real-time output."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Streaming with Real-time Display")
    print("=" * 60)

    # Callback to display chunks as they arrive
    def display_chunk(tool_name: str, chunk: str) -> None:
        print(chunk, end="", flush=True)

    provider = LocalProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "process_dataset", "parameters": {"size": 10}}',
            "Dataset processing complete!",
        ]
    )

    config = AgentConfig(
        verbose=False,
        hooks={"on_tool_chunk": display_chunk},
        max_iterations=3,
    )

    agent = Agent(tools=[process_dataset], provider=provider, config=config)

    print("\n📝 User: Process a dataset of 10 items\n")
    response = agent.run([Message(role=Role.USER, content="Process 10 items")])
    print(f"\n🤖 Agent: {response.content}\n")


def demo_log_search_streaming() -> None:
    """Demonstrate streaming log search."""
    print("\n" + "=" * 60)
    print("Demo 2: Streaming Log Search")
    print("=" * 60)

    def display_chunk(tool_name: str, chunk: str) -> None:
        print(chunk, end="", flush=True)

    provider = LocalProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "search_logs", "parameters": {"pattern": "ERROR", "max_results": 5}}',
            "Log search complete!",
        ]
    )

    config = AgentConfig(
        verbose=False,
        hooks={"on_tool_chunk": display_chunk},
        max_iterations=3,
    )

    agent = Agent(tools=[search_logs], provider=provider, config=config)

    print("\n📝 User: Search logs for ERROR messages\n")
    response = agent.run([Message(role=Role.USER, content="Find errors in logs")])
    print(f"\n🤖 Agent: {response.content}\n")


async def demo_async_streaming() -> None:
    """Demonstrate async streaming."""
    print("\n" + "=" * 60)
    print("Demo 3: Async Streaming")
    print("=" * 60)

    def display_chunk(tool_name: str, chunk: str) -> None:
        print(chunk, end="", flush=True)

    provider = LocalProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "fetch_data_async", "parameters": {"url": "https://api.example.com/data", "chunks": 5}}',
            "Data fetch complete!",
        ]
    )

    config = AgentConfig(
        verbose=False,
        hooks={"on_tool_chunk": display_chunk},
        max_iterations=3,
    )

    agent = Agent(tools=[fetch_data_async], provider=provider, config=config)

    print("\n📝 User: Fetch data from API\n")
    response = await agent.arun([Message(role=Role.USER, content="Fetch data")])
    print(f"\n🤖 Agent: {response.content}\n")


def demo_streaming_with_analytics() -> None:
    """Demonstrate streaming with analytics tracking."""
    print("\n" + "=" * 60)
    print("Demo 4: Streaming with Analytics")
    print("=" * 60)

    chunk_counter = {"count": 0}

    def count_chunks(tool_name: str, chunk: str) -> None:
        chunk_counter["count"] += 1
        # Display only milestone chunks to avoid clutter
        if "Milestone" in chunk or "🚀" in chunk or "🎉" in chunk:
            print(chunk, end="", flush=True)

    provider = LocalProvider(
        responses=[
            'TOOL_CALL: {"tool_name": "process_dataset", "parameters": {"size": 15}}',
            'TOOL_CALL: {"tool_name": "search_logs", "parameters": {"pattern": "INFO", "max_results": 10}}',
            "All tasks complete!",
        ]
    )

    config = AgentConfig(
        verbose=False,
        enable_analytics=True,
        hooks={"on_tool_chunk": count_chunks},
        max_iterations=5,
    )

    agent = Agent(tools=[process_dataset, search_logs], provider=provider, config=config)

    print("\n📝 User: Run multiple streaming tasks\n")
    response = agent.run([Message(role=Role.USER, content="Process data and search logs")])

    # Display analytics
    print("\n" + "=" * 60)
    print("Analytics Summary")
    print("=" * 60)
    analytics = agent.get_analytics()
    print(analytics.summary())


def demo_toolbox_streaming(tmp_path: Path) -> None:
    """Demonstrate streaming tools from toolbox."""
    print("\n" + "=" * 60)
    print("Demo 5: Toolbox Streaming Tools")
    print("=" * 60)

    # Create sample files
    test_file = tmp_path / "sample.txt"
    test_file.write_text(
        "This is line 1\n"
        "This is line 2\n"
        "This is line 3\n"
        "This is line 4\n"
        "This is line 5\n"
    )

    test_csv = tmp_path / "sample.csv"
    test_csv.write_text("name,age,city\n" "Alice,30,NYC\n" "Bob,25,SF\n" "Charlie,35,LA\n")

    from selectools.toolbox.data_tools import process_csv_stream
    from selectools.toolbox.file_tools import read_file_stream

    def display_chunk(tool_name: str, chunk: str) -> None:
        print(chunk, end="", flush=True)

    # Demo 5a: Read file stream
    print("\n--- 5a: read_file_stream ---\n")
    provider = LocalProvider(
        responses=[
            f'TOOL_CALL: {{"tool_name": "read_file_stream", "parameters": {{"filepath": "{test_file}"}}}}',
            "File read complete!",
        ]
    )

    config = AgentConfig(
        verbose=False,
        hooks={"on_tool_chunk": display_chunk},
        max_iterations=3,
    )

    agent = Agent(tools=[read_file_stream], provider=provider, config=config)
    response = agent.run([Message(role=Role.USER, content="Read the file")])
    print(f"\n🤖 Agent: {response.content}\n")

    # Demo 5b: Process CSV stream
    print("\n--- 5b: process_csv_stream ---\n")
    provider2 = LocalProvider(
        responses=[
            f'TOOL_CALL: {{"tool_name": "process_csv_stream", "parameters": {{"filepath": "{test_csv}"}}}}',
            "CSV processing complete!",
        ]
    )

    config2 = AgentConfig(
        verbose=False,
        hooks={"on_tool_chunk": display_chunk},
        max_iterations=3,
    )

    agent2 = Agent(tools=[process_csv_stream], provider=provider2, config=config2)
    response2 = agent2.run([Message(role=Role.USER, content="Process the CSV")])
    print(f"\n🤖 Agent: {response2.content}\n")


def main() -> None:
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# Streaming Tools Demo")
    print("#" * 60)

    # Demo 1: Basic streaming
    demo_basic_streaming()

    # Demo 2: Log search streaming
    demo_log_search_streaming()

    # Demo 3: Async streaming
    print("\nRunning async demo...")
    asyncio.run(demo_async_streaming())

    # Demo 4: Streaming with analytics
    demo_streaming_with_analytics()

    # Demo 5: Toolbox streaming tools
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_toolbox_streaming(Path(tmpdir))

    print("\n" + "#" * 60)
    print("# All Demos Complete!")
    print("#" * 60)


if __name__ == "__main__":
    main()
