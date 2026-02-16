"""
Async Agent Demo using Selectools

This example demonstrates the async capabilities of Selectools, including:
- Async tool functions
- Async agent execution with Agent.arun()
- Concurrent agent execution
- Async with conversation memory

Requirements:
- Set OPENAI_API_KEY in your environment or .env file
- Install: pip install selectools
"""

import asyncio
import time

from selectools import Agent, AgentConfig, ConversationMemory, Message, OpenAIProvider, Role, tool
from selectools.models import OpenAI


# Example 1: Async Tool Functions
@tool(description="Simulate an async API call")
async def fetch_weather(city: str) -> str:
    """Simulates fetching weather data asynchronously."""
    await asyncio.sleep(0.5)  # Simulate network delay
    return f"Weather in {city}: Sunny, 72°F"


@tool(description="Simulate an async database query")
async def get_user_info(user_id: str) -> str:
    """Simulates fetching user data asynchronously."""
    await asyncio.sleep(0.3)  # Simulate DB query
    return f"User {user_id}: John Doe, age 30, location: New York"


# Example 2: Sync and Async Tools Together
@tool(description="Synchronous calculation")
def calculate_sum(a: int, b: int) -> str:
    """A sync tool that works with async agent."""
    return f"The sum of {a} and {b} is {a + b}"


async def example_basic_async() -> None:
    """Basic async agent usage."""
    print("\n=== Example 1: Basic Async Agent ===")

    agent = Agent(
        tools=[fetch_weather, calculate_sum],
        provider=OpenAIProvider(),
        config=AgentConfig(model=OpenAI.GPT_4O_MINI.id, max_iterations=3),
    )

    start = time.time()
    response = await agent.arun(
        [Message(role=Role.USER, content="What's the weather in San Francisco?")]
    )
    elapsed = time.time() - start

    print(f"Response: {response.content}")
    print(f"Time taken: {elapsed:.2f}s")


async def example_concurrent_agents() -> None:
    """Run multiple agents concurrently."""
    print("\n=== Example 2: Concurrent Agent Execution ===")

    agent1 = Agent(
        tools=[fetch_weather],
        provider=OpenAIProvider(),
        config=AgentConfig(model=OpenAI.GPT_4O_MINI.id, max_iterations=2),
    )

    agent2 = Agent(
        tools=[get_user_info],
        provider=OpenAIProvider(),
        config=AgentConfig(model=OpenAI.GPT_4O_MINI.id, max_iterations=2),
    )

    start = time.time()

    # Run both agents concurrently
    results = await asyncio.gather(
        agent1.arun([Message(role=Role.USER, content="Weather in London?")]),
        agent2.arun([Message(role=Role.USER, content="Get info for user_123")]),
    )

    elapsed = time.time() - start

    print(f"Agent 1 response: {results[0].content}")
    print(f"Agent 2 response: {results[1].content}")
    print(f"Total time (concurrent): {elapsed:.2f}s")


async def example_async_with_memory() -> None:
    """Async agent with conversation memory."""
    print("\n=== Example 3: Async Agent with Memory ===")

    memory = ConversationMemory(max_messages=20)

    agent = Agent(
        tools=[fetch_weather, calculate_sum],
        provider=OpenAIProvider(),
        config=AgentConfig(model=OpenAI.GPT_4O_MINI.id, max_iterations=3),
        memory=memory,
    )

    # Turn 1
    response1 = await agent.arun([Message(role=Role.USER, content="What's 15 + 27?")])
    print(f"Turn 1: {response1.content}")
    print(f"Memory size: {len(memory)} messages")

    # Turn 2 - memory persists
    response2 = await agent.arun(
        [Message(role=Role.USER, content="Now check the weather in Tokyo")]
    )
    print(f"Turn 2: {response2.content}")
    print(f"Memory size: {len(memory)} messages")


async def example_async_streaming() -> None:
    """Async agent with streaming responses."""
    print("\n=== Example 4: Async Agent with Streaming ===")

    agent = Agent(
        tools=[fetch_weather],
        provider=OpenAIProvider(),
        config=AgentConfig(
            model=OpenAI.GPT_4O_MINI.id, max_iterations=2, stream=True
        ),  # Enable streaming
    )

    def stream_handler(chunk: str) -> None:
        print(chunk, end="", flush=True)

    print("Streaming response: ", end="")
    response = await agent.arun(
        [Message(role=Role.USER, content="What's the weather in Paris?")],
        stream_handler=stream_handler,
    )
    print()  # New line after streaming


async def example_fastapi_integration() -> None:
    """Example showing how to use async agent in FastAPI."""
    print("\n=== Example 5: FastAPI Integration Pattern ===")

    # This shows the pattern - not running actual FastAPI here
    print(
        """
# FastAPI Integration Example:

from fastapi import FastAPI
from selectools import Agent, Message, Role, tool, OpenAIProvider

app = FastAPI()

@tool(description="Fetch data")
async def fetch_data(query: str) -> str:
    # Your async logic here
    return f"Data for {query}"

@app.post("/chat")
async def chat(message: str):
    agent = Agent(
        tools=[fetch_data],
        provider=OpenAIProvider()
    )

    response = await agent.arun([
        Message(role=Role.USER, content=message)
    ])

    return {"response": response.content}
    """
    )


async def main() -> None:
    """Run all examples."""
    print("Async Selectools Examples")
    print("=" * 50)

    try:
        await example_basic_async()
        await example_concurrent_agents()
        await example_async_with_memory()
        await example_async_streaming()
        await example_fastapi_integration()

        print("\n" + "=" * 50)
        print("All examples completed!")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure OPENAI_API_KEY is set in your environment.")


if __name__ == "__main__":
    asyncio.run(main())
