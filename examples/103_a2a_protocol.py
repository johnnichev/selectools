#!/usr/bin/env python3
"""
A2A protocol — agent-to-agent communication (Google-backed standard).

Demonstrates A2AServer + A2AClient: one selectools agent exposes itself
over A2A (Agent Card discovery + JSON-RPC tasks) and another process
consumes it. Runs entirely offline — the client talks to the server
in-process via httpx's ASGITransport, no socket or API key needed.

Server endpoints:

    GET  /.well-known/agent.json  — Agent Card (name, capabilities, skills)
    POST /a2a                     — JSON-RPC 2.0 (message/send, tasks/get,
                                    tasks/cancel), optional bearer auth

In production, serve over a real socket:

    # server.py
    from selectools.serve import A2AServer
    server = A2AServer(agent=my_agent, auth_token="sk-...")
    server.serve(port=8000)

    # consumer.py
    from selectools.a2a import A2AClient
    client = A2AClient("https://other-agent.example.com", auth_token="sk-...")
    card = await client.discover()
    result = await client.send_task("Research quantum computing trends")

Prerequisites: pip install selectools[serve]
Run: python examples/103_a2a_protocol.py
"""

from __future__ import annotations

import asyncio

import httpx

from selectools import Agent, AgentConfig, tool
from selectools.a2a import A2AClient
from selectools.providers.stubs import LocalProvider
from selectools.serve import A2AServer

AUTH_TOKEN = "sk-demo-token"  # nosec B105


@tool()
def summarize(text: str) -> str:
    """Summarize a text in one sentence."""
    return f"Summary: {text[:60]}..."


def build_server() -> A2AServer:
    researcher = Agent(
        tools=[summarize],
        provider=LocalProvider(),
        config=AgentConfig(name="researcher", model="local-model"),
    )
    return A2AServer(
        agent=researcher,
        auth_token=AUTH_TOKEN,
        description="Research agent that summarizes topics",
    )


async def main() -> None:
    server = build_server()

    # In-process transport: the client speaks real A2A over ASGI, no socket.
    transport = httpx.ASGITransport(app=server)
    client = A2AClient("http://a2a-demo", auth_token=AUTH_TOKEN, transport=transport)

    print("=== 1. Discover the remote agent ===")
    card = await client.discover()
    print(f"name:         {card.name}")
    print(f"description:  {card.description}")
    print(f"protocol:     {card.protocol_version}")
    print(f"skills:       {[s.id for s in card.skills]}")

    print("\n=== 2. Send a task ===")
    task = await client.send_task("Research quantum computing trends")
    print(f"task id:      {task.id}")
    print(f"state:        {task.state}")
    print(f"answer:       {task.text[:120]}")

    print("\n=== 3. Retrieve the same task later (tasks/get) ===")
    fetched = await client.get_task(task.id)
    print(f"state:        {fetched.state}")

    print("\n=== 4. Sync wrappers for non-async code ===")
    sync_task = client.send_task_sync("Summarize the A2A protocol")
    print(f"state:        {sync_task.state}")
    print(f"answer:       {sync_task.text[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
