"""Safe synchronous-wrapper utilities for async code.

Calling :func:`asyncio.run` from a sync function that is itself reachable
from an async caller raises ``RuntimeError: asyncio.run() cannot be called
when another event loop is running``. This module provides a helper that
detects the surrounding event loop and executes the coroutine on a fresh
loop in a dedicated worker thread when one is already running.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code.

    If no event loop is running in the current thread, uses
    :func:`asyncio.run` directly. If one is running, spawns a worker
    thread, creates a fresh event loop there, and waits for the result.
    Safe to call from Jupyter notebooks, FastAPI handlers, async tests,
    and nested orchestration where a sync wrapper would otherwise crash.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    def _runner() -> T:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner)
        return future.result()
