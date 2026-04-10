"""Safe synchronous-wrapper utilities for async code.

Calling :func:`asyncio.run` from a sync function that is itself reachable
from an async caller raises ``RuntimeError: asyncio.run() cannot be called
when another event loop is running``. This module provides a helper that
detects the surrounding event loop and executes the coroutine on a fresh
loop in a dedicated worker thread when one is already running.

The worker thread lives in a module-level :class:`ThreadPoolExecutor`
singleton (never create a new ``ThreadPoolExecutor`` per call — pitfall #20).
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Optional, TypeVar

T = TypeVar("T")

# Module-level singleton for running coroutines from sync code that is
# reachable from an async caller. Creating a new ThreadPoolExecutor per
# call wastes resources and prevents thread reuse (pitfall #20).
_RUN_SYNC_EXECUTOR: Optional[ThreadPoolExecutor] = None
_RUN_SYNC_EXECUTOR_LOCK = threading.Lock()


def _get_run_sync_executor() -> ThreadPoolExecutor:
    """Return the shared worker pool, creating it once on first use."""
    global _RUN_SYNC_EXECUTOR
    if _RUN_SYNC_EXECUTOR is None:
        with _RUN_SYNC_EXECUTOR_LOCK:
            if _RUN_SYNC_EXECUTOR is None:
                _RUN_SYNC_EXECUTOR = ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="selectools-run-sync",
                )
    return _RUN_SYNC_EXECUTOR


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code.

    If no event loop is running in the current thread, uses
    :func:`asyncio.run` directly. If one is running, submits the coroutine
    to a module-level worker pool that executes it on a fresh loop in a
    dedicated thread. Safe to call from Jupyter notebooks, FastAPI handlers,
    async tests, and nested orchestration where a sync wrapper would
    otherwise crash.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    def _runner() -> T:
        return asyncio.run(coro)  # type: ignore[arg-type]

    executor = _get_run_sync_executor()
    future = executor.submit(_runner)
    return future.result()
