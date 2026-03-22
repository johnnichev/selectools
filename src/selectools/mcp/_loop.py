"""Background event loop for running MCP async operations from sync code."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class _BackgroundLoop:
    """Persistent background event loop for MCP sessions.

    MCP sessions are async and tied to their event loop. Using asyncio.run()
    per call would create and destroy the loop (and the session) each time.
    This class maintains a background thread with a persistent loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run an async coroutine on the background loop and return the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        """Stop the background loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


# Module-level singleton — created lazily
_bg_loop: _BackgroundLoop | None = None
_bg_lock = threading.Lock()


def get_background_loop() -> _BackgroundLoop:
    """Get or create the module-level background event loop."""
    global _bg_loop
    if _bg_loop is None:
        with _bg_lock:
            if _bg_loop is None:
                _bg_loop = _BackgroundLoop()
    return _bg_loop
