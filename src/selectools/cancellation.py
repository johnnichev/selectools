"""Thread-safe cancellation token for agent runs.

Provides a cooperative cancellation mechanism that allows external code
(UI handlers, supervisor agents, timeout managers) to signal a running
agent to stop at the next iteration boundary.
"""

from __future__ import annotations

import threading

from .exceptions import CancellationError


class CancellationToken:
    """Thread-safe signal for cancelling an agent run.

    The agent loop checks ``is_cancelled`` at the start of each iteration
    and after tool execution.  When cancelled, the agent returns a partial
    result with whatever content it has accumulated so far.

    Usage::

        token = CancellationToken()
        task = asyncio.create_task(agent.arun("do work", cancel_token=token))

        # Later, from any thread:
        token.cancel()  # agent stops at next iteration boundary

    The token can be reused across runs by calling :meth:`reset`.
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation.  Thread-safe."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled.is_set()

    def reset(self) -> None:
        """Clear the cancellation signal for reuse."""
        self._cancelled.clear()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`CancellationError` if cancellation was requested."""
        if self._cancelled.is_set():
            raise CancellationError("Agent run was cancelled")


__all__ = ["CancellationToken"]
