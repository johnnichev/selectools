"""Mixin providing observer notification and fallback wiring methods for the Agent class."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from ..memory import ConversationMemory

logger = logging.getLogger(__name__)


def _log_task_exception(task: "asyncio.Task[Any]") -> None:
    """Done-callback that logs exceptions from fire-and-forget observer tasks.

    Without this callback, exceptions raised inside a non-blocking async
    observer become unhandled-exception warnings on the event loop (Python
    3.12+) and are otherwise silently lost. BUG-18 / Agno #6236.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Async observer raised: %s", exc, exc_info=exc)


class _LifecycleMixin:
    """Mixin that provides observer lifecycle methods for the Agent class.

    All methods access ``self.*`` attributes (config, provider, etc.)
    which are expected to be provided by the Agent class that inherits
    from this mixin.
    """

    def _notify_observers(self, method: str, *args: Any) -> None:
        """Call *method* on every registered observer, swallowing errors."""
        for obs in self.config.observers:
            try:
                getattr(obs, method)(*args)
            except Exception:  # noqa: BLE001 # nosec B110
                pass

    async def _anotify_observers(self, method: str, *args: Any) -> None:
        """Call async ``a_{method}`` on every AsyncAgentObserver, swallowing errors.

        - ``blocking=True`` observers: awaited inline (sequential).
        - ``blocking=False`` observers: dispatched via ``asyncio.ensure_future``.
        """
        from ..observer import AsyncAgentObserver

        async_method = f"a_{method}"
        for obs in self.config.observers:
            if not isinstance(obs, AsyncAgentObserver):
                continue
            handler = getattr(obs, async_method, None)
            if handler is None:
                continue
            try:
                if obs.blocking:
                    await handler(*args)
                else:
                    task = asyncio.ensure_future(handler(*args))
                    task.add_done_callback(_log_task_exception)
            except Exception:  # noqa: BLE001 # nosec B110
                pass

    def _truncate_tool_result(self, result: Optional[str]) -> Optional[str]:
        """Truncate tool result text for trace storage."""
        if result is None:
            return None
        limit = self.config.trace_tool_result_chars
        if limit is None:
            return result
        return result[:limit]

    _fallback_run_id: threading.local = threading.local()
    _lock_type: type = type(threading.Lock())

    def _wire_fallback_observer(self, run_id: Optional[str]) -> None:
        """If the provider is a FallbackProvider, wire its on_fallback to observers.

        Thread-safe: uses a lock + refcount so multiple concurrent ``run()``
        calls (e.g. from ``batch()``) share a single callback on the provider
        while each thread's ``run_id`` is carried via a thread-local.
        The lock persists on the provider to avoid races from delete-then-recreate.
        """
        if not run_id or not self.config.observers:
            return
        provider = self.provider
        if not hasattr(provider, "on_fallback"):
            return

        _LifecycleMixin._fallback_run_id.value = run_id

        raw_lock = getattr(provider, "_fb_wire_lock", None)
        if not isinstance(raw_lock, _LifecycleMixin._lock_type):
            raw_lock = threading.Lock()
            provider._fb_wire_lock = raw_lock  # type: ignore[attr-defined]
        lock = cast(threading.Lock, raw_lock)

        agent_ref = self

        with lock:
            refcount: int = getattr(provider, "_fb_wire_refcount", 0)
            if refcount == 0:
                provider._fb_original_on_fallback = provider.on_fallback  # type: ignore[attr-defined]
                user_cb = provider.on_fallback

                def _observer_fallback(
                    failed: str,
                    next_p: str,
                    exc: Exception,
                ) -> None:
                    rid = getattr(_LifecycleMixin._fallback_run_id, "value", "")
                    agent_ref._notify_observers(
                        "on_provider_fallback",
                        rid,
                        failed,
                        next_p,
                        exc,
                    )
                    if user_cb:
                        try:
                            user_cb(failed, next_p, exc)
                        except Exception:  # nosec B110
                            pass

                provider.on_fallback = _observer_fallback  # type: ignore[attr-defined]

            provider._fb_wire_refcount = refcount + 1  # type: ignore[attr-defined]

    def _unwire_fallback_observer(self) -> None:
        """Restore FallbackProvider's original on_fallback callback (thread-safe).

        The lock is kept on the provider (never deleted) to prevent races when
        concurrent threads overlap wire / unwire calls.
        """
        provider = self.provider
        raw_lock = getattr(provider, "_fb_wire_lock", None)
        if not isinstance(raw_lock, _LifecycleMixin._lock_type):
            return
        lock = cast(threading.Lock, raw_lock)

        with lock:
            refcount: int = getattr(provider, "_fb_wire_refcount", 0) - 1
            if refcount < 0:
                refcount = 0
            provider._fb_wire_refcount = refcount  # type: ignore[attr-defined]
            if refcount == 0:
                original = getattr(provider, "_fb_original_on_fallback", None)
                provider.on_fallback = original  # type: ignore[attr-defined]
                if hasattr(provider, "_fb_original_on_fallback"):
                    try:
                        delattr(provider, "_fb_original_on_fallback")
                    except Exception:  # nosec B110
                        pass
