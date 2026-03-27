"""
Composable pipelines — plain Python functions with ``|`` operator.

The anti-LCEL: no Runnable base class, no magic, no paid debugger.
Steps are plain functions. ``|`` is thin sugar. Every step auto-traces.

Usage::

    from selectools import step, Pipeline, parallel, branch

    @step
    def summarize(text: str) -> str:
        return agent.run(f"Summarize: {text}").content

    @step(retry=3)
    def translate(text: str, lang: str = "es") -> str:
        return agent.run(f"Translate to {lang}: {text}").content

    # Compose with | operator
    pipeline = summarize | translate
    result = pipeline.run("Long article...")

    # Parallel fan-out
    pipeline = parallel(search_web, search_docs) | merge | summarize

    # Conditional branching
    pipeline = classify | branch(technical=review, creative=edit) | publish
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def _filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the kwargs that *fn* accepts.

    If *fn* has a ``**kwargs`` catch-all, all kwargs pass through.
    Otherwise only named parameters present in the signature are forwarded.
    """
    if not kwargs:
        return kwargs
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return kwargs
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _is_subtype(output: type, expected: type) -> bool:
    """Check if output type is compatible with expected input type."""
    try:
        # Skip generics (Dict[str, Any], List[int], etc.) — can't issubclass
        if hasattr(output, "__origin__") or hasattr(expected, "__origin__"):
            return True  # Can't validate generics, assume compatible
        if output is Any or expected is Any:
            return True
        return issubclass(output, expected)
    except TypeError:
        return True  # Unknown types — don't warn


def _get_type_hints(fn: Callable) -> Dict[str, Any]:
    """Extract input/return type hints from a function, safely."""
    try:
        import typing

        hints = typing.get_type_hints(fn)
        result: Dict[str, Any] = {}
        params = list(hints.keys())
        # First non-return param is the "input"
        for p in params:
            if p != "return":
                result["input"] = hints[p]
                break
        if "return" in hints:
            result["return"] = hints["return"]
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Step — decorator that wraps a plain function
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single step or pipeline execution.

    Attributes:
        output: The return value of the step/pipeline.
        trace: List of (step_name, input, output, duration_ms) tuples.
        steps_run: Number of steps executed.
    """

    output: Any
    trace: List[Dict[str, Any]] = field(default_factory=list)
    steps_run: int = 0


class Step:
    """A callable wrapper around a plain function with composition support.

    Created by the ``@step`` decorator. The wrapped function remains callable
    as normal Python — the Step wrapper adds ``|``, retry, and tracing.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        name: Optional[str] = None,
        retry: int = 0,
        on_error: str = "raise",
        input_type: Optional[type] = None,
        output_type: Optional[type] = None,
    ) -> None:
        self.fn = fn
        self.name = (
            name or getattr(fn, "__name__", None) or getattr(fn, "name", None) or type(fn).__name__
        )
        self.retry = retry
        self.on_error = on_error
        self.is_async = asyncio.iscoroutinefunction(fn)

        # Type contracts (optional)
        self.input_type = input_type
        self.output_type = output_type

        # Auto-infer from type hints if not explicitly set
        if input_type is None or output_type is None:
            hints = _get_type_hints(fn)
            if input_type is None and hints.get("input"):
                self.input_type = hints["input"]
            if output_type is None and hints.get("return"):
                self.output_type = hints["return"]

        # Preserve function metadata
        wraps(fn)(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly — zero overhead."""
        return self.fn(*args, **kwargs)

    def __or__(self, other: Union[Step, Pipeline, Callable]) -> Pipeline:
        """Compose with | operator: ``step_a | step_b`` creates a Pipeline."""
        return Pipeline(steps=[self])._append(other)

    def __ror__(self, other: Union[Step, Pipeline, Callable]) -> Pipeline:
        """Support ``callable | step``."""
        if isinstance(other, Pipeline):
            return other._append(self)
        return Pipeline(steps=[_ensure_step(other)])._append(self)

    def __repr__(self) -> str:  # noqa: D105
        return f"Step({self.name!r})"


def step(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    retry: int = 0,
    on_error: str = "raise",
) -> Union[Step, Callable[[Callable], Step]]:
    """Decorator that wraps a function as a composable Step.

    Can be used with or without arguments::

        @step
        def my_fn(text: str) -> str: ...

        @step(retry=3)
        def my_fn(text: str) -> str: ...

    Args:
        fn: The function to wrap (when used without parentheses).
        name: Override the step name (defaults to function name).
        retry: Number of retries on failure (0 = no retry).
        on_error: Error handling: "raise" (default) or "skip".
    """
    if fn is not None:
        return Step(fn, name=name, retry=retry, on_error=on_error)

    def decorator(f: Callable) -> Step:
        return Step(f, name=name, retry=retry, on_error=on_error)

    return decorator


def _ensure_step(fn: Any) -> Step:
    """Convert a callable to a Step if it isn't one already."""
    if isinstance(fn, Step):
        return fn
    if isinstance(fn, Pipeline):
        return fn  # type: ignore[return-value]
    if callable(fn):
        return Step(fn)
    raise TypeError(f"Expected callable or Step, got {type(fn).__name__}")


# ---------------------------------------------------------------------------
# Pipeline — thin list of steps with run()
# ---------------------------------------------------------------------------


class Pipeline:
    """A sequence of steps executed in order.

    Created via the ``|`` operator or directly::

        pipeline = step_a | step_b | step_c
        pipeline = Pipeline(steps=[step_a, step_b, step_c])

    Execute with::

        result = pipeline.run(input_data)
        result = await pipeline.arun(input_data)
    """

    def __init__(
        self,
        steps: Optional[Sequence[Union[Step, Pipeline, Callable]]] = None,
        *,
        name: str = "pipeline",
    ) -> None:
        self.name = name
        self._steps: List[Union[Step, Pipeline]] = []
        for s in steps or []:
            if isinstance(s, (Step, Pipeline)):
                self._steps.append(s)
            elif callable(s):
                self._steps.append(Step(s))
            else:
                raise TypeError(f"Pipeline step must be callable, got {type(s).__name__}")

        # Validate type contracts between adjacent steps
        self._validate_type_contracts()

    def _validate_type_contracts(self) -> None:
        """Check that adjacent steps have compatible types (when annotated)."""
        for i in range(len(self._steps) - 1):
            current = self._steps[i]
            next_step = self._steps[i + 1]

            out_type = getattr(current, "output_type", None)
            in_type = getattr(next_step, "input_type", None)

            if out_type is not None and in_type is not None:
                # Check compatibility — basic check, not full subtype analysis
                if out_type is not in_type and not _is_subtype(out_type, in_type):
                    current_name = getattr(current, "name", f"step_{i}")
                    next_name = getattr(next_step, "name", f"step_{i + 1}")
                    out_name = getattr(out_type, "__name__", str(out_type))
                    in_name = getattr(in_type, "__name__", str(in_type))
                    import warnings

                    warnings.warn(
                        f"Pipeline type mismatch: {current_name!r} outputs {out_name} "
                        f"but {next_name!r} expects {in_name}",
                        stacklevel=3,
                    )

    @property
    def steps(self) -> List[Union[Step, Pipeline]]:
        """Read-only view of the pipeline's step list."""
        return list(self._steps)

    def _append(self, other: Union[Step, Pipeline, Callable]) -> Pipeline:
        """Return a new Pipeline with ``other`` appended."""
        new_steps = list(self._steps)
        if isinstance(other, Pipeline):
            new_steps.extend(other._steps)
        elif isinstance(other, Step):
            new_steps.append(other)
        elif callable(other):
            new_steps.append(Step(other))
        else:
            raise TypeError(f"Cannot compose with {type(other).__name__}")
        return Pipeline(steps=new_steps, name=self.name)

    def __or__(self, other: Union[Step, Pipeline, Callable]) -> Pipeline:  # noqa: D105
        return self._append(other)

    def __ror__(self, other: Union[Step, Pipeline, Callable]) -> Pipeline:  # noqa: D105
        if isinstance(other, Pipeline):
            return Pipeline(steps=list(other._steps) + list(self._steps), name=self.name)
        return Pipeline(steps=[_ensure_step(other)] + list(self._steps), name=self.name)

    def run(self, input: Any, **kwargs: Any) -> StepResult:
        """Execute the pipeline synchronously.

        Each step receives the previous step's output as its first argument.
        Additional kwargs are passed to every step that accepts them.

        Returns:
            StepResult with .output, .trace, and .steps_run.
        """
        trace: List[Dict[str, Any]] = []
        current = input
        steps_run = 0

        for s in self._steps:
            step_name = getattr(s, "name", type(s).__name__)
            start = time.time()

            try:
                current = self._execute_step(s, current, kwargs)
                duration_ms = (time.time() - start) * 1000
                trace.append(
                    {
                        "step": step_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "ok",
                    }
                )
                steps_run += 1
            except Exception as exc:
                duration_ms = (time.time() - start) * 1000
                trace.append(
                    {
                        "step": step_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                retry_count = getattr(s, "retry", 0)
                on_error = getattr(s, "on_error", "raise")

                if retry_count > 0:
                    for attempt in range(retry_count):
                        retry_start = time.time()
                        try:
                            current = self._execute_step(s, current, kwargs)
                            trace.append(
                                {
                                    "step": step_name,
                                    "duration_ms": round((time.time() - retry_start) * 1000, 2),
                                    "status": "ok",
                                    "retry": attempt + 1,
                                }
                            )
                            steps_run += 1
                            break
                        except Exception:
                            if attempt == retry_count - 1:
                                if on_error == "skip":
                                    break
                                raise
                elif on_error == "skip":
                    continue
                else:
                    raise

        return StepResult(output=current, trace=trace, steps_run=steps_run)

    async def arun(self, input: Any, **kwargs: Any) -> StepResult:
        """Execute the pipeline asynchronously.

        Async steps are awaited; sync steps run directly.
        """
        trace: List[Dict[str, Any]] = []
        current = input
        steps_run = 0

        for s in self._steps:
            step_name = getattr(s, "name", type(s).__name__)
            start = time.time()

            try:
                current = await self._aexecute_step(s, current, kwargs)
                duration_ms = (time.time() - start) * 1000
                trace.append(
                    {
                        "step": step_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "ok",
                    }
                )
                steps_run += 1
            except Exception as exc:
                duration_ms = (time.time() - start) * 1000
                trace.append(
                    {
                        "step": step_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                retry_count = getattr(s, "retry", 0)
                on_error = getattr(s, "on_error", "raise")

                if retry_count > 0:
                    for attempt in range(retry_count):
                        retry_start = time.time()
                        try:
                            current = await self._aexecute_step(s, current, kwargs)
                            trace.append(
                                {
                                    "step": step_name,
                                    "duration_ms": round((time.time() - retry_start) * 1000, 2),
                                    "status": "ok",
                                    "retry": attempt + 1,
                                }
                            )
                            steps_run += 1
                            break
                        except Exception:
                            if attempt == retry_count - 1:
                                if on_error == "skip":
                                    break
                                raise
                elif on_error == "skip":
                    continue
                else:
                    raise

        return StepResult(output=current, trace=trace, steps_run=steps_run)

    async def astream(self, input: Any, **kwargs: Any):
        """Stream the pipeline — runs all steps, yields chunks from the last step.

        Earlier steps run to completion. The final step's output is yielded
        as it's produced (if the step is a generator). Otherwise yields the
        complete output as a single chunk.

        Usage::

            async for chunk in pipeline.astream("input"):
                print(chunk, end="")
        """
        # Run all steps except the last
        current = input
        for s in self._steps[:-1]:
            current = await self._aexecute_step(s, current, kwargs)

        # Stream the last step
        if not self._steps:
            yield input
            return

        last = self._steps[-1]
        fn: Callable = last.fn if isinstance(last, Step) else last  # type: ignore[assignment]
        filtered = _filter_kwargs(fn, kwargs)

        if asyncio.iscoroutinefunction(fn):
            result = await fn(current, **filtered)
            yield result
        elif inspect.isgeneratorfunction(fn):
            for chunk in fn(current, **filtered):
                yield chunk
        elif inspect.isasyncgenfunction(fn):
            async for chunk in fn(current, **filtered):
                yield chunk
        else:
            result = fn(current, **filtered)
            yield result

    def _execute_step(self, s: Any, current: Any, kwargs: Dict[str, Any]) -> Any:
        """Execute a single step synchronously."""
        if isinstance(s, Pipeline):
            result = s.run(current, **kwargs)
            return result.output
        fn = s.fn if isinstance(s, Step) else s
        filtered = _filter_kwargs(fn, kwargs)
        if asyncio.iscoroutinefunction(fn):
            return asyncio.run(fn(current, **filtered))
        return fn(current, **filtered)

    async def _aexecute_step(self, s: Any, current: Any, kwargs: Dict[str, Any]) -> Any:
        """Execute a single step asynchronously."""
        if isinstance(s, Pipeline):
            result = await s.arun(current, **kwargs)
            return result.output
        fn = s.fn if isinstance(s, Step) else s
        filtered = _filter_kwargs(fn, kwargs)
        if asyncio.iscoroutinefunction(fn):
            return await fn(current, **filtered)
        return fn(current, **filtered)

    def __repr__(self) -> str:  # noqa: D105
        names = [getattr(s, "name", type(s).__name__) for s in self._steps]
        return f"Pipeline({' | '.join(names)})"

    # -- AgentGraph bridge --------------------------------------------------

    def __call__(self, state: Any) -> Any:
        """Make Pipeline usable as an AgentGraph callable node.

        If state is a GraphState, extracts last_output, runs pipeline,
        writes result back. Otherwise runs directly.
        """
        from .orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState

        if isinstance(state, GraphState):
            input_text = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            if not input_text and state.messages:
                input_text = getattr(state.messages[-1], "content", str(state.messages[-1]))
            result = self.run(input_text)
            state.data[STATE_KEY_LAST_OUTPUT] = (
                result.output if isinstance(result.output, str) else str(result.output)
            )
            return state
        return self.run(state).output


# ---------------------------------------------------------------------------
# parallel() — fan-out composition primitive
# ---------------------------------------------------------------------------


def parallel(*steps_or_fns: Union[Step, Callable]) -> Step:
    """Run multiple steps in parallel on the same input, return dict of results.

    Usage::

        research = parallel(search_web, search_docs, search_db)
        result = research("quantum computing")
        # result == {"search_web": "...", "search_docs": "...", "search_db": "..."}

    In async context, steps run via asyncio.gather. In sync, they run sequentially.
    """
    wrapped = [_ensure_step(s) for s in steps_or_fns]
    names = [s.name for s in wrapped]

    def _parallel_sync(input: Any, **kwargs: Any) -> Dict[str, Any]:
        results = {}
        for s in wrapped:
            fn = s.fn if isinstance(s, Step) else s
            filtered = _filter_kwargs(fn, kwargs)
            results[s.name] = fn(input, **filtered)
        return results

    async def _parallel_async(input: Any, **kwargs: Any) -> Dict[str, Any]:
        async def _run(s: Step) -> Tuple[str, Any]:
            fn = s.fn if isinstance(s, Step) else s
            filtered = _filter_kwargs(fn, kwargs)
            if asyncio.iscoroutinefunction(fn):
                return s.name, await fn(input, **filtered)
            return s.name, fn(input, **filtered)

        pairs = await asyncio.gather(*[_run(s) for s in wrapped])
        return dict(pairs)

    # Check if any step is async
    has_async = any(getattr(s, "is_async", False) for s in wrapped)

    if has_async:
        result_step = Step(_parallel_async, name=f"parallel({','.join(names)})")
    else:
        result_step = Step(_parallel_sync, name=f"parallel({','.join(names)})")

    return result_step


# ---------------------------------------------------------------------------
# branch() — conditional routing primitive
# ---------------------------------------------------------------------------


def branch(
    router: Optional[Callable] = None,
    **named_steps: Union[Step, Callable],
) -> Step:
    """Route input to one of several steps based on a classifier.

    Usage::

        pipeline = classify | branch(
            router=lambda x: x["category"],
            technical=code_review,
            creative=copyedit,
            default=passthrough,
        )

    If no router is provided, input must be a string matching a branch name.

    Args:
        router: Function that takes input and returns a branch name (str).
        **named_steps: Named branches (key = branch name, value = step/callable).
    """
    branches = {k: _ensure_step(v) for k, v in named_steps.items()}

    def _branch_fn(input: Any, **kwargs: Any) -> Any:
        if router is not None:
            key = router(input)
        elif isinstance(input, str):
            key = input
        elif isinstance(input, dict) and "branch" in input:
            key = input["branch"]
        else:
            raise ValueError(
                f"branch() received {type(input).__name__} but no router function. "
                "Provide router= or pass a string/dict with 'branch' key."
            )

        target = branches.get(key)
        if target is None:
            target = branches.get("default")
        if target is None:
            raise KeyError(
                f"branch() got key {key!r} but no matching branch. "
                f"Available: {list(branches.keys())}"
            )

        fn = target.fn if isinstance(target, Step) else target
        filtered = _filter_kwargs(fn, kwargs)
        return fn(input, **filtered)

    branch_names = list(branches.keys())
    return Step(_branch_fn, name=f"branch({','.join(branch_names)})")


# ---------------------------------------------------------------------------
# retry() / cache() step wrappers
# ---------------------------------------------------------------------------


def retry(step_or_fn: Union[Step, Callable], attempts: int = 3) -> Step:
    """Wrap a step with retry logic.

    Usage::

        pipeline = preprocess | retry(flaky_api_call, 3) | postprocess

    Args:
        step_or_fn: Step or callable to wrap.
        attempts: Number of retry attempts (total calls = 1 + attempts).
    """
    wrapped = _ensure_step(step_or_fn)
    return Step(wrapped.fn, name=wrapped.name, retry=attempts, on_error="raise")


def cache_step(
    step_or_fn: Union[Step, Callable],
    ttl: int = 300,
) -> Step:
    """Wrap a step with result caching.

    Caches based on input value. Same input returns cached output.

    Usage::

        pipeline = preprocess | cache_step(expensive_call, ttl=600) | postprocess

    Args:
        step_or_fn: Step or callable to wrap.
        ttl: Cache time-to-live in seconds.
    """
    wrapped = _ensure_step(step_or_fn)
    _cache: Dict[str, Any] = {}
    _timestamps: Dict[str, float] = {}

    def cached_fn(input: Any, **kwargs: Any) -> Any:
        key = str(input)
        now = time.time()
        if key in _cache and (now - _timestamps.get(key, 0)) < ttl:
            return _cache[key]
        fn = wrapped.fn
        filtered = _filter_kwargs(fn, kwargs)
        result = fn(input, **filtered)
        _cache[key] = result
        _timestamps[key] = now
        return result

    return Step(cached_fn, name=f"cached({wrapped.name})")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "Step",
    "StepResult",
    "Pipeline",
    "step",
    "parallel",
    "branch",
    "retry",
    "cache_step",
]
