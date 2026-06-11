"""
Cost-optimized model router.

``RouterProvider`` wraps multiple providers organized in cost tiers
(cheapest to priciest) and routes each request to a tier based on a
deterministic, rule-based complexity classification of the input:

- **simple**: short, single-intent prompts with few tools
- **moderate**: multi-part questions, structured-output requests, moderate
  input size or tool counts
- **complex**: long inputs, code blocks, explicit reasoning requests
  ("step by step", "analyze", ...), or large tool inventories

The classifier is a transparent additive score over documented signals
(see :func:`classify_complexity`); thresholds and keyword lists are
configurable via :class:`RouterConfig`. An LLM-based classifier is
deliberately out of scope: deterministic and testable beats clever.

Failure escalation reuses :class:`~selectools.providers.fallback.FallbackProvider`
(retriable-error detection, circuit breaker, partial-stream safety) over the
escalation chain rather than reimplementing those semantics.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from ..pricing import get_model_pricing
from ..stability import beta, register_stability
from ..token_estimation import estimate_tokens
from ..types import Message, Role, ToolCall, text_content
from .fallback import FallbackProvider

if TYPE_CHECKING:
    from ..tools.base import Tool
    from ..usage import UsageStats

logger = logging.getLogger(__name__)

COMPLEXITY_SIMPLE = "simple"
COMPLEXITY_MODERATE = "moderate"
COMPLEXITY_COMPLEX = "complex"

# Constants cannot carry a __stability__ attribute: register instead (beta,
# matching the router domain).
register_stability("COMPLEXITY_SIMPLE", "beta")
register_stability("COMPLEXITY_MODERATE", "beta")
register_stability("COMPLEXITY_COMPLEX", "beta")

_STRATEGIES = ("cost_optimized", "quality_first", "balanced")

_DEFAULT_REASONING_KEYWORDS: Tuple[str, ...] = (
    "step by step",
    "step-by-step",
    "explain why",
    "explain how",
    "analyze",
    "analyse",
    "compare",
    "trade-off",
    "tradeoff",
    "refactor",
    "debug",
    "optimize",
    "architect",
    "prove",
    "derive",
    "design a",
    "implement",
    "reason about",
)
_DEFAULT_STRUCTURED_KEYWORDS: Tuple[str, ...] = (
    "json",
    "yaml",
    "xml",
    "csv",
    "schema",
    "structured output",
    "markdown table",
)

_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s")


@beta
@dataclass
class RouterConfig:
    """Thresholds and signal weights for the rule-based complexity classifier.

    Each signal adds points to an additive score; the score maps to a
    complexity class via ``moderate_score`` / ``complex_score`` boundaries.

    Attributes:
        moderate_token_threshold: Input token count at/above which +1.
        complex_token_threshold: Input token count at/above which +2.
        moderate_tool_threshold: Tool count at/above which +1.
        complex_tool_threshold: Tool count at/above which +2.
        reasoning_keywords: Case-insensitive substrings worth +2 (any match).
        structured_keywords: Case-insensitive substrings worth +1 (any match).
        multi_part_min_questions: Question marks at/above which +1.
        moderate_score: Total score at/above which the input is "moderate".
        complex_score: Total score at/above which the input is "complex".
    """

    moderate_token_threshold: int = 400
    complex_token_threshold: int = 1500
    moderate_tool_threshold: int = 4
    complex_tool_threshold: int = 8
    reasoning_keywords: Tuple[str, ...] = field(default=_DEFAULT_REASONING_KEYWORDS)
    structured_keywords: Tuple[str, ...] = field(default=_DEFAULT_STRUCTURED_KEYWORDS)
    multi_part_min_questions: int = 2
    moderate_score: int = 2
    complex_score: int = 4


@beta
def classify_complexity(
    text: str,
    *,
    total_tokens: int,
    tool_count: int,
    config: Optional[RouterConfig] = None,
) -> str:
    """Classify request complexity from deterministic, rule-based signals.

    Scoring (additive):

    - input size: +2 if ``total_tokens >= complex_token_threshold``,
      else +1 if ``>= moderate_token_threshold``
    - tool count: +2 if ``tool_count >= complex_tool_threshold``,
      else +1 if ``>= moderate_tool_threshold``
    - code block (three backticks) present: +2
    - any reasoning keyword present: +2
    - multi-part question (>= ``multi_part_min_questions`` question marks,
      or a numbered list): +1
    - structured-output keyword present: +1

    Score >= ``complex_score`` -> "complex"; >= ``moderate_score`` ->
    "moderate"; otherwise "simple".

    Args:
        text: The user-facing prompt text (typically the latest user message).
        total_tokens: Estimated token count of the full input (system prompt
            plus all messages).
        tool_count: Number of tools available to the agent.
        config: Threshold overrides; defaults to :class:`RouterConfig`.

    Returns:
        One of ``"simple"``, ``"moderate"``, ``"complex"``.
    """
    cfg = config if config is not None else RouterConfig()
    lowered = text.lower()
    score = 0

    if total_tokens >= cfg.complex_token_threshold:
        score += 2
    elif total_tokens >= cfg.moderate_token_threshold:
        score += 1

    if tool_count >= cfg.complex_tool_threshold:
        score += 2
    elif tool_count >= cfg.moderate_tool_threshold:
        score += 1

    if "```" in text:
        score += 2

    if any(kw in lowered for kw in cfg.reasoning_keywords):
        score += 2

    if lowered.count("?") >= cfg.multi_part_min_questions or _NUMBERED_LIST_RE.search(text):
        score += 1

    if any(kw in lowered for kw in cfg.structured_keywords):
        score += 1

    if score >= cfg.complex_score:
        return COMPLEXITY_COMPLEX
    if score >= cfg.moderate_score:
        return COMPLEXITY_MODERATE
    return COMPLEXITY_SIMPLE


class _TierBinding:
    """Binds a provider to a tier name and tier-specific model.

    Presents the tier name as ``name`` so FallbackProvider's bookkeeping
    (circuit breaker, ``provider_used``) operates on tier names, and
    overrides the incoming ``model`` argument with the tier's own model
    when one is known.
    """

    def __init__(self, tier: str, provider: Any, model: Optional[str]) -> None:
        self.name = tier
        self.provider = provider
        self.model = model

    @property
    def supports_streaming(self) -> bool:
        return bool(getattr(self.provider, "supports_streaming", False))

    @property
    def supports_async(self) -> bool:
        return bool(getattr(self.provider, "supports_async", False))

    def _model(self, incoming: str) -> str:
        return self.model if self.model is not None else incoming

    def complete(self, *, model: str, **kwargs: Any) -> tuple[Message, "UsageStats"]:
        result: tuple[Message, "UsageStats"] = self.provider.complete(
            model=self._model(model), **kwargs
        )
        return result

    async def acomplete(self, *, model: str, **kwargs: Any) -> tuple[Message, "UsageStats"]:
        result: tuple[Message, "UsageStats"] = await self.provider.acomplete(
            model=self._model(model), **kwargs
        )
        return result

    def stream(self, *, model: str, **kwargs: Any) -> Iterable[Union[str, ToolCall]]:
        return self.provider.stream(model=self._model(model), **kwargs)  # type: ignore[no-any-return]

    def astream(self, *, model: str, **kwargs: Any) -> AsyncIterable[Union[str, ToolCall]]:
        return self.provider.astream(model=self._model(model), **kwargs)  # type: ignore[no-any-return]


@beta
class RouterProvider:
    """Routes requests to the cheapest capable model tier.

    Providers are organized in tiers ordered cheapest -> priciest. Each
    request is classified (see :func:`classify_complexity`) and dispatched
    to a starting tier chosen by ``strategy``:

    - ``cost_optimized``: simple -> cheapest tier, moderate -> middle tier,
      complex -> top tier.
    - ``quality_first``: every request starts at the top tier and degrades
      down-tier on retriable failure (availability over cost).
    - ``balanced``: never routes below the middle tier; simple and moderate
      go to the middle tier, complex to the top tier. The middle tier is
      index ``len(tier_order) // 2`` (cheapest-first), which rounds toward
      the pricier tier for even tier counts: with 2 tiers the middle IS the
      top tier, with 4 tiers it is the upper-middle tier.

    On retriable failure (rate limit, timeout, 5xx, ...) the request
    escalates to the next tier in the chain, reusing
    :class:`FallbackProvider`'s retry detection and circuit breaker.
    Non-retriable errors (e.g. auth failures) propagate immediately.

    Tier order defaults to the ``providers`` dict insertion order. When
    every tier's model is known to the pricing registry, tiers are
    re-sorted by cost (a warning is logged if that disagrees with the
    insertion order). ``tier_order`` overrides both.

    Attributes:
        name: Always ``"router"``.
        tier_order: Resolved tier names, cheapest first.
        tier_used: Tier that served the most recent request. Diagnostic
            only: last-write-wins, unreliable under concurrent use. Use the
            ``on_route``/``on_escalation`` callbacks for per-request
            attribution.
        complexity_used: Complexity class of the most recent request. Same
            last-write-wins caveat as ``tier_used``.

    Note:
        ``stream`` and ``astream`` are generators: routing and the
        ``on_route`` callback fire at first iteration, not at call time,
        and ``tier_used`` is only updated once the stream is fully
        consumed, so it is stale if a stream is abandoned midway.

    Example::

        router = RouterProvider(
            providers={
                "fast": OpenAIProvider(default_model="gpt-5.4-nano"),
                "smart": AnthropicProvider(default_model="claude-sonnet-4-6"),
                "power": OpenAIProvider(default_model="gpt-5.4-pro"),
            },
            strategy="cost_optimized",
        )
        agent = Agent(tools, provider=router)
    """

    name: str = "router"

    def __init__(
        self,
        providers: Dict[str, Any],
        strategy: str = "cost_optimized",
        tier_order: Optional[List[str]] = None,
        tier_models: Optional[Dict[str, str]] = None,
        config: Optional[RouterConfig] = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown: float = 60.0,
        on_route: Optional[Callable[[str, str], None]] = None,
        on_escalation: Optional[Callable[[str, str, Exception], None]] = None,
    ) -> None:
        """
        Args:
            providers: Mapping of tier name -> provider, cheapest first by
                convention. Each tier's model is taken from ``tier_models``
                or the provider's ``default_model`` attribute.
            strategy: ``"cost_optimized"``, ``"quality_first"`` or
                ``"balanced"``.
            tier_order: Explicit cheapest-first tier ordering. Must be a
                permutation of ``providers`` keys. Overrides pricing-based
                ordering.
            tier_models: Per-tier model override, e.g. ``{"fast": "gpt-5.4-nano"}``.
            config: Classifier thresholds (see :class:`RouterConfig`).
            circuit_breaker_threshold: Consecutive failures before a tier is
                temporarily skipped.
            circuit_breaker_cooldown: Seconds a tripped tier is skipped.
            on_route: Callback ``(complexity, tier)`` invoked after routing.
            on_escalation: Callback ``(failed_tier, next_tier, exception)``
                invoked when a tier fails and the request escalates.
        """
        if not providers:
            raise ValueError("RouterProvider requires at least one provider tier.")
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose from {_STRATEGIES}.")
        if tier_models:
            unknown = sorted(set(tier_models) - set(providers))
            if unknown:
                raise ValueError(
                    f"tier_models keys {unknown!r} do not match any provider tier; "
                    f"known tiers: {sorted(providers)!r}."
                )

        self.providers = dict(providers)
        self.strategy = strategy
        self.config = config if config is not None else RouterConfig()
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self.on_route = on_route
        self.on_escalation = on_escalation
        self.tier_used: Optional[str] = None
        self.complexity_used: Optional[str] = None

        self._tier_models: Dict[str, Optional[str]] = {
            tier: (tier_models or {}).get(tier, getattr(provider, "default_model", None))
            for tier, provider in self.providers.items()
        }
        self.tier_order: List[str] = self._resolve_tier_order(tier_order)
        self._bindings: Dict[str, _TierBinding] = {
            tier: _TierBinding(tier, self.providers[tier], self._tier_models[tier])
            for tier in self.tier_order
        }
        self._chain_lock = threading.Lock()
        self._chains: Dict[Tuple[str, ...], FallbackProvider] = {}

    # -- capability flags ----------------------------------------------------

    @property
    def supports_streaming(self) -> bool:
        return any(getattr(p, "supports_streaming", False) for p in self.providers.values())

    @property
    def supports_async(self) -> bool:
        return any(getattr(p, "supports_async", False) for p in self.providers.values())

    # -- tier ordering ---------------------------------------------------------

    def _resolve_tier_order(self, tier_order: Optional[List[str]]) -> List[str]:
        names = list(self.providers.keys())
        if tier_order is not None:
            if sorted(tier_order) != sorted(names):
                raise ValueError(
                    f"tier_order {tier_order!r} must be a permutation of provider tiers {names!r}."
                )
            return list(tier_order)

        costs: Dict[str, Tuple[float, float]] = {}
        for tier in names:
            model = self._tier_models[tier]
            pricing = get_model_pricing(model) if model else None
            if pricing is None:
                logger.warning(
                    "RouterProvider: tier %r model %r not in pricing registry; "
                    "pricing verification skipped, keeping insertion order. "
                    "Pass tier_order to silence this warning.",
                    tier,
                    model,
                )
                return names
            costs[tier] = (pricing["prompt"], pricing["completion"])

        by_cost = sorted(names, key=lambda t: costs[t])
        if by_cost != names:
            logger.warning(
                "RouterProvider: providers dict order %r is not cheapest-first; "
                "re-sorted by pricing to %r. Pass tier_order to override.",
                names,
                by_cost,
            )
        return by_cost

    # -- routing ---------------------------------------------------------------

    def _start_index(self, complexity: str) -> int:
        n = len(self.tier_order)
        top = n - 1
        # Middle rounds toward the pricier tier for even tier counts
        # (2 tiers -> top, 4 tiers -> upper-middle), so "balanced" never
        # degenerates to "cost_optimized".
        middle = n // 2
        if self.strategy == "quality_first":
            return top
        if self.strategy == "balanced":
            return top if complexity == COMPLEXITY_COMPLEX else middle
        # cost_optimized
        if complexity == COMPLEXITY_COMPLEX:
            return top
        if complexity == COMPLEXITY_MODERATE:
            return middle
        return 0

    def _escalation_chain(self, complexity: str) -> Tuple[str, ...]:
        if self.strategy == "quality_first":
            # Start at the top, degrade down-tier on failure.
            return tuple(reversed(self.tier_order))
        return tuple(self.tier_order[self._start_index(complexity) :])

    def _chain_provider(self, chain: Tuple[str, ...]) -> FallbackProvider:
        with self._chain_lock:
            fb = self._chains.get(chain)
            if fb is None:
                fb = FallbackProvider(
                    providers=[self._bindings[t] for t in chain],
                    circuit_breaker_threshold=self.circuit_breaker_threshold,
                    circuit_breaker_cooldown=self.circuit_breaker_cooldown,
                    on_fallback=self.on_escalation,
                )
                self._chains[chain] = fb
            return fb

    def _route(
        self,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List["Tool"]],
    ) -> FallbackProvider:
        last_user = next((m for m in reversed(messages) if m.role == Role.USER), None)
        # text_content handles multimodal messages whose text lives in
        # content_parts (content="" by convention); reading .content alone
        # would score those requests as empty and route them to the
        # cheapest tier.
        text = text_content(last_user) if last_user else ""
        total_tokens = estimate_tokens(system_prompt) + sum(
            estimate_tokens(text_content(m)) for m in messages
        )
        complexity = classify_complexity(
            text,
            total_tokens=total_tokens,
            tool_count=len(tools) if tools else 0,
            config=self.config,
        )
        chain = self._escalation_chain(complexity)
        self.complexity_used = complexity
        if self.on_route:
            try:
                self.on_route(complexity, chain[0])
            except Exception:  # nosec B110
                pass
        return self._chain_provider(chain)

    # -- Provider protocol -------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, "UsageStats"]:
        chain_fb = self._route(system_prompt, messages, tools)
        result = chain_fb.complete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # chain_fb is shared by every request of the same complexity class,
        # so chain_fb.provider_used is last-write-wins under concurrency.
        # Capture it into a local immediately after the call returns to
        # minimize the race window. Truly per-request attribution would
        # require FallbackProvider's @stable API to return the serving
        # provider alongside the result, which is out of scope here; use
        # the on_route/on_escalation callbacks for reliable per-request
        # attribution.
        served_tier = chain_fb.provider_used
        self.tier_used = served_tier
        return result

    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, "UsageStats"]:
        chain_fb = self._route(system_prompt, messages, tools)
        result = await chain_fb.acomplete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # See complete() for the concurrency caveat on this capture.
        served_tier = chain_fb.provider_used
        self.tier_used = served_tier
        return result

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> Iterable[Union[str, ToolCall]]:
        """Stream a completion from the routed tier.

        This is a generator: routing and the ``on_route`` callback fire at
        first iteration, not at call time. ``tier_used`` is only updated
        once the stream is fully consumed; it is stale if the stream is
        abandoned midway.
        """
        chain_fb = self._route(system_prompt, messages, tools)
        yield from chain_fb.stream(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # See complete() for the concurrency caveat on this capture.
        served_tier = chain_fb.provider_used
        self.tier_used = served_tier

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        """Stream a completion asynchronously from the routed tier.

        This is an async generator: routing and the ``on_route`` callback
        fire at first iteration, not at call time. ``tier_used`` is only
        updated once the stream is fully consumed; it is stale if the
        stream is abandoned midway.
        """
        chain_fb = self._route(system_prompt, messages, tools)
        async for chunk in chain_fb.astream(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        ):
            yield chunk
        # See complete() for the concurrency caveat on this capture.
        served_tier = chain_fb.provider_used
        self.tier_used = served_tier


__stability__ = "beta"

__all__ = [
    "RouterProvider",
    "RouterConfig",
    "classify_complexity",
    "COMPLEXITY_SIMPLE",
    "COMPLEXITY_MODERATE",
    "COMPLEXITY_COMPLEX",
]
