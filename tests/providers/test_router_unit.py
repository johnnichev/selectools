"""
Unit tests for RouterProvider: complexity classification, strategy tier
selection, tier ordering (pricing + override), failure escalation, and
sync/async/stream delegation.
"""

from __future__ import annotations

from typing import Any, AsyncIterable, Iterable, List, Tuple, Union

import pytest

from selectools.providers.base import ProviderError
from selectools.providers.router import (
    COMPLEXITY_COMPLEX,
    COMPLEXITY_MODERATE,
    COMPLEXITY_SIMPLE,
    RouterConfig,
    RouterProvider,
    classify_complexity,
)
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats


def _usage(model: str = "mock") -> UsageStats:
    return UsageStats(0, 0, 0, 0.0, model, "mock")


def _user(text: str) -> List[Message]:
    return [Message(role=Role.USER, content=text)]


class _TierStub:
    """Recording stub provider for one router tier."""

    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        name: str,
        default_model: str | None = None,
        fail_with: str | None = None,
    ) -> None:
        self.name = name
        if default_model is not None:
            self.default_model = default_model
        self.fail_with = fail_with
        self.calls: List[dict] = []

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise ProviderError(self.fail_with)

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.calls.append(kwargs)
        self._maybe_fail()
        return Message(role=Role.ASSISTANT, content=f"from:{self.name}"), _usage()

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.calls.append(kwargs)
        self._maybe_fail()
        return Message(role=Role.ASSISTANT, content=f"from:{self.name}"), _usage()

    def stream(self, **kwargs: Any) -> Iterable[Union[str, ToolCall]]:
        self.calls.append(kwargs)
        self._maybe_fail()
        yield f"chunk:{self.name}"

    async def astream(self, **kwargs: Any) -> AsyncIterable[Union[str, ToolCall]]:
        self.calls.append(kwargs)
        self._maybe_fail()
        yield f"chunk:{self.name}"


def _three_tiers(**kwargs: Any) -> Tuple[_TierStub, _TierStub, _TierStub, RouterProvider]:
    fast = _TierStub("fast-stub")
    smart = _TierStub("smart-stub")
    power = _TierStub("power-stub")
    router = RouterProvider(
        providers={"fast": fast, "smart": smart, "power": power},
        **kwargs,
    )
    return fast, smart, power, router


SIMPLE_PROMPT = "What is the capital of France?"
COMPLEX_PROMPT = (
    "Analyze the following architecture step by step and explain why it fails "
    "under load. Compare the trade-offs of each option.\n"
    "```python\nprint('hello')\n```\n"
    "1. What is the bottleneck?\n2. How would you refactor it?\n"
)


# ---------------------------------------------------------------------------
# Complexity classifier
# ---------------------------------------------------------------------------


class TestClassifyComplexity:
    def test_short_plain_question_is_simple(self) -> None:
        assert classify_complexity(SIMPLE_PROMPT, total_tokens=10, tool_count=0) == (
            COMPLEXITY_SIMPLE
        )

    def test_reasoning_keywords_and_code_block_is_complex(self) -> None:
        assert classify_complexity(COMPLEX_PROMPT, total_tokens=100, tool_count=0) == (
            COMPLEXITY_COMPLEX
        )

    def test_token_threshold_bumps_to_moderate(self) -> None:
        config = RouterConfig(moderate_token_threshold=50, complex_token_threshold=10_000)
        # One signal (+1): below moderate_score=2 alone -> add structured keyword (+1).
        result = classify_complexity(
            "Give me the answer as JSON.", total_tokens=60, tool_count=0, config=config
        )
        assert result == COMPLEXITY_MODERATE

    def test_token_threshold_complex(self) -> None:
        config = RouterConfig(moderate_token_threshold=50, complex_token_threshold=100)
        result = classify_complexity(
            "Summarize this document as a markdown table.",
            total_tokens=5_000,
            tool_count=0,
            config=config,
        )
        # +2 (tokens >= complex threshold) +1 (structured) = 3 -> moderate
        assert result == COMPLEXITY_MODERATE

    def test_tool_count_thresholds(self) -> None:
        config = RouterConfig(moderate_tool_threshold=4, complex_tool_threshold=8)
        assert (
            classify_complexity("hi", total_tokens=1, tool_count=3, config=config)
            == COMPLEXITY_SIMPLE
        )
        # +1 (tools) alone is still simple; +1 structured makes it moderate
        assert (
            classify_complexity("hi as json", total_tokens=1, tool_count=4, config=config)
            == COMPLEXITY_MODERATE
        )

    def test_multi_part_question_signal(self) -> None:
        text = "What is X? And what is Y? Also explain why Z holds."
        # +1 (multi-part) +2 (reasoning keyword "explain why") = 3 -> moderate
        assert classify_complexity(text, total_tokens=20, tool_count=0) == COMPLEXITY_MODERATE

    def test_numbered_list_counts_as_multi_part(self) -> None:
        text = "Do the following:\n1. fetch data\n2. clean it\n3. plot it"
        config = RouterConfig(moderate_score=1)
        assert classify_complexity(text, total_tokens=10, tool_count=0, config=config) == (
            COMPLEXITY_MODERATE
        )

    def test_score_boundaries_configurable(self) -> None:
        # Lower the complex boundary so a single +2 signal is enough.
        config = RouterConfig(moderate_score=1, complex_score=2)
        text = "Walk me through this step by step."
        assert classify_complexity(text, total_tokens=5, tool_count=0, config=config) == (
            COMPLEXITY_COMPLEX
        )

    def test_keywords_configurable(self) -> None:
        config = RouterConfig(reasoning_keywords=("frobnicate",), moderate_score=1)
        assert (
            classify_complexity(
                "please frobnicate the data", total_tokens=5, tool_count=0, config=config
            )
            != COMPLEXITY_SIMPLE
        )

    def test_deterministic(self) -> None:
        results = {
            classify_complexity(COMPLEX_PROMPT, total_tokens=100, tool_count=2) for _ in range(10)
        }
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Construction + tier ordering
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            RouterProvider(providers={})

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            RouterProvider(providers={"a": _TierStub("a")}, strategy="yolo")

    def test_tier_order_must_be_permutation(self) -> None:
        with pytest.raises(ValueError, match="tier_order"):
            RouterProvider(
                providers={"a": _TierStub("a"), "b": _TierStub("b")},
                tier_order=["a", "nope"],
            )

    def test_insertion_order_when_models_unknown(self) -> None:
        _, _, _, router = _three_tiers()
        assert router.tier_order == ["fast", "smart", "power"]

    def test_pricing_reorders_known_models(self) -> None:
        # Deliberately misordered: pro ($30/1M) first, nano ($0.10/1M) last.
        router = RouterProvider(
            providers={
                "power": _TierStub("power", default_model="gpt-5.4-pro"),
                "smart": _TierStub("smart", default_model="gpt-5.4"),
                "fast": _TierStub("fast", default_model="gpt-5.4-nano"),
            },
        )
        assert router.tier_order == ["fast", "smart", "power"]

    def test_tier_order_override_wins(self) -> None:
        router = RouterProvider(
            providers={
                "power": _TierStub("power", default_model="gpt-5.4-pro"),
                "fast": _TierStub("fast", default_model="gpt-5.4-nano"),
            },
            tier_order=["power", "fast"],
        )
        assert router.tier_order == ["power", "fast"]

    def test_tier_models_param_overrides_default_model(self) -> None:
        router = RouterProvider(
            providers={"a": _TierStub("a"), "b": _TierStub("b")},
            tier_models={"a": "gpt-5.4-pro", "b": "gpt-5.4-nano"},
        )
        assert router.tier_order == ["b", "a"]

    def test_capability_flags_aggregate(self) -> None:
        _, _, _, router = _three_tiers()
        assert router.supports_streaming is True
        assert router.supports_async is True


# ---------------------------------------------------------------------------
# Strategy tier selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    def test_cost_optimized_simple_goes_to_cheapest(self) -> None:
        fast, smart, power, router = _three_tiers(strategy="cost_optimized")
        msg, _ = router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert msg.content == "from:fast-stub"
        assert router.tier_used == "fast"
        assert router.complexity_used == COMPLEXITY_SIMPLE
        assert not smart.calls and not power.calls

    def test_cost_optimized_complex_goes_to_top(self) -> None:
        fast, smart, power, router = _three_tiers(strategy="cost_optimized")
        msg, _ = router.complete(model="m", system_prompt="", messages=_user(COMPLEX_PROMPT))
        assert msg.content == "from:power-stub"
        assert router.tier_used == "power"
        assert router.complexity_used == COMPLEXITY_COMPLEX
        assert not fast.calls and not smart.calls

    def test_cost_optimized_moderate_goes_to_middle(self) -> None:
        fast, smart, power, router = _three_tiers(strategy="cost_optimized")
        text = "What is X? And what is Y? Also explain why Z holds."
        router.complete(model="m", system_prompt="", messages=_user(text))
        assert router.tier_used == "smart"
        assert router.complexity_used == COMPLEXITY_MODERATE

    def test_quality_first_always_starts_at_top(self) -> None:
        fast, smart, power, router = _three_tiers(strategy="quality_first")
        msg, _ = router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert msg.content == "from:power-stub"
        assert router.tier_used == "power"

    def test_balanced_simple_goes_to_middle(self) -> None:
        fast, smart, power, router = _three_tiers(strategy="balanced")
        router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert router.tier_used == "smart"
        assert not fast.calls

    def test_balanced_complex_goes_to_top(self) -> None:
        fast, smart, power, router = _three_tiers(strategy="balanced")
        router.complete(model="m", system_prompt="", messages=_user(COMPLEX_PROMPT))
        assert router.tier_used == "power"

    def test_on_route_callback(self) -> None:
        routed: List[Tuple[str, str]] = []
        _, _, _, router = _three_tiers(on_route=lambda c, t: routed.append((c, t)))
        router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert routed == [(COMPLEXITY_SIMPLE, "fast")]


# ---------------------------------------------------------------------------
# Escalation on provider failure
# ---------------------------------------------------------------------------


class TestEscalation:
    def test_retriable_failure_escalates_up_tier(self) -> None:
        fast = _TierStub("fast", fail_with="rate limit 429")
        smart = _TierStub("smart")
        power = _TierStub("power")
        router = RouterProvider(providers={"fast": fast, "smart": smart, "power": power})
        msg, _ = router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert msg.content == "from:smart"
        assert router.tier_used == "smart"
        assert len(fast.calls) == 1
        assert not power.calls

    def test_all_tiers_fail_raises_provider_error(self) -> None:
        router = RouterProvider(
            providers={
                "fast": _TierStub("fast", fail_with="500 server error"),
                "power": _TierStub("power", fail_with="503 unavailable"),
            }
        )
        with pytest.raises(ProviderError, match="exhausted"):
            router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))

    def test_non_retriable_error_propagates(self) -> None:
        fast = _TierStub("fast", fail_with="Authentication failed: invalid API key")
        power = _TierStub("power")
        router = RouterProvider(providers={"fast": fast, "power": power})
        with pytest.raises(ProviderError, match="Authentication"):
            router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert not power.calls

    def test_quality_first_degrades_downward(self) -> None:
        fast = _TierStub("fast")
        power = _TierStub("power", fail_with="overloaded 529")
        router = RouterProvider(providers={"fast": fast, "power": power}, strategy="quality_first")
        msg, _ = router.complete(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert msg.content == "from:fast"
        assert router.tier_used == "fast"


# ---------------------------------------------------------------------------
# Model resolution per tier
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_tier_model_overrides_incoming_model(self) -> None:
        fast = _TierStub("fast", default_model="gpt-5.4-nano")
        router = RouterProvider(providers={"fast": fast})
        router.complete(model="agent-model", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert fast.calls[0]["model"] == "gpt-5.4-nano"

    def test_incoming_model_used_when_tier_has_no_model(self) -> None:
        fast = _TierStub("fast")
        router = RouterProvider(providers={"fast": fast})
        router.complete(model="agent-model", system_prompt="", messages=_user(SIMPLE_PROMPT))
        assert fast.calls[0]["model"] == "agent-model"

    def test_tools_are_forwarded(self) -> None:
        fast = _TierStub("fast")
        router = RouterProvider(providers={"fast": fast})
        sentinel = ["tool-sentinel"]
        router.complete(
            model="m",
            system_prompt="",
            messages=_user(SIMPLE_PROMPT),
            tools=sentinel,  # type: ignore[arg-type]
        )
        assert fast.calls[0]["tools"] is sentinel


# ---------------------------------------------------------------------------
# Async + streaming delegation
# ---------------------------------------------------------------------------


class TestAsyncAndStreaming:
    @pytest.mark.asyncio
    async def test_acomplete_routes_and_delegates(self) -> None:
        fast, smart, power, router = _three_tiers()
        msg, _ = await router.acomplete(model="m", system_prompt="", messages=_user(COMPLEX_PROMPT))
        assert msg.content == "from:power-stub"
        assert router.tier_used == "power"

    def test_stream_routes_and_delegates(self) -> None:
        fast, smart, power, router = _three_tiers()
        chunks = list(router.stream(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT)))
        assert chunks == ["chunk:fast-stub"]
        assert router.tier_used == "fast"

    def test_stream_escalates_before_first_chunk(self) -> None:
        fast = _TierStub("fast", fail_with="timeout")
        power = _TierStub("power")
        router = RouterProvider(providers={"fast": fast, "power": power})
        chunks = list(router.stream(model="m", system_prompt="", messages=_user(SIMPLE_PROMPT)))
        assert chunks == ["chunk:power"]
        assert router.tier_used == "power"

    @pytest.mark.asyncio
    async def test_astream_routes_and_delegates(self) -> None:
        fast, smart, power, router = _three_tiers()
        chunks = [
            c
            async for c in router.astream(
                model="m", system_prompt="", messages=_user(SIMPLE_PROMPT)
            )
        ]
        assert chunks == ["chunk:fast-stub"]
        assert router.tier_used == "fast"

    def test_tools_forwarded_in_stream(self) -> None:
        fast = _TierStub("fast")
        router = RouterProvider(providers={"fast": fast})
        sentinel = ["tool-sentinel"]
        list(
            router.stream(
                model="m",
                system_prompt="",
                messages=_user(SIMPLE_PROMPT),
                tools=sentinel,  # type: ignore[arg-type]
            )
        )
        assert fast.calls[0]["tools"] is sentinel


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_exported_from_providers_package() -> None:
    from selectools.providers import RouterConfig as RC
    from selectools.providers import RouterProvider as RP

    assert RP is RouterProvider
    assert RC is RouterConfig
