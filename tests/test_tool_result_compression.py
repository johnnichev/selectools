"""Tests for tool result compression (ROADMAP P2: ToolConfig.compress_results)."""

import asyncio
from typing import Any, List, Optional, Tuple

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.config_groups import ToolConfig
from selectools.agent.core import Agent
from selectools.tools.base import Tool
from selectools.trace import StepType
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

BIG_RESULT = "row-1 id=42 url=https://example.com/x " + "data " * 800  # > 2000 chars
SMALL_RESULT = "small result"


def _make_tool(name: str = "fetch_data", result: str = BIG_RESULT, **kwargs: Any) -> Tool:
    return Tool(
        name=name,
        description=f"A tool named {name}",
        parameters=[],
        function=lambda: result,
        **kwargs,
    )


def _tool_call_msg(*tool_names: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(tool_name=name, parameters={}, id=f"tc-{i}")
            for i, name in enumerate(tool_names)
        ],
    )


class CompressorProvider:
    """Mock provider used as a dedicated compress_provider."""

    name = "compressor"
    supports_streaming = False
    supports_async = True

    def __init__(
        self,
        summary: str = "concise summary",
        fail: bool = False,
        finish_reason: Optional[str] = None,
        completion_tokens: int = 5,
    ) -> None:
        self.summary = summary
        self.fail = fail
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.calls: List[dict] = []

    def _respond(self, model: str, messages: List[Message]) -> Tuple[Message, UsageStats]:
        self.calls.append({"model": model, "messages": list(messages)})
        if self.fail:
            raise RuntimeError("compression provider down")
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=self.completion_tokens,
            total_tokens=10 + self.completion_tokens,
            cost_usd=0.0,
            model=model,
            provider=self.name,
        )
        msg = Message(role=Role.ASSISTANT, content=self.summary)
        if self.finish_reason is not None:
            msg.finish_reason = self.finish_reason  # type: ignore[attr-defined]
        return msg, usage

    def complete(self, *, model: str, messages: List[Message], **kwargs: Any) -> Any:
        return self._respond(model, messages)

    async def acomplete(self, *, model: str, messages: List[Message], **kwargs: Any) -> Any:
        return self._respond(model, messages)


def _tool_messages(agent: Agent) -> List[Message]:
    return [m for m in agent._history if m.role == Role.TOOL]


def _run_agent(
    fake_provider: Any,
    tool: Tool,
    tool_cfg: Optional[ToolConfig],
    *,
    responses: Optional[List[Any]] = None,
    **config_kwargs: Any,
) -> Tuple[Any, Agent]:
    provider = fake_provider(responses=responses or [_tool_call_msg(tool.name), "Done"])
    agent = Agent(
        tools=[tool],
        provider=provider,
        config=AgentConfig(max_iterations=3, tool=tool_cfg, **config_kwargs),
    )
    result = agent.run("fetch the data")
    return result, agent


class TestCompressionGating:
    def test_disabled_by_default_byte_identical(self, fake_provider):
        """With compression off (default), the TOOL message is the raw result."""
        tool = _make_tool()
        _, agent = _run_agent(fake_provider, tool, None)
        tool_msgs = _tool_messages(agent)
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == BIG_RESULT
        assert tool_msgs[0].tool_result == BIG_RESULT

    def test_result_at_threshold_not_compressed(self, fake_provider):
        """len(result) == threshold must NOT trigger compression (strict >)."""
        result = "x" * 2000
        tool = _make_tool(result=result)
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=2000,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        assert _tool_messages(agent)[0].content == result
        assert compressor.calls == []

    def test_result_over_threshold_compressed(self, fake_provider):
        """len(result) == threshold + 1 triggers compression."""
        result = "x" * 2001
        tool = _make_tool(result=result)
        compressor = CompressorProvider(summary="short summary")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=2000,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        content = _tool_messages(agent)[0].content
        assert content == "[compressed from 2001 chars] short summary"
        assert len(compressor.calls) == 1

    def test_marker_prefix_present(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider(summary="42 rows from https://example.com/x")
        cfg = ToolConfig(
            compress_results=True, compress_provider=compressor, compress_model="cheap-model"
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        content = _tool_messages(agent)[0].content
        assert content.startswith(f"[compressed from {len(BIG_RESULT)} chars] ")
        assert "42 rows from https://example.com/x" in content

    def test_trace_records_compression_step(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True, compress_provider=compressor, compress_model="cheap-model"
        )
        result, _ = _run_agent(fake_provider, tool, cfg)
        steps = [s for s in result.trace.steps if s.type == StepType.PROMPT_COMPRESSED]
        assert len(steps) == 1
        assert steps[0].tool_name == "fetch_data"


class TestCompressionFailure:
    def test_failure_falls_back_to_truncation_marker(self, fake_provider):
        """A failing compressor must never crash the loop — truncate instead."""
        tool = _make_tool()
        compressor = CompressorProvider(fail=True)
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        result, agent = _run_agent(fake_provider, tool, cfg)
        content = _tool_messages(agent)[0].content
        assert content.startswith(f"[truncated from {len(BIG_RESULT)} chars; compression failed] ")
        assert BIG_RESULT[:100] in content
        # Loop progressed to the final answer.
        assert result.content == "Done"

    def test_empty_summary_falls_back_to_truncation(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider(summary="   ")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        assert _tool_messages(agent)[0].content.startswith(
            f"[truncated from {len(BIG_RESULT)} chars; compression failed] "
        )

    def test_summary_longer_than_original_keeps_raw(self, fake_provider):
        """Compression must never make the result longer."""
        result = "y" * 150
        tool = _make_tool(result=result)
        compressor = CompressorProvider(summary="z" * 500)
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        assert _tool_messages(agent)[0].content == result


class TestCompressionExemptions:
    def test_terminal_tool_never_compressed(self, fake_provider):
        """Terminal tool results become the final answer and stay verbatim."""
        tool = _make_tool(name="final_answer", terminal=True)
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("final_answer"), "unused"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg),
        )
        result = agent.run("go")
        assert result.content == BIG_RESULT
        assert _tool_messages(agent)[0].content == BIG_RESULT
        assert compressor.calls == []

    def test_stop_condition_result_never_compressed(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("fetch_data"), "unused"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                tool=cfg,
                stop_condition=lambda name, result: "row-1" in result,
            ),
        )
        result = agent.run("go")
        assert result.content == BIG_RESULT
        assert compressor.calls == []

    def test_error_results_never_compressed(self, fake_provider):
        def boom() -> str:
            raise RuntimeError("kaboom " + "x" * 3000)

        tool = Tool(name="boom", description="boom", parameters=[], function=boom)
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("boom"), "Done"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg),
        )
        agent.run("go")
        assert compressor.calls == []
        assert _tool_messages(agent)[0].content.startswith("Error executing tool 'boom'")

    def test_raw_result_kept_in_all_tool_results(self, fake_provider):
        """Loop detection (ctx.all_tool_results) must see the RAW result."""
        from selectools.agent.core import _RunContext
        from selectools.trace import AgentTrace

        tool = _make_tool()
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=["unused"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg),
        )
        ctx = _RunContext(
            trace=AgentTrace(),
            run_id="r1",
            original_system_prompt="",
            history_checkpoint=0,
            response_format=None,
        )
        agent._execute_single_tool(ctx, ToolCall(tool_name="fetch_data", parameters={}))
        assert ctx.all_tool_results == [BIG_RESULT]
        assert agent._history[-1].content.startswith("[compressed from")


class TestCompressionProviderChoice:
    def test_dedicated_compress_provider_used(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, _ = _run_agent(fake_provider, tool, cfg)
        assert len(compressor.calls) == 1
        assert compressor.calls[0]["model"] == "cheap-model"
        # The oversized result is in the compression request.
        assert BIG_RESULT in compressor.calls[0]["messages"][-1].content

    def test_falls_back_to_agent_provider_with_model_override(self, recording_provider):
        """No compress_provider → agent's own provider with compress_model."""
        tool = _make_tool()
        cfg = ToolConfig(compress_results=True, compress_threshold=100, compress_model="mini-model")
        provider = recording_provider()
        provider._wrapped._responses = [_tool_call_msg("fetch_data"), "summary text", "Done"]
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, model="big-model"),
        )
        agent.run("go")
        models = [c["model"] for c in provider.complete_calls]
        assert "mini-model" in models
        compression_call = provider.complete_calls[models.index("mini-model")]
        assert "fetch_data" in compression_call["messages"][-1].content
        assert _tool_messages(agent)[0].content == (
            f"[compressed from {len(BIG_RESULT)} chars] summary text"
        )

    def test_falls_back_to_agent_model_without_override(self, recording_provider):
        tool = _make_tool()
        cfg = ToolConfig(compress_results=True, compress_threshold=100)
        provider = recording_provider()
        provider._wrapped._responses = [_tool_call_msg("fetch_data"), "summary text", "Done"]
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, model="big-model"),
        )
        agent.run("go")
        assert [c["model"] for c in provider.complete_calls] == ["big-model"] * 3


class TestCompressionAsyncAndParallel:
    def test_arun_compresses(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider(summary="async summary")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("fetch_data"), "Done"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg),
        )
        result = asyncio.run(agent.arun("go"))
        assert result.content == "Done"
        assert _tool_messages(agent)[0].content == (
            f"[compressed from {len(BIG_RESULT)} chars] async summary"
        )

    def test_parallel_sync_compresses_only_oversized(self, fake_provider):
        big = _make_tool(name="big_tool")
        small = _make_tool(name="small_tool", result=SMALL_RESULT)
        compressor = CompressorProvider(summary="parallel summary")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("big_tool", "small_tool"), "Done"])
        agent = Agent(
            tools=[big, small],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, parallel_tool_execution=True),
        )
        result = agent.run("go")
        assert result.content == "Done"
        contents = {m.tool_name: m.content for m in _tool_messages(agent)}
        assert contents["big_tool"] == (
            f"[compressed from {len(BIG_RESULT)} chars] parallel summary"
        )
        assert contents["small_tool"] == SMALL_RESULT
        assert len(compressor.calls) == 1

    def test_parallel_async_compresses(self, fake_provider):
        big = _make_tool(name="big_tool")
        small = _make_tool(name="small_tool", result=SMALL_RESULT)
        compressor = CompressorProvider(summary="parallel summary")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("big_tool", "small_tool"), "Done"])
        agent = Agent(
            tools=[big, small],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, parallel_tool_execution=True),
        )
        result = asyncio.run(agent.arun("go"))
        assert result.content == "Done"
        contents = {m.tool_name: m.content for m in _tool_messages(agent)}
        assert contents["big_tool"].startswith("[compressed from")
        assert contents["small_tool"] == SMALL_RESULT


class TestCompressionConfigPlumbing:
    def test_flat_config_keeps_defaults(self):
        cfg = AgentConfig()
        assert cfg.tool.compress_results is False
        assert cfg.tool.compress_threshold == 2000
        assert cfg.tool.compress_provider is None
        assert cfg.tool.compress_model is None

    def test_dict_config_unpacks(self):
        cfg = AgentConfig(tool={"compress_results": True, "compress_threshold": 500})
        assert cfg.tool.compress_results is True
        assert cfg.tool.compress_threshold == 500

    def test_usage_records_compression_call(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        result, _ = _run_agent(fake_provider, tool, cfg)
        assert result.usage is not None
        assert result.usage.total_completion_tokens >= 5  # includes compressor usage


class TestCompressionCacheInteraction:
    """Review S1: cache entries store (raw, compressed); hits reuse the summary."""

    @staticmethod
    def _cache_agent(fake_provider, tool, compressor, responses, **config_kwargs):
        from selectools.cache import InMemoryCache

        cache = InMemoryCache()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=responses)
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=5, tool=cfg, cache=cache, **config_kwargs),
        )
        return agent, cache

    def test_cache_hit_appends_stored_compressed(self, fake_provider):
        """A repeated call must reuse the stored summary, not re-flood raw."""
        tool = _make_tool(cacheable=True)
        compressor = CompressorProvider(summary="cached summary")
        agent, _ = self._cache_agent(
            fake_provider,
            tool,
            compressor,
            [_tool_call_msg("fetch_data"), _tool_call_msg("fetch_data"), "Done"],
        )
        result = agent.run("go")
        assert result.content == "Done"
        tool_msgs = _tool_messages(agent)
        expected = f"[compressed from {len(BIG_RESULT)} chars] cached summary"
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == expected  # miss: compressed + stored
        assert tool_msgs[1].content == expected  # hit: stored summary reused
        assert len(compressor.calls) == 1  # summarizer billed exactly once

    def test_cache_hit_stop_condition_sees_raw(self, fake_provider):
        """Terminal checks on a hit run against the RAW result, not the summary."""
        tool = _make_tool(cacheable=True)
        compressor = CompressorProvider()
        agent, _ = self._cache_agent(
            fake_provider,
            tool,
            compressor,
            [_tool_call_msg("fetch_data"), "unused"],
            stop_condition=lambda name, result: "row-1" in result,  # raw only
        )
        # Pre-populate the cache with a compressed entry whose summary does
        # NOT contain the stop marker.
        agent._store_tool_cache(
            tool, {}, BIG_RESULT, compressed="[compressed from 9 chars] no marker here"
        )
        result = agent.run("go")
        # stop_condition fired on the raw cached result and it became terminal.
        assert result.content == BIG_RESULT

    def test_legacy_tuple_cache_entry_tolerated(self, fake_provider):
        """Pre-compression (raw, None) entries behave exactly as before."""
        tool = _make_tool(cacheable=True)
        compressor = CompressorProvider()
        agent, cache = self._cache_agent(
            fake_provider, tool, compressor, [_tool_call_msg("fetch_data"), "Done"]
        )
        key = agent._build_tool_cache_key("fetch_data", {})
        cache.set(key, (BIG_RESULT, None), ttl=300)
        agent.run("go")
        assert _tool_messages(agent)[0].content == BIG_RESULT
        assert compressor.calls == []  # cached results are never re-compressed

    def test_legacy_plain_string_cache_entry_tolerated(self, fake_provider):
        """A plain-string cache entry (oldest format) is treated as raw."""
        tool = _make_tool(cacheable=True)
        compressor = CompressorProvider()
        agent, cache = self._cache_agent(
            fake_provider, tool, compressor, [_tool_call_msg("fetch_data"), "Done"]
        )
        key = agent._build_tool_cache_key("fetch_data", {})
        cache.set(key, BIG_RESULT, ttl=300)
        agent.run("go")
        assert _tool_messages(agent)[0].content == BIG_RESULT
        assert compressor.calls == []

    def test_truncation_fallback_not_cached_as_compressed(self, fake_provider):
        """A failed summarization must not freeze the truncation marker in cache."""
        tool = _make_tool(cacheable=True)
        compressor = CompressorProvider(fail=True)
        agent, _ = self._cache_agent(
            fake_provider, tool, compressor, [_tool_call_msg("fetch_data"), "Done"]
        )
        agent.run("go")
        cached = agent._check_tool_cache(tool, {})
        assert cached is not None
        assert cached[0] == BIG_RESULT
        assert cached[1] is None  # fallback text NOT stored as compressed


class TestCompressionFallbackWarning:
    """Review S3: truncation fallback logs a warning once per run."""

    def test_fallback_warns_once_per_run_with_exception_type(self, fake_provider, caplog):
        import logging as _logging

        big1 = _make_tool(name="big1")
        big2 = _make_tool(name="big2")
        compressor = CompressorProvider(fail=True)
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("big1", "big2"), "Done"])
        agent = Agent(
            tools=[big1, big2],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, parallel_tool_execution=True),
        )
        with caplog.at_level(_logging.WARNING, logger="selectools.agent._tool_executor"):
            agent.run("go")
        records = [r for r in caplog.records if "fell back to truncation" in r.getMessage()]
        assert len(records) == 1  # once per run, not per result
        assert "RuntimeError" in records[0].getMessage()
        assert "compression provider down" in records[0].getMessage()
        # Both results still degraded gracefully.
        for msg in _tool_messages(agent):
            assert msg.content.startswith("[truncated from")

    def test_compress_provider_without_model_raises(self):
        """compress_provider without an explicit compress_model is invalid."""
        with pytest.raises(ValueError, match="compress_model"):
            ToolConfig(compress_results=True, compress_provider=CompressorProvider())


class TestCompressionTokenAttribution:
    """Review S4: tool_tokens reflect the PARENT iteration, not compression calls."""

    PARENT = UsageStats(
        prompt_tokens=700,
        completion_tokens=77,
        total_tokens=777,
        cost_usd=0.0,
        model="fake",
        provider="fake",
    )

    def _two_big_agent(self, fake_provider, parallel: bool):
        big1 = _make_tool(name="big1")
        big2 = _make_tool(name="big2")
        compressor = CompressorProvider(summary="s")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[(_tool_call_msg("big1", "big2"), self.PARENT), "Done"])
        agent = Agent(
            tools=[big1, big2],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, parallel_tool_execution=parallel),
        )
        return agent, compressor

    def test_parallel_sync_both_tools_attributed_parent_tokens(self, fake_provider):
        agent, compressor = self._two_big_agent(fake_provider, parallel=True)
        result = agent.run("go")
        assert len(compressor.calls) == 2  # both oversized results compressed
        assert result.usage.tool_tokens["big1"] == 777
        assert result.usage.tool_tokens["big2"] == 777  # NOT the compressor's 15

    def test_parallel_async_both_tools_attributed_parent_tokens(self, fake_provider):
        agent, compressor = self._two_big_agent(fake_provider, parallel=True)
        result = asyncio.run(agent.arun("go"))
        assert len(compressor.calls) == 2
        assert result.usage.tool_tokens["big1"] == 777
        assert result.usage.tool_tokens["big2"] == 777

    def test_sequential_both_tools_attributed_parent_tokens(self, fake_provider):
        agent, compressor = self._two_big_agent(fake_provider, parallel=False)
        result = agent.run("go")
        assert len(compressor.calls) == 2
        assert result.usage.tool_tokens["big1"] == 777
        assert result.usage.tool_tokens["big2"] == 777


class TestStopConditionSemantics:
    """Review N1: stop_condition evaluated once per result, outside the execute try."""

    def test_stop_condition_called_once_per_result(self, fake_provider):
        calls: List[str] = []

        def stop(name: str, result: str) -> bool:
            calls.append(name)
            return False

        tool = _make_tool()
        compressor = CompressorProvider()
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        provider = fake_provider(responses=[_tool_call_msg("fetch_data"), "Done"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg, stop_condition=stop),
        )
        agent.run("go")
        assert calls == ["fetch_data"]  # once: compression gate + terminal share it

    def test_raising_stop_condition_propagates_sync(self, fake_provider):
        """A raising predicate must NOT be converted into a tool error."""

        def bad(name: str, result: str) -> bool:
            raise RuntimeError("predicate blew up")

        tool = _make_tool()
        provider = fake_provider(responses=[_tool_call_msg("fetch_data"), "Done"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, stop_condition=bad),
        )
        with pytest.raises(RuntimeError, match="predicate blew up"):
            agent.run("go")
        # No "Error executing tool" message was fabricated for the good result.
        assert not any(m.content.startswith("Error executing tool") for m in _tool_messages(agent))

    def test_raising_stop_condition_propagates_async(self, fake_provider):
        def bad(name: str, result: str) -> bool:
            raise RuntimeError("predicate blew up")

        tool = _make_tool()
        provider = fake_provider(responses=[_tool_call_msg("fetch_data"), "Done"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, stop_condition=bad),
        )
        with pytest.raises(RuntimeError, match="predicate blew up"):
            asyncio.run(agent.arun("go"))


class TestCompressionMaxTokenCap:
    """Review N2: summaries that hit the max-token cap are treated as failures."""

    def test_finish_reason_length_falls_back(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider(summary="cut off mid", finish_reason="length")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        assert _tool_messages(agent)[0].content.startswith(
            f"[truncated from {len(BIG_RESULT)} chars; compression failed] "
        )

    def test_completion_tokens_at_budget_falls_back(self, fake_provider):
        # threshold=100 → budget = max(128, min(1000, 100 // 4)) = 128
        tool = _make_tool()
        compressor = CompressorProvider(summary="suspiciously long", completion_tokens=128)
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        assert _tool_messages(agent)[0].content.startswith(
            f"[truncated from {len(BIG_RESULT)} chars; compression failed] "
        )

    def test_finish_reason_stop_is_fine(self, fake_provider):
        tool = _make_tool()
        compressor = CompressorProvider(summary="complete summary", finish_reason="stop")
        cfg = ToolConfig(
            compress_results=True,
            compress_threshold=100,
            compress_provider=compressor,
            compress_model="cheap-model",
        )
        _, agent = _run_agent(fake_provider, tool, cfg)
        assert _tool_messages(agent)[0].content == (
            f"[compressed from {len(BIG_RESULT)} chars] complete summary"
        )
