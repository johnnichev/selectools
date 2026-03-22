"""Tests for SimpleStepObserver (R4)."""

from selectools.observer import SimpleStepObserver
from selectools.types import AgentResult, Message, Role
from selectools.usage import UsageStats


class TestSimpleStepObserver:
    """Verify that SimpleStepObserver routes all events to a single callback."""

    def _make_observer(self):
        events = []

        def callback(event_name, run_id, **kwargs):
            events.append({"event": event_name, "run_id": run_id, **kwargs})

        return SimpleStepObserver(callback), events

    def test_run_start(self):
        obs, events = self._make_observer()
        obs.on_run_start("r1", [Message(role=Role.USER, content="hi")], "system")
        assert len(events) == 1
        assert events[0]["event"] == "run_start"
        assert events[0]["run_id"] == "r1"
        assert events[0]["message_count"] == 1

    def test_run_end(self):
        obs, events = self._make_observer()
        result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="done"),
            iterations=1,
            tool_calls=[],
        )
        obs.on_run_end("r1", result)
        assert events[0]["event"] == "run_end"
        assert events[0]["result"] is result

    def test_llm_start(self):
        obs, events = self._make_observer()
        obs.on_llm_start("r1", [], "gpt-4o", "system")
        assert events[0]["event"] == "llm_start"
        assert events[0]["model"] == "gpt-4o"

    def test_llm_end(self):
        obs, events = self._make_observer()
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            model="gpt-4o",
            provider="openai",
        )
        obs.on_llm_end("r1", "response text", usage)
        assert events[0]["event"] == "llm_end"
        assert events[0]["usage"] is usage

    def test_tool_start(self):
        obs, events = self._make_observer()
        obs.on_tool_start("r1", "c1", "search", {"query": "test"})
        assert events[0]["event"] == "tool_start"
        assert events[0]["tool_name"] == "search"
        assert events[0]["tool_args"] == {"query": "test"}

    def test_tool_end(self):
        obs, events = self._make_observer()
        obs.on_tool_end("r1", "c1", "search", "result", 150.0)
        assert events[0]["event"] == "tool_end"
        assert events[0]["tool_name"] == "search"
        assert events[0]["duration_ms"] == 150.0

    def test_tool_error(self):
        obs, events = self._make_observer()
        err = ValueError("boom")
        obs.on_tool_error("r1", "c1", "search", err, {"q": "x"}, 50.0)
        assert events[0]["event"] == "tool_error"
        assert events[0]["error"] is err

    def test_usage(self):
        obs, events = self._make_observer()
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            model="gpt-4o",
            provider="openai",
        )
        obs.on_usage("r1", usage)
        assert events[0]["event"] == "usage"
        assert events[0]["usage"] is usage

    def test_iteration_start(self):
        obs, events = self._make_observer()
        obs.on_iteration_start("r1", 1, [])
        assert events[0]["event"] == "iteration_start"
        assert events[0]["iteration"] == 1

    def test_iteration_end(self):
        obs, events = self._make_observer()
        obs.on_iteration_end("r1", 2, "response")
        assert events[0]["event"] == "iteration_end"
        assert events[0]["iteration"] == 2

    def test_policy_decision(self):
        obs, events = self._make_observer()
        obs.on_policy_decision("r1", "send_email", "review", "requires approval", {"to": "x"})
        assert events[0]["event"] == "policy_decision"
        assert events[0]["decision"] == "review"

    def test_memory_trim(self):
        obs, events = self._make_observer()
        obs.on_memory_trim("r1", 5, 15, "enforce_limits")
        assert events[0]["event"] == "memory_trim"
        assert events[0]["messages_removed"] == 5

    def test_session_load(self):
        obs, events = self._make_observer()
        obs.on_session_load("r1", "sess-1", 10)
        assert events[0]["event"] == "session_load"

    def test_session_save(self):
        obs, events = self._make_observer()
        obs.on_session_save("r1", "sess-1", 12)
        assert events[0]["event"] == "session_save"

    def test_error(self):
        obs, events = self._make_observer()
        obs.on_error("r1", RuntimeError("fail"), {"step": 3})
        assert events[0]["event"] == "error"

    def test_cache_hit(self):
        obs, events = self._make_observer()
        obs.on_cache_hit("r1", "gpt-4o", "cached response")
        assert events[0]["event"] == "cache_hit"

    def test_eval_start(self):
        obs, events = self._make_observer()
        obs.on_eval_start("suite1", 10, "gpt-4o")
        assert events[0]["event"] == "eval_start"
        assert events[0]["suite_name"] == "suite1"

    def test_multiple_events_accumulate(self):
        obs, events = self._make_observer()
        obs.on_run_start("r1", [], "sys")
        obs.on_iteration_start("r1", 1, [])
        obs.on_llm_start("r1", [], "gpt-4o", "sys")
        assert len(events) == 3
        assert [e["event"] for e in events] == ["run_start", "iteration_start", "llm_start"]

    def test_batch_start(self):
        obs, events = self._make_observer()
        obs.on_batch_start("b1", 5)
        assert events[0]["event"] == "batch_start"
        assert events[0]["prompts_count"] == 5

    def test_provider_fallback(self):
        obs, events = self._make_observer()
        obs.on_provider_fallback("r1", "openai", "anthropic", RuntimeError("timeout"))
        assert events[0]["event"] == "provider_fallback"

    def test_llm_retry(self):
        obs, events = self._make_observer()
        obs.on_llm_retry("r1", 1, 3, RuntimeError("rate limit"), 2.0)
        assert events[0]["event"] == "llm_retry"
        assert events[0]["attempt"] == 1

    def test_entity_extraction(self):
        obs, events = self._make_observer()
        obs.on_entity_extraction("r1", 3)
        assert events[0]["event"] == "entity_extraction"
        assert events[0]["entities_extracted"] == 3

    def test_kg_extraction(self):
        obs, events = self._make_observer()
        obs.on_kg_extraction("r1", 5)
        assert events[0]["event"] == "kg_extraction"

    def test_memory_summarize(self):
        obs, events = self._make_observer()
        obs.on_memory_summarize("r1", "Summary of conversation")
        assert events[0]["event"] == "memory_summarize"

    def test_structured_validate(self):
        obs, events = self._make_observer()
        obs.on_structured_validate("r1", True, 1)
        assert events[0]["event"] == "structured_validate"
        assert events[0]["success"] is True

    def test_tool_chunk(self):
        obs, events = self._make_observer()
        obs.on_tool_chunk("r1", "c1", "stream_tool", "chunk1")
        assert events[0]["event"] == "tool_chunk"

    def test_batch_end(self):
        obs, events = self._make_observer()
        obs.on_batch_end("b1", 5, 0, 1234.5)
        assert events[0]["event"] == "batch_end"
        assert events[0]["results_count"] == 5
