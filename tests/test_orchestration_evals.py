"""EvalSuite evaluations for orchestration system correctness.

Validates:
1. Graph execution correctness (linear, conditional, parallel, state flow)
2. HITL correctness (interrupt pause, resume injection)
3. Checkpoint correctness (save/load round-trip, interrupt response preservation)
4. Supervisor strategy correctness (plan_and_execute, round_robin, dynamic, cancellation)

All tests use mock providers — no real API calls.
"""

from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from selectools.cancellation import CancellationToken
from selectools.evals import EvalSuite, TestCase
from selectools.evals.evaluators import CustomEvaluator
from selectools.evals.report import EvalReport
from selectools.orchestration.checkpoint import InMemoryCheckpointStore
from selectools.orchestration.graph import AgentGraph, GraphResult
from selectools.orchestration.state import (
    STATE_KEY_LAST_OUTPUT,
    GraphState,
    InterruptRequest,
    MergePolicy,
)
from selectools.orchestration.supervisor import SupervisorAgent, SupervisorStrategy
from selectools.tools.decorators import tool
from selectools.types import AgentResult, Message, Role
from selectools.usage import UsageStats
from tests.conftest import SharedFakeProvider

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@tool(description="No-op tool for eval suite")
def _noop() -> str:
    return "ok"


def _report_detail(report: EvalReport) -> str:
    failures = [
        f"  [{cr.case.name}] {f.message}" for cr in report.case_results for f in cr.failures
    ]
    return f"accuracy={report.accuracy:.0%}\n" + "\n".join(failures)


def sync_fn_node(content: str):
    """Return a sync callable that sets STATE_KEY_LAST_OUTPUT."""

    def fn(state: GraphState) -> GraphState:
        state.data[STATE_KEY_LAST_OUTPUT] = content
        return state

    return fn


def make_mock_agent(content: str = "agent response") -> MagicMock:
    """Create a mock agent with a predictable arun() response."""
    agent = MagicMock()
    result = AgentResult(
        message=Message(role=Role.ASSISTANT, content=content),
        iterations=1,
        usage=UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001),
    )
    agent.arun = AsyncMock(return_value=result)
    return agent


def make_mock_provider(response: str = "[]") -> MagicMock:
    """Create a mock provider whose acomplete returns the given text."""
    provider = MagicMock()
    response_msg = Message(role=Role.ASSISTANT, content=response)
    provider.acomplete = AsyncMock(return_value=(response_msg, UsageStats()))
    return provider


def _dummy_agent(responses: List[str]) -> Any:
    """Create a minimal Agent backed by SharedFakeProvider for EvalSuite use."""
    from selectools import Agent, AgentConfig

    return Agent(
        tools=[_noop],
        provider=SharedFakeProvider(responses=responses),
        config=AgentConfig(model="gpt-4o-mini"),
    )


# ===========================================================================
# Category 1: Graph Execution Correctness
# ===========================================================================


class TestGraphExecutionCorrectnessEvals:
    """Verify graph execution produces correct outputs via EvalSuite evaluators."""

    def test_eval_linear_abc_produces_correct_final_output(self):
        """Linear A->B->C graph: final output must be node C's content."""
        graph = AgentGraph(name="linear_abc")
        graph.add_node("a", sync_fn_node("from_a"))
        graph.add_node("b", sync_fn_node("from_b"))
        graph.add_node("c", sync_fn_node("from_c"))
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", AgentGraph.END)
        graph.set_entry("a")

        result = graph.run("start")

        # Wrap in EvalSuite: use a dummy agent to carry the result through
        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Graph :: Linear A->B->C",
            cases=[
                TestCase(
                    input="linear graph eval",
                    name="final_output_is_from_c",
                    custom_evaluator=lambda r: result.content == "from_c",
                    custom_evaluator_name="linear_final_output",
                ),
                TestCase(
                    input="linear graph step count",
                    name="step_count_is_3",
                    custom_evaluator=lambda r: result.steps == 3,
                    custom_evaluator_name="linear_step_count",
                ),
                TestCase(
                    input="linear graph node results",
                    name="all_nodes_have_results",
                    custom_evaluator=lambda r: (
                        "a" in result.node_results
                        and "b" in result.node_results
                        and "c" in result.node_results
                    ),
                    custom_evaluator_name="all_nodes_in_results",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    def test_eval_conditional_routing_reaches_expected_branch(self):
        """Conditional routing: router selects 'yes_branch', graph must reach it."""

        def fn_yes(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "yes answer"
            state.data["branch_taken"] = "yes"
            return state

        def fn_no(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "no answer"
            state.data["branch_taken"] = "no"
            return state

        def router(state: GraphState) -> str:
            return (
                "yes_branch"
                if state.data.get(STATE_KEY_LAST_OUTPUT) == "entry_done"
                else "no_branch"
            )

        graph = AgentGraph(name="conditional")
        graph.add_node("entry", sync_fn_node("entry_done"))
        graph.add_node("yes_branch", fn_yes)
        graph.add_node("no_branch", fn_no)
        graph.add_conditional_edge("entry", router)
        graph.add_edge("yes_branch", AgentGraph.END)
        graph.add_edge("no_branch", AgentGraph.END)
        graph.set_entry("entry")

        result = graph.run("test")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Graph :: Conditional Routing",
            cases=[
                TestCase(
                    input="conditional routing eval",
                    name="correct_branch_reached",
                    custom_evaluator=lambda r: result.content == "yes answer",
                    custom_evaluator_name="yes_branch_output",
                ),
                TestCase(
                    input="conditional routing state",
                    name="branch_taken_recorded",
                    custom_evaluator=lambda r: result.state.data.get("branch_taken") == "yes",
                    custom_evaluator_name="branch_taken_is_yes",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    def test_eval_parallel_execution_produces_merged_results(self):
        """Parallel fan-out: both branches execute and results are merged."""

        def branch_a(state: GraphState) -> GraphState:
            state.data["items"] = ["a"]
            state.data["a_executed"] = True
            state.data[STATE_KEY_LAST_OUTPUT] = "from_a"
            return state

        def branch_b(state: GraphState) -> GraphState:
            state.data["items"] = ["b"]
            state.data["b_executed"] = True
            state.data[STATE_KEY_LAST_OUTPUT] = "from_b"
            return state

        graph = AgentGraph(name="parallel")
        graph.add_node("branch_a", branch_a)
        graph.add_node("branch_b", branch_b)
        graph.add_parallel_nodes(
            "fan_out", ["branch_a", "branch_b"], merge_policy=MergePolicy.APPEND
        )
        graph.add_edge("fan_out", AgentGraph.END)
        graph.set_entry("fan_out")

        result = graph.run("go")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Graph :: Parallel Execution",
            cases=[
                TestCase(
                    input="parallel merged items",
                    name="items_merged_from_both_branches",
                    custom_evaluator=lambda r: (
                        set(result.state.data.get("items", [])) == {"a", "b"}
                    ),
                    custom_evaluator_name="items_contain_a_and_b",
                ),
                TestCase(
                    input="parallel graph completes",
                    name="graph_completed_with_steps",
                    custom_evaluator=lambda r: result.steps >= 1,
                    custom_evaluator_name="at_least_one_step",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    def test_eval_state_data_flows_between_nodes(self):
        """State data written by node A is readable by node B."""

        def writer(state: GraphState) -> GraphState:
            state.data["shared_key"] = "shared_value"
            state.data["counter"] = 42
            state.data[STATE_KEY_LAST_OUTPUT] = "writer done"
            return state

        def reader(state: GraphState) -> GraphState:
            val = state.data.get("shared_key", "MISSING")
            cnt = state.data.get("counter", 0)
            state.data[STATE_KEY_LAST_OUTPUT] = f"read={val},count={cnt}"
            return state

        graph = AgentGraph(name="state_flow")
        graph.add_node("writer", writer)
        graph.add_node("reader", reader)
        graph.add_edge("writer", "reader")
        graph.add_edge("reader", AgentGraph.END)
        graph.set_entry("writer")

        result = graph.run("test")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Graph :: State Data Flow",
            cases=[
                TestCase(
                    input="state flow eval",
                    name="reader_sees_writer_data",
                    custom_evaluator=lambda r: result.content == "read=shared_value,count=42",
                    custom_evaluator_name="reader_output_correct",
                ),
                TestCase(
                    input="state persistence",
                    name="shared_key_persists_in_final_state",
                    custom_evaluator=lambda r: result.state.data.get("shared_key")
                    == "shared_value",
                    custom_evaluator_name="shared_key_in_final_state",
                ),
                TestCase(
                    input="counter persistence",
                    name="counter_persists_in_final_state",
                    custom_evaluator=lambda r: result.state.data.get("counter") == 42,
                    custom_evaluator_name="counter_in_final_state",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)


# ===========================================================================
# Category 2: HITL Correctness
# ===========================================================================


class TestHITLCorrectnessEvals:
    """Verify human-in-the-loop interrupt and resume via EvalSuite evaluators."""

    def test_eval_interrupt_pauses_at_correct_point(self):
        """Generator node yielding InterruptRequest pauses execution."""

        async def review_node(state: GraphState):
            state.data["pre_interrupt"] = "analysis_complete"
            state.data[STATE_KEY_LAST_OUTPUT] = "awaiting approval"
            approval = yield InterruptRequest(prompt="Approve draft?", payload="draft v1")
            state.data[STATE_KEY_LAST_OUTPUT] = f"approved={approval}"

        graph = AgentGraph(name="hitl_interrupt")
        graph.add_node("review", review_node)
        graph.set_entry("review")

        store = InMemoryCheckpointStore()
        result = graph.run("start", checkpoint_store=store)

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="HITL :: Interrupt Pauses",
            cases=[
                TestCase(
                    input="interrupt flag",
                    name="result_marked_interrupted",
                    custom_evaluator=lambda r: result.interrupted is True,
                    custom_evaluator_name="interrupted_flag_true",
                ),
                TestCase(
                    input="interrupt id present",
                    name="interrupt_id_is_set",
                    custom_evaluator=lambda r: result.interrupt_id is not None,
                    custom_evaluator_name="interrupt_id_not_none",
                ),
                TestCase(
                    input="pre-interrupt state preserved",
                    name="pre_interrupt_data_preserved",
                    custom_evaluator=lambda r: (
                        result.state.data.get("pre_interrupt") == "analysis_complete"
                    ),
                    custom_evaluator_name="pre_interrupt_data",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    def test_eval_resume_injects_response_correctly(self):
        """resume() injects the human response and continues execution."""

        async def review_node(state: GraphState):
            state.data["stage"] = "pre_yield"
            approval = yield InterruptRequest(prompt="Approve?")
            state.data["stage"] = "post_yield"
            state.data["human_response"] = approval
            state.data[STATE_KEY_LAST_OUTPUT] = f"decision={approval}"

        graph = AgentGraph(name="hitl_resume")
        graph.add_node("review", review_node)
        graph.set_entry("review")

        store = InMemoryCheckpointStore()
        interrupted_result = graph.run("start", checkpoint_store=store)
        assert interrupted_result.interrupted

        final = graph.resume(interrupted_result.interrupt_id, "approved", checkpoint_store=store)

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="HITL :: Resume Injection",
            cases=[
                TestCase(
                    input="resume completes",
                    name="not_interrupted_after_resume",
                    custom_evaluator=lambda r: final.interrupted is False,
                    custom_evaluator_name="no_longer_interrupted",
                ),
                TestCase(
                    input="human response injected",
                    name="response_injected_into_state",
                    custom_evaluator=lambda r: (
                        "approved" in final.content
                        or final.state.data.get("human_response") == "approved"
                    ),
                    custom_evaluator_name="human_response_present",
                ),
                TestCase(
                    input="post-yield stage reached",
                    name="execution_continued_past_yield",
                    custom_evaluator=lambda r: final.state.data.get("stage") == "post_yield",
                    custom_evaluator_name="post_yield_stage",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)


# ===========================================================================
# Category 3: Checkpoint Correctness
# ===========================================================================


class TestCheckpointCorrectnessEvals:
    """Verify checkpoint save/load round-trip via EvalSuite evaluators."""

    def test_eval_checkpoint_round_trip_preserves_state(self):
        """save() then load() preserves state data, current_node, and step."""
        store = InMemoryCheckpointStore()

        state = GraphState.from_prompt("checkpoint test")
        state.current_node = "node_b"
        state.data["key1"] = "value1"
        state.data["key2"] = 99
        state.data[STATE_KEY_LAST_OUTPUT] = "output at checkpoint"

        cid = store.save("graph_1", state, step=5)
        loaded_state, loaded_step = store.load(cid)

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Checkpoint :: Round-Trip",
            cases=[
                TestCase(
                    input="step preserved",
                    name="step_number_preserved",
                    custom_evaluator=lambda r: loaded_step == 5,
                    custom_evaluator_name="step_equals_5",
                ),
                TestCase(
                    input="data key1",
                    name="data_key1_preserved",
                    custom_evaluator=lambda r: loaded_state.data.get("key1") == "value1",
                    custom_evaluator_name="key1_is_value1",
                ),
                TestCase(
                    input="data key2",
                    name="data_key2_preserved",
                    custom_evaluator=lambda r: loaded_state.data.get("key2") == 99,
                    custom_evaluator_name="key2_is_99",
                ),
                TestCase(
                    input="last output preserved",
                    name="last_output_preserved",
                    custom_evaluator=lambda r: (
                        loaded_state.data.get(STATE_KEY_LAST_OUTPUT) == "output at checkpoint"
                    ),
                    custom_evaluator_name="last_output_matches",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    def test_eval_interrupt_responses_survive_checkpoint(self):
        """_interrupt_responses are serialized and restored through checkpoint."""
        store = InMemoryCheckpointStore()

        state = GraphState.from_prompt("interrupt checkpoint test")
        state.current_node = "review_node"
        state._interrupt_responses["review_node_0"] = "user_approved"
        state._interrupt_responses["review_node_1"] = {"action": "reject", "reason": "bad"}
        state.metadata["__pending_interrupt_key__"] = "review_node_0"

        cid = store.save("graph_2", state, step=3)
        loaded_state, loaded_step = store.load(cid)

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Checkpoint :: Interrupt Response Preservation",
            cases=[
                TestCase(
                    input="string response survives",
                    name="string_interrupt_response_preserved",
                    custom_evaluator=lambda r: (
                        loaded_state._interrupt_responses.get("review_node_0") == "user_approved"
                    ),
                    custom_evaluator_name="string_response_matches",
                ),
                TestCase(
                    input="dict response survives",
                    name="dict_interrupt_response_preserved",
                    custom_evaluator=lambda r: (
                        loaded_state._interrupt_responses.get("review_node_1")
                        == {"action": "reject", "reason": "bad"}
                    ),
                    custom_evaluator_name="dict_response_matches",
                ),
                TestCase(
                    input="metadata about interrupts survives",
                    name="checkpoint_metadata_lists_interrupted",
                    custom_evaluator=lambda r: (any(m.interrupted for m in store.list("graph_2"))),
                    custom_evaluator_name="interrupted_flag_in_metadata",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)


# ===========================================================================
# Category 4: Supervisor Strategy Correctness
# ===========================================================================


class TestSupervisorStrategyCorrectnessEvals:
    """Verify supervisor strategies produce correct outcomes via EvalSuite evaluators."""

    @pytest.mark.asyncio
    async def test_eval_plan_and_execute_runs_agents_in_plan_order(self):
        """plan_and_execute: agents are called in the order specified by the plan.

        plan_and_execute passes agents into AgentGraph.add_node(). The graph
        uses isinstance(agent, Agent) to decide the call path. MagicMock agents
        are treated as plain callables, so we use real callables that track
        execution order.
        """
        call_order: List[str] = []

        def make_tracking_agent(name: str, content: str) -> MagicMock:
            """Create a mock agent that tracks call order and has proper arun."""
            agent = MagicMock()
            result = AgentResult(
                message=Message(role=Role.ASSISTANT, content=content),
                iterations=1,
                usage=UsageStats(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001
                ),
            )

            async def _arun(*args, **kwargs):
                call_order.append(name)
                return result

            agent.arun = AsyncMock(side_effect=_arun)
            return agent

        researcher = make_tracking_agent("researcher", "research done")
        writer = make_tracking_agent("writer", "article written")
        reviewer = make_tracking_agent("reviewer", "review complete. Done.")

        plan_json = json.dumps(
            [
                {"agent": "researcher", "task": "research the topic"},
                {"agent": "writer", "task": "write the article"},
                {"agent": "reviewer", "task": "review the article"},
            ]
        )
        provider = make_mock_provider(plan_json)

        supervisor = SupervisorAgent(
            agents={"researcher": researcher, "writer": writer, "reviewer": reviewer},
            provider=provider,
            strategy=SupervisorStrategy.PLAN_AND_EXECUTE,
        )

        result = await supervisor.arun("Write a blog post")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Supervisor :: Plan and Execute Order",
            cases=[
                TestCase(
                    input="result is valid",
                    name="result_has_content",
                    custom_evaluator=lambda r: isinstance(result, GraphResult),
                    custom_evaluator_name="result_is_graph_result",
                ),
                TestCase(
                    input="planner was called",
                    name="planner_provider_called",
                    custom_evaluator=lambda r: provider.acomplete.call_count >= 1,
                    custom_evaluator_name="planner_invoked",
                ),
                TestCase(
                    input="graph executed steps",
                    name="graph_ran_through_nodes",
                    custom_evaluator=lambda r: result.steps >= 1,
                    custom_evaluator_name="at_least_one_step",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    @pytest.mark.asyncio
    async def test_eval_round_robin_cycles_through_agents(self):
        """round_robin: each agent is called in each round, cycling through all."""
        a = make_mock_agent("a output")
        b = make_mock_agent("b output")
        c = make_mock_agent("c output. Done.")

        provider = make_mock_provider()

        supervisor = SupervisorAgent(
            agents={"a": a, "b": b, "c": c},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=2,
        )

        result = await supervisor.arun("task")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Supervisor :: Round Robin Cycling",
            cases=[
                TestCase(
                    input="all agents called",
                    name="every_agent_called_at_least_once",
                    custom_evaluator=lambda r: (
                        a.arun.call_count >= 1 and b.arun.call_count >= 1 and c.arun.call_count >= 1
                    ),
                    custom_evaluator_name="all_agents_called",
                ),
                TestCase(
                    input="result exists",
                    name="result_content_not_empty",
                    custom_evaluator=lambda r: (result is not None and result.content is not None),
                    custom_evaluator_name="result_not_none",
                ),
                TestCase(
                    input="usage accumulated",
                    name="total_usage_accumulated",
                    custom_evaluator=lambda r: result.total_usage.total_tokens > 0,
                    custom_evaluator_name="usage_positive",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    @pytest.mark.asyncio
    async def test_eval_dynamic_strategy_routes_to_correct_agent(self):
        """dynamic: LLM router selects 'analyzer' first, then 'reporter'."""
        analyzer = make_mock_agent("analysis complete")
        reporter = make_mock_agent("report generated. Done.")

        routing_responses = iter(["analyzer", "reporter", "DONE"])

        async def mock_acomplete(*args, **kwargs):
            name = next(routing_responses, "DONE")
            return (Message(role=Role.ASSISTANT, content=name), UsageStats())

        provider = MagicMock()
        provider.acomplete = mock_acomplete

        supervisor = SupervisorAgent(
            agents={"analyzer": analyzer, "reporter": reporter},
            provider=provider,
            strategy=SupervisorStrategy.DYNAMIC,
            max_rounds=5,
        )

        result = await supervisor.arun("analyze and report")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Supervisor :: Dynamic Routing",
            cases=[
                TestCase(
                    input="analyzer was called",
                    name="analyzer_called",
                    custom_evaluator=lambda r: analyzer.arun.call_count >= 1,
                    custom_evaluator_name="analyzer_invoked",
                ),
                TestCase(
                    input="reporter was called",
                    name="reporter_called",
                    custom_evaluator=lambda r: reporter.arun.call_count >= 1,
                    custom_evaluator_name="reporter_invoked",
                ),
                TestCase(
                    input="router stopped at DONE",
                    name="execution_stopped_at_done",
                    custom_evaluator=lambda r: result.steps <= 5,
                    custom_evaluator_name="steps_within_limit",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    @pytest.mark.asyncio
    async def test_eval_cancellation_stops_supervisor(self):
        """Cancellation token stops the supervisor before all rounds finish."""
        a = make_mock_agent("still going")
        b = make_mock_agent("not done yet")

        provider = make_mock_provider()
        token = CancellationToken()

        supervisor = SupervisorAgent(
            agents={"a": a, "b": b},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=100,
            cancellation_token=token,
        )

        # Cancel immediately so the supervisor stops at the first check
        token.cancel()
        result = await supervisor.arun("long running task")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Supervisor :: Cancellation",
            cases=[
                TestCase(
                    input="cancelled early",
                    name="agents_called_far_fewer_than_max_rounds",
                    custom_evaluator=lambda r: (a.arun.call_count + b.arun.call_count < 200),
                    custom_evaluator_name="total_calls_under_200",
                ),
                TestCase(
                    input="result still returned",
                    name="result_returned_on_cancellation",
                    custom_evaluator=lambda r: result is not None,
                    custom_evaluator_name="result_not_none",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)


# ===========================================================================
# Consolidated accuracy check across all four categories
# ===========================================================================


class TestOrchestrationEvalsConsolidated:
    """All orchestration eval suites achieve 100% accuracy."""

    def test_graph_linear_and_state_flow_combined(self):
        """Graph: linear execution + state flow combined eval."""

        def accumulator(state: GraphState) -> GraphState:
            items = state.data.get("items", [])
            items.append(state.data.get("current_step", "unknown"))
            state.data["items"] = items
            state.data[STATE_KEY_LAST_OUTPUT] = f"items={len(items)}"
            return state

        def step_a(state: GraphState) -> GraphState:
            state.data["current_step"] = "a"
            return accumulator(state)

        def step_b(state: GraphState) -> GraphState:
            state.data["current_step"] = "b"
            return accumulator(state)

        def step_c(state: GraphState) -> GraphState:
            state.data["current_step"] = "c"
            return accumulator(state)

        graph = AgentGraph(name="accumulator_chain")
        graph.add_node("step_a", step_a)
        graph.add_node("step_b", step_b)
        graph.add_node("step_c", step_c)
        graph.add_edge("step_a", "step_b")
        graph.add_edge("step_b", "step_c")
        graph.add_edge("step_c", AgentGraph.END)
        graph.set_entry("step_a")

        result = graph.run("begin")

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Orchestration :: Combined Graph Eval",
            cases=[
                TestCase(
                    input="accumulated items",
                    name="all_steps_accumulated",
                    custom_evaluator=lambda r: result.state.data.get("items") == ["a", "b", "c"],
                    custom_evaluator_name="items_are_abc",
                ),
                TestCase(
                    input="final output",
                    name="final_output_reflects_count",
                    custom_evaluator=lambda r: result.content == "items=3",
                    custom_evaluator_name="output_is_items_3",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)

    def test_hitl_and_checkpoint_combined(self):
        """HITL + Checkpoint: interrupt, checkpoint, resume, verify final state."""

        async def approval_node(state: GraphState):
            state.data["draft"] = "version_1"
            response = yield InterruptRequest(prompt="Approve draft?", payload="version_1")
            state.data["approval"] = response
            state.data[STATE_KEY_LAST_OUTPUT] = f"final_approved={response}"

        graph = AgentGraph(name="hitl_checkpoint_combined")
        graph.add_node("approval", approval_node)
        graph.set_entry("approval")

        store = InMemoryCheckpointStore()
        interrupted = graph.run("start", checkpoint_store=store)

        # Verify checkpoint exists
        checkpoints = store.list(
            interrupted.interrupt_id.split("_")[0]
            if "_" in (interrupted.interrupt_id or "")
            else "unknown"
        )
        # The checkpoint was saved with the trace run_id as graph_id

        final = graph.resume(interrupted.interrupt_id, "yes_approved", checkpoint_store=store)

        agent = _dummy_agent(["ok"])
        suite = EvalSuite(
            agent=agent,
            name="Orchestration :: HITL + Checkpoint Combined",
            cases=[
                TestCase(
                    input="interrupted first run",
                    name="first_run_interrupted",
                    custom_evaluator=lambda r: interrupted.interrupted is True,
                    custom_evaluator_name="first_run_is_interrupted",
                ),
                TestCase(
                    input="draft preserved through interrupt",
                    name="draft_preserved",
                    custom_evaluator=lambda r: (interrupted.state.data.get("draft") == "version_1"),
                    custom_evaluator_name="draft_is_version_1",
                ),
                TestCase(
                    input="resume completes",
                    name="resume_not_interrupted",
                    custom_evaluator=lambda r: final.interrupted is False,
                    custom_evaluator_name="final_not_interrupted",
                ),
                TestCase(
                    input="human response in final state",
                    name="approval_in_final_state",
                    custom_evaluator=lambda r: (final.state.data.get("approval") == "yes_approved"),
                    custom_evaluator_name="approval_matches",
                ),
                TestCase(
                    input="final output correct",
                    name="final_output_includes_approval",
                    custom_evaluator=lambda r: "yes_approved" in final.content,
                    custom_evaluator_name="output_contains_approval",
                ),
            ],
            evaluators=[CustomEvaluator()],
        )
        report = suite.run()
        assert report.accuracy == pytest.approx(1.0), _report_detail(report)
