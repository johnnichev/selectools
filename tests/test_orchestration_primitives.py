"""Tests for orchestration primitives: GraphState, nodes, transforms, merge."""

from __future__ import annotations

import copy

import pytest

from selectools.orchestration.node import (
    GraphNode,
    ParallelGroupNode,
    SubgraphNode,
    build_context_messages,
    default_input_transform,
    default_output_transform,
)
from selectools.orchestration.state import (
    STATE_KEY_LAST_OUTPUT,
    ContextMode,
    GraphEvent,
    GraphEventType,
    GraphState,
    InterruptRequest,
    MergePolicy,
    Scatter,
    _Goto,
    _Update,
    goto,
    merge_states,
    update,
)
from selectools.types import Message, Role

# ------------------------------------------------------------------
# GraphState
# ------------------------------------------------------------------


class TestGraphState:
    def test_construction_defaults(self):
        state = GraphState()
        assert state.messages == []
        assert state.data == {}
        assert state.current_node == ""
        assert state.history == []
        assert state.metadata == {}
        assert state.errors == []
        assert state._interrupt_responses == {}

    def test_from_prompt(self):
        state = GraphState.from_prompt("Hello world")
        assert len(state.messages) == 1
        msg = state.messages[0]
        assert msg.content == "Hello world"
        assert msg.role == Role.USER

    def test_to_dict_excludes_interrupt_responses(self):
        state = GraphState.from_prompt("test")
        state._interrupt_responses["key1"] = "value1"
        d = state.to_dict()
        assert "__interrupt_responses" not in d
        assert "_interrupt_responses" not in d

    def test_to_dict_round_trip(self):
        state = GraphState.from_prompt("test prompt")
        state.data["key"] = "value"
        state.metadata["req_id"] = "abc123"
        state.current_node = "node_a"
        d = state.to_dict()
        assert d["data"]["key"] == "value"
        assert d["metadata"]["req_id"] == "abc123"
        assert d["current_node"] == "node_a"

    def test_from_dict_round_trip(self):
        state = GraphState.from_prompt("round trip test")
        state.data["foo"] = "bar"
        state.metadata["x"] = 42
        d = state.to_dict()
        restored = GraphState.from_dict(d)
        assert restored.data["foo"] == "bar"
        assert restored.metadata["x"] == 42

    def test_interrupt_responses_not_in_to_dict(self):
        """_interrupt_responses are NOT in to_dict — checkpoint saves them separately."""
        state = GraphState.from_prompt("test")
        state._interrupt_responses["node_0"] = "yes"
        d = state.to_dict()
        assert "__interrupt__" not in d
        # Must be restored separately (by checkpoint layer)
        restored = GraphState.from_dict(d)
        assert restored._interrupt_responses == {}

    def test_deep_copy_independence(self):
        state = GraphState.from_prompt("original")
        state.data["list"] = [1, 2, 3]
        copied = copy.deepcopy(state)
        copied.data["list"].append(4)
        assert state.data["list"] == [1, 2, 3]  # original unchanged


# ------------------------------------------------------------------
# MergePolicy
# ------------------------------------------------------------------


class TestMergeStates:
    def _make_state(self, content: str, data: dict) -> GraphState:
        s = GraphState.from_prompt(content)
        s.data.update(data)
        return s

    def test_last_wins(self):
        s1 = self._make_state("a", {"x": 1, "y": 10})
        s2 = self._make_state("b", {"x": 2, "z": 20})
        merged = merge_states([s1, s2], MergePolicy.LAST_WINS)
        assert merged.data["x"] == 2  # last wins
        assert merged.data["y"] == 10
        assert merged.data["z"] == 20

    def test_first_wins(self):
        s1 = self._make_state("a", {"x": 1})
        s2 = self._make_state("b", {"x": 2})
        merged = merge_states([s1, s2], MergePolicy.FIRST_WINS)
        assert merged.data["x"] == 1  # first wins

    def test_append_policy(self):
        s1 = self._make_state("a", {"items": [1, 2]})
        s2 = self._make_state("b", {"items": [3, 4]})
        merged = merge_states([s1, s2], MergePolicy.APPEND)
        assert merged.data["items"] == [1, 2, 3, 4]

    def test_append_non_list_fallback_last_wins(self):
        s1 = self._make_state("a", {"x": "first"})
        s2 = self._make_state("b", {"x": "second"})
        merged = merge_states([s1, s2], MergePolicy.APPEND)
        assert merged.data["x"] == "second"  # non-list → LAST_WINS

    def test_messages_always_concatenated(self):
        s1 = self._make_state("msg1", {})
        s2 = self._make_state("msg2", {})
        merged = merge_states([s1, s2], MergePolicy.LAST_WINS)
        assert len(merged.messages) == 2

    def test_history_always_concatenated(self):
        s1 = GraphState()
        s1.history.append(("node_a", None))
        s2 = GraphState()
        s2.history.append(("node_b", None))
        merged = merge_states([s1, s2], MergePolicy.LAST_WINS)
        assert len(merged.history) == 2

    def test_errors_always_concatenated(self):
        s1 = GraphState()
        s1.errors.append({"node": "a", "error": "fail"})
        s2 = GraphState()
        s2.errors.append({"node": "b", "error": "also fail"})
        merged = merge_states([s1, s2], MergePolicy.LAST_WINS)
        assert len(merged.errors) == 2

    def test_single_state_returns_independent_copy(self):
        """Single-state merge returns a deep copy to prevent mutation leaks."""
        s = self._make_state("x", {"a": 1})
        merged = merge_states([s], MergePolicy.LAST_WINS)
        assert merged is not s  # Must be a copy, not same object
        assert merged.data == s.data
        merged.data["a"] = 999
        assert s.data["a"] == 1  # Original unaffected

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            merge_states([], MergePolicy.LAST_WINS)


# ------------------------------------------------------------------
# InterruptRequest / Scatter / GraphEvent
# ------------------------------------------------------------------


class TestPrimitives:
    def test_interrupt_request_construction(self):
        ir = InterruptRequest(prompt="Approve?", payload={"draft": "text"})
        assert ir.prompt == "Approve?"
        assert ir.payload == {"draft": "text"}
        assert ir.interrupt_key == ""

    def test_scatter_construction(self):
        sc = Scatter(node_name="worker_a", state_patch={"task": "do x"})
        assert sc.node_name == "worker_a"
        assert sc.state_patch == {"task": "do x"}

    def test_scatter_default_empty_patch(self):
        sc = Scatter(node_name="worker")
        assert sc.state_patch == {}

    def test_graph_event_construction(self):
        ev = GraphEvent(type=GraphEventType.NODE_START, node_name="planner")
        assert ev.type == GraphEventType.NODE_START
        assert ev.node_name == "planner"
        assert ev.chunk == ""

    def test_goto_and_update(self):
        g = goto("next_node")
        assert isinstance(g, _Goto)
        assert g.node_name == "next_node"

        u = update({"key": "val"})
        assert isinstance(u, _Update)
        assert u.patch == {"key": "val"}

    def test_context_mode_values(self):
        assert ContextMode.LAST_MESSAGE == "last_message"
        assert ContextMode.LAST_N == "last_n"
        assert ContextMode.FULL == "full"
        assert ContextMode.SUMMARY == "summary"
        assert ContextMode.CUSTOM == "custom"

    def test_merge_policy_values(self):
        assert MergePolicy.LAST_WINS == "last_wins"
        assert MergePolicy.FIRST_WINS == "first_wins"
        assert MergePolicy.APPEND == "append"


# ------------------------------------------------------------------
# GraphNode / ParallelGroupNode / SubgraphNode
# ------------------------------------------------------------------


class TestNodeTypes:
    def test_graph_node_defaults(self):
        def my_fn(state):
            return state

        node = GraphNode(name="test", agent=my_fn)
        assert node.name == "test"
        assert node.context_mode == ContextMode.LAST_MESSAGE
        assert node.context_n == 6
        assert node.max_visits == 0
        assert node.max_iterations == 1
        assert node.input_transform is None
        assert node.output_transform is None

    def test_parallel_group_node_default_policy(self):
        node = ParallelGroupNode(name="fan_out", child_node_names=["a", "b"])
        assert node.merge_policy == MergePolicy.LAST_WINS

    def test_parallel_group_node_custom_policy(self):
        node = ParallelGroupNode(
            name="fan_out",
            child_node_names=["a", "b"],
            merge_policy=MergePolicy.APPEND,
        )
        assert node.merge_policy == MergePolicy.APPEND

    def test_subgraph_node_defaults(self):
        class FakeGraph:
            pass

        node = SubgraphNode(name="sub", graph=FakeGraph())
        assert node.input_map == {}
        assert node.output_map == {}


# ------------------------------------------------------------------
# Default transforms
# ------------------------------------------------------------------


class TestDefaultTransforms:
    def test_default_input_transform_first_node(self):
        """First node: falls back to last user message in state.messages."""
        state = GraphState.from_prompt("hello from user")
        messages = default_input_transform(state)
        assert len(messages) == 1
        assert messages[0].content == "hello from user"

    def test_default_input_transform_uses_last_output(self):
        """Subsequent nodes: reads STATE_KEY_LAST_OUTPUT."""
        state = GraphState.from_prompt("original")
        state.data[STATE_KEY_LAST_OUTPUT] = "result from previous node"
        messages = default_input_transform(state)
        assert len(messages) == 1
        assert messages[0].content == "result from previous node"
        assert messages[0].role == Role.USER

    def test_default_output_transform_writes_last_output(self):
        """Writes result.content to state.data[STATE_KEY_LAST_OUTPUT]."""
        from selectools.types import AgentResult

        state = GraphState.from_prompt("test")
        state.current_node = "my_node"
        result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="agent response"),
            iterations=1,
        )
        new_state = default_output_transform(result, state)
        assert new_state.data[STATE_KEY_LAST_OUTPUT] == "agent response"

    def test_default_output_transform_appends_history(self):
        from selectools.types import AgentResult

        state = GraphState.from_prompt("test")
        state.current_node = "node_a"
        result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="output"),
            iterations=1,
        )
        new_state = default_output_transform(result, state)
        assert len(new_state.history) == 1
        assert new_state.history[0][0] == "node_a"

    def test_default_output_transform_appends_message(self):
        from selectools.types import AgentResult

        state = GraphState.from_prompt("original")
        state.current_node = "node_a"
        result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="reply"),
            iterations=1,
        )
        new_state = default_output_transform(result, state)
        # messages: original user msg + new assistant msg
        assert len(new_state.messages) == 2
        assert new_state.messages[-1].role == Role.ASSISTANT


# ------------------------------------------------------------------
# build_context_messages
# ------------------------------------------------------------------


class TestBuildContextMessages:
    def _make_messages(self, *pairs):
        """Create a list of messages from (role, content) pairs."""
        msgs = []
        for role_str, content in pairs:
            role = Role.USER if role_str == "user" else Role.ASSISTANT
            msgs.append(Message(role=role, content=content))
        return msgs

    def test_last_message_mode(self):
        node = GraphNode(name="n", agent=lambda s: s, context_mode=ContextMode.LAST_MESSAGE)
        state = GraphState()
        state.messages = self._make_messages(
            ("user", "first"), ("assistant", "middle"), ("user", "last user msg")
        )
        result = build_context_messages(node, state)
        assert len(result) == 1
        assert result[0].content == "last user msg"

    def test_last_n_mode(self):
        node = GraphNode(name="n", agent=lambda s: s, context_mode=ContextMode.LAST_N, context_n=2)
        state = GraphState()
        state.messages = self._make_messages(
            ("user", "a"), ("user", "b"), ("user", "c"), ("user", "d")
        )
        result = build_context_messages(node, state)
        assert len(result) == 2
        assert result[-1].content == "d"

    def test_full_mode(self):
        node = GraphNode(name="n", agent=lambda s: s, context_mode=ContextMode.FULL)
        state = GraphState()
        state.messages = self._make_messages(("user", "a"), ("assistant", "b"), ("user", "c"))
        result = build_context_messages(node, state)
        assert len(result) == 3

    def test_empty_state_fallback(self):
        node = GraphNode(name="n", agent=lambda s: s, context_mode=ContextMode.LAST_MESSAGE)
        state = GraphState()
        result = build_context_messages(node, state)
        assert result == []
