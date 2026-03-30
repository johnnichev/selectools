"""
Property-based tests using Hypothesis.

These tests generate large volumes of arbitrary inputs to find edge cases that
hand-written tests miss. Each test asserts a *property* that must hold for all
valid inputs — not just the typical examples used in unit tests.

Run with: pytest tests/test_property_based.py -x -q
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectools.rag.vector_store import Document

# ============================================================================
# Strategies
# ============================================================================

# Safe text: printable ASCII, avoids newline-only or null-byte edge cases
safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=0,
    max_size=500,
)

# Non-empty safe text (for things that must not be blank)
nonempty_text = safe_text.filter(lambda t: t.strip() != "")

# Metadata dict: string keys, scalar values
scalar_value = st.one_of(st.text(max_size=50), st.integers(), st.booleans(), st.none())
metadata_dict = st.dictionaries(
    keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20),
    values=scalar_value,
    max_size=10,
)


# ============================================================================
# BM25
# ============================================================================


class TestBM25Properties:
    """BM25 must not crash or violate invariants on arbitrary inputs."""

    @given(texts=st.lists(safe_text, min_size=0, max_size=30), query=safe_text)
    @settings(max_examples=100)
    def test_search_never_crashes(self, texts: List[str], query: str) -> None:
        """search() must not raise for any combination of docs and query text."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        if texts:
            bm25.index_documents([Document(text=t) for t in texts])
        results = bm25.search(query, top_k=5)
        assert isinstance(results, list)

    @given(texts=st.lists(nonempty_text, min_size=1, max_size=20), query=nonempty_text)
    @settings(max_examples=100)
    def test_results_never_exceed_top_k(self, texts: List[str], query: str) -> None:
        """Results must never exceed top_k regardless of corpus size."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text=t) for t in texts])
        top_k = min(3, len(texts))
        results = bm25.search(query, top_k=top_k)
        assert len(results) <= top_k

    @given(text=safe_text)
    @settings(max_examples=200)
    def test_tokenize_always_returns_list_of_strings(self, text: str) -> None:
        """tokenize() must return List[str] for any input."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        tokens = bm25.tokenize(text)
        assert isinstance(tokens, list)
        for tok in tokens:
            assert isinstance(tok, str)
            assert tok  # no empty tokens

    @given(
        initial=st.lists(nonempty_text, min_size=1, max_size=10),
        extra=st.lists(nonempty_text, min_size=1, max_size=10),
    )
    @settings(max_examples=80)
    def test_add_documents_grows_count_exactly(self, initial: List[str], extra: List[str]) -> None:
        """document_count must grow by exactly len(extra) after add_documents."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text=t) for t in initial])
        before = bm25.document_count
        bm25.add_documents([Document(text=t) for t in extra])
        assert bm25.document_count == before + len(extra)

    @given(top_k=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_top_k_le_zero_raises(self, top_k: int) -> None:
        """top_k <= 0 must raise ValueError."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text="hello")])
        with pytest.raises(ValueError, match="top_k"):
            bm25.search("hello", top_k=top_k)


# ============================================================================
# InMemoryCache
# ============================================================================


class TestInMemoryCacheProperties:
    """InMemoryCache must respect size limits and return correct values."""

    @given(
        keys=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=50),
        max_size=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_size_never_exceeds_max(self, keys: List[str], max_size: int) -> None:
        """Cache must never hold more entries than max_size."""
        from selectools.cache import InMemoryCache

        cache = InMemoryCache(max_size=max_size)
        for k in keys:
            cache.set(k, (f"val_{k}", {}))
        assert cache.size <= max_size

    @given(key=st.text(min_size=1, max_size=50), value=st.text(max_size=100))
    @settings(max_examples=100)
    def test_get_after_set_returns_same_value(self, key: str, value: str) -> None:
        """get(key) after set(key, v) must return v (no TTL)."""
        from selectools.cache import InMemoryCache

        cache = InMemoryCache(max_size=1000)
        val = (value, {})
        cache.set(key, val)
        result = cache.get(key)
        assert result == val

    @given(key=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_delete_makes_key_miss(self, key: str) -> None:
        """After delete(key), get(key) must return None."""
        from selectools.cache import InMemoryCache

        cache = InMemoryCache(max_size=100)
        cache.set(key, ("v", {}))
        cache.delete(key)
        assert cache.get(key) is None


# ============================================================================
# ConversationMemory
# ============================================================================


class TestConversationMemoryProperties:
    """ConversationMemory must always stay within its configured limits."""

    @given(
        n_messages=st.integers(min_value=0, max_value=200),
        max_messages=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=80)
    def test_never_exceeds_max_messages(self, n_messages: int, max_messages: int) -> None:
        """len(get_history()) must never exceed max_messages regardless of how many are added."""
        from selectools.memory import ConversationMemory
        from selectools.types import Message, Role

        memory = ConversationMemory(max_messages=max_messages)
        for i in range(n_messages):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            memory.add(Message(role=role, content=f"message {i}"))
        assert len(memory.get_history()) <= max_messages

    @given(n=st.integers(min_value=0, max_value=30))
    @settings(max_examples=50)
    def test_clear_empties_history(self, n: int) -> None:
        """clear() must result in empty history."""
        from selectools.memory import ConversationMemory
        from selectools.types import Message, Role

        memory = ConversationMemory(max_messages=100)
        for i in range(n):
            memory.add(Message(role=Role.USER, content=f"msg {i}"))
        memory.clear()
        assert len(memory.get_history()) == 0

    @given(n=st.integers(min_value=2, max_value=30))
    @settings(max_examples=50)
    def test_branch_is_independent(self, n: int) -> None:
        """branch() must produce an independent copy — modifying one does not affect the other."""
        from selectools.memory import ConversationMemory
        from selectools.types import Message, Role

        memory = ConversationMemory(max_messages=100)
        for i in range(n):
            memory.add(Message(role=Role.USER, content=f"original {i}"))

        branch = memory.branch()
        original_len = len(memory.get_history())
        branch_len = len(branch.get_history())

        assert original_len == branch_len

        # Add to branch — original must not change
        branch.add(Message(role=Role.USER, content="branch only"))
        assert len(memory.get_history()) == original_len


# ============================================================================
# Policy
# ============================================================================


class TestPolicyProperties:
    """Policy.check() must never crash on valid tool names."""

    @given(tool_name=nonempty_text)
    @settings(max_examples=100)
    def test_check_never_crashes_on_nonempty_name(self, tool_name: str) -> None:
        """check() must return a PolicyResult for any non-empty tool name."""
        from selectools.policy import ToolPolicy

        policy = ToolPolicy()
        result = policy.evaluate(tool_name, {})
        assert result is not None

    @given(tool_name=nonempty_text, args=metadata_dict)
    @settings(max_examples=80)
    def test_check_always_returns_valid_decision(
        self, tool_name: str, args: Dict[str, Any]
    ) -> None:
        """check() must always return a result with a valid decision field."""
        from selectools.policy import PolicyDecision, ToolPolicy

        policy = ToolPolicy()
        result = policy.evaluate(tool_name, args)
        assert result.decision in (
            PolicyDecision.ALLOW,
            PolicyDecision.REVIEW,
            PolicyDecision.DENY,
        )


# ============================================================================
# Document metadata filter
# ============================================================================


def _check_filter(doc: Document, filter_dict: Dict[str, Any]) -> bool:
    """Helper: applies the same logic as InMemoryVectorStore._matches_filter."""
    for key, value in filter_dict.items():
        if doc.metadata.get(key) != value:
            return False
    return True


class TestMetadataFilterProperties:
    """Metadata filter matching must be consistent and reflexive."""

    @given(metadata=metadata_dict)
    @settings(max_examples=100)
    def test_doc_always_matches_its_own_metadata(self, metadata: Dict[str, Any]) -> None:
        """A document must always match a filter built from its own full metadata."""
        doc = Document(text="test", metadata=metadata)
        assert _check_filter(doc, metadata) is True

    @given(metadata=metadata_dict, extra_key=st.text(min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_filter_with_extra_key_does_not_match(
        self, metadata: Dict[str, Any], extra_key: str
    ) -> None:
        """A filter that requires a key not present in metadata must not match."""
        if extra_key in metadata:
            return

        doc = Document(text="test", metadata=metadata)
        filter_dict = {extra_key: "some_value_that_wont_match"}
        assert _check_filter(doc, filter_dict) is False

    @given(metadata=metadata_dict)
    @settings(max_examples=100)
    def test_empty_filter_always_matches(self, metadata: Dict[str, Any]) -> None:
        """An empty filter must match every document."""
        doc = Document(text="test", metadata=metadata)
        assert _check_filter(doc, {}) is True
