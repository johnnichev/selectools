"""
Tests for the KnowledgeBackend protocol and KnowledgeMemory backend integration.

Uses an in-memory fake backend — no network, no optional dependencies.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Optional

import pytest

from selectools.knowledge import KnowledgeBackend, KnowledgeMemory

# ======================================================================
# Fake backend
# ======================================================================


class InMemoryKnowledgeBackend:
    """Minimal KnowledgeBackend implementation backed by a bytes attribute."""

    def __init__(self) -> None:
        self.data: Optional[bytes] = None
        self.load_calls = 0
        self.save_calls = 0

    def load_bytes(self) -> Optional[bytes]:
        self.load_calls += 1
        return self.data

    def save_bytes(self, data: bytes) -> None:
        self.save_calls += 1
        self.data = data


# ======================================================================
# Protocol
# ======================================================================


class TestKnowledgeBackendProtocol:
    def test_protocol_is_importable_and_marked_beta(self) -> None:
        assert getattr(KnowledgeBackend, "__stability__", None) == "beta"

    def test_fake_satisfies_protocol_structurally(self) -> None:
        backend = InMemoryKnowledgeBackend()
        assert callable(backend.load_bytes)
        assert callable(backend.save_bytes)


# ======================================================================
# backend=None — behavior unchanged
# ======================================================================


class TestNoBackend:
    def test_default_backend_is_none(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path / "mem"))
        assert km.backend is None

    def test_remember_and_context_work_without_backend(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path / "mem"))
        km.remember("the sky is blue", persistent=True, importance=0.9)
        assert "the sky is blue" in km.build_context()
        assert "the sky is blue" in km.get_persistent_facts()

    def test_flush_is_noop_without_backend(self, tmp_path) -> None:
        km = KnowledgeMemory(directory=str(tmp_path / "mem"))
        km.flush()  # must not raise


# ======================================================================
# Persist hooks
# ======================================================================


class TestPersistHooks:
    def test_remember_persists_to_backend(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        km = KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        km.remember("fact one", importance=0.8)
        assert backend.save_calls >= 1
        assert backend.data is not None

    def test_init_loads_from_backend(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        assert backend.load_calls == 1

    def test_empty_backend_starts_fresh(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        km = KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        assert km.build_context() == ""

    def test_flush_persists_directory_state(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        directory = tmp_path / "mem"
        km = KnowledgeMemory(directory=str(directory), backend=backend)
        (directory / "extra.txt").write_text("hand-written", encoding="utf-8")
        km.flush()
        archive = json.loads(backend.data.decode("utf-8"))
        assert "extra.txt" in archive["files"]

    def test_prune_old_logs_persists_when_files_removed(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        directory = tmp_path / "mem"
        km = KnowledgeMemory(directory=str(directory), backend=backend)
        (directory / "2001-01-01.log").write_text("ancient", encoding="utf-8")
        saves_before = backend.save_calls
        removed = km.prune_old_logs(keep_days=2)
        assert removed == 1
        assert backend.save_calls > saves_before
        archive = json.loads(backend.data.decode("utf-8"))
        assert "2001-01-01.log" not in archive["files"]


# ======================================================================
# Round trips across instances (the ephemeral-infra scenario)
# ======================================================================


class TestRoundTrip:
    def test_full_round_trip_across_directories(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        km1 = KnowledgeMemory(directory=str(tmp_path / "deploy1"), backend=backend)
        km1.remember("user prefers dark mode", persistent=True, importance=0.9)
        km1.remember("ephemeral note", importance=0.3)

        # Fresh directory simulates a new deploy where /tmp was wiped.
        km2 = KnowledgeMemory(directory=str(tmp_path / "deploy2"), backend=backend)
        context = km2.build_context()
        assert "user prefers dark mode" in context
        assert "ephemeral note" in context
        assert "user prefers dark mode" in km2.get_persistent_facts()

    def test_restore_recreates_daily_logs(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        km1 = KnowledgeMemory(directory=str(tmp_path / "deploy1"), backend=backend)
        km1.remember("logged today", importance=0.5)

        km2 = KnowledgeMemory(directory=str(tmp_path / "deploy2"), backend=backend)
        assert "logged today" in km2.get_recent_logs()

    def test_binary_file_round_trip(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        d1 = tmp_path / "deploy1"
        km1 = KnowledgeMemory(directory=str(d1), backend=backend)
        payload = b"\x00\xff\xfe binary \x01"
        (d1 / "blob.bin").write_bytes(payload)
        km1.flush()

        d2 = tmp_path / "deploy2"
        KnowledgeMemory(directory=str(d2), backend=backend)
        assert (d2 / "blob.bin").read_bytes() == payload

    def test_nested_files_round_trip(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        d1 = tmp_path / "deploy1"
        km1 = KnowledgeMemory(directory=str(d1), backend=backend)
        nested = d1 / "sub" / "dir"
        nested.mkdir(parents=True)
        (nested / "note.md").write_text("nested content", encoding="utf-8")
        km1.flush()

        d2 = tmp_path / "deploy2"
        KnowledgeMemory(directory=str(d2), backend=backend)
        assert (d2 / "sub" / "dir" / "note.md").read_text(encoding="utf-8") == "nested content"

    def test_tmp_files_excluded_from_archive(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        d1 = tmp_path / "deploy1"
        km1 = KnowledgeMemory(directory=str(d1), backend=backend)
        (d1 / "entries.jsonl.tmp").write_text("partial write", encoding="utf-8")
        km1.flush()
        archive = json.loads(backend.data.decode("utf-8"))
        assert "entries.jsonl.tmp" not in archive["files"]


# ======================================================================
# Safety and corruption
# ======================================================================


class TestSafety:
    def test_path_traversal_in_archive_raises(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        backend.data = json.dumps(
            {"version": 1, "files": {"../evil.txt": {"encoding": "utf-8", "content": "pwned"}}}
        ).encode("utf-8")
        with pytest.raises(ValueError):
            KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        assert not (tmp_path / "evil.txt").exists()

    def test_absolute_path_in_archive_raises(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        backend.data = json.dumps(
            {"version": 1, "files": {"/etc/evil": {"encoding": "utf-8", "content": "pwned"}}}
        ).encode("utf-8")
        with pytest.raises(ValueError):
            KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)

    def test_corrupted_blob_starts_fresh(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        backend.data = b"definitely not json"
        km = KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        assert km.build_context() == ""

    def test_unknown_encoding_in_archive_raises(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        backend.data = json.dumps(
            {"version": 1, "files": {"x.txt": {"encoding": "rot13", "content": "abc"}}}
        ).encode("utf-8")
        with pytest.raises(ValueError):
            KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)

    def test_base64_encoding_marker_is_honored(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        payload = b"\x89PNG fake"
        backend.data = json.dumps(
            {
                "version": 1,
                "files": {
                    "img.bin": {
                        "encoding": "base64",
                        "content": base64.b64encode(payload).decode("ascii"),
                    }
                },
            }
        ).encode("utf-8")
        d = tmp_path / "mem"
        KnowledgeMemory(directory=str(d), backend=backend)
        assert (d / "img.bin").read_bytes() == payload


# ======================================================================
# Serialization metadata
# ======================================================================


class TestSerialization:
    def test_to_dict_unchanged(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        km = KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        d = km.to_dict()
        assert d["directory"] == str(tmp_path / "mem")
        assert "backend" not in d

    def test_backend_property(self, tmp_path) -> None:
        backend = InMemoryKnowledgeBackend()
        km = KnowledgeMemory(directory=str(tmp_path / "mem"), backend=backend)
        assert km.backend is backend
