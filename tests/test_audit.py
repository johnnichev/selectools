"""Tests for the AuditLogger."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from selectools.audit import AuditLogger, PrivacyLevel
from selectools.types import AgentResult, Message, Role
from selectools.usage import UsageStats


@pytest.fixture
def tmp_audit_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestAuditLoggerBasics:
    def test_creates_log_dir(self, tmp_audit_dir: str) -> None:
        sub = os.path.join(tmp_audit_dir, "sub", "audit")
        AuditLogger(log_dir=sub)
        assert os.path.isdir(sub)

    def test_writes_jsonl_on_run_start(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir)
        logger.on_run_start("run-1", [Message(role=Role.USER, content="hi")], "sys")
        files = os.listdir(tmp_audit_dir)
        assert len(files) == 1
        with open(os.path.join(tmp_audit_dir, files[0])) as f:
            entry = json.loads(f.readline())
        assert entry["event"] == "run_start"
        assert entry["run_id"] == "run-1"
        assert "ts" in entry

    def test_writes_run_end(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir)
        result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="done"),
            iterations=2,
            tool_name="search",
        )
        logger.on_run_end("run-1", result)
        files = os.listdir(tmp_audit_dir)
        with open(os.path.join(tmp_audit_dir, files[0])) as f:
            entry = json.loads(f.readline())
        assert entry["event"] == "run_end"
        assert entry["iterations"] == 2
        assert entry["tool_name"] == "search"


class TestPrivacyLevels:
    def test_full_privacy_logs_args(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, privacy=PrivacyLevel.FULL)
        logger.on_tool_start("r1", "c1", "search", {"query": "secret"})
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["tool_args"]["query"] == "secret"

    def test_keys_only_redacts_values(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, privacy=PrivacyLevel.KEYS_ONLY)
        logger.on_tool_start("r1", "c1", "search", {"query": "secret"})
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["tool_args"]["query"] == "<redacted>"

    def test_hashed_privacy(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, privacy=PrivacyLevel.HASHED)
        logger.on_tool_start("r1", "c1", "search", {"query": "secret"})
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["tool_args"]["query"] != "secret"
        assert len(entry["tool_args"]["query"]) == 16  # sha256[:16]

    def test_none_privacy_omits_args(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, privacy=PrivacyLevel.NONE)
        logger.on_tool_start("r1", "c1", "search", {"query": "secret"})
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["tool_args"] == {}


class TestDailyRotation:
    def test_daily_rotation_filename(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, daily_rotation=True)
        logger.on_run_start("r1", [], "sys")
        files = os.listdir(tmp_audit_dir)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert any(today in f for f in files)

    def test_no_rotation_uses_single_file(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, daily_rotation=False)
        logger.on_run_start("r1", [], "sys")
        logger.on_run_start("r2", [], "sys")
        assert "audit.jsonl" in os.listdir(tmp_audit_dir)


class TestToolEvents:
    def test_tool_end_records_success(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir)
        logger.on_tool_end("r1", "c1", "search", "result text", 42.5)
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["success"] is True
        assert entry["duration_ms"] == 42.5

    def test_tool_error_records_failure(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir)
        logger.on_tool_error("r1", "c1", "search", ValueError("bad"), {"q": "x"}, 10.0)
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["success"] is False
        assert entry["error_type"] == "ValueError"

    def test_policy_decision_logged(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir)
        logger.on_policy_decision("r1", "delete_user", "deny", "blocked", {})
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["decision"] == "deny"


class TestPathTraversalPrevention:
    def test_dotdot_path_is_normalized(self, tmp_audit_dir: str) -> None:
        """Regression: log_dir with .. components must be resolved to a canonical path."""
        traversal = os.path.join(tmp_audit_dir, "sub", "..", "..", "evil")
        logger = AuditLogger(log_dir=traversal)
        # The resolved path must not contain '..'
        assert ".." not in logger._log_dir
        # The resolved path is the real absolute path
        assert os.path.isabs(logger._log_dir)

    def test_relative_path_resolved_to_absolute(self, tmp_audit_dir: str) -> None:
        """log_dir as a relative path must be stored as an absolute path."""
        logger = AuditLogger(log_dir="./audit_relative_test")
        assert os.path.isabs(logger._log_dir)
        # Clean up
        import shutil

        if os.path.exists(logger._log_dir):
            shutil.rmtree(logger._log_dir)


class TestLLMEvents:
    def test_llm_end_with_usage(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir)
        usage = UsageStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            model="gpt-4o",
        )
        logger.on_llm_end("r1", "response text", usage)
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["model"] == "gpt-4o"
        assert entry["cost_usd"] == 0.001
        assert "response_length" not in entry  # include_content=False by default

    def test_include_content_adds_response_length(self, tmp_audit_dir: str) -> None:
        logger = AuditLogger(log_dir=tmp_audit_dir, include_content=True)
        logger.on_llm_end("r1", "response text", None)
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["response_length"] == 13


class TestNoneGuards:
    def test_on_tool_end_none_result_include_content(self, tmp_audit_dir: str) -> None:
        """Regression: on_tool_end with result=None and include_content=True previously raised
        TypeError: object of type 'NoneType' has no len().

        Tool executors can occasionally produce None results; the audit logger must
        treat a None result length as 0 rather than crashing.
        """
        logger = AuditLogger(log_dir=tmp_audit_dir, include_content=True)
        # Must not raise TypeError
        logger.on_tool_end("r1", "c1", "search", None, 10.0)  # type: ignore[arg-type]
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["event"] == "tool_end"
        assert entry["result_length"] == 0

    def test_on_tool_end_empty_string_result(self, tmp_audit_dir: str) -> None:
        """on_tool_end with empty string result must log result_length=0."""
        logger = AuditLogger(log_dir=tmp_audit_dir, include_content=True)
        logger.on_tool_end("r1", "c1", "search", "", 10.0)
        with open(os.path.join(tmp_audit_dir, os.listdir(tmp_audit_dir)[0])) as f:
            entry = json.loads(f.readline())
        assert entry["result_length"] == 0
