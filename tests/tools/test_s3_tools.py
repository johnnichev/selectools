"""
Tests for S3 tools (s3_list_objects, s3_get_object, s3_put_object).

boto3/botocore are mocked via sys.modules -- no network, no AWS credentials.
"""

from __future__ import annotations

import io
import sys
import types
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.s3_tools import s3_get_object, s3_list_objects, s3_put_object


class _FakeBotoCoreError(Exception):
    pass


class _FakeClientError(Exception):
    def __init__(self, error_response: Dict[str, Any], operation_name: str) -> None:
        super().__init__(f"{operation_name} failed")
        self.response = error_response
        self.operation_name = operation_name


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> MagicMock:
    boto3 = types.ModuleType("boto3")
    boto3.client = MagicMock(return_value=client)  # type: ignore[attr-defined]
    botocore = types.ModuleType("botocore")
    exceptions = types.ModuleType("botocore.exceptions")
    exceptions.BotoCoreError = _FakeBotoCoreError  # type: ignore[attr-defined]
    exceptions.ClientError = _FakeClientError  # type: ignore[attr-defined]
    botocore.exceptions = exceptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)
    return boto3.client  # type: ignore[attr-defined]


def _access_denied() -> _FakeClientError:
    return _FakeClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject"
    )


class TestS3ListObjects:
    def test_tool_metadata(self) -> None:
        assert s3_list_objects.name == "s3_list_objects"
        assert "S3" in s3_list_objects.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        result = s3_list_objects.function("my-bucket")
        assert "Error" in result
        assert "selectools[aws]" in result

    def test_list_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "reports/2026-06.csv", "Size": 1024},
                {"Key": "reports/2026-05.csv", "Size": 2048},
            ],
            "IsTruncated": False,
        }
        _install_fake_boto3(monkeypatch, client)
        result = s3_list_objects.function("my-bucket", prefix="reports/")
        assert "2 object(s)" in result
        assert "reports/2026-06.csv (1024 bytes)" in result
        assert "reports/2026-05.csv (2048 bytes)" in result
        kwargs = client.list_objects_v2.call_args[1]
        assert kwargs["Bucket"] == "my-bucket"
        assert kwargs["Prefix"] == "reports/"

    def test_truncation_noted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [{"Key": "a.txt", "Size": 1}],
            "IsTruncated": True,
        }
        _install_fake_boto3(monkeypatch, client)
        result = s3_list_objects.function("my-bucket", max_keys=1)
        assert "truncated" in result

    def test_empty_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.list_objects_v2.return_value = {}
        _install_fake_boto3(monkeypatch, client)
        result = s3_list_objects.function("my-bucket")
        assert "No objects found" in result

    def test_max_keys_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.list_objects_v2.return_value = {}
        _install_fake_boto3(monkeypatch, client)
        s3_list_objects.function("my-bucket", max_keys=99999)
        assert client.list_objects_v2.call_args[1]["MaxKeys"] == 1000

    def test_client_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.list_objects_v2.side_effect = _access_denied()
        _install_fake_boto3(monkeypatch, client)
        result = s3_list_objects.function("my-bucket")
        assert "Error" in result
        assert "AccessDenied" in result

    def test_empty_bucket_name_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, MagicMock())
        result = s3_list_objects.function("  ")
        assert "Error" in result


class TestS3GetObject:
    def test_tool_metadata(self) -> None:
        assert s3_get_object.name == "s3_get_object"

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        result = s3_get_object.function("my-bucket", "key.txt")
        assert "Error" in result
        assert "selectools[aws]" in result

    def test_get_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.get_object.return_value = {"Body": io.BytesIO(b"hello from s3")}
        _install_fake_boto3(monkeypatch, client)
        result = s3_get_object.function("my-bucket", "greeting.txt")
        assert "hello from s3" in result
        assert "s3://my-bucket/greeting.txt" in result
        client.get_object.assert_called_once_with(Bucket="my-bucket", Key="greeting.txt")

    def test_truncated_at_max_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.get_object.return_value = {"Body": io.BytesIO(b"A" * 100)}
        _install_fake_boto3(monkeypatch, client)
        result = s3_get_object.function("my-bucket", "big.txt", max_bytes=10)
        assert "AAAAAAAAAA" in result
        assert "A" * 11 not in result
        assert "truncated" in result

    def test_undecodable_bytes_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.get_object.return_value = {"Body": io.BytesIO(b"\xff\xfe ok")}
        _install_fake_boto3(monkeypatch, client)
        result = s3_get_object.function("my-bucket", "binary.bin")
        assert "Error" not in result
        assert "ok" in result

    def test_missing_key_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.get_object.side_effect = _FakeClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )
        _install_fake_boto3(monkeypatch, client)
        result = s3_get_object.function("my-bucket", "missing.txt")
        assert "Error" in result
        assert "NoSuchKey" in result

    def test_empty_key_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, MagicMock())
        result = s3_get_object.function("my-bucket", "")
        assert "Error" in result


class TestS3PutObject:
    def test_tool_metadata(self) -> None:
        assert s3_put_object.name == "s3_put_object"

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        result = s3_put_object.function("my-bucket", "key.txt", "content")
        assert "Error" in result
        assert "selectools[aws]" in result

    def test_put_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        _install_fake_boto3(monkeypatch, client)
        result = s3_put_object.function("my-bucket", "notes.txt", "hello world")
        assert "Uploaded 11 bytes to s3://my-bucket/notes.txt" in result
        kwargs = client.put_object.call_args[1]
        assert kwargs["Bucket"] == "my-bucket"
        assert kwargs["Key"] == "notes.txt"
        assert kwargs["Body"] == b"hello world"
        assert "ContentType" not in kwargs

    def test_put_with_content_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        _install_fake_boto3(monkeypatch, client)
        result = s3_put_object.function(
            "my-bucket", "data.json", '{"a": 1}', content_type="application/json"
        )
        assert "Uploaded" in result
        assert client.put_object.call_args[1]["ContentType"] == "application/json"

    def test_client_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.put_object.side_effect = _access_denied()
        _install_fake_boto3(monkeypatch, client)
        result = s3_put_object.function("my-bucket", "key.txt", "content")
        assert "Error" in result
        assert "AccessDenied" in result

    def test_botocore_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.put_object.side_effect = _FakeBotoCoreError("endpoint unreachable")
        _install_fake_boto3(monkeypatch, client)
        result = s3_put_object.function("my-bucket", "key.txt", "content")
        assert "Error" in result
        assert "Could not reach S3" in result

    def test_empty_bucket_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, MagicMock())
        result = s3_put_object.function("", "key.txt", "content")
        assert "Error" in result
