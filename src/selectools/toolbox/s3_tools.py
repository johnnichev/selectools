"""
Amazon S3 tools -- list objects, read object contents, and upload objects.

Requires the optional ``boto3`` library, installed via
``pip install selectools[aws]`` (or ``pip install boto3``). The import is
lazy: the module loads fine without the dependency.

Authentication uses the standard AWS credential chain: the
``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
``AWS_DEFAULT_REGION`` environment variables, shared config files
(``~/.aws/credentials``), or an instance/task role. Credentials are
never echoed in tool output or error messages.
"""

from __future__ import annotations

from typing import Any, Optional

from ..stability import beta
from ..tools import tool

_MISSING_DEP_ERROR = "Error: 'boto3' library not installed. Run: pip install selectools[aws]"
_MAX_GET_BYTES = 65536


def _s3_error(exc: Any) -> str:
    """Format a botocore ClientError into a readable string without leaking credentials."""
    error: dict = {}
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {}) or {}
    code = error.get("Code", "")
    message = error.get("Message", "")
    if code or message:
        return f"Error: S3 API returned '{code}': {message}"
    return f"Error: S3 API call failed: {type(exc).__name__}"


@beta
@tool(description="List objects in an S3 bucket")
def s3_list_objects(bucket: str, prefix: str = "", max_keys: int = 100) -> str:
    """
    List objects in an S3 bucket, optionally filtered by key prefix.

    Credentials come from the standard AWS chain (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION`` env vars, shared
    config, or an IAM role).

    Args:
        bucket: Name of the S3 bucket.
        prefix: Only list keys starting with this prefix (default: all keys).
        max_keys: Maximum number of objects to return (default: 100, max: 1000).

    Returns:
        Formatted list of object keys with sizes, or a readable error string.
    """
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.exceptions import (  # type: ignore[import-untyped]
            BotoCoreError,
            ClientError,
        )
    except ImportError:
        return _MISSING_DEP_ERROR

    if not bucket or not bucket.strip():
        return "Error: No bucket provided."

    max_keys = max(1, min(max_keys, 1000))

    try:
        client = boto3.client("s3")
        kwargs: dict = {"Bucket": bucket.strip(), "MaxKeys": max_keys}
        if prefix:
            kwargs["Prefix"] = prefix
        response = client.list_objects_v2(**kwargs)
        contents = response.get("Contents", [])

        if not contents:
            where = f" with prefix '{prefix}'" if prefix else ""
            return f"No objects found in bucket {bucket}{where}."

        lines = [f"{len(contents)} object(s) in bucket {bucket}:", ""]
        for obj in contents:
            key = obj.get("Key", "?")
            size = obj.get("Size", 0)
            lines.append(f"{key} ({size} bytes)")
        if response.get("IsTruncated"):
            lines.append("")
            lines.append(f"(truncated at {max_keys} objects)")

        return "\n".join(lines)
    except ClientError as exc:
        return _s3_error(exc)
    except BotoCoreError as exc:
        return f"Error: Could not reach S3: {type(exc).__name__}"
    except Exception as exc:
        return f"Error listing S3 objects: {type(exc).__name__}"


@beta
@tool(description="Read the contents of an S3 object")
def s3_get_object(bucket: str, key: str, max_bytes: int = _MAX_GET_BYTES) -> str:
    """
    Download an S3 object and return its contents as text.

    Bytes are decoded as UTF-8 (undecodable bytes are replaced); output is
    truncated to ``max_bytes``. Credentials come from the standard AWS
    chain (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``AWS_DEFAULT_REGION`` env vars, shared config, or an IAM role).

    Args:
        bucket: Name of the S3 bucket.
        key: Key of the object to read.
        max_bytes: Maximum number of bytes to read (default: 65536).

    Returns:
        The object contents as text, or a readable error string.
    """
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.exceptions import (  # type: ignore[import-untyped]
            BotoCoreError,
            ClientError,
        )
    except ImportError:
        return _MISSING_DEP_ERROR

    if not bucket or not bucket.strip():
        return "Error: No bucket provided."
    if not key or not key.strip():
        return "Error: No key provided."

    max_bytes = max(1, max_bytes)

    try:
        client = boto3.client("s3")
        response = client.get_object(Bucket=bucket.strip(), Key=key.strip())
        body = response["Body"].read(max_bytes + 1)
        truncated = len(body) > max_bytes
        text = body[:max_bytes].decode("utf-8", errors="replace")

        suffix = f"\n\n(truncated at {max_bytes} bytes)" if truncated else ""
        return f"Contents of s3://{bucket}/{key} ({len(body[:max_bytes])} bytes):\n{text}{suffix}"
    except ClientError as exc:
        return _s3_error(exc)
    except BotoCoreError as exc:
        return f"Error: Could not reach S3: {type(exc).__name__}"
    except Exception as exc:
        return f"Error reading S3 object: {type(exc).__name__}"


@beta
@tool(description="Upload a text object to an S3 bucket")
def s3_put_object(bucket: str, key: str, content: str, content_type: Optional[str] = None) -> str:
    """
    Upload text content to an S3 bucket under the given key.

    Credentials come from the standard AWS chain (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION`` env vars, shared
    config, or an IAM role).

    Args:
        bucket: Name of the S3 bucket.
        key: Destination key for the object.
        content: Text content to upload (encoded as UTF-8).
        content_type: Optional MIME type (e.g. ``"application/json"``).

    Returns:
        Confirmation with the object location and size, or a readable
        error string.
    """
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.exceptions import (  # type: ignore[import-untyped]
            BotoCoreError,
            ClientError,
        )
    except ImportError:
        return _MISSING_DEP_ERROR

    if not bucket or not bucket.strip():
        return "Error: No bucket provided."
    if not key or not key.strip():
        return "Error: No key provided."
    if content is None:
        return "Error: No content provided."

    body = content.encode("utf-8")

    try:
        client = boto3.client("s3")
        kwargs: dict = {"Bucket": bucket.strip(), "Key": key.strip(), "Body": body}
        if content_type:
            kwargs["ContentType"] = content_type
        client.put_object(**kwargs)
        return f"Uploaded {len(body)} bytes to s3://{bucket}/{key}"
    except ClientError as exc:
        return _s3_error(exc)
    except BotoCoreError as exc:
        return f"Error: Could not reach S3: {type(exc).__name__}"
    except Exception as exc:
        return f"Error uploading S3 object: {type(exc).__name__}"


__stability__ = "beta"

__all__ = [
    "s3_list_objects",
    "s3_get_object",
    "s3_put_object",
]
