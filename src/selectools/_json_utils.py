"""Internal JSON helpers shared across modules.

Home for JSON-parsing utilities that multiple subsystems need (orchestration
supervisor, team-lead and plan-and-execute patterns). Internal — not part of
the public API.
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK_RE = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)


def safe_json_parse(text: str, default: Any = None) -> Any:
    """Try to extract and parse JSON from an LLM response.

    Strips markdown code fences, then attempts a direct parse; falls back to
    extracting the first JSON array or object embedded in surrounding prose.
    Returns *default* when nothing parses.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return default
