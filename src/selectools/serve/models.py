"""Request/response models for the serve API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InvokeResponse:
    """Response body for POST /invoke."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None
    iterations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dict."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HealthResponse:
    """Response body for GET /health."""

    status: str = "ok"
    version: str = ""
    model: str = ""
    provider: str = ""
    tools: List[str] = field(default_factory=list)
