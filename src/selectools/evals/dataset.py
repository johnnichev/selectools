"""Load test cases from JSON, YAML, or dict lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .types import TestCase

# Fields that map directly to TestCase constructor
_TESTCASE_FIELDS = {f.name for f in TestCase.__dataclass_fields__.values()}


class DatasetLoader:
    """Load TestCase lists from files or dicts."""

    @staticmethod
    def from_json(filepath: Union[str, Path]) -> List[TestCase]:
        """Load test cases from a JSON file."""
        data = json.loads(Path(filepath).read_text())
        if isinstance(data, dict) and "cases" in data:
            data = data["cases"]
        return DatasetLoader.from_dicts(data)

    @staticmethod
    def from_yaml(filepath: Union[str, Path]) -> List[TestCase]:
        """Load test cases from a YAML file. Requires PyYAML."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("PyYAML is required for YAML datasets: pip install pyyaml")
        data = yaml.safe_load(Path(filepath).read_text())
        if isinstance(data, dict) and "cases" in data:
            data = data["cases"]
        return DatasetLoader.from_dicts(data)

    @staticmethod
    def from_dicts(data: List[Dict[str, Any]]) -> List[TestCase]:
        """Convert a list of dicts to TestCase objects.

        Unknown keys are stored in TestCase.metadata.
        """
        cases: List[TestCase] = []
        for item in data:
            known = {k: v for k, v in item.items() if k in _TESTCASE_FIELDS}
            unknown = {k: v for k, v in item.items() if k not in _TESTCASE_FIELDS}
            if unknown:
                meta = dict(known.get("metadata", {}))
                meta.update(unknown)
                known["metadata"] = meta
            cases.append(TestCase(**known))
        return cases

    @staticmethod
    def load(filepath: Union[str, Path]) -> List[TestCase]:
        """Auto-detect format from file extension (.json, .yaml, .yml)."""
        path = Path(filepath)
        if path.suffix == ".json":
            return DatasetLoader.from_json(path)
        if path.suffix in (".yaml", ".yml"):
            return DatasetLoader.from_yaml(path)
        raise ValueError(f"Unsupported file format: {path.suffix}")
