"""Snapshot testing — capture and diff agent outputs across runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SnapshotDiff:
    """A diff between the current output and the stored snapshot."""

    case_name: str
    field: str  # "content", "tool_calls", "iterations"
    expected: Any
    actual: Any

    @property
    def is_changed(self) -> bool:
        return self.expected != self.actual


@dataclass
class SnapshotResult:
    """Result of comparing current outputs against snapshots."""

    new_cases: List[str] = field(default_factory=list)
    removed_cases: List[str] = field(default_factory=list)
    diffs: List[SnapshotDiff] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_cases or self.removed_cases or self.diffs)

    @property
    def changed_count(self) -> int:
        changed_names = {d.case_name for d in self.diffs}
        return len(changed_names)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Snapshot Comparison:"]
        lines.append(f"  Unchanged: {len(self.unchanged)}")
        if self.new_cases:
            lines.append(f"  New cases: {len(self.new_cases)}")
            for name in self.new_cases:
                lines.append(f"    + {name}")
        if self.removed_cases:
            lines.append(f"  Removed cases: {len(self.removed_cases)}")
            for name in self.removed_cases:
                lines.append(f"    - {name}")
        if self.diffs:
            lines.append(f"  Changed cases: {self.changed_count}")
            for d in self.diffs:
                lines.append(f"    ~ {d.case_name}.{d.field}")
                lines.append(f"      expected: {str(d.expected)[:100]}")
                lines.append(f"      actual:   {str(d.actual)[:100]}")
        return "\n".join(lines)


class SnapshotStore:
    """Capture and compare exact agent outputs across runs.

    Similar to Jest snapshot testing. On first run, captures the agent's
    outputs as the "golden" snapshot. On subsequent runs, diffs against
    the snapshot and reports changes.

    Example::

        store = SnapshotStore("./snapshots")
        report = suite.run()

        result = store.compare(report, suite_name="my-agent")
        if result.has_changes:
            print(result.summary())
            # Review changes, then update:
            store.save(report, suite_name="my-agent")
    """

    def __init__(self, directory: Union[str, Path]) -> None:
        self._dir = Path(directory)

    def save(self, report: Any, suite_name: str = "default") -> Path:
        """Save current agent outputs as the snapshot."""
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(suite_name).name  # strip any directory components
        path = self._dir / f"{safe_name}.snapshot.json"

        snapshot: Dict[str, Any] = {}
        for idx, cr in enumerate(report.case_results):
            base_key = cr.case.name or cr.case.input[:60]
            key = f"{base_key}_{idx}" if not cr.case.name else base_key
            entry: Dict[str, Any] = {
                "input": cr.case.input,
                "verdict": cr.verdict.value,
                "tool_calls": cr.tool_calls,
            }
            if cr.agent_result:
                entry["content"] = cr.agent_result.content or ""
                entry["iterations"] = cr.agent_result.iterations
            else:
                entry["content"] = ""
                entry["iterations"] = 0
            if cr.error:
                entry["error"] = cr.error
            snapshot[key] = entry

        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        return path

    def load(self, suite_name: str = "default") -> Optional[Dict[str, Any]]:
        """Load a stored snapshot."""
        safe_name = Path(suite_name).name
        path = self._dir / f"{safe_name}.snapshot.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def compare(self, report: Any, suite_name: str = "default") -> SnapshotResult:
        """Compare current report against stored snapshot.

        Returns a SnapshotResult with new/removed/changed cases.
        """
        stored = self.load(Path(suite_name).name)
        if stored is None:
            # No snapshot exists — everything is new
            names = [
                cr.case.name or f"{cr.case.input[:60]}_{idx}"
                for idx, cr in enumerate(report.case_results)
            ]
            return SnapshotResult(new_cases=names)

        current_keys: Dict[str, Any] = {}
        for idx, cr in enumerate(report.case_results):
            base_key = cr.case.name or cr.case.input[:60]
            key = f"{base_key}_{idx}" if not cr.case.name else base_key
            entry: Dict[str, Any] = {
                "verdict": cr.verdict.value,
                "tool_calls": cr.tool_calls,
                "content": (cr.agent_result.content or "") if cr.agent_result else "",
                "iterations": cr.agent_result.iterations if cr.agent_result else 0,
            }
            current_keys[key] = entry

        new_cases = [k for k in current_keys if k not in stored]
        removed_cases = [k for k in stored if k not in current_keys]
        unchanged: List[str] = []
        diffs: List[SnapshotDiff] = []

        for key in current_keys:
            if key not in stored:
                continue
            old = stored[key]
            new = current_keys[key]
            case_changed = False

            for check_field in ["content", "tool_calls", "verdict", "iterations"]:
                old_val = old.get(check_field)
                new_val = new.get(check_field)
                if old_val != new_val:
                    diffs.append(
                        SnapshotDiff(
                            case_name=key,
                            field=check_field,
                            expected=old_val,
                            actual=new_val,
                        )
                    )
                    case_changed = True

            if not case_changed:
                unchanged.append(key)

        return SnapshotResult(
            new_cases=new_cases,
            removed_cases=removed_cases,
            diffs=diffs,
            unchanged=unchanged,
        )
