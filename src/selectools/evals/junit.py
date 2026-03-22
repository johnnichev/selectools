"""JUnit XML output for CI integration."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path
from typing import Any, Union

from .types import CaseVerdict


def render_junit_xml(report: Any, filepath: Union[str, Path]) -> None:
    """Render an EvalReport as JUnit XML."""
    suite_el = ET.Element(
        "testsuite",
        name=report.metadata.suite_name,
        tests=str(report.metadata.total_cases),
        failures=str(report.fail_count),
        errors=str(report.error_count),
        skipped=str(report.skip_count),
        time=f"{report.metadata.duration_ms / 1000:.3f}",
    )

    for cr in report.case_results:
        case_name = cr.case.name or cr.case.input[:80]
        tc_el = ET.SubElement(
            suite_el,
            "testcase",
            name=case_name,
            classname=report.metadata.suite_name,
            time=f"{cr.latency_ms / 1000:.3f}",
        )

        if cr.verdict == CaseVerdict.FAIL:
            failure_msgs = "\n".join(f.message for f in cr.failures)
            ET.SubElement(tc_el, "failure", message=failure_msgs).text = failure_msgs
        elif cr.verdict == CaseVerdict.ERROR:
            ET.SubElement(tc_el, "error", message=cr.error or "Unknown error").text = cr.error or ""
        elif cr.verdict == CaseVerdict.SKIP:
            ET.SubElement(tc_el, "skipped")

    tree = ET.ElementTree(suite_el)
    ET.indent(tree, space="  ")
    tree.write(str(filepath), encoding="unicode", xml_declaration=True)
