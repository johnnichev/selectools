"""Tests for selectools.evals.__main__ — CLI entry point."""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from selectools.evals.__main__ import _build_parser, _create_agent, main
from selectools.evals.regression import RegressionResult
from selectools.evals.report import EvalReport
from selectools.evals.types import CaseResult, CaseVerdict, EvalFailure, EvalMetadata, TestCase

# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for _build_parser()."""

    def test_returns_parser(self):
        parser = _build_parser()
        assert parser is not None

    def test_run_command_parses(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "cases.json"])
        assert args.command == "run"
        assert args.cases == "cases.json"

    def test_run_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "cases.json"])
        assert args.concurrency == 1
        assert args.name == "eval"
        assert args.provider == "local"
        assert args.verbose is False
        assert args.html is None
        assert args.junit is None
        assert args.json_out is None
        assert args.baseline is None
        assert args.model is None

    def test_run_all_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "cases.json",
                "--html",
                "report.html",
                "--junit",
                "results.xml",
                "--json",
                "out.json",
                "--baseline",
                "/baselines",
                "--concurrency",
                "4",
                "--name",
                "my-suite",
                "--verbose",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
            ]
        )
        assert args.html == "report.html"
        assert args.junit == "results.xml"
        assert args.json_out == "out.json"
        assert args.baseline == "/baselines"
        assert args.concurrency == 4
        assert args.name == "my-suite"
        assert args.verbose is True
        assert args.provider == "openai"
        assert args.model == "gpt-4o"

    def test_compare_command_parses(self):
        parser = _build_parser()
        args = parser.parse_args(["compare", "cases.json", "--baseline", "/base"])
        assert args.command == "compare"
        assert args.cases == "cases.json"
        assert args.baseline == "/base"
        assert args.save is False

    def test_compare_with_save(self):
        parser = _build_parser()
        args = parser.parse_args(["compare", "cases.json", "--baseline", "/base", "--save"])
        assert args.save is True

    def test_no_command_gives_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# _create_agent tests
# ---------------------------------------------------------------------------


class TestCreateAgent:
    """Tests for _create_agent()."""

    @patch("selectools.Agent")
    def test_local_provider(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        agent = _create_agent("local", None)
        assert agent is not None
        # Verify Agent was called with model="local"
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["config"].model == "local"

    @patch("selectools.Agent")
    def test_local_provider_with_model(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        agent = _create_agent("local", "custom-local")
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["config"].model == "custom-local"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _create_agent("unknown", None)

    @patch("selectools.Agent")
    def test_openai_provider(self, mock_agent_cls):
        """Verify OpenAI provider instantiation."""
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.OpenAIProvider") as mock_prov:
            mock_prov.return_value = MagicMock()
            agent = _create_agent("openai", None)
            mock_prov.assert_called_once()

    @patch("selectools.Agent")
    def test_anthropic_provider(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.AnthropicProvider") as mock_prov:
            mock_prov.return_value = MagicMock()
            agent = _create_agent("anthropic", None)
            mock_prov.assert_called_once()

    @patch("selectools.Agent")
    def test_gemini_provider(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.GeminiProvider") as mock_prov:
            mock_prov.return_value = MagicMock()
            agent = _create_agent("gemini", None)
            mock_prov.assert_called_once()

    @patch("selectools.Agent")
    def test_ollama_provider(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.OllamaProvider") as mock_prov:
            mock_prov.return_value = MagicMock()
            agent = _create_agent("ollama", None)
            mock_prov.assert_called_once()

    @patch("selectools.Agent")
    def test_openai_default_model(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.OpenAIProvider"):
            _create_agent("openai", None)
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["config"].model == "gpt-4.1-mini"

    @patch("selectools.Agent")
    def test_anthropic_default_model(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.AnthropicProvider"):
            _create_agent("anthropic", None)
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["config"].model == "claude-sonnet-4-6"

    @patch("selectools.Agent")
    def test_gemini_default_model(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.GeminiProvider"):
            _create_agent("gemini", None)
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["config"].model == "gemini-2.5-flash"

    @patch("selectools.Agent")
    def test_ollama_default_model(self, mock_agent_cls):
        mock_agent_cls.return_value = MagicMock()
        with patch("selectools.providers.OllamaProvider"):
            _create_agent("ollama", None)
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["config"].model == "llama3"


# ---------------------------------------------------------------------------
# Helpers for main() tests
# ---------------------------------------------------------------------------


def _mock_report(
    accuracy: float = 0.5,
    total_cases: int = 2,
    suite_name: str = "eval",
):
    """Build a mock EvalReport with realistic structure."""
    meta = EvalMetadata(
        suite_name=suite_name,
        model="local",
        provider="local",
        timestamp=0.0,
        run_id="test-run",
        total_cases=total_cases,
        duration_ms=100.0,
        selectools_version="0.20.0",
    )
    pass_case = CaseResult(
        case=TestCase(input="What is 1+1?", name="math"),
        verdict=CaseVerdict.PASS,
        latency_ms=50.0,
    )
    fail_case = CaseResult(
        case=TestCase(input="What is 2+2?", name="math2"),
        verdict=CaseVerdict.FAIL,
        latency_ms=60.0,
        failures=[
            EvalFailure(
                evaluator_name="contains",
                expected="4",
                actual="5",
                message="Expected '4' in output",
            )
        ],
    )
    report = EvalReport(metadata=meta, case_results=[pass_case, fail_case])
    # Attach mock methods for serialization
    report.to_html = MagicMock()
    report.to_junit_xml = MagicMock()
    report.to_json = MagicMock()
    return report


def _patch_main(argv, report=None, regression_result=None):
    """Return a set of patches for running main() with given argv."""
    if report is None:
        report = _mock_report()
    if regression_result is None:
        regression_result = RegressionResult()

    mock_suite = MagicMock()
    mock_suite.run.return_value = report

    patches = {
        "argv": patch("sys.argv", ["selectools.evals"] + argv),
        "dataset": patch(
            "selectools.evals.__main__.DatasetLoader.load",
            return_value=[TestCase(input="q1"), TestCase(input="q2")],
        ),
        "agent": patch(
            "selectools.evals.__main__._create_agent",
            return_value=MagicMock(),
        ),
        "suite": patch(
            "selectools.evals.__main__.EvalSuite",
            return_value=mock_suite,
        ),
        "baseline_store": patch(
            "selectools.evals.__main__.BaselineStore",
        ),
    }
    return patches, mock_suite, report, regression_result


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMainNoCommand:
    """Test main() with no subcommand."""

    def test_no_command_exits_1(self):
        with patch("sys.argv", ["selectools.evals"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestMainRunCommand:
    """Test main() 'run' subcommand."""

    def test_basic_run(self, capsys):
        patches, mock_suite, report, _ = _patch_main(["run", "cases.json"])
        with patches["argv"], patches["dataset"], patches["agent"], patches["suite"]:
            main()
        captured = capsys.readouterr()
        assert "Loaded 2 test cases" in captured.out
        assert "Running eval suite" in captured.out

    def test_run_verbose(self, capsys):
        report = _mock_report()
        patches, mock_suite, _, _ = _patch_main(["run", "cases.json", "--verbose"], report=report)
        with patches["argv"], patches["dataset"], patches["agent"], patches["suite"]:
            main()
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "FAIL" in captured.out
        assert "contains:" in captured.out

    def test_run_html_output(self, capsys):
        report = _mock_report()
        patches, mock_suite, _, _ = _patch_main(
            ["run", "cases.json", "--html", "report.html"], report=report
        )
        with patches["argv"], patches["dataset"], patches["agent"], patches["suite"]:
            main()
        report.to_html.assert_called_once_with("report.html")

    def test_run_junit_output(self, capsys):
        report = _mock_report()
        patches, mock_suite, _, _ = _patch_main(
            ["run", "cases.json", "--junit", "results.xml"], report=report
        )
        with patches["argv"], patches["dataset"], patches["agent"], patches["suite"]:
            main()
        report.to_junit_xml.assert_called_once_with("results.xml")

    def test_run_json_output(self, capsys):
        report = _mock_report()
        patches, mock_suite, _, _ = _patch_main(
            ["run", "cases.json", "--json", "out.json"], report=report
        )
        with patches["argv"], patches["dataset"], patches["agent"], patches["suite"]:
            main()
        report.to_json.assert_called_once_with("out.json")

    def test_run_baseline_no_regression(self, capsys):
        report = _mock_report()
        regression = RegressionResult(accuracy_delta=0.1, improvements=["math improved"])
        patches, mock_suite, _, _ = _patch_main(
            ["run", "cases.json", "--baseline", "/baselines"], report=report
        )
        mock_store_instance = MagicMock()
        mock_store_instance.compare.return_value = regression

        with (
            patches["argv"],
            patches["dataset"],
            patches["agent"],
            patches["suite"],
            patches["baseline_store"] as mock_store_cls,
        ):
            mock_store_cls.return_value = mock_store_instance
            main()
        captured = capsys.readouterr()
        assert "No regressions" in captured.out
        assert "Improvements" in captured.out
        mock_store_instance.save.assert_called_once_with(report)

    def test_run_baseline_with_regression(self, capsys):
        report = _mock_report()
        regression = RegressionResult(
            regressions=["math regressed"],
            accuracy_delta=-0.2,
        )
        patches, mock_suite, _, _ = _patch_main(
            ["run", "cases.json", "--baseline", "/baselines"], report=report
        )
        mock_store_instance = MagicMock()
        mock_store_instance.compare.return_value = regression

        with (
            patches["argv"],
            patches["dataset"],
            patches["agent"],
            patches["suite"],
            patches["baseline_store"] as mock_store_cls,
        ):
            mock_store_cls.return_value = mock_store_instance
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "REGRESSIONS DETECTED" in captured.out

    def test_run_zero_accuracy_exits_1(self, capsys):
        report = _mock_report(accuracy=0.0, total_cases=2)
        # Override case_results to be all-fail for 0% accuracy
        report.case_results = [
            CaseResult(
                case=TestCase(input="q1", name="q1"),
                verdict=CaseVerdict.FAIL,
                latency_ms=10.0,
            ),
            CaseResult(
                case=TestCase(input="q2", name="q2"),
                verdict=CaseVerdict.FAIL,
                latency_ms=10.0,
            ),
        ]
        report.metadata.total_cases = 2
        patches, mock_suite, _, _ = _patch_main(["run", "cases.json"], report=report)
        with patches["argv"], patches["dataset"], patches["agent"], patches["suite"]:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestMainCompareCommand:
    """Test main() 'compare' subcommand."""

    def test_compare_no_regression(self, capsys):
        report = _mock_report()
        regression = RegressionResult(
            accuracy_delta=0.05,
            improvements=["math improved"],
        )
        patches, mock_suite, _, _ = _patch_main(
            ["compare", "cases.json", "--baseline", "/baselines"], report=report
        )
        mock_store_instance = MagicMock()
        mock_store_instance.compare.return_value = regression

        with (
            patches["argv"],
            patches["dataset"],
            patches["agent"],
            patches["suite"],
            patches["baseline_store"] as mock_store_cls,
        ):
            mock_store_cls.return_value = mock_store_instance
            main()
        captured = capsys.readouterr()
        assert "No regressions detected" in captured.out
        assert "Improvements" in captured.out

    def test_compare_with_regression_exits_1(self, capsys):
        report = _mock_report()
        regression = RegressionResult(
            regressions=["test1"],
            accuracy_delta=-0.1,
            latency_p50_delta=50.0,
            cost_delta=0.001,
        )
        patches, mock_suite, _, _ = _patch_main(
            ["compare", "cases.json", "--baseline", "/baselines"], report=report
        )
        mock_store_instance = MagicMock()
        mock_store_instance.compare.return_value = regression

        with (
            patches["argv"],
            patches["dataset"],
            patches["agent"],
            patches["suite"],
            patches["baseline_store"] as mock_store_cls,
        ):
            mock_store_cls.return_value = mock_store_instance
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "REGRESSIONS DETECTED" in captured.out
        assert "test1" in captured.out
        assert "Accuracy:" in captured.out
        assert "Latency p50:" in captured.out
        assert "Cost:" in captured.out

    def test_compare_save_flag(self, capsys):
        report = _mock_report()
        regression = RegressionResult(accuracy_delta=0.0)
        patches, mock_suite, _, _ = _patch_main(
            [
                "compare",
                "cases.json",
                "--baseline",
                "/baselines",
                "--save",
            ],
            report=report,
        )
        mock_store_instance = MagicMock()
        mock_store_instance.compare.return_value = regression

        with (
            patches["argv"],
            patches["dataset"],
            patches["agent"],
            patches["suite"],
            patches["baseline_store"] as mock_store_cls,
        ):
            mock_store_cls.return_value = mock_store_instance
            main()
        mock_store_instance.save.assert_called_once_with(report)
        captured = capsys.readouterr()
        assert "Baseline updated" in captured.out


class TestMainSuiteCreation:
    """Verify EvalSuite is constructed with correct args."""

    def test_suite_receives_correct_params(self, capsys):
        patches, mock_suite, report, _ = _patch_main(
            ["run", "cases.json", "--name", "my-suite", "--concurrency", "3"]
        )

        with (
            patches["argv"],
            patches["dataset"],
            patches["agent"],
            patches["suite"] as mock_suite_cls,
        ):
            main()

        mock_suite_cls.assert_called_once()
        call_kwargs = mock_suite_cls.call_args
        assert call_kwargs.kwargs["name"] == "my-suite"
        assert call_kwargs.kwargs["max_concurrency"] == 3
        assert call_kwargs.kwargs["on_progress"] is not None


class TestMainModuleInvocation:
    """Test `python -m selectools.evals` invocation guard."""

    def test_if_name_main_guard(self):
        """The __main__.py must have the if __name__ == '__main__' guard."""
        import inspect

        import selectools.evals.__main__ as mod

        source = inspect.getsource(mod)
        assert 'if __name__ == "__main__"' in source
