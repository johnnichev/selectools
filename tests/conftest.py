"""
Pytest configuration for selectools tests.

This file configures pytest with custom markers and command-line options
for running different types of tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests with real API calls",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires real API keys, use --run-e2e to run)"
    )
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI API key")
    config.addinivalue_line("markers", "anthropic: mark test as requiring Anthropic API key")
    config.addinivalue_line("markers", "gemini: mark test as requiring Gemini API key")


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests unless --run-e2e is passed."""
    if config.getoption("--run-e2e"):
        # --run-e2e given: do not skip e2e tests
        return

    skip_e2e = pytest.mark.skip(reason="Need --run-e2e option to run end-to-end tests")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
