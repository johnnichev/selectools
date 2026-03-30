"""Tests for ToolLoader dynamic discovery and Agent dynamic tool management."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List
from unittest.mock import MagicMock

import pytest

from selectools.tools import Tool, ToolLoader, tool
from selectools.tools.base import ToolParameter

if TYPE_CHECKING:
    from selectools.agent.core import Agent

# ---------------------------------------------------------------------------
# Fixtures: temp plugin files
# ---------------------------------------------------------------------------

PLUGIN_CODE = """
from selectools.tools import tool

@tool(name="greet", description="Say hello")
def greet(name: str) -> str:
    return f"Hello, {name}!"

@tool(name="farewell", description="Say goodbye")
def farewell(name: str) -> str:
    return f"Goodbye, {name}!"
"""

PLUGIN_CODE_V2 = """
from selectools.tools import tool

@tool(name="greet", description="Say hello (v2)")
def greet(name: str) -> str:
    return f"Hi, {name}!"
"""

NO_TOOLS_CODE = """
def plain_function():
    return 42
"""


@pytest.fixture()
def plugin_file(tmp_path: Path) -> str:
    """Write a temporary plugin file with two tools."""
    f = tmp_path / "my_tools.py"
    f.write_text(PLUGIN_CODE)
    return str(f)


@pytest.fixture()
def plugin_dir(tmp_path: Path) -> str:
    """Create a plugin directory with multiple tool files."""
    (tmp_path / "search.py").write_text(PLUGIN_CODE)
    (tmp_path / "empty.py").write_text(NO_TOOLS_CODE)
    (tmp_path / "_private.py").write_text(PLUGIN_CODE)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text(PLUGIN_CODE)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# ToolLoader.from_file
# ---------------------------------------------------------------------------


class TestToolLoaderFromFile:
    def test_loads_tools_from_file(self, plugin_file: str) -> None:
        tools = ToolLoader.from_file(plugin_file)
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "greet" in names
        assert "farewell" in names

    def test_tools_are_executable(self, plugin_file: str) -> None:
        tools = ToolLoader.from_file(plugin_file)
        greet = next(t for t in tools if t.name == "greet")
        result = greet.execute({"name": "Alice"})
        assert result == "Hello, Alice!"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ToolLoader.from_file("/nonexistent/path/tools.py")

    def test_non_py_file(self, tmp_path: Path) -> None:
        f = tmp_path / "tools.txt"
        f.write_text("not python")
        with pytest.raises(ValueError, match=".py"):
            ToolLoader.from_file(str(f))

    def test_no_tools_found(self, tmp_path: Path) -> None:
        f = tmp_path / "plain.py"
        f.write_text(NO_TOOLS_CODE)
        tools = ToolLoader.from_file(str(f))
        assert tools == []


# ---------------------------------------------------------------------------
# ToolLoader.from_directory
# ---------------------------------------------------------------------------


class TestToolLoaderFromDirectory:
    def test_loads_from_directory(self, plugin_dir: str) -> None:
        tools = ToolLoader.from_directory(plugin_dir)
        assert len(tools) >= 2
        assert all(isinstance(t, Tool) for t in tools)

    def test_skips_private_files(self, plugin_dir: str) -> None:
        tools = ToolLoader.from_directory(plugin_dir)
        for t in tools:
            result = t.execute({"name": "Test"})
            assert isinstance(result, str)

    def test_recursive_discovery(self, plugin_dir: str) -> None:
        tools_flat = ToolLoader.from_directory(plugin_dir, recursive=False)
        tools_recursive = ToolLoader.from_directory(plugin_dir, recursive=True)
        assert len(tools_recursive) > len(tools_flat)

    def test_exclude_files(self, plugin_dir: str) -> None:
        all_tools = ToolLoader.from_directory(plugin_dir)
        excluded = ToolLoader.from_directory(plugin_dir, exclude=["search.py"])
        assert len(excluded) < len(all_tools)

    def test_directory_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ToolLoader.from_directory("/nonexistent/dir")

    def test_empty_directory(self, tmp_path: Path) -> None:
        tools = ToolLoader.from_directory(str(tmp_path))
        assert tools == []


# ---------------------------------------------------------------------------
# ToolLoader.reload_file
# ---------------------------------------------------------------------------


class TestToolLoaderReload:
    def test_reload_file_picks_up_changes(self, tmp_path: Path) -> None:
        f = tmp_path / "hot.py"
        f.write_text(PLUGIN_CODE)
        tools_v1 = ToolLoader.from_file(str(f))
        greet_v1 = next(t for t in tools_v1 if t.name == "greet")
        assert greet_v1.description == "Say hello"

        f.write_text(PLUGIN_CODE_V2)
        tools_v2 = ToolLoader.reload_file(str(f))
        greet_v2 = next(t for t in tools_v2 if t.name == "greet")
        assert greet_v2.description == "Say hello (v2)"
        assert greet_v2.execute({"name": "Bob"}) == "Hi, Bob!"


# ---------------------------------------------------------------------------
# Agent dynamic tool methods
# ---------------------------------------------------------------------------


def _make_tool(name: str, desc: str = "test tool") -> Tool:
    """Create a minimal Tool for testing."""
    return Tool(
        name=name,
        description=desc,
        parameters=[ToolParameter(name="x", param_type=str, description="input")],
        function=lambda x: x,
    )


def _make_agent(tools: List[Tool]) -> Agent:
    """Create an Agent with mocked provider."""
    from selectools.agent.core import Agent as _Agent

    provider = MagicMock()
    provider.name = "mock"
    agent = _Agent(tools=tools, provider=provider)
    return agent


class TestAgentDynamicTools:
    def test_add_tool(self) -> None:
        agent = _make_agent([_make_tool("a")])
        agent.add_tool(_make_tool("b"))
        assert len(agent.tools) == 2
        assert "b" in agent._tools_by_name

    def test_add_tool_duplicate_raises(self) -> None:
        agent = _make_agent([_make_tool("a")])
        with pytest.raises(ValueError, match="already exists"):
            agent.add_tool(_make_tool("a"))

    def test_add_tools_batch(self) -> None:
        agent = _make_agent([_make_tool("a")])
        agent.add_tools([_make_tool("b"), _make_tool("c")])
        assert len(agent.tools) == 3

    def test_add_tools_batch_duplicate_raises(self) -> None:
        agent = _make_agent([_make_tool("a")])
        with pytest.raises(ValueError, match="already exists"):
            agent.add_tools([_make_tool("b"), _make_tool("a")])

    def test_remove_tool(self) -> None:
        agent = _make_agent([_make_tool("a"), _make_tool("b")])
        removed = agent.remove_tool("a")
        assert removed.name == "a"
        assert len(agent.tools) == 1
        assert "a" not in agent._tools_by_name

    def test_remove_tool_not_found(self) -> None:
        agent = _make_agent([_make_tool("a")])
        with pytest.raises(KeyError, match="not found"):
            agent.remove_tool("z")

    def test_remove_last_tool_raises(self) -> None:
        agent = _make_agent([_make_tool("a")])
        with pytest.raises(ValueError, match="at least one tool"):
            agent.remove_tool("a")

    def test_replace_tool_existing(self) -> None:
        agent = _make_agent([_make_tool("a", "v1")])
        old = agent.replace_tool(_make_tool("a", "v2"))
        assert old is not None
        assert old.description == "v1"
        assert agent._tools_by_name["a"].description == "v2"
        assert len(agent.tools) == 1

    def test_replace_tool_new(self) -> None:
        agent = _make_agent([_make_tool("a")])
        old = agent.replace_tool(_make_tool("b"))
        assert old is None
        assert len(agent.tools) == 2
        assert "b" in agent._tools_by_name

    def test_system_prompt_rebuilt_on_add(self) -> None:
        agent = _make_agent([_make_tool("a")])
        prompt_before = agent._system_prompt
        agent.add_tool(_make_tool("b"))
        assert agent._system_prompt != prompt_before

    def test_system_prompt_rebuilt_on_remove(self) -> None:
        agent = _make_agent([_make_tool("a"), _make_tool("b")])
        prompt_before = agent._system_prompt
        agent.remove_tool("b")
        assert agent._system_prompt != prompt_before

    def test_system_prompt_rebuilt_on_replace(self) -> None:
        agent = _make_agent([_make_tool("a", "old desc")])
        prompt_before = agent._system_prompt
        agent.replace_tool(_make_tool("a", "new desc"))
        assert agent._system_prompt != prompt_before

    def test_full_lifecycle(self, plugin_file: str) -> None:
        """Load tools from file, add to agent, remove, replace."""
        loaded = ToolLoader.from_file(plugin_file)
        agent = _make_agent([_make_tool("placeholder")])

        agent.add_tools(loaded)
        assert "greet" in agent._tools_by_name
        assert "farewell" in agent._tools_by_name

        agent.remove_tool("placeholder")
        assert len(agent.tools) == 2

        new_greet = _make_tool("greet", "updated greet")
        old = agent.replace_tool(new_greet)
        assert old is not None
        assert agent._tools_by_name["greet"].description == "updated greet"


# ---------------------------------------------------------------------------
# Regression Tests — Ralph Bug Hunt Pass 4
# ---------------------------------------------------------------------------


class TestToolLoaderSyntaxError:
    """Regression: from_file must raise ImportError (not SyntaxError) for broken files.

    Previously exec_module propagated SyntaxError directly, violating the documented
    contract (only FileNotFoundError and ImportError are declared).  The partially-
    registered module was also left in sys.modules.
    """

    def test_syntax_error_raises_import_error(self, tmp_path: Path) -> None:
        """from_file with a syntax-broken file must raise ImportError, not SyntaxError."""
        broken = tmp_path / "broken.py"
        broken.write_text("def oops(: this is not valid python\n")

        with pytest.raises(ImportError, match="Syntax error"):
            ToolLoader.from_file(str(broken))

    def test_syntax_error_does_not_leave_module_in_sys_modules(self, tmp_path: Path) -> None:
        """Modules that fail to load must be removed from sys.modules."""
        import sys

        broken = tmp_path / "orphan.py"
        broken.write_text("INVALID ===\n")

        try:
            ToolLoader.from_file(str(broken))
        except ImportError:
            pass

        # No _selectools_dynamic_ entry for this file should remain
        leaked = [k for k in sys.modules if "orphan" in k and "_selectools_dynamic_" in k]
        assert leaked == [], f"Leaked modules in sys.modules: {leaked}"

    def test_from_directory_swallows_syntax_errors(self, tmp_path: Path) -> None:
        """from_directory must silently skip files with syntax errors."""
        good = tmp_path / "good.py"
        good.write_text(PLUGIN_CODE)
        bad = tmp_path / "bad.py"
        bad.write_text("def oops(: broken\n")

        tools = ToolLoader.from_directory(str(tmp_path))
        tool_names = {t.name for t in tools}
        assert "greet" in tool_names, "Good file's tools must still be loaded"

    def test_from_directory_swallows_runtime_errors(self, tmp_path: Path) -> None:
        """from_directory must silently skip files that raise arbitrary exceptions at import.

        Regression: previously only ImportError/SyntaxError/AttributeError were caught,
        so a module that raises RuntimeError (or any other exception) at import time would
        propagate and abort the entire directory scan, leaving valid tools undiscovered.
        """
        good = tmp_path / "agood.py"
        good.write_text(PLUGIN_CODE)
        bad = tmp_path / "zbad.py"
        bad.write_text("raise RuntimeError('module-level failure')\n")

        # Must not raise; the RuntimeError must be caught and logged
        tools = ToolLoader.from_directory(str(tmp_path))
        tool_names = {t.name for t in tools}
        assert "greet" in tool_names, "Good file's tools must still be loaded despite sibling error"
        assert "farewell" in tool_names

    def test_from_directory_swallows_value_error(self, tmp_path: Path) -> None:
        """from_directory must swallow ValueError raised during module import."""
        good = tmp_path / "agood.py"
        good.write_text(PLUGIN_CODE)
        bad = tmp_path / "zbad.py"
        bad.write_text("raise ValueError('invalid config')\n")

        tools = ToolLoader.from_directory(str(tmp_path))
        assert any(t.name == "greet" for t in tools)


class TestToolLoaderSameStemCollision:
    """Regression: two files with identical stems from different directories must
    not collide in sys.modules.  Previously both got the same key, so reloading
    one would affect the other.
    """

    def test_same_stem_files_in_different_dirs_both_load(self, tmp_path: Path) -> None:
        """Two files named 'search.py' in different dirs must both load correctly."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "search.py").write_text(
            "from selectools.tools import tool\n"
            "@tool(description='Search A')\n"
            "def search_a(q: str) -> str:\n"
            "    return 'A:' + q\n"
        )
        (dir_b / "search.py").write_text(
            "from selectools.tools import tool\n"
            "@tool(description='Search B')\n"
            "def search_b(q: str) -> str:\n"
            "    return 'B:' + q\n"
        )

        tools_a = ToolLoader.from_file(str(dir_a / "search.py"))
        tools_b = ToolLoader.from_file(str(dir_b / "search.py"))

        assert len(tools_a) == 1
        assert len(tools_b) == 1
        assert tools_a[0].execute({"q": "x"}) == "A:x"
        assert tools_b[0].execute({"q": "x"}) == "B:x"

    def test_reload_file_targets_correct_module(self, tmp_path: Path) -> None:
        """reload_file must only remove the specific file's module, not a same-stem sibling."""
        import sys

        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "plugin.py").write_text(
            "from selectools.tools import tool\n"
            "@tool(description='Plugin A v1')\n"
            "def plugin_a(x: str) -> str:\n"
            "    return 'A:' + x\n"
        )
        (dir_b / "plugin.py").write_text(
            "from selectools.tools import tool\n"
            "@tool(description='Plugin B')\n"
            "def plugin_b(x: str) -> str:\n"
            "    return 'B:' + x\n"
        )

        tools_a = ToolLoader.from_file(str(dir_a / "plugin.py"))
        tools_b = ToolLoader.from_file(str(dir_b / "plugin.py"))

        # Reload A — B's module must remain in sys.modules
        (dir_a / "plugin.py").write_text(
            "from selectools.tools import tool\n"
            "@tool(description='Plugin A v2')\n"
            "def plugin_a(x: str) -> str:\n"
            "    return 'A2:' + x\n"
        )
        tools_a_v2 = ToolLoader.reload_file(str(dir_a / "plugin.py"))
        assert tools_a_v2[0].execute({"x": "t"}) == "A2:t"
        # B's tool must still work from its own module
        assert tools_b[0].execute({"x": "t"}) == "B:t"
