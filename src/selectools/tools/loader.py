"""Dynamic tool loading from Python modules and directories."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .base import Tool


def _is_tool(obj: object) -> bool:
    """Check if an object is a selectools Tool instance."""
    return isinstance(obj, Tool)


class ToolLoader:
    """
    Discover and load ``Tool`` instances from Python modules and directories.

    Supports three discovery strategies:

    1. **From a module path** (dotted import string)::

        tools = ToolLoader.from_module("myproject.tools.search")

    2. **From a file path** (``*.py`` file)::

        tools = ToolLoader.from_file("/abs/path/to/search_tools.py")

    3. **From a directory** (all ``*.py`` files, optionally recursive)::

        tools = ToolLoader.from_directory("./plugins/")

    In every case the loader imports the module and collects all module-level
    attributes that are ``Tool`` instances (i.e. functions decorated with
    ``@tool``).
    """

    @staticmethod
    def from_module(module_path: str) -> List[Tool]:
        """
        Import a dotted module path and return all ``Tool`` objects found.

        Args:
            module_path: Dotted Python module path, e.g. ``"myapp.tools"``.

        Returns:
            List of Tool instances discovered in the module.

        Raises:
            ImportError: If the module cannot be imported.
        """
        module = importlib.import_module(module_path)
        return [obj for obj in vars(module).values() if _is_tool(obj)]

    @staticmethod
    def from_file(file_path: str) -> List[Tool]:
        """
        Load a single Python file and return all ``Tool`` objects found.

        The file is imported as a module with a name derived from its filename.

        Args:
            file_path: Absolute or relative path to a ``.py`` file.

        Returns:
            List of Tool instances discovered in the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If the file cannot be imported.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Tool file not found: {path}")
        if not path.suffix == ".py":
            raise ValueError(f"Expected a .py file, got: {path}")

        module_name = f"_selectools_dynamic_.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        return [obj for obj in vars(module).values() if _is_tool(obj)]

    @staticmethod
    def from_directory(
        directory: str,
        *,
        recursive: bool = False,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[Tool]:
        """
        Discover and load ``Tool`` objects from all ``.py`` files in a directory.

        Files whose names start with ``_`` are skipped unless explicitly included.

        Args:
            directory: Path to the directory to scan.
            recursive: If ``True``, also scan subdirectories (default ``False``).
            exclude: Optional sequence of filenames to skip.

        Returns:
            List of Tool instances discovered across all loaded files.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Tool directory not found: {dir_path}")

        exclude_set = set(exclude or [])
        pattern = "**/*.py" if recursive else "*.py"

        tools: List[Tool] = []
        for py_file in sorted(dir_path.glob(pattern)):
            if py_file.name.startswith("_"):
                continue
            if py_file.name in exclude_set:
                continue
            try:
                tools.extend(ToolLoader.from_file(str(py_file)))
            except Exception:  # nosec B112
                continue

        return tools

    @staticmethod
    def reload_module(module_path: str) -> List[Tool]:
        """
        Re-import a module and return the freshly loaded ``Tool`` objects.

        Useful for hot-reloading tools after code changes without restarting.

        Args:
            module_path: Dotted Python module path to reload.

        Returns:
            List of Tool instances from the reloaded module.

        Raises:
            ImportError: If the module is not already imported or cannot be reloaded.
        """
        if module_path not in sys.modules:
            return ToolLoader.from_module(module_path)

        module = importlib.reload(sys.modules[module_path])
        return [obj for obj in vars(module).values() if _is_tool(obj)]

    @staticmethod
    def reload_file(file_path: str) -> List[Tool]:
        """
        Re-import a Python file and return the freshly loaded ``Tool`` objects.

        Useful for hot-reloading tools from plugin files after edits.

        Args:
            file_path: Path to the ``.py`` file to reload.

        Returns:
            List of Tool instances from the reloaded file.
        """
        path = Path(file_path).resolve()
        module_name = f"_selectools_dynamic_.{path.stem}"

        if module_name in sys.modules:
            del sys.modules[module_name]

        return ToolLoader.from_file(file_path)


__all__ = ["ToolLoader"]
