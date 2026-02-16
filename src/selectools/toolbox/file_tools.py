"""
File operation tools for reading, writing, and listing files.

All file operations are relative to the current working directory by default.
"""

from pathlib import Path
from typing import Generator

from ..tools import tool


@tool(description="Read the contents of a text file")
def read_file(filepath: str, encoding: str = "utf-8") -> str:
    """
    Read and return the contents of a text file.

    Args:
        filepath: Path to the file to read
        encoding: Text encoding (default: utf-8)

    Returns:
        The file contents as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
    """
    try:
        path = Path(filepath)
        content = path.read_text(encoding=encoding)
        return f"File: {filepath}\nSize: {len(content)} characters\n\n{content}"
    except FileNotFoundError:
        return f"❌ Error: File not found: {filepath}"
    except PermissionError:
        return f"❌ Error: Permission denied reading: {filepath}"
    except Exception as e:
        return f"❌ Error reading file: {e}"


@tool(description="Write text content to a file")
def write_file(filepath: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str:
    """
    Write text content to a file.

    Args:
        filepath: Path to the file to write
        content: Text content to write
        mode: Write mode ('w' = overwrite, 'a' = append)
        encoding: Text encoding (default: utf-8)

    Returns:
        Success message with file path and size
    """
    try:
        if mode not in ["w", "a"]:
            return f"❌ Error: Invalid mode '{mode}'. Use 'w' (write) or 'a' (append)"

        path = Path(filepath)
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "w":
            path.write_text(content, encoding=encoding)
            action = "Written"
        else:  # append
            with path.open("a", encoding=encoding) as f:
                f.write(content)
            action = "Appended"

        return f"✅ {action} {len(content)} characters to: {filepath}"
    except PermissionError:
        return f"❌ Error: Permission denied writing to: {filepath}"
    except Exception as e:
        return f"❌ Error writing file: {e}"


@tool(description="List files and directories in a path")
def list_files(
    directory: str = ".",
    pattern: str = "*",
    show_hidden: bool = False,
    recursive: bool = False,
) -> str:
    """
    List files and directories matching a pattern.

    Args:
        directory: Directory to list (default: current directory)
        pattern: Glob pattern to match (default: *)
        show_hidden: Include hidden files starting with '.'
        recursive: Search subdirectories recursively

    Returns:
        Formatted list of matching files and directories
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"❌ Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"❌ Error: Not a directory: {directory}"

        # Use glob or rglob based on recursive flag
        glob_func = path.rglob if recursive else path.glob
        items = sorted(glob_func(pattern))

        # Filter hidden files if needed
        if not show_hidden:
            items = [item for item in items if not any(part.startswith(".") for part in item.parts)]

        if not items:
            return f"No files found matching '{pattern}' in {directory}"

        # Format output
        lines = [f"Files in {directory} (pattern: {pattern}):"]
        for item in items:
            rel_path = item.relative_to(path) if recursive else item.name
            if item.is_dir():
                lines.append(f"  📁 {rel_path}/")
            else:
                size_kb = item.stat().st_size / 1024
                lines.append(f"  📄 {rel_path} ({size_kb:.1f} KB)")

        lines.append(f"\nTotal: {len(items)} items")
        return "\n".join(lines)
    except PermissionError:
        return f"❌ Error: Permission denied accessing: {directory}"
    except Exception as e:
        return f"❌ Error listing files: {e}"


@tool(description="Check if a file or directory exists")
def file_exists(path: str) -> str:
    """
    Check if a file or directory exists.

    Args:
        path: Path to check

    Returns:
        Information about the path's existence and type
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"❌ Path does not exist: {path}"

        if p.is_file():
            size_kb = p.stat().st_size / 1024
            return f"✅ File exists: {path} ({size_kb:.1f} KB)"
        elif p.is_dir():
            items = list(p.iterdir())
            return f"✅ Directory exists: {path} ({len(items)} items)"
        else:
            return f"✅ Path exists: {path} (special file)"
    except PermissionError:
        return f"❌ Error: Permission denied accessing: {path}"
    except Exception as e:
        return f"❌ Error checking path: {e}"


@tool(description="Read a file line by line with streaming", streaming=True)
def read_file_stream(filepath: str, encoding: str = "utf-8") -> Generator[str, None, None]:
    """
    Read a file line by line and yield each line progressively.

    This is useful for large files where you want to see results as they're processed.

    Args:
        filepath: Path to the file to read
        encoding: Text encoding (default: utf-8)

    Yields:
        Each line from the file, prefixed with line number

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
    """
    try:
        path = Path(filepath)
        yield f"📄 Reading file: {filepath}\n"
        yield f"📏 Size: {path.stat().st_size} bytes\n\n"

        with path.open("r", encoding=encoding) as f:
            for i, line in enumerate(f, 1):
                yield f"[Line {i:4d}] {line}"

        yield f"\n✅ Finished reading {filepath}\n"
    except FileNotFoundError:
        yield f"❌ Error: File not found: {filepath}\n"
    except PermissionError:
        yield f"❌ Error: Permission denied reading: {filepath}\n"
    except Exception as e:
        yield f"❌ Error reading file: {e}\n"
