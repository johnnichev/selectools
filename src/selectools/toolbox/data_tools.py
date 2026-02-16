"""
Data processing tools for JSON, CSV, and data formatting.
"""

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Generator

from ..tools import tool


@tool(description="Parse and validate JSON data")
def parse_json(json_string: str, pretty: bool = True) -> str:
    """
    Parse a JSON string and optionally pretty-print it.

    Args:
        json_string: The JSON string to parse
        pretty: Whether to format output with indentation

    Returns:
        Formatted JSON or error message
    """
    try:
        data = json.loads(json_string)
        if pretty:
            output = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            output = json.dumps(data, ensure_ascii=False)
        return f"✅ Valid JSON:\n{output}"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"
    except Exception as e:
        return f"❌ Error parsing JSON: {e}"


@tool(description="Convert JSON to CSV format")
def json_to_csv(json_string: str, delimiter: str = ",") -> str:
    """
    Convert JSON array of objects to CSV format.

    Args:
        json_string: JSON string containing an array of objects
        delimiter: CSV delimiter (default: comma)

    Returns:
        CSV formatted string or error message
    """
    try:
        data = json.loads(json_string)

        if not isinstance(data, list):
            return "❌ Error: JSON must be an array of objects"

        if not data:
            return "❌ Error: Empty array"

        if not isinstance(data[0], dict):
            return "❌ Error: Array items must be objects/dictionaries"

        # Get all keys from all objects (in case some have different keys)
        all_keys: set[str] = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        fieldnames = sorted(all_keys)

        # Write CSV
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)

        csv_content = output.getvalue()
        return f"✅ Converted {len(data)} rows to CSV:\n\n{csv_content}"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"
    except Exception as e:
        return f"❌ Error converting to CSV: {e}"


@tool(description="Parse CSV data into JSON format")
def csv_to_json(csv_string: str, delimiter: str = ",", pretty: bool = True) -> str:
    """
    Parse CSV data and convert to JSON array of objects.

    Args:
        csv_string: CSV data as a string
        delimiter: CSV delimiter (default: comma)
        pretty: Whether to format JSON output

    Returns:
        JSON formatted string or error message
    """
    try:
        # Parse CSV
        reader = csv.DictReader(StringIO(csv_string), delimiter=delimiter)
        data = list(reader)

        if not data:
            return "❌ Error: No data found in CSV"

        # Convert to JSON
        if pretty:
            json_output = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            json_output = json.dumps(data, ensure_ascii=False)

        return f"✅ Converted {len(data)} rows to JSON:\n\n{json_output}"
    except csv.Error as e:
        return f"❌ CSV parsing error: {e}"
    except Exception as e:
        return f"❌ Error converting to JSON: {e}"


@tool(description="Extract specific fields from JSON data")
def extract_json_field(json_string: str, field_path: str) -> str:
    """
    Extract a specific field from JSON using dot notation.

    Args:
        json_string: JSON string to extract from
        field_path: Path to field using dot notation (e.g., 'user.name' or 'items.0.price')

    Returns:
        Extracted value as JSON or error message
    """
    try:
        data = json.loads(json_string)

        # Navigate through the path
        current = data
        for key in field_path.split("."):
            # Handle array indices
            if isinstance(current, list):
                try:
                    index = int(key)
                    current = current[index]
                except (ValueError, IndexError):
                    return f"❌ Error: Invalid array index '{key}' in path"
            elif isinstance(current, dict):
                if key not in current:
                    return f"❌ Error: Field '{key}' not found"
                current = current[key]
            else:
                return f"❌ Error: Cannot navigate through '{key}' - not an object or array"

        # Format output
        result = json.dumps(current, indent=2, ensure_ascii=False)
        return f"✅ Extracted '{field_path}':\n{result}"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"
    except Exception as e:
        return f"❌ Error extracting field: {e}"


@tool(description="Format tabular data as a readable table")
def format_table(data: str, format_type: str = "simple") -> str:
    """
    Format JSON array of objects as a readable table.

    Args:
        data: JSON string containing array of objects
        format_type: Table format ('simple', 'markdown', or 'csv')

    Returns:
        Formatted table or error message
    """
    try:
        items = json.loads(data)

        if not isinstance(items, list):
            return "❌ Error: Data must be a JSON array"

        if not items:
            return "❌ Error: Empty array"

        if not isinstance(items[0], dict):
            return "❌ Error: Array items must be objects"

        # Get all keys
        all_keys: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        headers = sorted(all_keys)

        if format_type == "csv":
            # CSV format
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            writer.writerows(items)
            return output.getvalue()

        elif format_type == "markdown":
            # Markdown table format
            lines = []
            # Header
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            # Rows
            for item in items:
                row = [str(item.get(h, "")) for h in headers]
                lines.append("| " + " | ".join(row) + " |")
            return "\n".join(lines)

        else:  # simple format
            # Calculate column widths
            col_widths = {h: len(h) for h in headers}
            for item in items:
                for h in headers:
                    val_len = len(str(item.get(h, "")))
                    col_widths[h] = max(col_widths[h], val_len)

            # Build table
            lines = []
            # Header
            header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))
            # Rows
            for item in items:
                row = [str(item.get(h, "")).ljust(col_widths[h]) for h in headers]
                lines.append(" | ".join(row))

            return "\n".join(lines)

    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"
    except Exception as e:
        return f"❌ Error formatting table: {e}"


@tool(description="Process a CSV file row by row with streaming", streaming=True)
def process_csv_stream(
    filepath: str, delimiter: str = ",", encoding: str = "utf-8"
) -> Generator[str, None, None]:
    """
    Process a CSV file row by row and yield each row progressively.

    This is useful for large CSV files where you want to see results as they're processed.

    Args:
        filepath: Path to the CSV file
        delimiter: CSV delimiter (default: comma)
        encoding: Text encoding (default: utf-8)

    Yields:
        Each row from the CSV file as a formatted string
    """
    try:
        path = Path(filepath)
        yield f"📊 Processing CSV file: {filepath}\n"
        yield f"📏 Size: {path.stat().st_size} bytes\n\n"

        with path.open("r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            # Yield header info
            if reader.fieldnames:
                yield f"📋 Columns: {', '.join(reader.fieldnames)}\n\n"

            # Process each row
            for i, row in enumerate(reader, 1):
                yield f"--- Row {i} ---\n"
                for key, value in row.items():
                    yield f"  {key}: {value}\n"
                yield "\n"

        yield f"✅ Finished processing {filepath}\n"
    except FileNotFoundError:
        yield f"❌ Error: File not found: {filepath}\n"
    except PermissionError:
        yield f"❌ Error: Permission denied reading: {filepath}\n"
    except csv.Error as e:
        yield f"❌ Error parsing CSV: {e}\n"
    except Exception as e:
        yield f"❌ Error processing file: {e}\n"
