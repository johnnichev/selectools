"""
Date and time utility tools.
"""

from datetime import datetime, timedelta
from typing import Optional

from ..tools import tool


@tool(description="Get the current date and time")
def get_current_time(timezone: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Get the current date and time in a specified timezone.

    Args:
        timezone: Timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London')
        format: Datetime format string (default: ISO-like format with timezone)

    Returns:
        Formatted current datetime or error message
    """
    try:
        try:
            import pytz  # type: ignore[import-untyped]
        except ImportError:
            # Fallback to basic UTC if pytz not available
            if timezone != "UTC":
                return (
                    "❌ Error: 'pytz' library required for timezone support. Run: pip install pytz"
                )
            now = datetime.utcnow()
            formatted = now.strftime(format).replace("%Z", "UTC")
            return f"Current time: {formatted}"

        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        formatted = now.strftime(format)
        return f"Current time in {timezone}: {formatted}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"❌ Error: Unknown timezone '{timezone}'"
    except Exception as e:
        return f"❌ Error getting current time: {e}"


@tool(description="Parse a date/time string into a standard format")
def parse_datetime(
    datetime_string: str,
    input_format: Optional[str] = None,
    output_format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Parse a date/time string and reformat it.

    Args:
        datetime_string: The datetime string to parse
        input_format: Expected format (if None, tries common formats)
        output_format: Desired output format (default: YYYY-MM-DD HH:MM:SS)

    Returns:
        Formatted datetime string or error message
    """
    try:
        # If format specified, use it
        if input_format:
            dt = datetime.strptime(datetime_string, input_format)
            return f"✅ Parsed: {dt.strftime(output_format)}"

        # Try common formats
        common_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M",
        ]

        for fmt in common_formats:
            try:
                dt = datetime.strptime(datetime_string, fmt)
                return f"✅ Parsed: {dt.strftime(output_format)}"
            except ValueError:
                continue

        return f"❌ Error: Could not parse '{datetime_string}' with any common format"
    except ValueError as e:
        return f"❌ Error parsing datetime: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


@tool(description="Calculate time difference between two dates")
def time_difference(start_date: str, end_date: str, unit: str = "days") -> str:
    """
    Calculate the difference between two dates.

    Args:
        start_date: Start date/time string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end_date: End date/time string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        unit: Unit for difference ('days', 'hours', 'minutes', 'seconds')

    Returns:
        Time difference in specified unit or error message
    """
    try:
        # Try to parse dates
        formats_to_try = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]

        start_dt = None
        for fmt in formats_to_try:
            try:
                start_dt = datetime.strptime(start_date, fmt)
                break
            except ValueError:
                continue

        if start_dt is None:
            return f"❌ Error: Could not parse start date '{start_date}'"

        end_dt = None
        for fmt in formats_to_try:
            try:
                end_dt = datetime.strptime(end_date, fmt)
                break
            except ValueError:
                continue

        if end_dt is None:
            return f"❌ Error: Could not parse end date '{end_date}'"

        # Calculate difference
        diff = end_dt - start_dt
        total_seconds = diff.total_seconds()

        if unit == "days":
            result = total_seconds / 86400
        elif unit == "hours":
            result = total_seconds / 3600
        elif unit == "minutes":
            result = total_seconds / 60
        elif unit == "seconds":
            result = total_seconds
        else:
            return f"❌ Error: Invalid unit '{unit}'. Use 'days', 'hours', 'minutes', or 'seconds'"

        return f"✅ Time difference: {result:.2f} {unit}"
    except Exception as e:
        return f"❌ Error calculating difference: {e}"


@tool(description="Add or subtract time from a date")
def date_arithmetic(
    date: str,
    operation: str,
    value: int,
    unit: str = "days",
    output_format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Add or subtract time from a date.

    Args:
        date: Starting date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        operation: 'add' or 'subtract'
        value: Amount to add/subtract
        unit: Time unit ('days', 'hours', 'minutes', 'seconds')
        output_format: Format for result (default: YYYY-MM-DD HH:MM:SS)

    Returns:
        Calculated date or error message
    """
    try:
        # Parse date
        formats_to_try = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
        dt = None
        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(date, fmt)
                break
            except ValueError:
                continue

        if dt is None:
            return f"❌ Error: Could not parse date '{date}'"

        # Create timedelta
        if unit == "days":
            delta = timedelta(days=value)
        elif unit == "hours":
            delta = timedelta(hours=value)
        elif unit == "minutes":
            delta = timedelta(minutes=value)
        elif unit == "seconds":
            delta = timedelta(seconds=value)
        else:
            return f"❌ Error: Invalid unit '{unit}'"

        # Apply operation
        if operation == "add":
            result_dt = dt + delta
        elif operation == "subtract":
            result_dt = dt - delta
        else:
            return f"❌ Error: Invalid operation '{operation}'. Use 'add' or 'subtract'"

        return f"✅ Result: {result_dt.strftime(output_format)}"
    except Exception as e:
        return f"❌ Error performing date arithmetic: {e}"
