"""
Calculator tools -- safe math expression evaluation and unit conversion.

``evaluate_expression`` parses input with :mod:`ast` and walks the tree
against an explicit whitelist of node types, operators, functions, and
constants. It never calls ``eval()`` or ``exec()``, so arbitrary code
(imports, attribute access, lambdas, comprehensions, names outside the
whitelist) is structurally impossible to execute.

``unit_convert`` is a pure dict-based converter for length, mass,
temperature, and data units. No external dependencies.
"""

from __future__ import annotations

import ast
import math
from typing import Callable, Dict, Union

from ..stability import beta
from ..tools import tool

_MAX_EXPRESSION_LENGTH = 2000
_MAX_POW_EXPONENT = 1000.0
_MAX_POW_BASE = 1e100
_MAX_NUMBER = 1e308

_Number = Union[int, float]

# Whitelist: the ONLY callables reachable from an expression.
_ALLOWED_FUNCTIONS: Dict[str, Callable[..., _Number]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "degrees": math.degrees,
    "radians": math.radians,
}

# Whitelist: the ONLY names resolvable in an expression.
_ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Whitelist: binary operators. Bitwise/shift operators are intentionally
# excluded (1 << huge allocates unbounded memory).
_BINARY_OPS: Dict[type, Callable[[_Number, _Number], _Number]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}


def _checked_pow(base: _Number, exponent: _Number) -> _Number:
    """Exponentiation with operand-size guards against memory/CPU bombs."""
    if abs(exponent) > _MAX_POW_EXPONENT:
        raise ValueError(f"exponent magnitude exceeds the limit of {int(_MAX_POW_EXPONENT)}")
    if abs(base) > _MAX_POW_BASE:
        raise ValueError("base magnitude is too large for exponentiation")
    return base**exponent


def _eval_node(node: ast.AST) -> _Number:
    """Recursively evaluate a whitelisted AST node.

    Raises:
        ValueError: for any node, operator, name, or call outside the whitelist.
        ZeroDivisionError: propagated for division/modulo by zero.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(
                f"only int/float constants are allowed, got {type(node.value).__name__}"
            )
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ValueError(
            f"unknown name '{node.id}' (allowed: {', '.join(sorted(_ALLOWED_CONSTANTS))})"
        )

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"unary operator '{type(node.op).__name__}' is not allowed")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        # Evaluate right-hand side lazily for Pow so chained exponents like
        # 2 ** 100 ** 100 are guarded before Python computes the inner pow.
        op_type = type(node.op)
        if isinstance(node.op, ast.Pow):
            right = _eval_node(node.right)
            result = _checked_pow(left, right)
        elif op_type in _BINARY_OPS:
            right = _eval_node(node.right)
            result = _BINARY_OPS[op_type](left, right)
        else:
            raise ValueError(f"operator '{op_type.__name__}' is not allowed")
        if isinstance(result, (int, float)) and abs(result) > _MAX_NUMBER:
            raise ValueError("result magnitude is too large")
        return result

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            func_repr = (
                node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            )
            raise ValueError(
                f"function '{func_repr}' is not allowed "
                f"(allowed: {', '.join(sorted(_ALLOWED_FUNCTIONS))})"
            )
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in function calls")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    raise ValueError(f"expression element '{type(node).__name__}' is not allowed")


@beta
@tool(description="Safely evaluate a math expression (arithmetic, sqrt, log, trig, pi/e)")
def evaluate_expression(expression: str) -> str:
    """
    Evaluate a mathematical expression safely (no code execution).

    Supports: ``+ - * / // % **``, parentheses, unary minus, the constants
    ``pi``, ``e``, ``tau``, and the functions ``abs``, ``round``, ``min``,
    ``max``, ``sqrt``, ``sin``, ``cos``, ``tan``, ``asin``, ``acos``,
    ``atan``, ``log``, ``log2``, ``log10``, ``exp``, ``floor``, ``ceil``,
    ``degrees``, ``radians``.

    Examples: ``"2 + 3 * 4"``, ``"sqrt(16) + 2 ** 10"``, ``"sin(pi / 2)"``.

    Args:
        expression: The math expression to evaluate.

    Returns:
        The result as ``expression = value``, or an error message.
    """
    if not expression or not expression.strip():
        return "Error: No expression provided."

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return f"Error: Expression exceeds the {_MAX_EXPRESSION_LENGTH}-character limit."

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Error: Invalid expression syntax: {exc.msg}"

    try:
        result = _eval_node(tree)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Result is too large to compute."
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"

    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return f"{expression.strip()} = {int(result)}"
    return f"{expression.strip()} = {result}"


# Conversion factors to a base unit per category (length: meter, mass: gram,
# data: byte). Temperature is handled separately (affine, not linear).
_LENGTH_UNITS: Dict[str, float] = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}

_MASS_UNITS: Dict[str, float] = {
    "mg": 0.001,
    "g": 1.0,
    "kg": 1000.0,
    "t": 1_000_000.0,
    "tonne": 1_000_000.0,
    "oz": 28.349523125,
    "ounce": 28.349523125,
    "ounces": 28.349523125,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}

_DATA_UNITS: Dict[str, float] = {
    "b": 1.0,
    "byte": 1.0,
    "bytes": 1.0,
    "kb": 1000.0,
    "mb": 1000.0**2,
    "gb": 1000.0**3,
    "tb": 1000.0**4,
    "kib": 1024.0,
    "mib": 1024.0**2,
    "gib": 1024.0**3,
    "tib": 1024.0**4,
}

_TEMPERATURE_UNITS = {
    "c": "c",
    "celsius": "c",
    "f": "f",
    "fahrenheit": "f",
    "k": "k",
    "kelvin": "k",
}

_CATEGORIES: Dict[str, Dict[str, float]] = {
    "length": _LENGTH_UNITS,
    "mass": _MASS_UNITS,
    "data": _DATA_UNITS,
}


def _to_celsius(value: float, unit: str) -> float:
    if unit == "c":
        return value
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15


def _from_celsius(value: float, unit: str) -> float:
    if unit == "c":
        return value
    if unit == "f":
        return value * 9.0 / 5.0 + 32.0
    return value + 273.15


def _format_quantity(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


@beta
@tool(description="Convert between length, mass, temperature, and data units")
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a value between units of the same category.

    Supported categories and units:

    - **length**: mm, cm, m, km, in, ft, yd, mi
    - **mass**: mg, g, kg, t, oz, lb
    - **temperature**: c (celsius), f (fahrenheit), k (kelvin)
    - **data**: b, kb, mb, gb, tb (decimal) and kib, mib, gib, tib (binary)

    Examples: ``unit_convert(10, "km", "mi")``, ``unit_convert(100, "c", "f")``.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (case-insensitive, e.g. ``"km"``).
        to_unit: Target unit (case-insensitive, e.g. ``"mi"``).

    Returns:
        Conversion result as ``<value> <from> = <result> <to>``, or an error
        message listing valid units.
    """
    src = from_unit.strip().lower()
    dst = to_unit.strip().lower()

    if src in _TEMPERATURE_UNITS and dst in _TEMPERATURE_UNITS:
        celsius = _to_celsius(float(value), _TEMPERATURE_UNITS[src])
        converted = _from_celsius(celsius, _TEMPERATURE_UNITS[dst])
        return f"{_format_quantity(float(value))} {src} = {_format_quantity(converted)} {dst}"

    src_category = next((name for name, units in _CATEGORIES.items() if src in units), None)
    dst_category = next((name for name, units in _CATEGORIES.items() if dst in units), None)

    if src_category is None and src not in _TEMPERATURE_UNITS:
        all_units = sorted(set().union(_TEMPERATURE_UNITS, *_CATEGORIES.values()))
        return f"Error: Unknown unit '{from_unit}'. Supported units: {', '.join(all_units)}"
    if dst_category is None and dst not in _TEMPERATURE_UNITS:
        all_units = sorted(set().union(_TEMPERATURE_UNITS, *_CATEGORIES.values()))
        return f"Error: Unknown unit '{to_unit}'. Supported units: {', '.join(all_units)}"

    if src_category is None or dst_category is None or src_category != dst_category:
        return (
            f"Error: Cannot convert between '{from_unit}' and '{to_unit}' -- "
            "units belong to different categories."
        )

    units = _CATEGORIES[src_category]
    converted = float(value) * units[src] / units[dst]
    return f"{_format_quantity(float(value))} {src} = {_format_quantity(converted)} {dst}"


__stability__ = "stable"

__all__ = [
    "evaluate_expression",
    "unit_convert",
]
