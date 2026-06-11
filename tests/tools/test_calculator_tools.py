"""
Tests for calculator tools (evaluate_expression, unit_convert).

evaluate_expression is security-critical: it must never reach eval()/exec()
and must reject any AST node outside the documented whitelist. The
adversarial tests below lock that contract in.
"""

from __future__ import annotations

from selectools.toolbox import calculator_tools
from selectools.toolbox.calculator_tools import evaluate_expression, unit_convert

# =============================================================================
# evaluate_expression — happy path
# =============================================================================


class TestEvaluateExpressionHappyPath:
    """Valid arithmetic must evaluate correctly."""

    def test_tool_metadata(self) -> None:
        assert evaluate_expression.name == "evaluate_expression"
        assert evaluate_expression.description

    def test_basic_arithmetic(self) -> None:
        assert "7" in evaluate_expression.function("3 + 4")

    def test_operator_precedence(self) -> None:
        assert "14" in evaluate_expression.function("2 + 3 * 4")

    def test_parentheses(self) -> None:
        assert "20" in evaluate_expression.function("(2 + 3) * 4")

    def test_float_division(self) -> None:
        assert "2.5" in evaluate_expression.function("5 / 2")

    def test_floor_division_and_modulo(self) -> None:
        assert "3" in evaluate_expression.function("7 // 2")
        assert "1" in evaluate_expression.function("7 % 2")

    def test_unary_minus(self) -> None:
        assert "-5" in evaluate_expression.function("-5")
        assert "5" in evaluate_expression.function("--5")

    def test_power(self) -> None:
        assert "1024" in evaluate_expression.function("2 ** 10")

    def test_whitelisted_functions(self) -> None:
        assert "4" in evaluate_expression.function("sqrt(16)")
        assert "120" in evaluate_expression.function("max(120, 5)")
        assert "3" in evaluate_expression.function("abs(-3)")
        assert "2" in evaluate_expression.function("round(2.4)")
        assert "1" in evaluate_expression.function("cos(0)")

    def test_constants(self) -> None:
        result = evaluate_expression.function("pi")
        assert "3.14159" in result

    def test_nested_function_calls(self) -> None:
        assert "5" in evaluate_expression.function("sqrt(abs(-25))")


# =============================================================================
# evaluate_expression — adversarial (security)
# =============================================================================


class TestEvaluateExpressionAdversarial:
    """Hostile inputs must be rejected with readable errors, never executed."""

    def test_dunder_import_rejected(self) -> None:
        result = evaluate_expression.function("__import__('os').system('id')")
        assert "Error" in result

    def test_attribute_access_rejected(self) -> None:
        result = evaluate_expression.function("(1).__class__.__bases__")
        assert "Error" in result

    def test_lambda_rejected(self) -> None:
        result = evaluate_expression.function("(lambda: 1)()")
        assert "Error" in result

    def test_arbitrary_name_rejected(self) -> None:
        result = evaluate_expression.function("os")
        assert "Error" in result

    def test_non_whitelisted_call_rejected(self) -> None:
        result = evaluate_expression.function("eval('1+1')")
        assert "Error" in result

    def test_exec_rejected(self) -> None:
        result = evaluate_expression.function("exec('x = 1')")
        assert "Error" in result

    def test_getattr_rejected(self) -> None:
        result = evaluate_expression.function("getattr(1, '__class__')")
        assert "Error" in result

    def test_subscript_rejected(self) -> None:
        result = evaluate_expression.function("[1, 2][0]")
        assert "Error" in result

    def test_string_constant_rejected(self) -> None:
        result = evaluate_expression.function("'a' * 1000000")
        assert "Error" in result

    def test_fstring_rejected(self) -> None:
        result = evaluate_expression.function("f'{1+1}'")
        assert "Error" in result

    def test_comprehension_rejected(self) -> None:
        result = evaluate_expression.function("[x for x in range(10)]")
        assert "Error" in result

    def test_walrus_rejected(self) -> None:
        result = evaluate_expression.function("(x := 5)")
        assert "Error" in result

    def test_statement_rejected(self) -> None:
        result = evaluate_expression.function("import os")
        assert "Error" in result

    def test_huge_exponent_rejected(self) -> None:
        result = evaluate_expression.function("9 ** 9999999")
        assert "Error" in result

    def test_chained_huge_exponent_rejected(self) -> None:
        result = evaluate_expression.function("2 ** 100 ** 100")
        assert "Error" in result

    def test_huge_base_rejected(self) -> None:
        result = evaluate_expression.function("(10 ** 200) ** 10")
        assert "Error" in result

    def test_division_by_zero_readable(self) -> None:
        result = evaluate_expression.function("1 / 0")
        assert "Error" in result
        assert "zero" in result.lower()

    def test_modulo_by_zero_readable(self) -> None:
        result = evaluate_expression.function("1 % 0")
        assert "Error" in result

    def test_syntax_error_readable(self) -> None:
        result = evaluate_expression.function("2 +")
        assert "Error" in result

    def test_empty_expression(self) -> None:
        result = evaluate_expression.function("")
        assert "Error" in result

    def test_oversized_expression_rejected(self) -> None:
        result = evaluate_expression.function("1+" * 6000 + "1")
        assert "Error" in result

    def test_boolean_op_rejected(self) -> None:
        result = evaluate_expression.function("True and False")
        assert "Error" in result

    def test_bitwise_shift_rejected(self) -> None:
        result = evaluate_expression.function("1 << 1000000")
        assert "Error" in result

    def test_module_has_no_eval_or_exec(self) -> None:
        """The module source must never call the eval() or exec() builtins."""
        import ast as ast_mod
        import inspect

        source = inspect.getsource(calculator_tools)
        for node in ast_mod.walk(ast_mod.parse(source)):
            if isinstance(node, ast_mod.Call) and isinstance(node.func, ast_mod.Name):
                assert node.func.id not in ("eval", "exec"), (
                    f"calculator_tools calls forbidden builtin {node.func.id}()"
                )


# =============================================================================
# unit_convert
# =============================================================================


class TestUnitConvert:
    """Unit conversion across length, mass, temperature, and data."""

    def test_tool_metadata(self) -> None:
        assert unit_convert.name == "unit_convert"
        assert unit_convert.description

    def test_length_km_to_miles(self) -> None:
        result = unit_convert.function(10, "km", "mi")
        assert "6.21" in result

    def test_length_feet_to_meters(self) -> None:
        result = unit_convert.function(10, "ft", "m")
        assert "3.048" in result

    def test_mass_kg_to_lb(self) -> None:
        result = unit_convert.function(1, "kg", "lb")
        assert "2.20" in result

    def test_temperature_c_to_f(self) -> None:
        result = unit_convert.function(100, "celsius", "fahrenheit")
        assert "212" in result

    def test_temperature_f_to_c(self) -> None:
        result = unit_convert.function(32, "f", "c")
        assert "0" in result

    def test_temperature_c_to_k(self) -> None:
        result = unit_convert.function(0, "c", "k")
        assert "273.15" in result

    def test_data_gb_to_mb(self) -> None:
        result = unit_convert.function(1, "gb", "mb")
        assert "1000" in result

    def test_data_gib_to_bytes(self) -> None:
        result = unit_convert.function(1, "gib", "b")
        assert "1073741824" in result

    def test_unknown_unit(self) -> None:
        result = unit_convert.function(1, "parsec", "m")
        assert "Error" in result

    def test_cross_category_rejected(self) -> None:
        result = unit_convert.function(1, "kg", "m")
        assert "Error" in result

    def test_case_insensitive(self) -> None:
        result = unit_convert.function(1, "KM", "M")
        assert "1000" in result
