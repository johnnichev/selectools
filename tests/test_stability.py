"""Tests for selectools.stability decorators."""

import warnings

import pytest

from selectools.stability import beta, deprecated, stable

# ---------------------------------------------------------------------------
# @stable
# ---------------------------------------------------------------------------


def test_stable_function_sets_attribute():
    @stable
    def my_func():
        return 42

    assert my_func.__stability__ == "stable"


def test_stable_function_behavior_unchanged():
    @stable
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_stable_class_sets_attribute():
    @stable
    class MyClass:
        pass

    assert MyClass.__stability__ == "stable"


def test_stable_class_instantiation_unchanged():
    @stable
    class MyClass:
        def __init__(self, x):
            self.x = x

    obj = MyClass(10)
    assert obj.x == 10


def test_stable_emits_no_warning():
    @stable
    def my_func():
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        my_func()  # should not raise


# ---------------------------------------------------------------------------
# @beta
# ---------------------------------------------------------------------------


def test_beta_function_sets_attribute():
    @beta
    def my_func():
        return "beta"

    assert my_func.__stability__ == "beta"


def test_beta_class_sets_attribute():
    @beta
    class MyClass:
        pass

    assert MyClass.__stability__ == "beta"


def test_beta_emits_no_warning():
    @beta
    def my_func():
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        my_func()  # should not raise


# ---------------------------------------------------------------------------
# @deprecated — functions
# ---------------------------------------------------------------------------


def test_deprecated_function_emits_warning():
    @deprecated(since="0.19", replacement="new_func")
    def old_func():
        return "old"

    with pytest.warns(DeprecationWarning, match="deprecated since v0.19"):
        result = old_func()

    assert result == "old"


def test_deprecated_function_warning_contains_replacement():
    @deprecated(since="0.19", replacement="NewThing")
    def old_func():
        pass

    with pytest.warns(DeprecationWarning, match="NewThing"):
        old_func()


def test_deprecated_function_no_replacement_omits_use_line():
    @deprecated(since="0.18")
    def old_func():
        pass

    with pytest.warns(DeprecationWarning) as record:
        old_func()

    assert "Use" not in str(record[0].message)


def test_deprecated_function_preserves_name_and_doc():
    @deprecated(since="0.19", replacement="new_func")
    def old_func():
        """Old docstring."""
        pass

    assert old_func.__name__ == "old_func"
    assert old_func.__doc__ == "Old docstring."


def test_deprecated_function_sets_stability_attribute():
    @deprecated(since="0.19")
    def old_func():
        pass

    assert old_func.__stability__ == "deprecated"
    assert old_func.__deprecated_since__ == "0.19"


def test_deprecated_function_passes_args():
    @deprecated(since="0.19")
    def add(a, b):
        return a + b

    with pytest.warns(DeprecationWarning):
        assert add(3, 4) == 7


# ---------------------------------------------------------------------------
# @deprecated — classes
# ---------------------------------------------------------------------------


def test_deprecated_class_emits_warning_on_instantiation():
    @deprecated(since="0.19", replacement="NewClass")
    class OldClass:
        pass

    with pytest.warns(DeprecationWarning, match="deprecated since v0.19"):
        OldClass()


def test_deprecated_class_sets_stability_attribute():
    @deprecated(since="0.19", replacement="NewClass")
    class OldClass:
        pass

    assert OldClass.__stability__ == "deprecated"
    assert OldClass.__deprecated_since__ == "0.19"
    assert OldClass.__deprecated_replacement__ == "NewClass"


def test_deprecated_class_init_still_runs():
    @deprecated(since="0.19")
    class OldClass:
        def __init__(self, value):
            self.value = value

    with pytest.warns(DeprecationWarning):
        obj = OldClass(99)

    assert obj.value == 99


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------


def test_importable_from_selectools():
    from selectools import beta, deprecated, stable  # noqa: F401
