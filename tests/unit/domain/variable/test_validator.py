"""Unit tests for VariableValidator."""

import pytest

from curvefit.domain.variable import VariableValidator


class TestSymbol:
    """Tests for VariableValidator.symbol()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("x", "x"),
            ("T", "T"),
            ("x_1", "x_1"),
            (r"x_{2}", r"x_{2}"),
            (r"\theta", r"\theta"),
            (r"\Delta_T", r"\Delta_T"),
            ("  t  ", "t"),
        ],
    )
    def test_accepts_atomic_symbols(self, value, expected) -> None:
        assert VariableValidator.symbol(value) == expected

    @pytest.mark.parametrize("value", ["", " ", "\t", "\n", " \t \n "])
    def test_rejects_blank_symbols(self, value) -> None:
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            VariableValidator.symbol(value)

    @pytest.mark.parametrize(
        "value",
        ["x+y", "x^2", "sin(x)", r"\frac{x}{t}", "x y"],
    )
    def test_rejects_expressions(self, value) -> None:
        with pytest.raises(
            ValueError,
            match="valid atomic mathematical symbol",
        ):
            VariableValidator.symbol(value)

    @pytest.mark.parametrize("value", [None, 1, 1.0, True, object()])
    def test_rejects_non_string_symbols(self, value) -> None:
        with pytest.raises(ValueError, match="symbol must be a string"):
            VariableValidator.symbol(value)


class TestName:
    """Tests for VariableValidator.name()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("Tempo", "Tempo"),
            ("Temperatura", "Temperatura"),
            ("Pressão média", "Pressão média"),
            ("  Tempo de residência  ", "Tempo de residência"),
            ("", None),
            ("   ", None),
        ],
    )
    def test_accepts_optional_names(self, value, expected) -> None:
        assert VariableValidator.name(value) == expected

    @pytest.mark.parametrize("value", [1, 1.0, True, object()])
    def test_rejects_non_string_names(self, value) -> None:
        with pytest.raises(ValueError, match="name must be a string or None"):
            VariableValidator.name(value)


class TestUnit:
    """Tests for VariableValidator.unit()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("s", "s"),
            ("°C", "°C"),
            ("m/s", "m/s"),
            ("  Pa.s  ", "Pa.s"),
            ("", None),
            ("   ", None),
        ],
    )
    def test_accepts_optional_units(self, value, expected) -> None:
        assert VariableValidator.unit(value) == expected

    @pytest.mark.parametrize("value", [1, 1.0, True, object()])
    def test_rejects_non_string_units(self, value) -> None:
        with pytest.raises(ValueError, match="unit must be a string or None"):
            VariableValidator.unit(value)
