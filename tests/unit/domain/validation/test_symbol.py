"""Unit tests for SymbolValidator."""

import pytest

from curvefit.domain.validation import SymbolValidator


class TestValidSymbols:
    """Tests for supported atomic mathematical symbols."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("A", "A"),
            ("k", "k"),
            ("mu", "mu"),
            ("T0", "T0"),
            ("μ", "μ"),
            (r"\mu", r"\mu"),
            (r"\Delta", r"\Delta"),
            ("C_p", "C_p"),
            (r"C_{p}", r"C_{p}"),
            (r"\sigma_1", r"\sigma_1"),
            (r"C_{12}", r"C_{12}"),
            (r"C_{\alpha}", r"C_{\alpha}"),
            ("  k  ", "k"),
        ],
    )
    def test_accepts_atomic_symbols(self, value, expected) -> None:
        assert SymbolValidator.validate(value) == expected


class TestInvalidSymbols:
    """Tests for invalid or non-atomic mathematical symbols."""

    @pytest.mark.parametrize(
        "value",
        [
            "A+B",
            "x^2",
            "sin(x)",
            r"\frac{A}{B}",
            "A B",
            "=x",
            "A/B",
            "C_p_q",
            "_A",
            "1A",
            r"\unknown",
            r"C_{p+1}",
        ],
    )
    def test_rejects_expressions_and_unsupported_syntax(self, value) -> None:
        with pytest.raises(
            ValueError,
            match="valid atomic mathematical symbol",
        ):
            SymbolValidator.validate(value)

    @pytest.mark.parametrize(
        "value",
        ["", " ", "\t", "\n"],
    )
    def test_rejects_blank_symbols(self, value) -> None:
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            SymbolValidator.validate(value)

    @pytest.mark.parametrize(
        "value",
        [None, 1, 1.0, True, object()],
    )
    def test_rejects_non_string_symbols(self, value) -> None:
        with pytest.raises(ValueError, match="symbol must be a string"):
            SymbolValidator.validate(value)

    def test_uses_custom_field_name(self) -> None:
        with pytest.raises(
            ValueError,
            match="variable_symbol must be a valid atomic mathematical symbol",
        ):
            SymbolValidator.validate(
                "x+y",
                field_name="variable_symbol",
            )
