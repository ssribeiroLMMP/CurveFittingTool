"""Unit tests for TextValidator."""

import pytest

from curvefit.domain.validation import TextValidator


class TestRequired:
    """Tests for TextValidator.required()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("A", "A"),
            ("Viscosidade dinâmica", "Viscosidade dinâmica"),
            ("  Tempo  ", "Tempo"),
            ("μ", "μ"),
        ],
    )
    def test_accepts_non_blank_text(self, value, expected) -> None:
        assert TextValidator.required(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["", " ", "\t", "\n", " \t \n "],
    )
    def test_rejects_blank_text(self, value) -> None:
        with pytest.raises(ValueError, match="value cannot be empty"):
            TextValidator.required(value)

    @pytest.mark.parametrize(
        "value",
        [None, 1, 1.0, True, object()],
    )
    def test_rejects_non_string_values(self, value) -> None:
        with pytest.raises(ValueError, match="value must be a string"):
            TextValidator.required(value)

    def test_uses_custom_field_name(self) -> None:
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            TextValidator.required(" ", field_name="symbol")


class TestOptional:
    """Tests for TextValidator.optional()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("Viscosidade", "Viscosidade"),
            ("  Pa.s  ", "Pa.s"),
            ("", None),
            (" \t \n ", None),
        ],
    )
    def test_accepts_optional_text(self, value, expected) -> None:
        assert TextValidator.optional(value) == expected

    @pytest.mark.parametrize(
        "value",
        [1, 1.0, True, object()],
    )
    def test_rejects_non_string_values(self, value) -> None:
        with pytest.raises(
            ValueError,
            match="value must be a string or None",
        ):
            TextValidator.optional(value)

    def test_uses_custom_field_name(self) -> None:
        with pytest.raises(
            ValueError,
            match="unit must be a string or None",
        ):
            TextValidator.optional(10, field_name="unit")
