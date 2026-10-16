"""Unit tests for the generic numeric domain validators."""

from math import inf, nan

import pytest

from curvefit.domain.validation import NumericValidator


class TestReal:
    """Tests for NumericValidator.real()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (1, 1.0),
            (-1, -1.0),
            (2.5, 2.5),
            (-3.75, -3.75),
            (inf, inf),
            (-inf, -inf),
        ],
    )
    def test_accepts_real_numbers(self, value, expected) -> None:
        result = NumericValidator.real(value)
        assert result == expected
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "value",
        [True, False, "1.0", None, object()],
    )
    def test_rejects_non_real_values(self, value) -> None:
        with pytest.raises(ValueError, match="must be a real numeric value"):
            NumericValidator.real(value)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="cannot be NaN"):
            NumericValidator.real(nan)

    def test_uses_field_name_in_error_message(self) -> None:
        with pytest.raises(
            ValueError,
            match="initial_value must be a real numeric value",
        ):
            NumericValidator.real(
                "invalid",
                field_name="initial_value",
            )


class TestFinite:
    """Tests for NumericValidator.finite()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (1, 1.0),
            (-1, -1.0),
            (2.5, 2.5),
            (-3.75, -3.75),
        ],
    )
    def test_accepts_finite_values(self, value, expected) -> None:
        assert NumericValidator.finite(value) == expected

    @pytest.mark.parametrize("value", [inf, -inf])
    def test_rejects_infinite_values(self, value) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            NumericValidator.finite(value)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="cannot be NaN"):
            NumericValidator.finite(nan)


class TestNonNegative:
    """Tests for NumericValidator.non_negative()."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0.0), (1, 1.0), (1.5, 1.5)],
    )
    def test_accepts_non_negative_values(self, value, expected) -> None:
        assert NumericValidator.non_negative(value) == expected

    @pytest.mark.parametrize("value", [-1, -0.001])
    def test_rejects_negative_values(self, value) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            NumericValidator.non_negative(value)

    def test_rejects_infinity_by_default(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            NumericValidator.non_negative(inf)

    def test_accepts_positive_infinity_when_allowed(self) -> None:
        result = NumericValidator.non_negative(
            inf,
            finite_only=False,
        )
        assert result == inf

    def test_still_rejects_negative_infinity_when_infinity_allowed(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            NumericValidator.non_negative(
                -inf,
                finite_only=False,
            )


class TestLessThanOrEqual:
    """Tests for NumericValidator.less_than_or_equal()."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (-1, 0),
            (0, 1),
            (1, 1),
            (-inf, inf),
            (-inf, 0),
            (0, inf),
        ],
    )
    def test_accepts_valid_order(self, left, right) -> None:
        result = NumericValidator.less_than_or_equal(left, right)
        assert result == (float(left), float(right))

    def test_rejects_invalid_order(self) -> None:
        with pytest.raises(
            ValueError,
            match="left cannot be greater than right",
        ):
            NumericValidator.less_than_or_equal(2, 1)

    def test_uses_operand_names_in_error_message(self) -> None:
        with pytest.raises(
            ValueError,
            match="min_value cannot be greater than max_value",
        ):
            NumericValidator.less_than_or_equal(
                10,
                5,
                left_name="min_value",
                right_name="max_value",
            )


class TestBetween:
    """Tests for NumericValidator.between()."""

    @pytest.mark.parametrize("value", [0, 5, 10])
    def test_accepts_values_inside_inclusive_interval(self, value) -> None:
        result = NumericValidator.between(value, 0, 10)
        assert result == float(value)

    @pytest.mark.parametrize("value", [-1, 11])
    def test_rejects_values_outside_interval(self, value) -> None:
        with pytest.raises(ValueError, match="must be between"):
            NumericValidator.between(value, 0, 10)

    @pytest.mark.parametrize("value", [0, 10])
    def test_excludes_boundaries_for_open_interval(self, value) -> None:
        with pytest.raises(ValueError, match="must be strictly between"):
            NumericValidator.between(
                value,
                0,
                10,
                inclusive=False,
            )

    def test_accepts_value_inside_open_interval(self) -> None:
        result = NumericValidator.between(
            5,
            0,
            10,
            inclusive=False,
        )
        assert result == 5.0

    def test_accepts_infinite_bounds(self) -> None:
        result = NumericValidator.between(5, -inf, inf)
        assert result == 5.0

    def test_accepts_infinite_value_when_finite_not_required(self) -> None:
        result = NumericValidator.between(inf, 0, inf)
        assert result == inf

    def test_rejects_infinite_value_when_finite_is_required(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            NumericValidator.between(
                inf,
                0,
                inf,
                finite_value=True,
            )

    def test_rejects_invalid_interval(self) -> None:
        with pytest.raises(
            ValueError,
            match="minimum cannot be greater than maximum",
        ):
            NumericValidator.between(5, 10, 0)

    def test_rejects_nan_as_interval_value(self) -> None:
        with pytest.raises(ValueError, match="cannot be NaN"):
            NumericValidator.between(nan, 0, 10)

    def test_rejects_nan_as_minimum(self) -> None:
        with pytest.raises(ValueError, match="cannot be NaN"):
            NumericValidator.between(5, nan, 10)
