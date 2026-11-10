"""Unit tests for ParameterValidator."""

from math import inf, nan

import pytest

from curvefit.domain.parameter import ParameterValidator


class TestSymbol:
    """Check validation and normalization of mathematical symbols."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("A", "A"),
            ("k", "k"),
            (r"\mu", r"\mu"),
            ("C_p", "C_p"),
            (r"C_{p}", r"C_{p}"),
            (r"\sigma_1", r"\sigma_1"),
            ("  k  ", "k"),
        ],
    )
    def test_accepts_non_blank_symbols(
        self,
        value,
        expected,
    ) -> None:
        """Check that supported atomic symbols are accepted and trimmed."""
        assert ParameterValidator.symbol(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "",
            " ",
            "\t",
            "\n",
            " \t \n ",
        ],
    )
    def test_rejects_blank_symbols(
        self,
        value,
    ) -> None:
        """Check that symbols containing only whitespace are rejected."""
        with pytest.raises(
            ValueError,
            match="symbol cannot be empty",
        ):
            ParameterValidator.symbol(value)

    @pytest.mark.parametrize(
        "value",
        [
            "A+B",
            "x^2",
            "sin(x)",
            r"\frac{A}{B}",
            "A B",
        ],
    )
    def test_rejects_non_atomic_symbols(
        self,
        value,
    ) -> None:
        """Check that expressions and compound text are not accepted as symbols."""
        with pytest.raises(
            ValueError,
            match="valid atomic mathematical symbol",
        ):
            ParameterValidator.symbol(value)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            1,
            1.0,
            True,
            object(),
        ],
    )
    def test_rejects_non_string_symbols(
        self,
        value,
    ) -> None:
        """Check that values other than strings are rejected as symbols."""
        with pytest.raises(
            ValueError,
            match="symbol must be a string",
        ):
            ParameterValidator.symbol(value)


class TestName:
    """Check validation and normalization of optional descriptive names."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("Viscosidade", "Viscosidade"),
            ("Viscosidade dinâmica", "Viscosidade dinâmica"),
            ("Coeficiente de transferência", "Coeficiente de transferência"),
            ("  Tempo característico  ", "Tempo característico"),
        ],
    )
    def test_accepts_optional_descriptive_names(
        self,
        value,
        expected,
    ) -> None:
        """Check that names are preserved, trimmed, or left unspecified as expected."""
        assert ParameterValidator.name(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "",
            " ",
            "\t",
            "\n",
            " \t \n ",
        ],
    )
    def test_blank_names_become_none(
        self,
        value,
    ) -> None:
        """Check that blank optional names normalize to None."""
        assert ParameterValidator.name(value) is None

    @pytest.mark.parametrize(
        "value",
        [
            1,
            1.0,
            True,
            object(),
        ],
    )
    def test_rejects_non_string_names(
        self,
        value,
    ) -> None:
        """Check that values other than strings or None are rejected as names."""
        with pytest.raises(
            ValueError,
            match="name must be a string or None",
        ):
            ParameterValidator.name(value)


class TestBounds:
    """Check validation and normalization of lower and upper bounds."""

    @pytest.mark.parametrize(
        ("minimum", "maximum", "expected"),
        [
            (0, 1, (0.0, 1.0)),
            (-1, 1, (-1.0, 1.0)),
            (1, 1, (1.0, 1.0)),
            (-inf, inf, (-inf, inf)),
            (-inf, 10, (-inf, 10.0)),
            (0, inf, (0.0, inf)),
        ],
    )
    def test_accepts_valid_bounds(
        self,
        minimum,
        maximum,
        expected,
    ) -> None:
        """Check that ordered finite and infinite bounds are normalized to floats."""
        assert ParameterValidator.bounds(
            minimum,
            maximum,
        ) == expected

    def test_rejects_reversed_bounds(self) -> None:
        """Check that a lower bound greater than the upper bound is rejected."""
        with pytest.raises(
            ValueError,
            match="min_value cannot be greater than max_value",
        ):
            ParameterValidator.bounds(
                10,
                5,
            )

    @pytest.mark.parametrize(
        ("minimum", "maximum"),
        [
            (nan, 1),
            (0, nan),
        ],
    )
    def test_rejects_nan_bounds(
        self,
        minimum,
        maximum,
    ) -> None:
        """Check that NaN is rejected in either bound position."""
        with pytest.raises(
            ValueError,
            match="cannot be NaN",
        ):
            ParameterValidator.bounds(
                minimum,
                maximum,
            )

    @pytest.mark.parametrize(
        ("minimum", "maximum"),
        [
            ("0", 1),
            (0, "1"),
            (True, 1),
            (0, False),
        ],
    )
    def test_rejects_non_numeric_bounds(
        self,
        minimum,
        maximum,
    ) -> None:
        """Check that non-numeric values and booleans are rejected as bounds."""
        with pytest.raises(ValueError):
            ParameterValidator.bounds(
                minimum,
                maximum,
            )


class TestInitialValue:
    """Check validation of the configured starting value."""

    @pytest.mark.parametrize(
        ("value", "minimum", "maximum", "expected"),
        [
            (0, 0, 10, 0.0),
            (5, 0, 10, 5.0),
            (10, 0, 10, 10.0),
            (-5, -10, 0, -5.0),
            (2.5, -inf, inf, 2.5),
        ],
    )
    def test_accepts_values_inside_bounds(
        self,
        value,
        minimum,
        maximum,
        expected,
    ) -> None:
        """Check that finite starting values within the inclusive bounds are accepted."""
        assert ParameterValidator.initial_value(
            value,
            minimum,
            maximum,
        ) == expected

    @pytest.mark.parametrize(
        "value",
        [
            -1,
            11,
        ],
    )
    def test_rejects_values_outside_bounds(
        self,
        value,
    ) -> None:
        """Check that starting values below or above the configured bounds are rejected."""
        with pytest.raises(
            ValueError,
            match="initial_value must be between",
        ):
            ParameterValidator.initial_value(
                value,
                0,
                10,
            )

    @pytest.mark.parametrize(
        "value",
        [
            inf,
            -inf,
        ],
    )
    def test_rejects_infinite_initial_values(
        self,
        value,
    ) -> None:
        """Check that infinite starting values are rejected even with infinite bounds."""
        with pytest.raises(
            ValueError,
            match="initial_value must be finite",
        ):
            ParameterValidator.initial_value(
                value,
                -inf,
                inf,
            )

    def test_rejects_nan_initial_value(self) -> None:
        """Check that NaN is rejected as a starting value."""
        with pytest.raises(
            ValueError,
            match="initial_value cannot be NaN",
        ):
            ParameterValidator.initial_value(
                nan,
                0,
                10,
            )


class TestVary:
    """Check validation of the parameter variation flag."""

    @pytest.mark.parametrize(
        "value",
        [
            True,
            False,
        ],
    )
    def test_accepts_boolean_values(
        self,
        value,
    ) -> None:
        """Check that both boolean values are accepted unchanged."""
        assert ParameterValidator.vary(value) is value

    @pytest.mark.parametrize(
        "value",
        [
            0,
            1,
            "True",
            None,
        ],
    )
    def test_rejects_non_boolean_values(
        self,
        value,
    ) -> None:
        """Check that non-boolean values are rejected as variation flags."""
        with pytest.raises(
            ValueError,
            match="vary must be a boolean",
        ):
            ParameterValidator.vary(value)

class TestUnit:
    """Check validation of an optional parameter unit."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("Pa.s", "Pa.s"),
            ("  kg/m³  ", "kg/m³"),
            ("", None),
            ("   ", None),
        ],
    )
    def test_accepts_optional_units(
        self,
        value,
        expected,
    ) -> None:
        """Check that units are trimmed and blank or omitted units become None."""
        assert ParameterValidator.unit(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            1,
            2.5,
            True,
            object(),
        ],
    )
    def test_rejects_non_string_units(
        self,
        value,
    ) -> None:
        """Check that values other than strings or None are rejected as units."""
        with pytest.raises(
            ValueError,
            match="unit must be a string or None",
        ):
            ParameterValidator.unit(value)


class TestFitState:
    """Check that the validator does not handle fitting results."""

    def test_does_not_validate_fit_values_or_uncertainties(self) -> None:
        """Check that fit values and uncertainties have no validator methods."""
        assert not hasattr(ParameterValidator, "fit_value")
        assert not hasattr(ParameterValidator, "stderr")
