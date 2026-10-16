"""Unit tests for ParameterValidator."""

from math import inf, nan

import pytest

from curvefit.domain.parameter import ParameterValidator


class TestSymbol:
    """Tests for ParameterValidator.symbol()."""

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
        with pytest.raises(
            ValueError,
            match="symbol must be a string",
        ):
            ParameterValidator.symbol(value)


class TestName:
    """Tests for ParameterValidator.name()."""

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
        with pytest.raises(
            ValueError,
            match="name must be a string or None",
        ):
            ParameterValidator.name(value)


class TestBounds:
    """Tests for ParameterValidator.bounds()."""

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
        assert ParameterValidator.bounds(
            minimum,
            maximum,
        ) == expected

    def test_rejects_reversed_bounds(self) -> None:
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
        with pytest.raises(ValueError):
            ParameterValidator.bounds(
                minimum,
                maximum,
            )


class TestInitialValue:
    """Tests for ParameterValidator.initial_value()."""

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
        with pytest.raises(
            ValueError,
            match="initial_value cannot be NaN",
        ):
            ParameterValidator.initial_value(
                nan,
                0,
                10,
            )


class TestFitValue:
    """Tests for ParameterValidator.fit_value()."""

    @pytest.mark.parametrize(
        ("value", "minimum", "maximum", "expected"),
        [
            (0, 0, 10, 0.0),
            (5, 0, 10, 5.0),
            (10, 0, 10, 10.0),
            (-2.5, -5, 5, -2.5),
            (1.5, -inf, inf, 1.5),
        ],
    )
    def test_accepts_valid_fit_values(
        self,
        value,
        minimum,
        maximum,
        expected,
    ) -> None:
        assert ParameterValidator.fit_value(
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
    def test_rejects_fit_values_outside_bounds(
        self,
        value,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="value must be between",
        ):
            ParameterValidator.fit_value(
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
    def test_rejects_infinite_fit_values(
        self,
        value,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="value must be finite",
        ):
            ParameterValidator.fit_value(
                value,
                -inf,
                inf,
            )


class TestStderr:
    """Tests for ParameterValidator.stderr()."""

    def test_accepts_none(self) -> None:
        assert ParameterValidator.stderr(None) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (0.1, 0.1),
            (5, 5.0),
        ],
    )
    def test_accepts_non_negative_finite_values(
        self,
        value,
        expected,
    ) -> None:
        assert ParameterValidator.stderr(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            -1,
            -0.001,
        ],
    )
    def test_rejects_negative_values(
        self,
        value,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="stderr cannot be negative",
        ):
            ParameterValidator.stderr(value)

    @pytest.mark.parametrize(
        "value",
        [
            inf,
            -inf,
        ],
    )
    def test_rejects_infinite_values(
        self,
        value,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="stderr must be finite",
        ):
            ParameterValidator.stderr(value)

    def test_rejects_nan(self) -> None:
        with pytest.raises(
            ValueError,
            match="stderr cannot be NaN",
        ):
            ParameterValidator.stderr(nan)


class TestVary:
    """Tests for ParameterValidator.vary()."""

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
        with pytest.raises(
            ValueError,
            match="vary must be a boolean",
        ):
            ParameterValidator.vary(value)



class TestUnit:
    """Tests for ParameterValidator.unit()."""

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
        with pytest.raises(
            ValueError,
            match="unit must be a string or None",
        ):
            ParameterValidator.unit(value)
